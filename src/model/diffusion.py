import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from transformers import CLIPTokenizer, CLIPTextModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm.auto import tqdm

# Assuming VQ-VAE implementation in vqvae.py
from vq_vae import VQVAE, VQVAEConfig, load_best_model

LABEL_MAP = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine"
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class DiffusionConfig:
    image_size: int = 28
    channels: int = 64
    latent_channels: int = 64  # Add this to specify VQ-VAE latent dimensions
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 128
    lr: float = 1e-4
    epochs: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir: str = './diffusion_checkpoints'
    # Early stopping parameters
    patience: int = 10
    min_delta: float = 1e-6
    validation_freq: int = 1  # Validate every N epochs
    # Text conditioning flag
    use_text_conditioning: bool = True

@dataclass
class OptunaConfig:
    n_trials: int = 100
    study_name: str = "diffusion_optimization"
    storage: str = "sqlite:///diffusion_optuna.db"
    max_epochs_trial: int = 5  # Shorter epochs for hyperparameter search
    patience_trial: int = 5     # Shorter patience for trials

class FiLM(nn.Module):
    """Produce γ (scale) and β (shift) from text embeddings."""
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, num_features * 2),
            nn.SiLU(),
            nn.Linear(num_features * 2, num_features * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: [B, C, H, W]; cond: [B, cond_dim]
        b, c, h, w = x.shape
        gamma, beta = self.net(cond).chunk(2, dim=1)
        gamma = gamma.view(b, c, 1, 1)
        beta  = beta.view(b, c, 1, 1)
        return x * (1 + gamma) + beta


class NoiseSchedule(ABC):
    @abstractmethod
    def betas(self, timesteps: int) -> torch.Tensor:
        """Generates beta schedule."""
        pass

class LinearSchedule(NoiseSchedule):
    def __init__(self, beta_start: float, beta_end: float):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def betas(self, timesteps: int) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, timesteps)

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding module"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class UNetBlock(nn.Module):
    """Non-conditional UNet block with time conditioning"""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, out_ch * 2),
            nn.SiLU(),
            nn.Linear(out_ch * 2, out_ch * 2)
        )
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        # First conv
        x = self.act1(self.norm1(self.conv1(x)))
        
        # Apply time conditioning
        time_scale, time_shift = self.time_proj(time_emb).chunk(2, dim=1)
        time_scale = time_scale.view(-1, x.shape[1], 1, 1)
        time_shift = time_shift.view(-1, x.shape[1], 1, 1)
        x = x * (1 + time_scale) + time_shift
        
        # Second conv
        x = self.act2(self.norm2(self.conv2(x)))
        return x
    
class ConditionalUNetBlock(nn.Module):
    """Conditional UNet block with FiLM conditioning and time embedding"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, time_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act  = nn.SiLU()
        self.film = FiLM(cond_dim, out_ch)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, out_ch * 2),
            nn.SiLU(),
            nn.Linear(out_ch * 2, out_ch * 2)
        )
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2  = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor):
        # First conv with text conditioning
        x = self.conv(x)
        x = self.act(self.film(self.norm(x), cond))
        
        # Apply time conditioning
        time_scale, time_shift = self.time_proj(time_emb).chunk(2, dim=1)
        time_scale = time_scale.view(-1, x.shape[1], 1, 1)
        time_shift = time_shift.view(-1, x.shape[1], 1, 1)
        x = x * (1 + time_scale) + time_shift
        
        # Second conv with text conditioning
        x = self.conv2(x)
        x = self.act2(self.film(self.norm2(x), cond))
        return x

class UNetModel(nn.Module):
    """
    A simple U-Net for diffusion, conditioned on timestep embeddings.
    """
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        c = config.channels
        time_dim = c * 4  # Time embedding dimension
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Input projection layer to match VQ-VAE latent dimensions
        self.input_proj = nn.Conv2d(config.latent_channels, c, 1)
        
        # Down
        self.down1 = UNetBlock(c, c, time_dim)
        self.down2 = UNetBlock(c, c * 2, time_dim)
        self.pool = nn.AvgPool2d(2)
        # Bottleneck
        self.bot = UNetBlock(c * 2, c * 2, time_dim)
        # Up
        self.up1 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec1 = UNetBlock(c * 4, c * 2, time_dim)  # c*2 (up1) + c*2 (d2) = c*4
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec2 = UNetBlock(c * 2, c, time_dim)     # c (up2) + c (d1) = c*2
        # Final output projection back to latent space
        self.out = nn.Conv2d(c, config.latent_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, latent_channels, H, W], t: [B]
        # Embed timesteps
        time_emb = self.time_embedding(t)  # [B, time_dim]
        
        # Project input to model dimensions
        x = self.input_proj(x)  # [B, c, H, W]
        
        d1 = self.down1(x, time_emb)           # [B, c, H, W]
        d2 = self.down2(self.pool(d1), time_emb)  # [B, c*2, H/2, W/2]
        b = self.bot(self.pool(d2), time_emb)     # [B, c*2, H/4, W/4]
        
        u1 = self.up1(b)            # [B, c*2, H/2, W/2]
        # Ensure u1 and d2 have the same spatial dimensions
        if u1.shape[2:] != d2.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d2], dim=1)  # [B, c*4, H/2, W/2]
        u1 = self.dec1(u1, time_emb)          # [B, c*2, H/2, W/2]
        
        u2 = self.up2(u1)           # [B, c, H, W]
        # Ensure u2 and d1 have the same spatial dimensions
        if u2.shape[2:] != d1.shape[2:]:
            u2 = nn.functional.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d1], dim=1)  # [B, c*2, H, W]
        u2 = self.dec2(u2, time_emb)          # [B, c, H, W]
        
        # Project back to latent space
        out = self.out(u2)          # [B, latent_channels, H, W]
        return out
    
class ConditionalUNetModel(nn.Module):
    def __init__(self, config: DiffusionConfig, text_emb_dim: int):
        super().__init__()
        c = config.channels
        time_dim = c * 4  # Time embedding dimension
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        self.input_proj = nn.Conv2d(config.latent_channels, c, 1)
        # load text encoder
        self.tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze CLIP parameters
        self.text_encoder.requires_grad_(False)
        # pass cond_dim = text_embedding_dim
        cond_dim = self.text_encoder.config.hidden_size
        
        # down blocks now take cond_dim and time_dim - using ConditionalUNetBlock
        self.down1 = ConditionalUNetBlock(c,    c,    cond_dim, time_dim)
        self.down2 = ConditionalUNetBlock(c,  2*c,    cond_dim, time_dim)
        self.pool  = nn.AvgPool2d(2)
        self.bot   = ConditionalUNetBlock(2*c,2*c,    cond_dim, time_dim)
        self.up1   = nn.ConvTranspose2d(2*c,2*c,2, stride=2)
        self.dec1  = ConditionalUNetBlock(4*c,2*c,    cond_dim, time_dim)
        self.up2   = nn.ConvTranspose2d(2*c,  c,2, stride=2)
        self.dec2  = ConditionalUNetBlock(2*c,  c,    cond_dim, time_dim)
        self.out   = nn.Conv2d(c, config.latent_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, texts: List[str]) -> torch.Tensor:
        # 1) encode text
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(x.device)
        text_emb = self.text_encoder(**tokens).last_hidden_state[:,0]  # [B, cond_dim]
        
        # 2) embed timesteps
        time_emb = self.time_embedding(t)  # [B, time_dim]
        
        # 3) standard UNet but pass text_emb and time_emb into each block
        x = self.input_proj(x)
        d1 = self.down1(x,         text_emb, time_emb)
        d2 = self.down2(self.pool(d1), text_emb, time_emb)
        b  = self.bot(self.pool(d2),    text_emb, time_emb)
        
        u1 = self.up1(b)
        if u1.shape[-2:] != d2.shape[-2:]:
            u1 = F.interpolate(u1, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.dec1(torch.cat([u1,d2],1), text_emb, time_emb)
        
        u2 = self.up2(u1)
        if u2.shape[-2:] != d1.shape[-2:]:
            u2 = F.interpolate(u2, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.dec2(torch.cat([u2,d1],1), text_emb, time_emb)
        
        return self.out(u2)

class GaussianDiffusion:
    """Non-conditional diffusion for regular UNetModel"""
    def __init__(self, config: DiffusionConfig, schedule: NoiseSchedule):
        self.timesteps = config.timesteps
        self.betas = schedule.betas(self.timesteps).to(config.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.device = config.device

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data (forward process): q(x_t | x_0)
        """
        sqrt_alpha_hat = self.alpha_hat[t]**0.5
        sqrt_one_minus = (1 - self.alpha_hat[t])**0.5
        return sqrt_alpha_hat.view(-1, 1, 1, 1) * x_start + sqrt_one_minus.view(-1, 1, 1, 1) * noise

    def p_loss(self, model: nn.Module, x_start: torch.Tensor) -> torch.Tensor:
        """Non-conditional loss function"""
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = model(x_noisy, t)  # Only 2 arguments for non-conditional model
        return nn.functional.mse_loss(predicted, noise)

    def sample(self, model: nn.Module, shape: List[int]) -> torch.Tensor:
        """Generate samples from model (reverse process)"""
        model.eval()
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            x = (1 / alpha**0.5) * (x - (beta / (1 - alpha_hat)**0.5) * pred_noise)
            if i > 0:
                x += (beta**0.5) * torch.randn_like(x)
        return x

class ConditionalGaussianDiffusion:
    """Conditional diffusion for ConditionalUNetModel"""
    def __init__(self, config: DiffusionConfig, schedule: NoiseSchedule):
        self.timesteps = config.timesteps
        self.betas = schedule.betas(self.timesteps).to(config.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.device = config.device

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse the data (forward process): q(x_t | x_0)
        """
        sqrt_alpha_hat = self.alpha_hat[t]**0.5
        sqrt_one_minus = (1 - self.alpha_hat[t])**0.5
        return sqrt_alpha_hat.view(-1, 1, 1, 1) * x_start + sqrt_one_minus.view(-1, 1, 1, 1) * noise

    def p_loss(self, model: nn.Module, x_start: torch.Tensor, text_labels: List[str]) -> torch.Tensor:
        """Conditional loss function"""
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = model(x_noisy, t, text_labels)  # 3 arguments for conditional model
        return nn.functional.mse_loss(predicted, noise)

    def sample(self, model: nn.Module, shape: List[int], text_labels: List[str]) -> torch.Tensor:
        """Generate samples from model (reverse process)"""
        model.eval()
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t, text_labels)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_hat = self.alpha_hat[i]
            x = (1 / alpha**0.5) * (x - (beta / (1 - alpha_hat)**0.5) * pred_noise)
            if i > 0:
                x += (beta**0.5) * torch.randn_like(x)
        return x

class RawDataset(Dataset):
    def __init__(self, base_dataset: Dataset, label_map: dict):
        self.dataset   = base_dataset
        self.label_map = label_map  # e.g. {0:"zero", …, 9:"nine"}
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.label_map[int(y)]
    def __len__(self):
        return len(self.dataset)

class Trainer:
    """Template Method pattern for training."""
    def __init__(self, model: nn.Module, diffusion: GaussianDiffusion,
                 optimizer: optim.Optimizer, train_loader: DataLoader,
                 val_loader: DataLoader, config: DiffusionConfig, vqvae: VQVAE):
        self.model = model.to(config.device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vqvae = vqvae.to(config.device)
        self.vqvae.eval()
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # Compute actual latent spatial dimensions from VQ-VAE
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, config.image_size, config.image_size, device=config.device)
            z_e = self.vqvae.encode(dummy_input)
            self.latent_h, self.latent_w = z_e.shape[-2:]
        
        logger.info(f"VQ-VAE latent dimensions: {self.latent_h}x{self.latent_w}")

        # Sample generation setup
        self.sample_prompts = ["zero", "one", "two", "three", "four"]  # Sample prompts for generation
        self.samples_dir = os.path.join(config.checkpoint_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def generate_samples(self, epoch: int, num_samples: int = 5, prefix: str = "epoch"):
        """Generate sample images and save them"""
        try:
            self.model.eval()
            self.vqvae.eval()  # Ensure VQ-VAE stays in eval mode
            with torch.no_grad():
                # Use correct latent dimensions from VQ-VAE
                latent_shape = [num_samples, self.config.latent_channels, self.latent_h, self.latent_w]
                
                # Generate latent samples
                latent_samples = self.diffusion.sample(self.model, latent_shape)
                
                # Decode with VQ-VAE
                reconstructed_images = self.vqvae.decode(latent_samples)
                
                # Save images
                sample_path = os.path.join(self.samples_dir, f"{prefix}_{epoch:03d}_samples.png")
                vutils.save_image(reconstructed_images, sample_path, nrow=num_samples, normalize=True, 
                                value_range=(-1, 1) if reconstructed_images.min() < 0 else (0, 1))
                
                logger.info(f"Generated samples saved to: {sample_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate samples: {e}")
        finally:
            self.model.train()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            if epoch % self.config.validation_freq == 0:
                val_loss = self._validate_epoch(epoch)
                
                # Generate samples every few epochs
                if epoch % 5 == 0 or epoch == 1:
                    self.generate_samples(epoch)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_checkpoint(epoch, val_loss)
                    logger.info(f"New best validation loss: {val_loss:.6f}")
                    # Generate samples for best model
                    self.generate_samples(epoch, prefix="best")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement. Patience: {self.patience_counter}/{self.config.patience}")
                
                # Early stopping check
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered. Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.6f}")
                    break
            
            # Save regular checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss if epoch % self.config.validation_freq == 0 else None)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.vqvae.eval()  # Ensure VQ-VAE stays in eval mode
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        total_loss = 0.0
        num_batches = 0
        
        for raw_images, text_labels in pbar:
            raw_images = raw_images.to(self.config.device)
            
            # Encode to latents using consistent VQ-VAE API
            with torch.no_grad():
                z_e = self.vqvae.encode(raw_images)  # Use consistent encode() API
                z_q, _ = self.vqvae.vector_quantizer(z_e)
            
            self.optimizer.zero_grad()
            loss = self.diffusion.p_loss(self.model, z_q)  # Non-conditional training
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / num_batches
        logger.info(f"Train Loss: {avg_loss:.6f}")
        return avg_loss

    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        self.vqvae.eval()  # Ensure VQ-VAE stays in eval mode
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation", leave=False)
            for raw_images, text_labels in pbar:
                raw_images = raw_images.to(self.config.device)
                
                # Encode to latents using consistent VQ-VAE API
                z_e = self.vqvae.encode(raw_images)  # Consistent with training
                z_q, _ = self.vqvae.vector_quantizer(z_e)
                
                loss = self.diffusion.p_loss(self.model, z_q)  # Non-conditional validation
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(val_loss=loss.item())
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, f"diffusion_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def _save_best_checkpoint(self, epoch: int, val_loss: float):
        # Save best checkpoint with only essential data, not config object
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            # Save config as dict instead of object
            'config_dict': {
                'image_size': self.config.image_size,
                'channels': self.config.channels,
                'latent_channels': self.config.latent_channels,
                'timesteps': self.config.timesteps,
                'beta_start': self.config.beta_start,
                'beta_end': self.config.beta_end,
                'batch_size': self.config.batch_size,
                'lr': self.config.lr,
                'device': self.config.device,
            }
        }
        path = os.path.join(self.config.checkpoint_dir, "best_diffusion_model.pth")
        torch.save(checkpoint, path)
        logger.info(f"Saved best checkpoint: {path}")

    def load_best_checkpoint(self):
        """Load the best saved checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, "best_diffusion_model.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.6f}")
            return checkpoint
        else:
            logger.warning("No best checkpoint found")
            return None
        
class ConditionalTrainer(Trainer):
    """Template Method pattern for conditional training."""
    def __init__(self, model: nn.Module, diffusion: ConditionalGaussianDiffusion,
                 optimizer: optim.Optimizer, train_loader: DataLoader,
                 val_loader: DataLoader, config: DiffusionConfig, vqvae: VQVAE):
        self.model = model.to(config.device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vqvae = vqvae.to(config.device)
        self.vqvae.eval()
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # Compute actual latent spatial dimensions from VQ-VAE
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, config.image_size, config.image_size, device=config.device)
            z_e = self.vqvae.encode(dummy_input)
            self.latent_h, self.latent_w = z_e.shape[-2:]
        
        logger.info(f"VQ-VAE latent dimensions: {self.latent_h}x{self.latent_w}")

        # Sample generation setup for conditional model
        self.sample_prompts = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.samples_dir = os.path.join(config.checkpoint_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def generate_samples(self, epoch: int, num_samples: int = 10, prefix: str = "epoch"):
        """Generate conditional sample images and save them"""
        try:
            self.model.eval()
            self.vqvae.eval()  # Ensure VQ-VAE stays in eval mode
            with torch.no_grad():
                # Use subset of prompts based on num_samples
                prompts = self.sample_prompts[:num_samples]
                batch_size = len(prompts)
                
                # Use correct latent dimensions from VQ-VAE
                latent_shape = [batch_size, self.config.latent_channels, self.latent_h, self.latent_w]
                
                # Generate conditional latent samples
                latent_samples = self.diffusion.sample(self.model, latent_shape, prompts)
                
                # Decode with VQ-VAE
                reconstructed_images = self.vqvae.decode(latent_samples)
                
                # Save images with labels
                sample_path = os.path.join(self.samples_dir, f"{prefix}_{epoch:03d}_conditional_samples.png")
                vutils.save_image(reconstructed_images, sample_path, nrow=5, normalize=True,
                                value_range=(-1, 1) if reconstructed_images.min() < 0 else (0, 1))
                
                logger.info(f"Generated conditional samples saved to: {sample_path}")
                logger.info(f"Prompts used: {prompts}")
                
        except Exception as e:
            logger.warning(f"Failed to generate conditional samples: {e}")
        finally:
            self.model.train()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            if epoch % self.config.validation_freq == 0:
                val_loss = self._validate_epoch(epoch)
                
                # Generate samples every few epochs or at the beginning
                if epoch % 5 == 0 or epoch == 1:
                    self.generate_samples(epoch)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_checkpoint(epoch, val_loss)
                    logger.info(f"New best validation loss: {val_loss:.6f}")
                    # Generate samples for best model
                    self.generate_samples(epoch, prefix="best")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement. Patience: {self.patience_counter}/{self.config.patience}")
                
                # Early stopping check
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered. Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.6f}")
                    # Generate final samples before stopping
                    self.generate_samples(epoch, prefix="final")
                    break
            
            # Save regular checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss if epoch % self.config.validation_freq == 0 else None)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.vqvae.eval()  # Ensure VQ-VAE stays in eval mode
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        total_loss = 0.0
        num_batches = 0
        
        for raw_images, text_labels in pbar:
            raw_images = raw_images.to(self.config.device)
            
            # Encode to latents using consistent VQ-VAE API
            with torch.no_grad():
                z_e = self.vqvae.encode(raw_images)  # Use consistent encode() API
                z_q, _ = self.vqvae.vector_quantizer(z_e)
            
            self.optimizer.zero_grad()
            loss = self.diffusion.p_loss(self.model, z_q, text_labels)  # Conditional training with text_labels
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / num_batches
        logger.info(f"Train Loss: {avg_loss:.6f}")
        return avg_loss

    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        self.vqvae.eval()  # Ensure VQ-VAE stays in eval mode
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation", leave=False)
            for raw_images, text_labels in pbar:
                raw_images = raw_images.to(self.config.device)
                
                # Encode to latents using consistent VQ-VAE API
                z_e = self.vqvae.encode(raw_images)  # Consistent with training
                z_q, _ = self.vqvae.vector_quantizer(z_e)
                
                loss = self.diffusion.p_loss(self.model, z_q, text_labels)  # Conditional validation with text_labels
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(val_loss=loss.item())
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, f"conditional_diffusion_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def _save_best_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, "best_conditional_diffusion_model.pth")
        torch.save(checkpoint, path)
        logger.info(f"Saved best checkpoint: {path}")

    def load_best_checkpoint(self):
        """Load the best saved checkpoint"""
        path = os.path.join(self.config.checkpoint_dir, "best_conditional_diffusion_model.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.6f}")
            return checkpoint
        else:
            logger.warning("No best checkpoint found")
            return None

class OptunaTrainer(Trainer):
    """Extended trainer for Optuna optimization"""
    
    def __init__(self, model: nn.Module, diffusion: GaussianDiffusion,
                 optimizer: optim.Optimizer, train_loader: DataLoader,
                 val_loader: DataLoader, config: DiffusionConfig, vqvae: VQVAE,
                 trial: optuna.Trial = None):
        super().__init__(model, diffusion, optimizer, train_loader, val_loader, config, vqvae)
        self.trial = trial
    
    def train_for_optuna(self) -> float:
        """Train for a limited number of epochs and return best validation loss"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            val_loss = self._validate_epoch(epoch)
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Report intermediate value to Optuna
            if self.trial:
                self.trial.report(val_loss, epoch)
                
                # Handle pruning based on the intermediate value
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping for trials
            if patience_counter >= self.config.patience:
                logger.info(f"Trial early stopping at epoch {epoch}")
                break
        
        # Generate samples at the end of trial
        try:
            self.generate_samples(self.trial.number if self.trial else 0, num_samples=5, prefix="trial")
        except Exception as e:
            logger.warning(f"Failed to generate trial samples: {e}")
        
        return best_val_loss
    
    # Override save methods to prevent checkpoint saving during trials
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float = None):
        """Override to disable checkpoint saving during trials"""
        pass
    
    def _save_best_checkpoint(self, epoch: int, val_loss: float):
        """Override to disable best checkpoint saving during trials"""
        pass

class ConditionalOptunaTrainer(ConditionalTrainer):
    """Extended conditional trainer for Optuna optimization"""
    
    def __init__(self, model: nn.Module, diffusion: ConditionalGaussianDiffusion,
                 optimizer: optim.Optimizer, train_loader: DataLoader,
                 val_loader: DataLoader, config: DiffusionConfig, vqvae: VQVAE,
                 trial: optuna.Trial = None):
        super().__init__(model, diffusion, optimizer, train_loader, val_loader, config, vqvae)
        self.trial = trial
    
    def train_for_optuna(self) -> float:
        """Train for a limited number of epochs and return best validation loss"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            val_loss = self._validate_epoch(epoch)
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Report intermediate value to Optuna
            if self.trial:
                self.trial.report(val_loss, epoch)
                
                # Handle pruning based on the intermediate value
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Early stopping for trials
            if patience_counter >= self.config.patience:
                logger.info(f"Trial early stopping at epoch {epoch}")
                break
        
        # Generate conditional samples at the end of trial
        try:
            trial_num = self.trial.number if self.trial else 0
            self.generate_samples(trial_num, num_samples=5, prefix="trial")
            logger.info(f"Generated samples for trial {trial_num}")
        except Exception as e:
            logger.warning(f"Failed to generate trial samples: {e}")
        
        return best_val_loss
    
    # Override save methods to prevent checkpoint saving during trials
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float = None):
        """Override to disable checkpoint saving during trials"""
        pass
    
    def _save_best_checkpoint(self, epoch: int, val_loss: float):
        """Override to disable best checkpoint saving during trials"""
        pass

def create_config_from_trial(trial: optuna.Trial, base_config: DiffusionConfig) -> DiffusionConfig:
    """Create a DiffusionConfig based on Optuna trial suggestions"""
    
    # Suggest hyperparameters
    channels = trial.suggest_categorical('channels', [32, 64, 128, 256])
    timesteps = trial.suggest_categorical('timesteps', [500, 1000, 1500, 2000])
    beta_start = trial.suggest_float('beta_start', 1e-5, 1e-3, log=True)
    beta_end = trial.suggest_float('beta_end', 0.01, 0.05)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    # Create new config with suggested parameters
    config = DiffusionConfig(
        image_size=base_config.image_size,
        channels=channels,
        latent_channels=base_config.latent_channels,  # Use base_config's latent_channels
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        batch_size=batch_size,
        lr=lr,
        epochs=3,  # Even shorter for faster trials
        device=base_config.device,
        checkpoint_dir=f"./optuna_trials/trial_{trial.number}",
        patience=2,  # Very short patience for trials
        min_delta=base_config.min_delta,
        validation_freq=base_config.validation_freq
    )
    
    return config

def objective(trial: optuna.Trial, vqvae: VQVAE, train_ds: Dataset, val_ds: Dataset, base_config: DiffusionConfig) -> float:
    """Optuna objective function"""
    
    # Create config from trial
    config = create_config_from_trial(trial, base_config)
    
    # Create data loaders with trial-specific batch size
    train_raw_ds = RawDataset(train_ds, LABEL_MAP)
    val_raw_ds = RawDataset(val_ds, LABEL_MAP)
    
    train_loader = DataLoader(train_raw_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_raw_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Build model and diffusion based on conditioning flag
    scheduling = LinearSchedule(config.beta_start, config.beta_end)
    
    if config.use_text_conditioning:
        diffusion = ConditionalGaussianDiffusion(config, scheduling)
        model = ConditionalUNetModel(config, text_emb_dim=512)
    else:
        diffusion = GaussianDiffusion(config, scheduling)
        model = UNetModel(config)
    
    # Create optimizer after model is defined
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Create trainer based on conditioning flag
    if config.use_text_conditioning:
        trainer = ConditionalOptunaTrainer(model, diffusion, optimizer, train_loader, val_loader, config, vqvae, trial)
    else:
        trainer = OptunaTrainer(model, diffusion, optimizer, train_loader, val_loader, config, vqvae, trial)
    
    # Create trial directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Train with appropriate trainer
    try:
        best_val_loss = trainer.train_for_optuna()
        return best_val_loss
    except optuna.TrialPruned:
        # Clean up trial directory if pruned
        import shutil
        if os.path.exists(config.checkpoint_dir):
            shutil.rmtree(config.checkpoint_dir)
        raise

def optimize_hyperparameters(vqvae: VQVAE, train_ds: Dataset, val_ds: Dataset, optuna_config: OptunaConfig, vq_embedding_dim: int) -> DiffusionConfig:
    """Run Optuna optimization to find best hyperparameters"""
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name=optuna_config.study_name,
        storage=optuna_config.storage,
        load_if_exists=False,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)  # Reduced warmup
    )
    
    # Create base config with correct embedding dimensions
    base_config = DiffusionConfig(latent_channels=vq_embedding_dim)
    
    # Optimize
    logger.info(f"Starting Optuna optimization with {optuna_config.n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, vqvae, train_ds, val_ds, base_config),
        n_trials=optuna_config.n_trials,
        timeout=None,
        n_jobs=1  # Keep as 1 to avoid CUDA conflicts
    )
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_value:.6f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Create best config
    best_config = DiffusionConfig(
        image_size=28,
        channels=best_params['channels'],
        latent_channels=vq_embedding_dim,  # Use the passed embedding dimension
        timesteps=best_params['timesteps'],
        beta_start=best_params['beta_start'],
        beta_end=best_params['beta_end'],
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        epochs=100,  # Full epochs for final training
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='./best_diffusion_checkpoints',
        patience=15,  # More patience for final training
        min_delta=1e-6,
        validation_freq=1
    )
    
    return best_config


def train_final_model(best_config: DiffusionConfig, vqvae: VQVAE, train_ds: Dataset, val_ds: Dataset):
    """Train the final model with the best configuration"""
    
    logger.info("Starting final training with best configuration")
    logger.info(f"Best config: {best_config}")
    logger.info(f"Text conditioning enabled: {best_config.use_text_conditioning}")
    
    # Create data loaders
    train_raw_ds = RawDataset(train_ds, LABEL_MAP)
    val_raw_ds = RawDataset(val_ds, LABEL_MAP)
    
    train_loader = DataLoader(train_raw_ds, batch_size=best_config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_raw_ds, batch_size=best_config.batch_size, shuffle=False, num_workers=0)
    
    # Build model and diffusion based on conditioning flag
    scheduling = LinearSchedule(best_config.beta_start, best_config.beta_end)
    
    if best_config.use_text_conditioning:
        logger.info("Training conditional diffusion model")
        diffusion = ConditionalGaussianDiffusion(best_config, scheduling)
        model = ConditionalUNetModel(best_config, text_emb_dim=512)
    else:
        logger.info("Training non-conditional diffusion model")
        diffusion = GaussianDiffusion(best_config, scheduling)
        model = UNetModel(best_config)
    
    # Create optimizer after model is defined
    optimizer = optim.Adam(model.parameters(), lr=best_config.lr)
    
    # Create trainer based on conditioning flag
    if best_config.use_text_conditioning:
        trainer = ConditionalTrainer(model, diffusion, optimizer, train_loader, val_loader, best_config, vqvae)
    else:
        trainer = Trainer(model, diffusion, optimizer, train_loader, val_loader, best_config, vqvae)
    
    # Generate initial samples before training
    trainer.generate_samples(0, prefix="initial")
    
    trainer.train()
    
    # Generate final samples after training
    trainer.generate_samples(999, prefix="final_complete")
    
    logger.info("Final training completed")

def load_config_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> DiffusionConfig:
    """Load DiffusionConfig from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config_dict' in checkpoint:
        config_dict = checkpoint['config_dict']
        return DiffusionConfig(
            image_size=config_dict['image_size'],
            channels=config_dict['channels'],
            latent_channels=config_dict['latent_channels'],
            timesteps=config_dict['timesteps'],
            beta_start=config_dict['beta_start'],
            beta_end=config_dict['beta_end'],
            batch_size=config_dict['batch_size'],
            lr=config_dict['lr'],
            device=device
        )
    else:
        # Fallback for old checkpoints
        return checkpoint.get('config', None)
    
def generate_samples(model_path: str, vqvae: VQVAE, text_prompts: List[str] = None, device: str = 'cuda'):
    # Load config from checkpoint
    config = load_config_from_checkpoint(model_path, device)
    if config is None:
        raise ValueError("Could not load config from checkpoint")
    
    # Compute actual latent dimensions from VQ-VAE
    vqvae.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, config.image_size, config.image_size, device=device)
        z_e = vqvae.encode(dummy_input)
        latent_h, latent_w = z_e.shape[-2:]
    
    # Create diffusion and model based on conditioning
    scheduling = LinearSchedule(config.beta_start, config.beta_end)
    
    if config.use_text_conditioning and text_prompts is not None:
        logger.info("Generating conditional samples")
        diffusion = ConditionalGaussianDiffusion(config, scheduling)
        model = ConditionalUNetModel(config, text_emb_dim=512).to(device)
        batch_size = len(text_prompts)
    else:
        logger.info("Generating non-conditional samples")
        diffusion = GaussianDiffusion(config, scheduling)
        model = UNetModel(config).to(device)
        batch_size = 10 if text_prompts is None else len(text_prompts)
        text_prompts = None  # Disable text prompts for non-conditional
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate samples with correct latent dimensions
    latent_shape = [batch_size, config.latent_channels, latent_h, latent_w]
    
    with torch.no_grad():
        # Generate latent samples
        if config.use_text_conditioning and text_prompts is not None:
            latent_samples = diffusion.sample(model, latent_shape, text_prompts)
        else:
            latent_samples = diffusion.sample(model, latent_shape)
        
        # Decode with VQ-VAE
        reconstructed_images = vqvae.decode(latent_samples)
    
    return reconstructed_images, latent_samples

def main():
    # Load VQ-VAE
    vqvae, vq_cfg = load_best_model('/home/nair-group/abian_torres/repositories/tool_generation/final_model_checkpoints')
    vqvae.eval()
    print(f"Loaded model with embedding_dim: {vq_cfg.embedding_dim}")

    # Set seed
    set_seed()

    # Load datasets
    transform = transforms.Compose([transforms.ToTensor()])
    base_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split into train/validation
    train_size = int(0.8 * len(base_ds))
    val_size = len(base_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(base_ds, [train_size, val_size])

    # Configuration flag - SET THIS TO CONTROL TEXT CONDITIONING
    USE_TEXT_CONDITIONING = False  # Change to False to disable text conditioning
    
    # Create base config with VQ-VAE latent dimensions and conditioning flag
    base_config = DiffusionConfig(
        latent_channels=vq_cfg.embedding_dim,
        use_text_conditioning=USE_TEXT_CONDITIONING
    )

    # Optuna configuration - adjust study name based on conditioning
    study_name = "conditional_diffusion_hyperopt" if USE_TEXT_CONDITIONING else "unconditional_diffusion_hyperopt"
    storage_name = "conditional_diffusion_optuna.db" if USE_TEXT_CONDITIONING else "unconditional_diffusion_optuna.db"
    
    optuna_config = OptunaConfig(
        n_trials=50,
        study_name=study_name,
        storage=f"sqlite:///{storage_name}"
    )
    
    # Phase 1: Hyperparameter optimization
    conditioning_type = "conditional" if USE_TEXT_CONDITIONING else "unconditional"
    logger.info(f"Phase 1: Hyperparameter optimization for {conditioning_type} diffusion")
    best_config = optimize_hyperparameters(vqvae, train_ds, val_ds, optuna_config, vq_cfg.embedding_dim)
    
    # Ensure the conditioning flag is preserved in best_config
    best_config.use_text_conditioning = USE_TEXT_CONDITIONING
    best_config.checkpoint_dir = f'./best_{conditioning_type}_diffusion_checkpoints'
    
    # Phase 2: Final training with best configuration
    logger.info(f"Phase 2: Final training with best configuration ({conditioning_type})")
    train_final_model(best_config, vqvae, train_ds, val_ds)
    
    # Phase 3: Generate some samples
    logger.info("Phase 3: Generating samples")
    model_filename = f"best_{'conditional' if USE_TEXT_CONDITIONING else 'unconditional'}_diffusion_model.pth"
    model_path = os.path.join(best_config.checkpoint_dir, model_filename)
    
    if os.path.exists(model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if USE_TEXT_CONDITIONING:
            text_prompts = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            images, latents = generate_samples(model_path, vqvae, text_prompts, device)
            output_filename = 'generated_conditional_samples.png'
        else:
            images, latents = generate_samples(model_path, vqvae, None, device)
            output_filename = 'generated_unconditional_samples.png'
        
        # Save generated images
        import torchvision.utils as vutils
        vutils.save_image(images, output_filename, nrow=5, normalize=True)
        logger.info(f"Generated samples saved to '{output_filename}'")
    else:
        logger.warning("No trained model found for sample generation")

if __name__ == "__main__":
    main()