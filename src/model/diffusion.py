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

class UNetBlock(nn.Module):
    """Non-conditional UNet block for regular diffusion"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
class ConditionalUNetBlock(nn.Module):
    """Conditional UNet block with FiLM conditioning"""
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act  = nn.SiLU()
        self.film = FiLM(cond_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2  = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.conv(x)
        x = self.act(self.film(self.norm(x), cond))
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
        # Input projection layer to match VQ-VAE latent dimensions
        self.input_proj = nn.Conv2d(config.latent_channels, c, 1)
        
        # Down
        self.down1 = UNetBlock(c, c)
        self.down2 = UNetBlock(c, c * 2)
        self.pool = nn.AvgPool2d(2)
        # Bottleneck
        self.bot = UNetBlock(c * 2, c * 2)
        # Up
        self.up1 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec1 = UNetBlock(c * 4, c * 2)  # c*2 (up1) + c*2 (d2) = c*4
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec2 = UNetBlock(c * 2, c)     # c (up2) + c (d1) = c*2
        # Final output projection back to latent space
        self.out = nn.Conv2d(c, config.latent_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, latent_channels, H, W], t: [B]
        # Project input to model dimensions
        x = self.input_proj(x)  # [B, c, H, W]
        
        d1 = self.down1(x)           # [B, c, H, W]
        d2 = self.down2(self.pool(d1))  # [B, c*2, H/2, W/2]
        b = self.bot(self.pool(d2))     # [B, c*2, H/4, W/4]
        
        u1 = self.up1(b)            # [B, c*2, H/2, W/2]
        # Ensure u1 and d2 have the same spatial dimensions
        if u1.shape[2:] != d2.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d2], dim=1)  # [B, c*4, H/2, W/2]
        u1 = self.dec1(u1)          # [B, c*2, H/2, W/2]
        
        u2 = self.up2(u1)           # [B, c, H, W]
        # Ensure u2 and d1 have the same spatial dimensions
        if u2.shape[2:] != d1.shape[2:]:
            u2 = nn.functional.interpolate(u2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d1], dim=1)  # [B, c*2, H, W]
        u2 = self.dec2(u2)          # [B, c, H, W]
        
        # Project back to latent space
        out = self.out(u2)          # [B, latent_channels, H, W]
        return out
    
class ConditionalUNetModel(nn.Module):
    def __init__(self, config: DiffusionConfig, text_emb_dim: int):
        super().__init__()
        c = config.channels
        self.input_proj = nn.Conv2d(config.latent_channels, c, 1)
        # load text encoder
        self.tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # pass cond_dim = text_embedding_dim
        cond_dim = self.text_encoder.config.hidden_size
        
        # down blocks now take cond_dim - using ConditionalUNetBlock
        self.down1 = ConditionalUNetBlock(c,    c,    cond_dim)
        self.down2 = ConditionalUNetBlock(c,  2*c,    cond_dim)
        self.pool  = nn.AvgPool2d(2)
        self.bot   = ConditionalUNetBlock(2*c,2*c,    cond_dim)
        self.up1   = nn.ConvTranspose2d(2*c,2*c,2, stride=2)
        self.dec1  = ConditionalUNetBlock(4*c,2*c,    cond_dim)
        self.up2   = nn.ConvTranspose2d(2*c,  c,2, stride=2)
        self.dec2  = ConditionalUNetBlock(2*c,  c,    cond_dim)
        self.out   = nn.Conv2d(c, config.latent_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, texts: List[str]) -> torch.Tensor:
        # 1) encode text
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(x.device)
        text_emb = self.text_encoder(**tokens).last_hidden_state[:,0]  # [B, cond_dim]
        
        # 2) standard UNet but pass text_emb into each block
        x = self.input_proj(x)
        d1 = self.down1(x,         text_emb)
        d2 = self.down2(self.pool(d1), text_emb)
        b  = self.bot(self.pool(d2),    text_emb)
        
        u1 = self.up1(b)
        if u1.shape[-2:] != d2.shape[-2:]:
            u1 = F.interpolate(u1, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.dec1(torch.cat([u1,d2],1), text_emb)
        
        u2 = self.up2(u1)
        if u2.shape[-2:] != d1.shape[-2:]:
            u2 = F.interpolate(u2, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.dec2(torch.cat([u2,d1],1), text_emb)
        
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

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            if epoch % self.config.validation_freq == 0:
                val_loss = self._validate_epoch(epoch)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_checkpoint(epoch, val_loss)
                    logger.info(f"New best validation loss: {val_loss:.6f}")
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
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        total_loss = 0.0
        num_batches = 0
        
        for raw_images, text_labels in pbar:
            raw_images = raw_images.to(self.config.device)
            
            # Encode to latents using VQ-VAE on GPU
            with torch.no_grad():
                z_e = self.vqvae.encoder(raw_images)
                latents, _ = self.vqvae.vector_quantizer(z_e)
            
            self.optimizer.zero_grad()
            loss = self.diffusion.p_loss(self.model, latents)  # Non-conditional training
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
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation", leave=False)
            for raw_images, text_labels in pbar:
                raw_images = raw_images.to(self.config.device)
                
                # Encode to latents using VQ-VAE on GPU
                z_e = self.vqvae.encoder(raw_images)
                latents, _ = self.vqvae.vector_quantizer(z_e)
                
                loss = self.diffusion.p_loss(self.model, latents)  # Non-conditional validation
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
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

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            if epoch % self.config.validation_freq == 0:
                val_loss = self._validate_epoch(epoch)
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_best_checkpoint(epoch, val_loss)
                    logger.info(f"New best validation loss: {val_loss:.6f}")
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
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        total_loss = 0.0
        num_batches = 0
        
        for raw_images, text_labels in pbar:
            raw_images = raw_images.to(self.config.device)
            
            # Encode to latents using VQ-VAE on GPU
            with torch.no_grad():
                z_e = self.vqvae.encoder(raw_images)
                latents, _ = self.vqvae.vector_quantizer(z_e)
            
            self.optimizer.zero_grad()
            loss = self.diffusion.p_loss(self.model, latents, text_labels)  # Conditional training
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
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation", leave=False)
            for raw_images, text_labels in pbar:
                raw_images = raw_images.to(self.config.device)
                
                # Encode to latents using VQ-VAE on GPU
                z_e = self.vqvae.encoder(raw_images)
                latents, _ = self.vqvae.vector_quantizer(z_e)
                
                loss = self.diffusion.p_loss(self.model, latents, text_labels)  # Conditional validation
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
        latent_channels=base_config.latent_channels,  # Keep latent channels fixed
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

def objective(trial: optuna.Trial, vqvae: VQVAE, train_ds: Dataset, val_ds: Dataset, latent_channels: int) -> float:
    """Optuna objective function"""
    
    # Create config from trial
    base_config = DiffusionConfig(latent_channels=latent_channels)
    config = create_config_from_trial(trial, base_config)
    
    # Create data loaders with trial-specific batch size
    train_raw_ds = RawDataset(train_ds, LABEL_MAP)
    val_raw_ds = RawDataset(val_ds, LABEL_MAP)
    
    train_loader = DataLoader(train_raw_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_raw_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Build model and diffusion with trial parameters - USE CONDITIONAL VERSIONS
    scheduling = LinearSchedule(config.beta_start, config.beta_end)
    diffusion = ConditionalGaussianDiffusion(config, scheduling)  # Conditional diffusion
    model = ConditionalUNetModel(config, text_emb_dim=512)  # Conditional model
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Create trial directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Train with trial-specific trainer
    trainer = ConditionalOptunaTrainer(model, diffusion, optimizer, train_loader, val_loader, config, vqvae, trial)
    
    try:
        best_val_loss = trainer.train_for_optuna()
        return best_val_loss
    except optuna.TrialPruned:
        # Clean up trial directory if pruned
        import shutil
        if os.path.exists(config.checkpoint_dir):
            shutil.rmtree(config.checkpoint_dir)
        raise

def optimize_hyperparameters(vqvae: VQVAE, train_ds: Dataset, val_ds: Dataset, optuna_config: OptunaConfig) -> DiffusionConfig:
    """Run Optuna optimization to find best hyperparameters"""
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name=optuna_config.study_name,
        storage=optuna_config.storage,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)  # Reduced warmup
    )
    
    # Get VQ-VAE config for latent dimensions
    _, vq_cfg = load_best_model('/home/nair-group/abian_torres/repositories/tool_generation/final_model_checkpoints')
    
    # Optimize
    logger.info(f"Starting Optuna optimization with {optuna_config.n_trials} trials")
    study.optimize(
        lambda trial: objective(trial, vqvae, train_ds, val_ds, vq_cfg.embedding_dim),
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
        latent_channels=vq_cfg.embedding_dim,  # Use VQ-VAE embedding dimension
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
    
    # Create data loaders with text labels for conditional training
    train_raw_ds = RawDataset(train_ds, LABEL_MAP)
    val_raw_ds = RawDataset(val_ds, LABEL_MAP)
    
    train_loader = DataLoader(train_raw_ds, batch_size=best_config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_raw_ds, batch_size=best_config.batch_size, shuffle=False, num_workers=0)
    
    # Build conditional model and diffusion
    scheduling = LinearSchedule(best_config.beta_start, best_config.beta_end)
    diffusion = ConditionalGaussianDiffusion(best_config, scheduling)  # Conditional diffusion
    model = ConditionalUNetModel(best_config, text_emb_dim=512)  # Conditional model
    optimizer = optim.Adam(model.parameters(), lr=best_config.lr)
    
    # Train final model with conditional trainer
    trainer = ConditionalTrainer(model, diffusion, optimizer, train_loader, val_loader, best_config, vqvae)
    trainer.train()
    
    logger.info("Final training completed")

def generate_samples(model_path: str, vqvae: VQVAE, config: DiffusionConfig, text_prompts: List[str], num_samples: int = 4):
    """Generate samples using the trained conditional model"""
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=config.device)
    model = ConditionalUNetModel(config, text_emb_dim=512).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion
    scheduling = LinearSchedule(config.beta_start, config.beta_end)
    diffusion = ConditionalGaussianDiffusion(config, scheduling)
    
    # Generate samples
    batch_size = len(text_prompts)
    latent_shape = [batch_size, config.latent_channels, config.image_size // 4, config.image_size // 4]  # Assuming 4x downsampling
    
    with torch.no_grad():
        # Generate latent samples
        latent_samples = diffusion.sample(model, latent_shape, text_prompts)
        
        # Decode with VQ-VAE
        reconstructed_images = vqvae.decoder(latent_samples)
    
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

    # Create base config with VQ-VAE latent dimensions
    base_config = DiffusionConfig(latent_channels=vq_cfg.embedding_dim)

    # Optuna configuration
    optuna_config = OptunaConfig(
        n_trials=50,  # Adjust based on your computational budget
        study_name="conditional_diffusion_hyperopt",  # Different study name
        storage="sqlite:///conditional_diffusion_optuna.db"  # Different database
    )
    
    # Phase 1: Hyperparameter optimization
    logger.info("Phase 1: Hyperparameter optimization for conditional diffusion")
    best_config = optimize_hyperparameters(vqvae, train_ds, val_ds, optuna_config)
    
    # Phase 2: Final training with best configuration
    logger.info("Phase 2: Final training with best configuration")
    train_final_model(best_config, vqvae, train_ds, val_ds)
    
    # Phase 3: Generate some samples
    logger.info("Phase 3: Generating samples")
    text_prompts = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    model_path = os.path.join(best_config.checkpoint_dir, "best_conditional_diffusion_model.pth")
    
    if os.path.exists(model_path):
        images, latents = generate_samples(model_path, vqvae, best_config, text_prompts)
        
        # Save generated images
        import torchvision.utils as vutils
        vutils.save_image(images, 'generated_conditional_samples.png', nrow=5, normalize=True)
        logger.info("Generated samples saved to 'generated_conditional_samples.png'")
    else:
        logger.warning("No trained model found for sample generation")

if __name__ == "__main__":
    main()