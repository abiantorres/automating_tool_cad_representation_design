import logging
import os
from abc import ABC, abstractmethod
import typing
from dataclasses import dataclass
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # progress bars
from torch.utils.data import random_split
import optuna
from optuna.trial import TrialState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Global variables to store consistent data splits
TRAIN_DATASET = None
VAL_DATASET = None
# Alternative approach: Store the best trial's seed for exact reproduction
BEST_TRIAL_SEED = None


@dataclass
class VQVAEConfig:
    in_channels: int = 3
    hidden_dims: list = (128, 256)
    embedding_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25  # Fixed to paper value
    decay: float = 0.99
    learning_rate: float = 2e-4
    batch_size: int = 32
    num_epochs: int = 50
    use_bernoulli_loss: bool = False  # New parameter

class BaseVAE(ABC, nn.Module):
    """
    Abstract base class for Variational Autoencoders
    """
    @abstractmethod
    def encode(self, x: torch.Tensor) -> typing.Any:
        pass

    @abstractmethod
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, ...]:
        z_e = self.encode(x)
        z_q, loss_q = self.vector_quantizer(z_e)
        x_recon = self.decode(z_q)
        return x_recon, loss_q

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list, embedding_dim: int):
        super().__init__()
        modules = []
        prev_channels = in_channels
        
        # Reduce downsampling for better spatial resolution
        # Instead of 28x28 -> 7x7, go 28x28 -> 14x14
        for i, h_dim in enumerate(hidden_dims):
            stride = 2 if i < len(hidden_dims) - 1 else 1  # Last layer doesn't downsample
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, h_dim, kernel_size=4, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = h_dim
            
        # Final conv to embedding dim
        modules.append(nn.Conv2d(prev_channels, embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list, out_channels: int):
        super().__init__()
        modules = []
        prev_channels = embedding_dim
        
        # Mirror the encoder structure
        reversed_dims = list(reversed(hidden_dims))
        for i, h_dim in enumerate(reversed_dims):
            stride = 2 if i > 0 else 1  # First layer doesn't upsample
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_channels, h_dim, kernel_size=4, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = h_dim
            
        # Final conv to reconstruct
        modules.append(nn.Conv2d(prev_channels, out_channels, kernel_size=1))
        modules.append(nn.Sigmoid())  # For Bernoulli loss
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float = 0.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Make embeddings trainable parameters instead of buffers
        self.embedding = nn.Parameter(torch.randn(embedding_dim, num_embeddings))
        
        # Only keep EMA buffers if using EMA updates
        if self.decay > 0:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', self.embedding.data.clone())

    def forward(self, z: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Flatten input: (B, D, H, W) -> (B*H*W, D)
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        # Compute distances
        distances = (flat_z.pow(2).sum(1, keepdim=True)
                     - 2 * flat_z @ self.embedding
                     + self.embedding.pow(2).sum(0, keepdim=True))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = (encodings @ self.embedding.t()).view(z_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # EMA updates (if decay > 0)
        if self.training and self.decay > 0:
            self._ema_update(flat_z, encodings)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        return quantized, loss
    
    def _ema_update(self, flat_z: torch.Tensor, encodings: torch.Tensor):
        """EMA update for the codebook"""
        with torch.no_grad():
            # Update cluster sizes
            cluster_size = encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            
            # Update embeddings
            embed_sum = flat_z.t() @ encodings
            self.ema_w.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # Normalize embeddings
            n = self.ema_cluster_size.sum()
            smoothed_cluster_size = (
                (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            )
            embed_normalized = self.ema_w / smoothed_cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embed_normalized)

class VQVAE(BaseVAE):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.in_channels, config.hidden_dims, config.embedding_dim)
        
        # Add residual blocks before quantization
        self.pre_quantize = nn.Sequential(
            nn.Conv2d(config.embedding_dim, config.embedding_dim, kernel_size=1),
            ResidualBlock(config.embedding_dim),
            ResidualBlock(config.embedding_dim),
            nn.Conv2d(config.embedding_dim, config.embedding_dim, kernel_size=1)
        )
        
        self.vector_quantizer = VectorQuantizer(config.num_embeddings, config.embedding_dim,
                                                config.commitment_cost, config.decay)
        
        # Add residual blocks after quantization
        self.post_quantize = nn.Sequential(
            nn.Conv2d(config.embedding_dim, config.embedding_dim, kernel_size=1),
            ResidualBlock(config.embedding_dim),
            ResidualBlock(config.embedding_dim),
            nn.Conv2d(config.embedding_dim, config.embedding_dim, kernel_size=1)
        )
        
        self.decoder = Decoder(config.embedding_dim, config.hidden_dims, config.in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        return self.pre_quantize(z_e)

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        z_q = self.post_quantize(quantized)
        return self.decoder(z_q)
    
class Trainer:
    """
    Trainer using Template Method pattern
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader, val_loader: DataLoader = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 checkpoint_dir: str = "./checkpoints",
                 patience: int = 5, min_delta: float = 1e-4,
                 save_checkpoints: bool = True):
        
        # Better device detection
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logger.info("CUDA not available, using CPU")
                
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints  

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False

        # Create checkpoint directory only if saving is enabled
        if self.save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, epochs: int):
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            self.on_epoch_start(epoch)
            self._train_epoch(epoch)
            
            if self.val_loader:
                val_loss = self._validate_epoch(epoch)
                
                # Check for improvement and early stopping
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best checkpoint only if enabled
                    if self.save_checkpoints:
                        self.save_checkpoint(epoch, filename="best_model.pth")
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                
                # Early stopping check
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    self.early_stopped = True
                    break
        
        # Save final checkpoint only if enabled and training completed without early stopping
        if not self.early_stopped and self.save_checkpoints:
            self.save_checkpoint(epochs, filename="final_model.pth")

    def on_epoch_start(self, epoch: int):
        pass

    def _train_epoch(self, epoch: int):
        self.model.train()
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)
        for batch in loop:
            x, _ = batch
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_recon, loss_q = self.model(x)
            
            # Use Bernoulli loss for binary images like MNIST
            if hasattr(self.model.config, 'use_bernoulli_loss') and self.model.config.use_bernoulli_loss:
                recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
            else:
                recon_loss = F.mse_loss(x_recon, x)
                
            loss = recon_loss + loss_q
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(train_loss=loss.item())
        logger.info(f"Train Loss: {loss.item():.4f}")

    def _validate_epoch(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        loop = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation", leave=False)
        with torch.no_grad():
            for batch in loop:
                x, _ = batch
                x = x.to(self.device)
                x_recon, loss_q = self.model(x)
                
                # Use Bernoulli loss for binary images like MNIST
                if hasattr(self.model.config, 'use_bernoulli_loss') and self.model.config.use_bernoulli_loss:
                    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
                else:
                    recon_loss = F.mse_loss(x_recon, x)
                    
                batch_loss = (recon_loss + loss_q).item()
                val_loss += batch_loss
                loop.set_postfix(val_loss=batch_loss)
        val_loss /= len(self.val_loader)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        return val_loss

    def save_checkpoint(self, epoch: int, filename: str = None):
        """Save model and optimizer state"""
        if not self.save_checkpoints:
            return  # Skip saving if disabled
            
        if filename is None:
            filename = f"vqvae_checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config if hasattr(self.model, 'config') else None
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


    def load_checkpoint(self, checkpoint_path: str):
        """Load model and optimizer state from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
        return epoch

def get_data_splits():
    """Get consistent train/val splits across all trials"""
    global TRAIN_DATASET, VAL_DATASET
    
    if TRAIN_DATASET is None or VAL_DATASET is None:
        # Load data once and split consistently
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        
        # Use fixed seed for consistent splits
        set_seed(42)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        TRAIN_DATASET, VAL_DATASET = random_split(full_dataset, [train_size, val_size])
        
    return TRAIN_DATASET, VAL_DATASET

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Use fixed seed for reproducible model initialization across all trials
    set_seed(42)
    
    # Suggest hyperparameters with better ranges
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256, 512])
    num_embeddings = trial.suggest_categorical('num_embeddings', [512, 1024, 2048])
    commitment_cost = trial.suggest_float('commitment_cost', 0.1, 0.3)  # Narrower range
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Larger batches
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [128, 256])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [256, 512])
    decay = trial.suggest_float('decay', 0.9, 0.999)  # EMA decay rate
    
    # Create config with suggested parameters
    config = VQVAEConfig(
        in_channels=1,  # MNIST
        hidden_dims=[hidden_dim_1, hidden_dim_2],
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=5,  # Increased for better evaluation
        use_bernoulli_loss=True  # Use Bernoulli loss for MNIST
    )
    
    # Get consistent data splits
    train_dataset, val_dataset = get_data_splits()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = VQVAE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize trainer WITHOUT checkpoint saving for optimization
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="",
        patience=10,
        min_delta=1e-5,
        save_checkpoints=False
    )
    
    # Train model
    trainer.train(config.num_epochs)
    
    # Return best validation loss for optimization
    return trainer.best_val_loss

def objective_with_trial_seeds(trial):
    """Alternative objective function that stores the seed of the best trial"""
    global BEST_TRIAL_SEED
    
    # Use trial-specific seed
    trial_seed = 42 + trial.number
    set_seed(trial_seed)
    
    # Suggest hyperparameters
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256, 512, 1024])
    num_embeddings = trial.suggest_categorical('num_embeddings', [512, 1024, 2048, 4096])
    commitment_cost = trial.suggest_float('commitment_cost', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [128, 256, 512])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [256, 512, 1024, 2048])
    
    # Create config with suggested parameters
    config = VQVAEConfig(
        in_channels=1,
        hidden_dims=[hidden_dim_1, hidden_dim_2],
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=1
    )
    
    # Get consistent data splits
    train_dataset, val_dataset = get_data_splits()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = VQVAE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="",
        patience=10,
        min_delta=1e-5,
        save_checkpoints=False
    )
    
    # Train model
    trainer.train(config.num_epochs)
    
    # Store the seed if this is the best trial so far
    if BEST_TRIAL_SEED is None or trainer.best_val_loss < getattr(objective_with_trial_seeds, 'best_loss', float('inf')):
        BEST_TRIAL_SEED = trial_seed
        objective_with_trial_seeds.best_loss = trainer.best_val_loss
    
    return trainer.best_val_loss

def run_optuna_optimization(n_trials=50, use_consistent_seed=True):
    """Run Optuna hyperparameter optimization"""
    
    # Set global seed
    set_seed(42)
    
    # Delete existing database to avoid distribution compatibility issues
    db_path = 'vqvae_optuna.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Removed existing database: {db_path}")
    
    # Create study with in-memory storage to avoid conflicts
    study = optuna.create_study(direction='minimize')
    
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    
    # Choose objective function based on seed strategy
    if use_consistent_seed:
        study.optimize(objective, n_trials=n_trials)
        best_seed = 42  # Consistent seed used
    else:
        study.optimize(objective_with_trial_seeds, n_trials=n_trials)
        best_seed = BEST_TRIAL_SEED  # Seed from best trial
    
    # Print results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info(f"  Best seed: {best_seed}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Return best params, best value, and best seed
    return study.best_params, study.best_value, best_seed

def evaluate_best_config_short(best_params, epochs=1, seed=42):
    """Evaluate the best config with the same number of epochs and seed as optimization"""
    
    logger.info(f"Evaluating best config with {epochs} epochs and seed {seed} for comparison...")
    
    # Set same seed as the best trial
    set_seed(seed)
    
    # Create config with best parameters
    config = VQVAEConfig(
        in_channels=1,
        hidden_dims=[best_params['hidden_dim_1'], best_params['hidden_dim_2']],
        embedding_dim=best_params['embedding_dim'],
        num_embeddings=best_params['num_embeddings'],
        commitment_cost=best_params['commitment_cost'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        num_epochs=epochs
    )
    
    # Use the same data splits
    train_dataset, val_dataset = get_data_splits()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = VQVAE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize trainer without checkpoints
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="",
        patience=10,
        min_delta=1e-5,
        save_checkpoints=False
    )
    
    # Train model
    trainer.train(config.num_epochs)
    
    logger.info(f"Best config evaluation ({epochs} epochs, seed {seed}) - Final validation loss: {trainer.best_val_loss:.6f}")
    return trainer.best_val_loss

def train_with_best_config(best_params, seed=42):
    """Train the final model with the best hyperparameters"""
    
    logger.info("Training final model with best hyperparameters...")
    
    # Set seed for reproducible final training
    set_seed(seed)
    
    # Create config with best parameters
    config = VQVAEConfig(
        in_channels=1,  # MNIST
        hidden_dims=[best_params['hidden_dim_1'], best_params['hidden_dim_2']],
        embedding_dim=best_params['embedding_dim'],
        num_embeddings=best_params['num_embeddings'],
        commitment_cost=best_params['commitment_cost'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        num_epochs=100  # Full training epochs
    )
    
    # Use the same data splits as optimization
    train_dataset, val_dataset = get_data_splits()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = VQVAE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize trainer WITH checkpoint saving for final training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="./final_model_checkpoints",
        patience=15,  # More patience for final training
        min_delta=1e-5,
        save_checkpoints=True  # Enable checkpoint saving for final training
    )
    
    # Train final model
    trainer.train(config.num_epochs)
    
    logger.info("Final training completed!")
    return trainer, config

if __name__ == "__main__":
    # Import required modules for main execution
    from torchvision import datasets, transforms
    
    # Step 1: Run Optuna optimization with consistent seed
    best_params, best_optuna_value, best_seed = run_optuna_optimization(n_trials=20, use_consistent_seed=True)
    
    # Step 2: Evaluate best config with same conditions as optimization
    comparison_loss = evaluate_best_config_short(best_params, epochs=1, seed=best_seed)
    logger.info(f"Comparison - Optuna best: {best_optuna_value:.6f}, Re-evaluation: {comparison_loss:.6f}")
    
    # The difference should now be minimal (ideally zero)
    difference = abs(best_optuna_value - comparison_loss)
    logger.info(f"Difference: {difference:.6f} (should be close to 0)")
    
    # Step 3: Train final model with best hyperparameters
    final_trainer, final_config = train_with_best_config(best_params, seed=best_seed)
    
    # Save final config for future reference
    import json
    config_dict = {
        'in_channels': final_config.in_channels,
        'hidden_dims': list(final_config.hidden_dims),
        'embedding_dim': final_config.embedding_dim,
        'num_embeddings': final_config.num_embeddings,
        'commitment_cost': final_config.commitment_cost,
        'decay': final_config.decay,
        'learning_rate': final_config.learning_rate,
        'batch_size': final_config.batch_size,
        'num_epochs': final_config.num_epochs,
        'best_seed': best_seed  # Save the seed used for reproducibility
    }
    
    with open('./final_model_checkpoints/best_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("Best configuration saved to './final_model_checkpoints/best_config.json'")