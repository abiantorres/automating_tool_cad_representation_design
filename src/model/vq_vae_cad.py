import logging
import os
import json
from abc import ABC, abstractmethod
import typing
from dataclasses import dataclass
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm  # progress bars
from torch.utils.data import random_split
import optuna
from optuna.trial import TrialState

# MongoDB imports
import pymongo
from pymongo import MongoClient

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
    in_channels: int = 1  # Changed from 3 to 1 for CAD data
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
        
        # Special handling for 128x17 input size
        # First layer: 128x17 -> 64x9 (stride 2, padding adjusted for width)
        modules.append(
            nn.Sequential(
                nn.Conv2d(prev_channels, hidden_dims[0], kernel_size=(4, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )
        )
        prev_channels = hidden_dims[0]
        
        # Second layer: 64x9 -> 32x5 (stride 2, padding adjusted)
        if len(hidden_dims) > 1:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, hidden_dims[1], kernel_size=(4, 3), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(hidden_dims[1]),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = hidden_dims[1]
            
        # Additional layers if more hidden dims
        for i, h_dim in enumerate(hidden_dims[2:], 2):
            # Use stride 1 for additional layers to avoid going too small
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
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
        
        # Additional layers if more than 2 hidden dims (use conv, not transpose)
        for i, h_dim in enumerate(reversed_dims[:-2]):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = h_dim
        
        # Second to last layer: 32x5 -> 64x9
        if len(reversed_dims) > 1:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_channels, reversed_dims[-2], kernel_size=(4, 3), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(reversed_dims[-2]),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = reversed_dims[-2]
            
        # Final layer: 64x9 -> 128x17
        modules.append(
            nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=(4, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1))
        )
        
        # No sigmoid since we're not using Bernoulli loss for CAD data
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

class CADDataset(Dataset):
    """Custom dataset for CAD data from MongoDB"""
    
    def __init__(self, db_name: str = "omni_cad", collection_name: str = "processed_vectors", 
                 split: str = "train", mongo_uri: str = "mongodb://localhost:27017/",
                 transform=None, max_samples: int = None):
        self.db_name = db_name
        self.collection_name = collection_name
        self.split = split
        self.transform = transform
        
        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Get all documents for the specified split
        query = {"split": split} if split else {}
        self.documents = list(self.collection.find(query))
        
        if max_samples:
            self.documents = self.documents[:max_samples]
            
        logger.info(f"Loaded {len(self.documents)} documents from {db_name}.{collection_name} (split: {split})")
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Extract vector and convert to numpy array
        vector = np.array(doc['vector'], dtype=np.float32)  # Shape: (128, 17)
        
        # Process the data according to your specifications
        processed_vector = self._process_vector(vector)
        
        # Convert to tensor and add channel dimension: (128, 17) -> (1, 128, 17)
        tensor = torch.from_numpy(processed_vector).unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            tensor = self.transform(tensor)
            
        # Return tensor and text caption
        text_caption = doc.get('text caption', '')
        
        return tensor, text_caption
    
    def _process_vector(self, vector):
        """Process the vector according to specifications:
        - First column: values 0-5 (no change needed)
        - Other columns: shift -1 values, range 0-256 after shift
        """
        processed = vector.copy()
        
        # Process columns 1-16 (indices 1:17)
        # Shift values that are not -1 by adding 1
        mask = processed[:, 1:] != -1
        processed[:, 1:][mask] += 1
        
        # Now -1 becomes 0, and original 0-255 becomes 1-256
        # Replace remaining -1 with 0
        processed[processed == -1] = 0
        
        # Normalize to [0, 1] range
        # First column: 0-5 -> normalize by 5
        processed[:, 0] = processed[:, 0] / 5.0
        
        # Other columns: 0-256 -> normalize by 256
        processed[:, 1:] = processed[:, 1:] / 256.0
        
        return processed
    
    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()

def get_data_splits():
    """Get consistent train/val splits for CAD data"""
    global TRAIN_DATASET, VAL_DATASET
    
    if TRAIN_DATASET is None or VAL_DATASET is None:
        try:
            # Create datasets from MongoDB
            TRAIN_DATASET = CADDataset(split="train", max_samples=1000)  # Limit for testing
            VAL_DATASET = CADDataset(split="val", max_samples=200)  # Limit for testing
            
            logger.info(f"Loaded train dataset: {len(TRAIN_DATASET)} samples")
            logger.info(f"Loaded val dataset: {len(VAL_DATASET)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load CAD datasets: {e}")
            # Fallback to dummy data for testing
            logger.warning("Creating dummy datasets for testing")
            TRAIN_DATASET = DummyCADDataset(1000)
            VAL_DATASET = DummyCADDataset(200)
    
    return TRAIN_DATASET, VAL_DATASET

class DummyCADDataset(Dataset):
    """Dummy dataset for testing when MongoDB is not available"""
    
    def __init__(self, size: int):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create dummy 128x17 data
        # First column: random values 0-5
        first_col = np.random.randint(0, 6, size=(128, 1))
        # Other columns: random values 0-256, with some -1 values
        other_cols = np.random.randint(-1, 257, size=(128, 16))
        
        # Combine
        vector = np.concatenate([first_col, other_cols], axis=1).astype(np.float32)
        
        # Process the vector
        processed = self._process_vector(vector)
        
        # Convert to tensor with channel dimension
        tensor = torch.from_numpy(processed).unsqueeze(0)
        
        return tensor, f"dummy_caption_{idx}"
    
    def _process_vector(self, vector):
        """Same processing as CADDataset"""
        processed = vector.copy()
        
        # Process columns 1-16
        mask = processed[:, 1:] != -1
        processed[:, 1:][mask] += 1
        processed[processed == -1] = 0
        
        # Normalize
        processed[:, 0] = processed[:, 0] / 5.0
        processed[:, 1:] = processed[:, 1:] / 256.0
        
        return processed

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Use fixed seed for reproducible model initialization across all trials
    set_seed(42)
    
    # Suggest hyperparameters adapted for CAD data
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    num_embeddings = trial.suggest_categorical('num_embeddings', [512, 1024, 2048])
    commitment_cost = trial.suggest_float('commitment_cost', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Smaller batches for large images
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [64, 128, 256])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [128, 256, 512])
    decay = trial.suggest_float('decay', 0.9, 0.999)
    
    # Create config with suggested parameters
    config = VQVAEConfig(
        in_channels=1,  # CAD data
        hidden_dims=[hidden_dim_1, hidden_dim_2],
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=3,  # Reduced for faster trials
        use_bernoulli_loss=False  # MSE loss for CAD data
    )
    
    # Get consistent data splits
    train_dataset, val_dataset = get_data_splits()
    
    # Create data loaders with reduced num_workers for stability
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
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
        patience=5,  # Reduced patience for trials
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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
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
        study.optimize(objective, n_trials=n_trials, n_jobs=4)  # Use single job for consistent seed
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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
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
        in_channels=1,  # CAD data
        hidden_dims=[best_params['hidden_dim_1'], best_params['hidden_dim_2']],
        embedding_dim=best_params['embedding_dim'],
        num_embeddings=best_params['num_embeddings'],
        commitment_cost=best_params['commitment_cost'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        num_epochs=50,  # Reduced from 100 for CAD data
        use_bernoulli_loss=False  # MSE loss for CAD data
    )
    
    # Use the same data splits as optimization
    train_dataset, val_dataset = get_data_splits()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
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
        patience=10,  # Reduced patience for CAD data
        min_delta=1e-5,
        save_checkpoints=True
    )
    
    # Train final model
    trainer.train(config.num_epochs)
    
    logger.info("Final training completed!")
    return trainer, config

def load_config_from_json(config_path: str) -> VQVAEConfig:
    """Load VQVAEConfig from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert list back to tuple for hidden_dims if needed
    if 'hidden_dims' in config_dict:
        config_dict['hidden_dims'] = tuple(config_dict['hidden_dims'])
    
    # Remove non-config fields if present
    config_dict.pop('best_seed', None)
    
    return VQVAEConfig(**config_dict)

def load_config_from_checkpoint(checkpoint_path: str, device: str = 'cpu') -> VQVAEConfig:
    """Load VQVAEConfig from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint and checkpoint['config'] is not None:
        return checkpoint['config']
    else:
        raise ValueError("No config found in checkpoint file")

def load_best_model(checkpoint_dir: str = './final_model_checkpoints', 
                   device: str = None) -> tuple[VQVAE, VQVAEConfig]:
    """Load the best model with its configuration"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Try to load config from JSON first (more reliable)
    json_config_path = os.path.join(checkpoint_dir, 'best_config.json')
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if os.path.exists(json_config_path):
        config = load_config_from_json(json_config_path)
    elif os.path.exists(checkpoint_path):
        config = load_config_from_checkpoint(checkpoint_path, device)
    else:
        raise FileNotFoundError("Neither config.json nor checkpoint file found")
    
    # Initialize model
    model = VQVAE(config)
    
    # Load weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from {checkpoint_path}")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, config

if __name__ == "__main__":
    # Step 1: Test data loading
    logger.info("Testing CAD data loading...")
    try:
        train_dataset = CADDataset(split="train", max_samples=10)
        sample_data, sample_caption = train_dataset[0]
        logger.info(f"Sample data shape: {sample_data.shape}")
        logger.info(f"Sample caption: {sample_caption}")
        logger.info(f"Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        train_dataset.close()
    except Exception as e:
        logger.warning(f"MongoDB not available, will use dummy data: {e}")
    
    # Step 2: Run Optuna optimization with consistent seed
    best_params, best_optuna_value, best_seed = run_optuna_optimization(n_trials=10, use_consistent_seed=True)  # Reduced trials
    
    # Step 3: Train final model with best hyperparameters
    final_trainer, final_config = train_with_best_config(best_params, seed=best_seed)
    
    # Save final config for future reference
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
        'use_bernoulli_loss': final_config.use_bernoulli_loss,
        'best_seed': best_seed
    }
    
    with open('./final_model_checkpoints/best_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("Best configuration saved to './final_model_checkpoints/best_config.json'")
    
    # Clean up datasets
    train_dataset, val_dataset = get_data_splits()
    if hasattr(train_dataset, 'close'):
        train_dataset.close()
    if hasattr(val_dataset, 'close'):
        val_dataset.close()