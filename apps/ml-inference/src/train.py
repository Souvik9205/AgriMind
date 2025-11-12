"""
Training pipeline for plant disease detection model.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Logging will be local only.")

from config import MODEL_CONFIG, BEST_MODEL_PATH, LATEST_MODEL_PATH, WANDB_PROJECT
from models import create_model, count_parameters
from data_loader import create_data_loaders
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseTrainer:
    """
    Trainer class for plant disease detection model.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or MODEL_CONFIG.__dict__
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model, dataloaders, optimizer, etc.
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Initialize wandb if available
        self.wandb_run = None
        if WANDB_AVAILABLE:
            self.init_wandb()
    
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            # Set wandb API key from environment if available
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
                wandb.login(key=wandb_api_key)
            
            self.wandb_run = wandb.init(
                project=WANDB_PROJECT,
                config=self.config,
                name=f"plant-disease-{int(time.time())}"
            )
            logger.info("Weights & Biases initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.wandb_run = None
    
    def setup_model(self, num_classes: int):
        """Setup model architecture."""
        logger.info(f"Creating model with {num_classes} classes")
        
        self.model = create_model(self.config, num_classes)
        
        # Count parameters
        total_params, trainable_params = count_parameters(self.model)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup for multi-GPU if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
    
    def setup_data_loaders(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            batch_size=self.config.get('batch_size', 32),
            num_workers=4
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Test batches: {len(self.test_loader)}")
    
    def setup_criterion(self):
        """Setup loss function."""
        # Use weighted cross entropy if class imbalance exists
        self.criterion = nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Using {optimizer_name.upper()} optimizer with lr={learning_rate}")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        epochs = self.config.get('epochs', 100)
        
        if scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        if self.scheduler:
            logger.info(f"Using {scheduler_name} learning rate scheduler")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            acc = accuracy(outputs, targets, topk=(1,))[0]
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accuracies.avg:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return losses.avg, accuracies.avg
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                acc = accuracy(outputs, targets, topk=(1,))[0]
                
                # Update meters
                losses.update(loss.item(), images.size(0))
                accuracies.update(acc.item(), images.size(0))
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{accuracies.avg:.2f}%'
                })
        
        return losses.avg, accuracies.avg
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup everything
        self.setup_data_loaders()
        
        # Get number of classes from data loader
        num_classes = self.train_loader.dataset.num_classes
        
        self.setup_model(num_classes)
        self.setup_criterion()
        self.setup_optimizer()
        self.setup_scheduler()
        
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    'epoch': self.current_epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Check for best model
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoints
            save_checkpoint({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_accuracy': self.best_accuracy,
                'training_history': self.training_history,
                'config': self.config
            }, is_best, LATEST_MODEL_PATH, BEST_MODEL_PATH)
            
            # Log progress
            elapsed_time = time.time() - start_time
            logger.info(
                f'Epoch {self.current_epoch}/{epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                f'Best Acc: {self.best_accuracy:.2f}% - '
                f'Time: {elapsed_time:.0f}s'
            )
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.0f} seconds")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        # Save final training history
        history_path = Path(BEST_MODEL_PATH).parent / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        if self.wandb_run:
            self.wandb_run.finish()
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Setup model first
        num_classes = len(checkpoint.get('class_mapping', {}))
        self.setup_model(num_classes)
        
        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        self.setup_scheduler()
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}, best accuracy: {self.best_accuracy:.2f}%")

def main():
    """Main training function."""
    trainer = PlantDiseaseTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
