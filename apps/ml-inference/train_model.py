"""
Model Training Pipeline
Train custom plant disease classification model
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlantDiseaseDataset(Dataset):
    """Custom dataset for plant disease images"""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        self._load_samples()
        
    def _load_samples(self):
        """Load all samples and create class mapping"""
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.classes = sorted([d.name for d in class_dirs])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PlantDiseaseModel(nn.Module):
    """Custom plant disease classification model based on ResNet"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(PlantDiseaseModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, data_root: str = "data", model_root: str = "models"):
        self.data_root = Path(data_root)
        self.model_root = Path(model_root)
        self.model_root.mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.num_classes = self.metadata["num_classes"]
        self.class_names = self.metadata["class_names"]
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.patience = 10  # Early stopping patience
        
        print(f"🎯 Training setup: {self.num_classes} classes")
        print(f"📝 Classes: {', '.join(self.class_names)}")
        
    def _load_metadata(self):
        """Load preprocessing metadata"""
        metadata_path = self.data_root / "preprocessing_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Preprocessing metadata not found. Please run preprocess_data.py first."
            )
            
        with open(metadata_path, 'r') as f:
            return json.load(f)
            
    def get_transforms(self):
        """Get data transforms for training and validation"""
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
        
    def create_data_loaders(self):
        """Create data loaders for training and validation"""
        
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = PlantDiseaseDataset(
            self.data_root / "train", 
            transform=train_transform
        )
        
        val_dataset = PlantDiseaseDataset(
            self.data_root / "validation", 
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
        
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
        
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
        
    def train_model(self, save_name: str = None):
        """Train the complete model"""
        
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"plant_disease_model_{timestamp}"
            
        print(f"\n🚀 Starting training...")
        print(f"🔧 Device: {device}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Epochs: {self.num_epochs}")
        print(f"📈 Learning rate: {self.learning_rate}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()
        
        # Initialize model
        model = PlantDiseaseModel(num_classes=self.num_classes, pretrained=True)
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"{'EPOCH':<6} {'TRAIN_LOSS':<12} {'TRAIN_ACC':<12} {'VAL_LOSS':<12} {'VAL_ACC':<12}")
        print(f"{'='*60}")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                model_path = self.model_root / f"{save_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': self.class_names,
                    'num_classes': self.num_classes
                }, model_path)
                
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
                
        print(f"\n✅ Training completed!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
        
        # Save final model and metadata
        self._save_model_artifacts(model, history, save_name, best_val_acc)
        
        # Plot training history
        self._plot_training_history(history, save_name)
        
        return model, history
        
    def _save_model_artifacts(self, model, history, save_name, best_val_acc):
        """Save model artifacts and metadata"""
        
        # Save final model
        final_model_path = self.model_root / f"{save_name}_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'final_val_acc': history['val_acc'][-1]
        }, final_model_path)
        
        # Save training history
        history_path = self.model_root / f"{save_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        # Save model info
        model_info = {
            'model_name': save_name,
            'architecture': 'ResNet50',
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'best_val_accuracy': best_val_acc,
            'final_val_accuracy': history['val_acc'][-1],
            'training_epochs': len(history['train_loss']),
            'training_date': datetime.now().isoformat(),
            'model_files': {
                'best_model': f"{save_name}_best.pth",
                'final_model': f"{save_name}_final.pth",
                'history': f"{save_name}_history.json",
                'plot': f"{save_name}_training_plot.png"
            }
        }
        
        info_path = self.model_root / f"{save_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print(f"\n💾 Model artifacts saved:")
        print(f"   📁 Best model: {final_model_path.name}")
        print(f"   📁 Model info: {info_path.name}")
        print(f"   📁 Training history: {history_path.name}")
        
    def _plot_training_history(self, history, save_name):
        """Plot and save training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_root / f"{save_name}_training_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Training plot: {plot_path.name}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Train plant disease classification model')
    parser.add_argument('--data-root', default='data',
                       help='Root directory containing processed data (default: data)')
    parser.add_argument('--model-root', default='models',
                       help='Directory to save trained models (default: models)')
    parser.add_argument('--model-name', default=None,
                       help='Name for the saved model (default: auto-generated)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Check for processed data
        data_root = Path(args.data_root)
        if not (data_root / "train").exists():
            print("❌ Processed training data not found.")
            print("   Please run preprocess_data.py first to prepare your data.")
            return 1
            
        # Initialize trainer
        trainer = ModelTrainer(args.data_root, args.model_root)
        
        # Update parameters
        trainer.batch_size = args.batch_size
        trainer.num_epochs = args.epochs
        trainer.learning_rate = args.learning_rate
        trainer.patience = args.patience
        
        # Train model
        model, history = trainer.train_model(args.model_name)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Models saved in: {args.model_root}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
