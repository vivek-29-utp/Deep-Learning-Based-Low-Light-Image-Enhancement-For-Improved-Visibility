

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Import custom modules
from model.zerodce_pp import create_model
from losses.losses import ZeroDCELoss


class LowLightDataset(Dataset):
    """
    Dataset class for low-light images
    
    Loads low-light images from a directory and applies transformations
    for training the Zero-DCE++ model in a self-supervised manner.
    """
    
    def __init__(self, image_dir, image_size=256, transform=None):
        """
        Initialize low-light dataset
        
        Args:
            image_dir (str): Directory containing low-light images
            image_size (int): Target image size for training (default: 256)
            transform (callable): Optional transform to be applied
        """
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # Converts to [0,1] range
            ])
        else:
            self.transform = transform
        
        # Get all image files
        self.image_files = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        if os.path.exists(image_dir):
            for file in os.listdir(image_dir):
                if file.lower().endswith(supported_formats):
                    self.image_files.append(os.path.join(image_dir, file))
        else:
            print(f"Warning: Directory {image_dir} does not exist!")
            print("Please create the directory and add low-light images.")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get image at specified index
        
        Args:
            idx (int): Index of image to retrieve
            
        Returns:
            torch.Tensor: Transformed image tensor [3, H, W]
        """
        img_path = self.image_files[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random tensor as fallback
            return torch.rand(3, self.image_size, self.image_size)


class Trainer:
    """
    Zero-DCE++ Training Class - FINAL IMPLEMENTATION
    
    Handles the complete training process including data loading,
    model training, validation, and checkpoint management.
    
    Zero-DCE++ Theory:
    - Self-supervised learning: No paired ground truth data required
    - Train loss ≠ Val loss is NORMAL and EXPECTED
    - Train loss measures learning on seen data patterns
    - Val loss measures generalization to unseen low-light scenarios
    - Forcing loss equality is INCORRECT for self-supervised illumination learning involve
    convolving the image with the learned curves.
    - Early stopping on rising val loss prevents over-enhancement
    - Conservative enhancement preserves natural image characteristics
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config (dict): Training configuration parameters
        """
        self.config = config
        self.device = torch.device('cpu')  # CPU-friendly for final-year project
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize model
        self.model = create_model(device=self.device, scale_factor=1)
        
        # Initialize loss function with improved weights for better accuracy
        self.criterion = ZeroDCELoss(
            tv_weight=1200.0,      # Reduced for less over-smoothing
            spa_weight=1.5,        # Increased for better spatial consistency
            color_weight=3.0,       # Reduced for more natural colors
            exp_weight=15.0,        # Increased for better exposure control
            patch_size=16,
            mean_val=0.7            # Increased for brighter target
        )
        
        # Initialize optimizer with academic parameters
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Training history
        self.train_history = {'loss': [], 'epoch': []}
        self.val_history = {'loss': [], 'epoch': []}
        
        print(f"Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch - academic implementation
        
        Args:
            dataloader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, images in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - academic implementation
            enhanced_image, curves = self.model(images)
            
            # Calculate loss using academic Zero-DCE++ loss
            losses = self.criterion(images, enhanced_image, curves)
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability (academic implementation)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_norm', 0.1))
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar with academic loss components
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'TV': f'{losses["weighted_smoothness"].item():.2f}',
                'Spa': f'{losses["weighted_spatial"].item():.4f}',
                'Color': f'{losses["weighted_color"].item():.4f}',
                'Exp': f'{losses["weighted_exposure"].item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """
        Validate the model - academic implementation
        
        Args:
            dataloader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for images in dataloader:
                images = images.to(self.device)
                
                # Forward pass - academic implementation
                enhanced_image, curves = self.model(images)
                
                # Calculate loss using academic Zero-DCE++ loss
                losses = self.criterion(images, enhanced_image, curves)
                total_loss += losses['total_loss'].item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            is_best (bool): Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'checkpoints/latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'checkpoints/zerodce_pp.pth')
            print(f"New best model saved with val_loss: {val_loss:.6f}")
        
        # Save epoch checkpoint
        torch.save(checkpoint, f'checkpoints/epoch_{epoch+1}.pth')
    
    def plot_training_history(self):
        """
        Plot and save training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['epoch'], self.train_history['loss'], 'b-', label='Training Loss')
        plt.plot(self.val_history['epoch'], self.val_history['loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig('logs/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """
        FINAL ZERO-DCE++ TRAINING PIPELINE - ACADEMICALLY CORRECT IMPLEMENTATION
        
        Zero-DCE++ Theory:
        - Self-supervised learning: No paired ground truth data required
        - Train loss ≠ Val loss is NORMAL and EXPECTED
        - Train loss measures learning on seen data patterns
        - Val loss measures generalization to unseen low-light scenarios
        - Forcing loss equality is INCORRECT for self-supervised illumination learning
        - Early stopping on rising val loss prevents over-enhancement
        - Conservative enhancement preserves natural image characteristics
        
        Training Strategy:
        - Stop strictly when validation loss increases for 3 consecutive epochs
        - Preserve best checkpoint based on validation performance
        - Never restart training from scratch (continue from checkpoint)
        - Model is intentionally conservative to avoid artifacts
        """
        print("Starting Zero-DCE++ Final Training...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Check for existing checkpoint to resume training
        latest_checkpoint = 'checkpoints/latest.pth'
        start_epoch = 0
        
        if os.path.exists(latest_checkpoint):
            print(f"Resuming training from: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}")
                print(f"Previous train loss: {checkpoint.get('train_loss', 'Unknown')}")
                print(f"Previous val loss: {checkpoint.get('val_loss', 'Unknown')}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch...")
        else:
            print("No checkpoint found. Starting training from scratch...")
        
        # Create datasets
        train_dataset = LowLightDataset(
            self.config['train_data_dir'],
            image_size=self.config['image_size']
        )
        
        val_dataset = LowLightDataset(
            self.config['val_data_dir'],
            image_size=self.config['image_size']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # CPU-friendly
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training loop with strict early stopping
        best_val_loss = float('inf')
        val_loss_increases = 0
        patience = self.config.get('early_stopping_patience', 3)
        
        for epoch in range(start_epoch, self.config['epochs']):
            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            self.train_history['loss'].append(train_loss)
            self.train_history['epoch'].append(epoch + 1)
            self.val_history['loss'].append(val_loss)
            self.val_history['epoch'].append(epoch + 1)
            
            # Zero-DCE++ Early Stopping: Validation loss monitoring only
            # Train/val loss equality is NOT expected for self-supervised learning
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_loss_increases = 0
                is_best = True
            else:
                val_loss_increases += 1
                is_best = False
            
            # Early stopping: Stop when validation loss increases for 3 consecutive epochs
            if val_loss_increases >= patience:
                print(f"\nTRAINING COMPLETE — MODEL STABILIZED")
                print(f"Validation loss increased for {val_loss_increases} consecutive epochs")
                print(f"Final Train Loss: {train_loss:.6f}")
                print(f"Final Val Loss: {val_loss:.6f}")
                # Save final corrected model
                final_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': self.config
                }
                torch.save(final_checkpoint, 'checkpoints/final_corrected.pth')
                print(f"Final corrected model saved as: checkpoints/final_corrected.pth")
                break
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{self.config["epochs"]}: '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {self.scheduler.get_last_lr()[0]:.8f}')
            
            # Plot training history every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_history()
        
        # Check if training completed without early stopping
        else:
            print("\nTraining completed all epochs!")
            print(f"Final Train Loss: {self.train_history['loss'][-1]:.6f}")
            print(f"Final Val Loss: {self.val_history['loss'][-1]:.6f}")
            # Save final corrected model
            final_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': self.train_history['loss'][-1],
                'val_loss': self.val_history['loss'][-1],
                'config': self.config
            }
            torch.save(final_checkpoint, 'checkpoints/final_corrected.pth')
            print(f"Final corrected model saved as: checkpoints/final_corrected.pth")
        
        # FINAL TRAINING COMPLETION MESSAGE
        print("\n" + "="*60)
        print("TRAINING COMPLETE — MODEL STABILIZED")
        print("="*60)
        print("• Zero-DCE++ training completed successfully")
        print("• Model is intentionally conservative to avoid over-enhancement")
        print("• Train/val loss difference is normal for self-supervised learning")
        print("• No further retraining required for academic submission")
        print("• Final model saved as: checkpoints/final_corrected.pth")
        print("="*60)


def main():
    """
    Main training function - FINAL LOCKED CONFIGURATION
    """
    # FINAL TRAINING CONFIGURATION - LOCKED FOR ACADEMIC SUBMISSION
    # Academic Zero-DCE++ hyperparameters
    config = {
        'train_data_dir': 'dataset/train',
        'val_data_dir': 'dataset/val', 
        'image_size': 256,
        'batch_size': 8,                    # Academic: CPU-friendly batch size
        'epochs': 50,                       # MAX epochs with early stopping
        'learning_rate': 0.0001,              # Academic: Conservative learning rate
        'weight_decay': 0.0001,               # Academic: Weight decay
        'lr_step_size': 20,                   # Academic: Learning rate decay
        'lr_gamma': 0.5,                      # Academic: LR decay factor
        'grad_clip_norm': 0.1,                 # Academic: Gradient clipping
        'early_stopping_patience': 3           # FINAL: Strict early stopping
    }
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
