

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class lowlight_loader(Dataset):
    """
    Academic Zero-DCE++ Dataset Class
    
    This is an academic implementation for educational purposes.
    Loads low-light images for self-supervised training without paired data.
    """
    
    def __init__(self, images_path, transform=None):
        """
        Initialize low-light dataset
        
        Args:
            images_path (str): Directory containing low-light images
            transform (callable): Optional transform to be applied
        """
        self.image_paths = []
        self.transform = transform
        
        # Get all image files from directory
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        if os.path.exists(images_path):
            for file in os.listdir(images_path):
                if file.lower().endswith(supported_formats):
                    self.image_paths.append(os.path.join(images_path, file))
        else:
            print(f"Warning: Directory {images_path} does not exist!")
            print("Please create the directory and add low-light images.")
        
        print(f"Found {len(self.image_paths)} images in {images_path}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get image at specified index
        
        Args:
            idx (int): Index of image to retrieve
            
        Returns:
            torch.Tensor: Transformed image tensor [3, H, W]
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random tensor as fallback
            return torch.rand(3, 256, 256)


def get_data_loaders(train_dir, val_dir, batch_size=8, image_size=256, num_workers=0):
    """
    Create training and validation data loaders
    
    Args:
        train_dir (str): Directory containing training images
        val_dir (str): Directory containing validation images
        batch_size (int): Batch size for data loading
        image_size (int): Target image size for training
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Converts to [0,1] range
    ])
    
    # Create datasets
    train_dataset = lowlight_loader(train_dir, transform=transform)
    val_dataset = lowlight_loader(val_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # CPU-friendly
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing Zero-DCE++ Academic Data Loader...")
    
    # Test with dummy directories
    train_loader, val_loader = get_data_loaders(
        'dataset/train', 
        'dataset/val',
        batch_size=2
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Batch range: [{batch.min().item():.3f}, {batch.max().item():.3f}]")
        break
    
    print("Data loader test completed successfully!")
