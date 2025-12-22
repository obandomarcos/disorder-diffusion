"""Dataset download and preparation utilities."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms


class DatasetDownloader:
    """Download and prepare benchmark datasets."""
    
    def __init__(self, cache_dir: str = "datasets"):
        """
        Initialize downloader.
        
        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_ffhq_local(self, num_samples: int = 100) -> np.ndarray:
        """
        Load FFHQ from local directory.
        
        Assumes FFHQ images are in datasets/ffhq/images/
        Download from: https://github.com/NVlabs/ffhq-dataset
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            np.ndarray of shape (num_samples, 3, 256, 256)
        """
        ffhq_dir = self.cache_dir / "ffhq" / "images"
        
        if not ffhq_dir.exists():
            raise FileNotFoundError(
                f"FFHQ not found at {ffhq_dir}\n"
                "Download from: https://github.com/NVlabs/ffhq-dataset\n"
                "Or use synthetic dataset instead: download_and_prepare_benchmark('synthetic')"
            )
        
        # Collect image files
        image_files = sorted(ffhq_dir.glob("*.png"))[:num_samples]
        
        if not image_files:
            raise FileNotFoundError(f"No PNG images found in {ffhq_dir}")
        
        images = []
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue
        
        return torch.stack(images).numpy()
    
    def _create_synthetic_dataset(self, num_samples: int = 100, 
                                  image_size: int = 32,
                                  channels: int = 1) -> np.ndarray:
        """
        Create synthetic dataset.
        
        Args:
            num_samples: Number of synthetic images
            image_size: Image height/width
            channels: Number of channels (1 or 3)
            
        Returns:
            np.ndarray of shape (num_samples, channels, image_size, image_size)
        """
        images = np.random.randn(num_samples, channels, image_size, image_size).astype(np.float32)
        # Normalize to [0, 1]
        images = (images - images.min()) / (images.max() - images.min())
        return images
    
    def _create_cifar10_like_dataset(self, num_samples: int = 100) -> np.ndarray:
        """
        Create CIFAR-10-like dataset (synthetic).
        
        Args:
            num_samples: Number of samples
            
        Returns:
            np.ndarray of shape (num_samples, 3, 32, 32)
        """
        images = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)
        # Normalize to [0, 1]
        images = (images - images.min()) / (images.max() - images.min())
        return images
    
    def _create_imagenet_subset(self, num_samples: int = 100) -> np.ndarray:
        """
        Create ImageNet-like dataset (synthetic).
        
        Args:
            num_samples: Number of samples
            
        Returns:
            np.ndarray of shape (num_samples, 3, 224, 224)
        """
        images = np.random.randn(num_samples, 3, 224, 224).astype(np.float32)
        # Normalize to [0, 1]
        images = (images - images.min()) / (images.max() - images.min())
        return images
    
    def download(self, dataset_name: str, num_samples: int = 100,
                 image_size: int = 32, channels: int = 1) -> np.ndarray:
        """
        Download or create dataset.
        
        Args:
            dataset_name: 'ffhq', 'cifar10', 'synthetic', 'imagenet_subset'
            num_samples: Number of samples
            image_size: Image size (for synthetic)
            channels: Number of channels (for synthetic)
            
        Returns:
            np.ndarray of images
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name == 'ffhq':
            return self._load_ffhq_local(num_samples)
        
        elif dataset_name in ['synthetic', 'random']:
            return self._create_synthetic_dataset(num_samples, image_size, channels)
        
        elif dataset_name in ['cifar10', 'cifar']:
            return self._create_cifar10_like_dataset(num_samples)
        
        elif dataset_name in ['imagenet', 'imagenet_subset']:
            return self._create_imagenet_subset(num_samples)
        
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}\n"
                f"Supported: 'ffhq', 'cifar10', 'synthetic', 'imagenet_subset'"
            )
    
    def save_dataset(self, dataset: np.ndarray, name: str, split: str = 'train'):
        """
        Save dataset as NPZ file.
        
        Args:
            dataset: np.ndarray of images
            name: Dataset name
            split: 'train', 'val', 'test'
        """
        save_path = self.cache_dir / f"{name}_{split}.npz"
        np.savez_compressed(save_path, images=dataset)
        print(f"Saved to {save_path} ({dataset.nbytes / 1e9:.2f} GB)")
        return save_path
    
    def load_dataset(self, name: str, split: str = 'train') -> np.ndarray:
        """Load dataset from NPZ file."""
        load_path = self.cache_dir / f"{name}_{split}.npz"
        if not load_path.exists():
            raise FileNotFoundError(f"Dataset not found at {load_path}")
        return np.load(load_path)['images']

def download_and_prepare_benchmark(dataset_name: str = "synthetic",
                                   num_samples: int = 100,
                                   cache_dir: str = "datasets",
                                   image_size: int = 32,
                                   channels: int = 1) -> Tuple[np.ndarray, str]:
    """
    Download and prepare benchmark dataset.
    ...
    """
    downloader = DatasetDownloader(cache_dir=cache_dir)
    
    print(f"\nDownloading {dataset_name} dataset...")
    print(f"Number of samples: {num_samples}")
    
    dataset = downloader.download(dataset_name, num_samples=num_samples, 
                                  image_size=image_size, channels=channels)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Data type: {dataset.dtype}")
    print(f"Value range: [{dataset.min():.4f}, {dataset.max():.4f}]")
    
    return dataset, cache_dir


if __name__ == '__main__':
    # Example: Create synthetic dataset
    data, cache_dir = download_and_prepare_benchmark('imagenet_subset', num_samples=100)
    print(f"\nDataset created successfully!")
    print(f"Shape: {data.shape}")
