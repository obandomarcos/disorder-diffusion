"""Device and reproducibility utilities."""

import torch
import numpy as np
from typing import Optional


class DeviceUtils:
    """Device and reproducibility utilities."""
    
    @staticmethod
    def set_seed(seed: int):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def get_device(use_cuda: bool = True) -> torch.device:
        """
        Get torch device.
        
        Args:
            use_cuda: Whether to use CUDA if available
            
        Returns:
            torch.device instance
        """
        if use_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    @staticmethod
    def move_to_device(obj, device: torch.device):
        """Move object to device (handles tensors and modules)."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(DeviceUtils.move_to_device(item, device) for item in obj)
        elif isinstance(obj, dict):
            return {k: DeviceUtils.move_to_device(v, device) for k, v in obj.items()}
        else:
            return obj
    
    @staticmethod
    def print_device_info():
        """Print device information."""
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")