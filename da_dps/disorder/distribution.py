"""Disorder distribution classes for DA-DPS."""
import torch
import numpy as np
from typing import Optional, Dict, List
import warnings


class DisorderDistribution:
    """Base class for disorder distributions."""
    
    def __init__(self, distribution_type: str = 'uniform', **kwargs):
        """
        Initialize disorder distribution.
        
        Args:
            distribution_type: Type of distribution ('uniform', 'normal', 'lognormal')
            **kwargs: Distribution-specific parameters
                - For 'uniform': low, high
                - For 'normal': mean, std
                - For 'lognormal': mu, sigma
        """
        self.distribution_type = distribution_type
        self.params = kwargs
        self._validate_params()
    
    def _validate_params(self):
        """Validate distribution parameters."""
        if self.distribution_type == 'uniform':
            self.low = self.params.get('low', 0.8)
            self.high = self.params.get('high', 1.2)
            assert self.low < self.high, "low must be < high"
            assert self.low > 0, "low must be positive"
        
        elif self.distribution_type == 'normal':
            self.mean = self.params.get('mean', 1.0)
            self.std = self.params.get('std', 0.1)
            assert self.std > 0, "std must be positive"
        
        elif self.distribution_type == 'lognormal':
            self.mu = self.params.get('mu', 0.0)
            self.sigma = self.params.get('sigma', 0.2)
            assert self.sigma > 0, "sigma must be positive"
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from disorder distribution.
        
        Args:
            n_samples: Number of samples
            device: Torch device ('cpu', 'cuda', etc.)
            
        Returns:
            torch.Tensor: Disorder samples of shape (n_samples,)
        """
        if self.distribution_type == 'uniform':
            return self.low + (self.high - self.low) * torch.rand(
                n_samples, device=device
            )
        
        elif self.distribution_type == 'normal':
            return torch.randn(n_samples, device=device) * self.std + self.mean
        
        elif self.distribution_type == 'lognormal':
            normal_samples = torch.randn(n_samples, device=device)
            return torch.exp(self.mu + self.sigma * normal_samples)
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution_type}")
    
    def validate_extreme_values(self, samples: torch.Tensor) -> bool:
        """
        Validate handling of extreme disorder values.
        
        Args:
            samples: Disorder samples to validate
            
        Returns:
            bool: True if valid
        """
        min_val = samples.min().item()
        max_val = samples.max().item()
        
        # Check bounds
        assert min_val >= 1e-10, f"Minimum disorder {min_val} too small"
        assert max_val <= 1e6, f"Maximum disorder {max_val} too large"
        
        # Check for numerical issues
        assert not torch.isnan(samples).any(), "Samples contain NaN"
        assert not torch.isinf(samples).any(), "Samples contain Inf"
        
        return True
    
    def get_statistics(self, n_samples: int = 10000) -> Dict[str, float]:
        """
        Get theoretical statistics for the distribution.
        
        Returns:
            Dict with 'mean', 'std', 'min', 'max'
        """
        samples = self.sample(n_samples)
        return {
            'mean': float(samples.mean()),
            'std': float(samples.std()),
            'min': float(samples.min()),
            'max': float(samples.max()),
        }