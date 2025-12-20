"""Disorder ensemble management for DA-DPS."""

import torch
import numpy as np
from typing import Optional, Dict, List, Iterator
import warnings


class DisorderEnsemble:
    """Manages disorder ensemble for DA-DPS."""
    
    def __init__(self, n_disorder: int, disorder_dist, device: str = 'cpu'):
        """
        Create disorder ensemble.
        
        Args:
            n_disorder: Number of disorder realizations
            disorder_dist: DisorderDistribution instance
            device: Torch device
        """
        if n_disorder <= 0:
            raise ValueError(f"n_disorder must be > 0, got {n_disorder}")
        
        self.n_disorder = n_disorder
        self.disorder_dist = disorder_dist
        self.device = device
        
        # Sample disorder ensemble
        self.disorder_samples = disorder_dist.sample(n_disorder, device=device)
        
        # Validation
        self.validate_numerical_stability()
    
    def validate_numerical_stability(self) -> bool:
        """
        Check for numerical issues in disorder samples.
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If NaN or Inf detected
        """
        samples = self.disorder_samples
        
        # Check for NaN and Inf
        if torch.isnan(samples).any():
            raise ValueError("Disorder samples contain NaN")
        if torch.isinf(samples).any():
            raise ValueError("Disorder samples contain Inf")
        
        # Check mean is reasonable
        mean = samples.mean()
        if mean < 0.1 or mean > 10.0:
            warnings.warn(
                f"Disorder mean {mean:.4f} is outside expected range [0.1, 10.0]"
            )
        
        return True
    
    def __len__(self) -> int:
        """Return number of disorder samples."""
        return self.n_disorder
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get disorder sample by index."""
        return self.disorder_samples[idx]
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over disorder samples."""
        return iter(self.disorder_samples)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics of disorder ensemble.
        
        Returns:
            Dict with 'mean', 'std', 'min', 'max', 'median'
        """
        return {
            'mean': float(self.disorder_samples.mean()),
            'std': float(self.disorder_samples.std()),
            'min': float(self.disorder_samples.min()),
            'max': float(self.disorder_samples.max()),
            'median': float(torch.median(self.disorder_samples)),
        }
    
    @staticmethod
    def estimate_convergence(
        true_value: float,
        disorder_dist,
        n_samples_list: List[int],
        n_trials: int = 100
    ) -> Dict[int, float]:
        """
        Estimate convergence of disorder averaging (bias-variance tradeoff).
        
        Args:
            true_value: True value to estimate
            disorder_dist: DisorderDistribution instance
            n_samples_list: List of sample counts to test
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Dict mapping n_samples to variance of estimates
        """
        variances = {}
        
        for n_samples in n_samples_list:
            estimates = []
            for _ in range(n_trials):
                samples = disorder_dist.sample(n_samples)
                estimate = samples.mean().item()
                estimates.append(estimate)
            
            # Compute variance
            variances[n_samples] = float(np.var(estimates))
        
        return variances