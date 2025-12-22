"""Score network with stability checks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import torch
from typing import List, Optional
import warnings


class DisorderAveragingEMA:
    """Effective Medium Approximation for disorder averaging."""
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize EMA averaging.
        
        Args:
            reduction: 'mean' for simple mean, 'weighted' for weighted averaging
        """
        if reduction not in ['mean', 'weighted']:
            raise ValueError(f"Unknown reduction: {reduction}")
        self.reduction = reduction
    
    def average(self, values: List[torch.Tensor],
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Average values over disorder ensemble.
        
        Args:
            values: List of tensors from disorder samples
            weights: Optional weights for weighted averaging
            
        Returns:
            torch.Tensor: Averaged value
        """
        if len(values) == 0:
            raise ValueError("Cannot average empty list")
        
        if self.reduction == 'mean':
            return torch.stack(values).mean(dim=0)
        
        elif self.reduction == 'weighted':
            if weights is None:
                raise ValueError("Weights required for weighted averaging")
            
            stacked = torch.stack(values)  # (n_disorder, *shape)
            
            # Reshape weights to match stacked tensor dimensions
            while weights.dim() < stacked.dim():
                weights = weights.unsqueeze(-1)
            
            # Ensure weights are on same device
            weights = weights.to(stacked.device)
            
            # Compute weighted average
            weighted_sum = (stacked * weights).sum(dim=0)
            weight_sum = weights.sum()
            
            if weight_sum == 0:
                warnings.warn("Sum of weights is zero, falling back to mean")
                return stacked.mean(dim=0)
            
            return weighted_sum / weight_sum
        
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def average_with_confidence(
        self,
        values: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Average values and compute confidence (std).
        
        Args:
            values: List of tensors from disorder samples
            
        Returns:
            Tuple of (averaged_value, standard_deviation)
        """
        stacked = torch.stack(values)  # (n_disorder, *shape)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std


class DisorderWeightingScheme:
    """Different weighting schemes for disorder averaging."""
    
    @staticmethod
    def uniform_weights(n_disorder: int, device: str = 'cpu') -> torch.Tensor:
        """Create uniform weights."""
        return torch.ones(n_disorder, device=device) / n_disorder
    
    @staticmethod
    def confidence_weights(
        values: List[torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Create weights based on confidence (inverse variance).
        
        Args:
            values: List of tensors
            temperature: Temperature parameter for softmax
            
        Returns:
            torch.Tensor: Normalized weights
        """
        # Compute variance for each value
        stacked = torch.stack(values)  # (n_disorder, *shape)
        variances = stacked.var(dim=tuple(range(1, stacked.dim())))
        
        # Inverse variance weighting with softmax
        inv_var = 1.0 / (variances + 1e-8)
        weights = torch.softmax(inv_var / temperature, dim=0)
        
        return weights
