"""Score network with stability checks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScoreNetwork(nn.Module):
    """Score/noise prediction network with stability checks."""
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128,
                 n_layers: int = 3, max_norm: float = 1e6):
        """
        Initialize score network.
        
        Args:
            in_channels: Number of input channels
            hidden_dim: Hidden dimension size
            n_layers: Number of conv layers
            max_norm: Maximum allowed gradient norm
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.max_norm = max_norm
        
        # Build convolutional layers
        layers = []
        in_ch = in_channels
        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = hidden_dim
        
        # Output layer
        layers.append(nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute score with stability checks.
        
        Args:
            x: Input sample
            t: Timestep (currently unused, for API compatibility)
            
        Returns:
            torch.Tensor: Score estimate
        """
        score = self.net(x)
        
        # Check for numerical issues
        if torch.isnan(score).any():
            raise RuntimeError("Score computation produced NaN")
        if torch.isinf(score).any():
            raise RuntimeError("Score computation produced Inf")
        
        return score
    
    def compute_with_stability_check(self, x: torch.Tensor,
                                     t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute score with gradient clipping.
        
        Args:
            x: Input sample
            t: Timestep
            
        Returns:
            torch.Tensor: Clipped score
        """
        score = self.forward(x, t)
        
        # Clip to prevent explosion
        score = torch.clamp(score, -self.max_norm, self.max_norm)
        
        return score
    
    def compute_with_grad_norm_check(self, x: torch.Tensor,
                                     t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute score and check gradient norms.
        
        Args:
            x: Input sample
            t: Timestep
            
        Returns:
            torch.Tensor: Score
        """
        x.requires_grad_(True)
        score = self.forward(x, t)
        
        # Compute gradient norm
        if score.requires_grad:
            grad_norm = torch.autograd.grad(
                outputs=score.sum(),
                inputs=x,
                create_graph=False,
                retain_graph=True
            )[0].norm()
            
            if grad_norm > self.max_norm:
                import warnings
                warnings.warn(f"Gradient norm {grad_norm:.4f} exceeds max {self.max_norm}")
        
        return score