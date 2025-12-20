"""Standard Diffusion Posterior Sampling (no disorder)."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings


class DPS_Sampler:
    """Standard Diffusion Posterior Sampling (no disorder)."""
    
    def __init__(self, score_network: nn.Module,
                 measurement_operator,
                 scheduler,
                 guidance_scale: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize DPS sampler.
        
        Args:
            score_network: Score network module
            measurement_operator: Measurement operator with forward/adjoint
            scheduler: Diffusion scheduler
            guidance_scale: Strength of measurement guidance
            device: Torch device
        """
        self.score_network = score_network.to(device)
        self.measurement_operator = measurement_operator
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.device = device
    
    def _compute_likelihood_gradient(self, x: torch.Tensor,
                                     y: torch.Tensor) -> torch.Tensor:
        """
        Compute likelihood gradient.
        
        Args:
            x: Current sample
            y: Measurements
            
        Returns:
            torch.Tensor: ∇_x ||A(x) - y||^2
        """
        # Forward operator
        Ax = self.measurement_operator(x)
        
        # Residual
        residual = Ax - y
        
        # Likelihood gradient: ∇ ||Ax - y||^2 = 2 * A^T * (Ax - y)
        grad = 2.0 * self.measurement_operator.adjoint(residual)
        
        return grad
    
    def sample(self, x_T: torch.Tensor, y: torch.Tensor,
               num_steps: int = 100) -> torch.Tensor:
        """
        Standard DPS sampling.
        
        Args:
            x_T: Initial noise sample
            y: Measurements
            num_steps: Number of diffusion steps
            
        Returns:
            torch.Tensor: Reconstructed sample
        """
        x = x_T.clone()
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        for t in self.scheduler.timesteps:
            # Compute score
            score = self.score_network(x, t)
            
            # Compute likelihood gradient
            likelihood_grad = self._compute_likelihood_gradient(x, y)
            
            # Combined update
            x_next = x + self.guidance_scale * likelihood_grad + score
            
            # Apply scheduler step
            x = self.scheduler.step(x_next, t, x)[0]
        
        return x
    
    def sample_with_guidance_scale(self, x_T: torch.Tensor, y: torch.Tensor,
                                   num_steps: int = 100,
                                   guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Sample with custom guidance scale."""
        original_scale = self.guidance_scale
        if guidance_scale is not None:
            self.guidance_scale = guidance_scale
        
        try:
            result = self.sample(x_T, y, num_steps)
        finally:
            self.guidance_scale = original_scale
        
        return result