
import sys
import os

# Add parent directory to path so we can import da_dps
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Optional, List
import warnings



class DA_DPS_Sampler:
    """Disorder-Averaged Diffusion Posterior Sampling."""
    
    def __init__(self, score_network: nn.Module,
                 measurement_operator,
                 scheduler,
                 disorder_ensemble,
                 guidance_scale: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize DA-DPS sampler.
        
        Args:
            score_network: Score network module
            measurement_operator: Measurement operator
            scheduler: Diffusion scheduler
            disorder_ensemble: DisorderEnsemble instance
            guidance_scale: Strength of measurement guidance
            device: Torch device
        """
        self.score_network = score_network.to(device)
        self.measurement_operator = measurement_operator
        self.scheduler = scheduler
        self.disorder_ensemble = disorder_ensemble
        self.guidance_scale = guidance_scale
        self.device = device
        
        # Import averaging here to avoid circular imports
        from da_dps.disorder.averaging import DisorderAveragingEMA
        self.averaging = DisorderAveragingEMA(reduction='mean')
    
    def compute_disorder_averaged_score(self, x: torch.Tensor,
                                        t: torch.Tensor) -> torch.Tensor:
        """
        Compute score averaged over disorder ensemble.
        
        Args:
            x: Input sample of shape (batch, channels, height, width)
            t: Timestep tensor
            
        Returns:
            torch.Tensor: Disorder-averaged score
        """
        scores = []
        
        for disorder_sample in self.disorder_ensemble.disorder_samples:
            # Apply disorder scaling
            x_disordered = disorder_sample * x
            
            # Compute score for this disorder realization
            score = self.score_network(x_disordered, t)
            scores.append(score)
        
        # Average over disorder ensemble
        score_avg = self.averaging.average(scores)
        
        return score_avg
    
    def _compute_disorder_averaged_likelihood_gradient(self, x: torch.Tensor,
                                                       y: torch.Tensor) -> torch.Tensor:
        """
        Compute disorder-averaged likelihood gradient.
        
        Args:
            x: Current sample
            y: Measurements
            
        Returns:
            torch.Tensor: ∇_x ||A(disorder)*x - y||^2 (disorder-averaged)
        """
        grads = []
        
        for disorder_sample in self.disorder_ensemble.disorder_samples:
            # Apply forward operator
            Ax = self.measurement_operator(x)
            
            # Apply disorder weighting
            Ax_disorder = disorder_sample * Ax
            
            # Compute residual
            residual = Ax_disorder - y
            
            # Compute gradient: ∇ ||Ax - y||^2 = 2 * A^T * residual
            grad = 2.0 * self.measurement_operator.adjoint(residual)
            grads.append(grad)
        
        # Average gradients over disorder
        grad_avg = self.averaging.average(grads)
        
        return grad_avg
    
    def sample(self, x_T: torch.Tensor, y: Optional[torch.Tensor] = None,
               num_steps: int = 100) -> torch.Tensor:
        """
        Disorder-Averaged DPS sampling.
        
        Args:
            x_T: Initial noise sample
            y: Measurements (optional for pure prior sampling)
            num_steps: Number of diffusion steps
            
        Returns:
            torch.Tensor: Reconstructed sample
        """
        x = x_T.clone()
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        for t in self.scheduler.timesteps:
            # Disorder-averaged score
            score_avg = self.compute_disorder_averaged_score(x, t)
            
            # Compute update
            x_next = x + score_avg
            
            # Add likelihood guidance if measurements provided
            if y is not None:
                likelihood_grad_avg = self._compute_disorder_averaged_likelihood_gradient(x, y)
                x_next = x_next + self.guidance_scale * likelihood_grad_avg
            
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