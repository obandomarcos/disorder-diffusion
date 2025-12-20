"""Diffusion-based generative model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings


class DiffusionScheduler:
    """Simple diffusion scheduler."""
    
    def __init__(self, num_train_timesteps: int = 1000, schedule: str = 'linear'):
        """
        Initialize scheduler.
        
        Args:
            num_train_timesteps: Total number of timesteps
            schedule: 'linear' or 'quadratic' noise schedule
        """
        self.num_train_timesteps = num_train_timesteps
        self.schedule = schedule
        
        # Initialize noise schedule
        if schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        elif schedule == 'quadratic':
            betas = torch.linspace(0.0001 ** 0.5, 0.02 ** 0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        self.timesteps = torch.linspace(num_train_timesteps - 1, 0, num_train_timesteps).long()
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer (simple version)."""
        setattr(self, name, tensor)
    
    def set_timesteps(self, num_steps: int, device: str = 'cpu'):
        """Set number of inference timesteps."""
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_steps,
            device=device
        ).long()
    
    def step(self, model_output: torch.Tensor, timestep: int,
             sample: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Perform diffusion step.
        
        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Tuple of (prev_sample, dict with step info)
        """
        timestep_int = int(timestep) if torch.is_tensor(timestep) else timestep
        
        # Simple Euler step
        alpha = self.alphas_cumprod[timestep_int]
        prev_sample = sample - 0.01 * model_output
        
        return prev_sample, {'alpha': alpha.item()}


class DiffusionGenerator:
    """Diffusion-based generative model."""
    
    def __init__(self, score_network: Optional[nn.Module] = None,
                 num_timesteps: int = 1000, device: str = 'cpu'):
        """
        Initialize diffusion generator.
        
        Args:
            score_network: Score network module (if None, creates default)
            num_timesteps: Number of diffusion timesteps
            device: Torch device
        """
        self.device = device
        
        # Create score network if not provided
        if score_network is None:
            from .score_network import ScoreNetwork
            self.score_network = ScoreNetwork(
                in_channels=1,
                hidden_dim=128,
                n_layers=3
            ).to(device)
        else:
            self.score_network = score_network.to(device)
        
        # Initialize scheduler
        self.scheduler = DiffusionScheduler(num_timesteps)
    
    def get_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute score (gradient of log probability).
        
        Args:
            x: Input sample
            t: Timestep
            
        Returns:
            torch.Tensor: Score estimate
        """
        return self.score_network(x, t)
    
    def sample_step(self, x: torch.Tensor, t: torch.Tensor,
                    guidance: Optional[torch.Tensor] = None,
                    guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Perform single sampling step.
        
        Args:
            x: Current sample
            t: Current timestep
            guidance: Optional guidance signal
            guidance_scale: Guidance strength
            
        Returns:
            torch.Tensor: Next sample
        """
        # Compute score
        score = self.get_score(x, t)
        
        # Add guidance if provided
        if guidance is not None:
            score = score + guidance_scale * guidance
        
        # Scheduler step
        x_next, _ = self.scheduler.step(score, t, x)
        
        return x_next