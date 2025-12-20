"""Configuration for DA-DPS."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import dataclass
import torch


@dataclass
class DADPSConfig:
    """Configuration for DA-DPS."""
    
    # Disorder parameters
    disorder_type: str = 'uniform'  # 'uniform', 'normal', 'lognormal'
    disorder_low: float = 0.8
    disorder_high: float = 1.2
    n_disorder: int = 10
    
    # Diffusion parameters
    num_diffusion_steps: int = 100
    diffusion_model: str = 'default'
    guidance_scale: float = 1.0
    
    # Measurement parameters
    measurement_operator: str = 'gaussian_cs'  # 'gaussian_cs', 'blur', 'inpainting'
    n_measurements: int = 256
    n_pixels: int = 1024
    noise_std: float = 0.01
    
    # Network parameters
    score_network_hidden_dim: int = 128
    score_network_layers: int = 3
    
    # Computation parameters
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_disorder > 0, "n_disorder must be > 0"
        assert self.num_diffusion_steps > 0, "num_diffusion_steps must be > 0"
        assert self.guidance_scale >= 0, "guidance_scale must be >= 0"
        assert self.noise_std > 0, "noise_std must be > 0"