"""Score network with stability checks and Diffusers integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

from diffusers import DiffusionPipeline, DDPMScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass


@dataclass
class UNet2DOutput(BaseOutput):
    """Output of UNet2DModel forward pass."""
    sample: torch.FloatTensor


class ScoreNetwork(nn.Module):
    """Score/noise prediction network with stability checks and Diffusers compatibility."""
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128,
                 n_layers: int = 3, max_norm: float = 1e6, out_channels: int = None):
        """
        Initialize score network.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            hidden_dim: Hidden dimension size
            n_layers: Number of conv layers
            max_norm: Maximum allowed gradient norm
            out_channels: Output channels (defaults to in_channels)
            
        Example:
            # Grayscale (32x32)
            net = ScoreNetwork(in_channels=1, hidden_dim=64)
            
            # RGB (32x32)
            net = ScoreNetwork(in_channels=3, hidden_dim=64)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.max_norm = max_norm
        self.out_channels = out_channels if out_channels is not None else in_channels
        
        # Time embedding (for Diffusers compatibility)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Build convolutional layers with residual connections
        self.layers = nn.ModuleList()
        in_ch = in_channels
        
        for i in range(n_layers - 1):
            self.layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            in_ch = hidden_dim
        
        # Output layer
        self.layers.append(nn.Conv2d(in_ch, self.out_channels, kernel_size=3, padding=1))
        
        # Option: use Sequential for simpler case
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, sample: torch.Tensor, timestep: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> torch.Tensor:
        """
        Compute score with stability checks. Compatible with Diffusers API.
        
        Args:
            sample: Input sample of shape (batch, channels, height, width)
            timestep: Diffusers timestep (optional, single value or batch)
            return_dict: Whether to return dict (Diffusers compatibility)
            
        Returns:
            torch.Tensor or UNet2DOutput: Score estimate of same shape as input
        """
        score = self.net(sample)
        
        # Check for numerical issues
        if torch.isnan(score).any():
            raise RuntimeError("Score computation produced NaN")
        if torch.isinf(score).any():
            raise RuntimeError("Score computation produced Inf")
        
        if return_dict:
            return UNet2DOutput(sample=score)
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
        score = self.forward(x, t, return_dict=False)
        
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
        score = self.forward(x, t, return_dict=False)
        
        # Compute gradient norm
        if score.requires_grad:
            grad_norm = torch.autograd.grad(
                outputs=score.sum(),
                inputs=x,
                create_graph=False,
                retain_graph=True
            )[0].norm()
            
            if grad_norm > self.max_norm:
                warnings.warn(f"Gradient norm {grad_norm:.4f} exceeds max {self.max_norm}")
        
        return score


# ==============================================================================
# DIFFUSERS INTEGRATION EXAMPLES
# ==============================================================================

def load_cifar10_diffusion_pipeline(model_path: str = None, device: str = "cpu"):
    """
    Load a diffusion pipeline with ScoreNetwork.
    
    Args:
        model_path: Path to saved ScoreNetwork checkpoint
        device: Device to load on ("cpu" or "cuda")
        
    Returns:
        tuple: (pipeline, scheduler, model)
    """
    # Initialize score network
    score_net = ScoreNetwork(in_channels=3, hidden_dim=128, n_layers=4)
    
    # Load checkpoint if provided
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        score_net.load_state_dict(checkpoint)
    
    score_net = score_net.to(device)
    score_net.eval()
    
    # Initialize scheduler (DDPM for CIFAR-10)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )
    
    return score_net, scheduler


def inference_with_custom_scheduler(score_net: ScoreNetwork, 
                                     num_inference_steps: int = 50,
                                     batch_size: int = 4,
                                     image_size: int = 32,
                                     device: str = "cpu") -> torch.Tensor:
    """
    Generate images using custom score network with Diffusers scheduler.
    
    Args:
        score_net: Trained ScoreNetwork model
        num_inference_steps: Number of denoising steps
        batch_size: Number of images to generate
        image_size: Image resolution (32 for CIFAR-10)
        device: Device to generate on
        
    Returns:
        torch.Tensor: Generated images (batch_size, 3, 32, 32)
    """
    from diffusers.schedulers import DDPMScheduler
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Start from pure noise
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    score_net.eval()
    with torch.no_grad():
        for t in scheduler.timesteps:
            # Get score prediction
            score = score_net(x, t)
            if isinstance(score, UNet2DOutput):
                score = score.sample
            
            # Apply scheduler step
            x = scheduler.step(score, t, x).prev_sample
    
    # Clamp to valid image range
    x = torch.clamp(x, -1.0, 1.0)
    return x


def training_loop_with_diffusers(score_net: ScoreNetwork,
                                  train_loader,
                                  num_epochs: int = 100,
                                  learning_rate: float = 1e-3,
                                  device: str = "cpu"):
    """
    Training loop using Diffusers scheduler and loss computation.
    
    Args:
        score_net: ScoreNetwork model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        dict: Training history
    """
    from diffusers.schedulers import DDPMScheduler
    
    optimizer = torch.optim.Adam(score_net.parameters(), lr=learning_rate)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    score_net.train()
    history = {"loss": []}
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, 1000, (images.shape[0],), device=device)
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # Add noise to images (forward process)
            noisy_images = scheduler.add_noise(images, noise, t)
            
            # Predict noise with score network
            predicted_noise = score_net(noisy_images, t)
            if isinstance(predicted_noise, UNet2DOutput):
                predicted_noise = predicted_noise.sample
            
            # MSE loss between predicted and actual noise
            loss = F.mse_loss(predicted_noise, noise)
            
            # Stability check
            loss = score_net.compute_with_stability_check(noisy_images, t)
            loss = F.mse_loss(loss, noise)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
    
    return history


# ==============================================================================
# PRETRAINED MODEL LOADING (Google DDPM CIFAR-10)
# ==============================================================================

def load_pretrained_diffusion_model(model_id: str = "google/ddpm-cifar10-32",
                                    device: str = "cpu"):
    """
    Load pretrained model from Hugging Face Hub.
    
    Args:
        model_id: Model identifier on Hugging Face (e.g., "google/ddpm-cifar10-32")
        device: Device to load on
        
    Returns:
        DiffusionPipeline: Ready-to-use pipeline
    """
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    return pipeline


def generate_with_pretrained(model_id: str = "google/ddpm-cifar10-32",
                             num_samples: int = 4,
                             device: str = "cpu"):
    """
    Generate images using pretrained Hugging Face model.
    
    Args:
        model_id: Model from Hugging Face Hub
        num_samples: Number of images to generate
        device: Device to use
        
    Returns:
        PIL.Image: Generated image
    """
    pipeline = load_pretrained_diffusion_model(model_id, device)
    
    # Generate
    with torch.no_grad():
        image = pipeline(batch_size=num_samples).images
    
    return image


# ==============================================================================
# QUICK START EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Option 1: Use custom ScoreNetwork with Diffusers scheduler
    print("=== Custom ScoreNetwork with Diffusers ===")
    score_net, scheduler = load_cifar10_diffusion_pipeline(device=device)
    images = inference_with_custom_scheduler(score_net, device=device)
    print(f"Generated images shape: {images.shape}")
    
    # Option 2: Use pretrained Google DDPM from Hugging Face
    print("\n=== Pretrained Google DDPM ===")
    try:
        images = generate_with_pretrained("google/ddpm-cifar10-32", device=device)
        print(f"Generated {len(images)} images from pretrained model")
    except Exception as e:
        print(f"Pretrained model loading failed: {e}")
        print("Install with: pip install diffusers transformers")
