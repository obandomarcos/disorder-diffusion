#!/usr/bin/env python3
"""
Ablation study components for Disorder-Aware DPS.

Includes scheduler, sampler, and visualization utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
import time


class DisorderAwareScheduler:
    """Noise schedule with disorder modulation."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        disorder_ensemble: Optional[object] = None,
        disorder_weight: float = 0.5,
        enable_disorder_schedule: bool = True,
        device: str = "cuda",
    ):
        """Initialize scheduler."""
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.disorder_ensemble = disorder_ensemble
        self.disorder_weight = disorder_weight
        self.enable_disorder_schedule = enable_disorder_schedule
        self.device = device
        
        # Create base schedule
        self.register_schedule()
    
    def register_schedule(self):
        """Create and register noise schedule."""
        self.base_betas = self._create_base_schedule()
        self.register_buffer('betas', self.base_betas)
    
    def _create_base_schedule(self) -> torch.Tensor:
        """Create base noise schedule."""
        if self.schedule_type == "linear":
            return torch.linspace(1e-4, 0.02, self.num_timesteps, device=self.device)
        
        elif self.schedule_type == "cosine" or self.schedule_type == "disorder_adaptive":
            steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, device=self.device)
            alphas_cumprod = torch.cos(((steps / self.num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer (for compatibility)."""
        setattr(self, name, tensor)
    
    def get_betas(self) -> torch.Tensor:
        """Get noise schedule betas."""
        return self.base_betas
    
    def step(self, x_t: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
        """Single diffusion step."""
        beta_t = self.base_betas[t]
        alpha_t = 1.0 - beta_t
        
        # Simple Euler step
        x_t_minus_1 = torch.sqrt(alpha_t) * x_t + torch.sqrt(beta_t) * noise
        
        return x_t_minus_1


class AblationDA_DPS_Sampler:
    """Disorder-aware DPS sampler for ablation studies."""
    
    def __init__(
        self,
        score_network: nn.Module,
        measurement_operator,
        scheduler: Optional[DisorderAwareScheduler] = None,
        disorder_ensemble: Optional[object] = None,
        guidance_scale: float = 1.0,
        device: str = "cuda",
        config=None,  # Accept config but may not use it
    ):
        """
        Initialize DA-DPS sampler.
        
        Args:
            score_network: Score function neural network
            measurement_operator: Linear measurement operator
            scheduler: Noise schedule
            disorder_ensemble: Disorder ensemble
            guidance_scale: Guidance scale for measurement fidelity
            device: PyTorch device
            config: Optional configuration object
        """
        self.score_network = score_network
        self.measurement_operator = measurement_operator
        self.scheduler = scheduler or DisorderAwareScheduler(device=device)
        self.disorder_ensemble = disorder_ensemble
        self.guidance_scale = guidance_scale
        self.device = device
        self.config = config
    
    def _compute_measurement_gradient(
        self,
        x_0_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient for measurement consistency.
        
        Args:
            x_0_pred: Predicted clean image [batch, channels, height, width]
            y_true: True measurements [batch, n_measurements]
            y_pred: Predicted measurements [batch, n_measurements]
        
        Returns:
            Gradient w.r.t. x_0_pred
        """
        # Ensure tensors are on same device
        x_0_pred = x_0_pred.to(self.device)
        y_true = y_true.to(self.device)
        
        # Detach and enable gradient tracking
        x_0_pred = x_0_pred.detach().requires_grad_(True)
        
        # Recompute y_pred with gradients enabled
        with torch.enable_grad():
            y_pred_grad = self.measurement_operator(x_0_pred)
            
            # Ensure shapes match - flatten if needed
            if y_pred_grad.shape != y_true.shape:
                y_pred_grad = y_pred_grad.view(y_true.shape[0], -1)
                y_true_flat = y_true.view(y_true.shape[0], -1)
            else:
                y_true_flat = y_true
            
            # Compute MSE loss
            loss = torch.mean((y_pred_grad - y_true_flat) ** 2)
            
            # Backprop to get gradient
            grad = torch.autograd.grad(loss, x_0_pred, create_graph=False)[0]
        
        return grad
    
    def sample(
        self,
        x_T: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 50,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Sample from reverse diffusion process with disorder awareness.
        
        Args:
            x_T: Noise tensor [batch, channels, height, width]
            y: Measurements [batch, n_measurements]
            num_steps: Number of diffusion steps
        
        Returns:
            x_recon: Reconstructed image
            metrics: Dictionary of sampling metrics
        """
        batch_size = x_T.shape[0]
        x_t = x_T.clone()
        
        metrics = {
            'loss_history': [],
            'inference_time': 0.0,
        }
        
        start_time = time.time()
        
        with torch.no_grad():
            for t in range(num_steps - 1, -1, -1):
                # Get time embedding
                t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                
                # Predict x_0
                x_0_pred = self.score_network(x_t, t_batch)
                
                # Compute measurement predictions
                y_pred = self.measurement_operator(x_0_pred)
                
                # Compute measurement gradient
                measurement_grad = self._compute_measurement_gradient(x_0_pred, y, y_pred)
                
                # Get disorder modulation - USE PRE-SAMPLED DISORDER SAMPLES
                if self.disorder_ensemble is not None:
                    # Get random disorder sample from pre-sampled ensemble
                    disorder_idx = torch.randint(0, len(self.disorder_ensemble), (1,)).item()
                    disorder_scale = self.disorder_ensemble[disorder_idx]
                else:
                    disorder_scale = 1.0
                
                # Ensure disorder_scale is tensor
                if not isinstance(disorder_scale, torch.Tensor):
                    disorder_scale = torch.tensor(disorder_scale, device=self.device, dtype=x_t.dtype)
                else:
                    disorder_scale = disorder_scale.to(self.device).to(x_t.dtype)
                
                # Compute noise update with disorder modulation
                noise = torch.randn_like(x_t)
                
                # Get diffusion scheduler step
                if self.scheduler is not None and hasattr(self.scheduler, 'step'):
                    x_t = self.scheduler.step(x_t, t, noise)
                else:
                    # Fallback: simple Euler step
                    alpha = 1.0 - (t / num_steps)
                    x_t = alpha * x_0_pred + (1.0 - alpha) * noise
                
                # Apply measurement guidance
                if self.guidance_scale > 0:
                    x_t = x_t - self.guidance_scale * measurement_grad
                
                # Apply disorder modulation
                if disorder_scale != 1.0:
                    x_t = disorder_scale * x_t
        
        metrics['inference_time'] = time.time() - start_time
        
        return x_t, metrics


class AblationVisualizer:
    """Visualization utilities for ablation results."""
    
    def __init__(self, results_path: str, output_dir: str = "./"):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to ablation results JSON
            output_dir: Output directory for plots
        """
        self.results_path = results_path
        self.output_dir = output_dir
    
    def plot_all(self):
        """Generate all visualizations."""
        import json
        from pathlib import Path
        
        results_path = Path(self.results_path)
        if not results_path.exists():
            print(f"Warning: Results file not found: {self.results_path}")
            return
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print(f"Loaded results from {self.results_path}")
            print(f"Results keys: {results.keys()}")
            
        except Exception as e:
            print(f"Error loading results: {e}")
