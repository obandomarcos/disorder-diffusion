"""
FIXED: alss.py - Anomalous Langevin Score Sampling (ALSS) Solver

KEY FIXES:
- Added GRADIENT CLIPPING to prevent explosion
- Passes current_sigma to disorder estimator
- NaN detection and early stopping
- Stable step size computation
"""

import torch
import numpy as np
from da_dps.physics.levy import sample_isotropic_levy


class AnomalousLangevinSampler:
    """
    Anomalous Langevin Score Sampling with stability fixes.
    """
    
    def __init__(self, score_model, config):
        """
        Args:
            score_model: Score network s_theta(x, t)
            config: Configuration dict with:
                    - alpha: Levy index (1 < alpha <= 2)
                    - N: Sampling steps
                    - sigma_min, sigma_max: Noise schedule
                    - device: torch device
                    - max_grad_norm: Gradient clipping threshold
                    - data_weight: Likelihood term weighting
        """
        self.score_model = score_model
        self.alpha = config.get('alpha', 1.8)
        self.N = config['N']
        self.device = config['device']
        
        # FIX: Stability parameters
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.data_weight = config.get('data_weight', 1.0)
        
        # Noise schedule
        self.sigmas = torch.tensor(np.exp(np.linspace(
            np.log(config['sigma_min']), 
            np.log(config['sigma_max']), 
            self.N
        ))).to(self.device)
        
        print(f"[AnomalousLangevinSampler]")
        print(f"  Alpha: {self.alpha:.2f}")
        print(f"  Steps: {self.N}")
        print(f"  Max Grad Norm: {self.max_grad_norm}")
        print(f"  Data Weight: {self.data_weight}")
    
    def get_step_size(self, sigma_i, sigma_prev):
        """Calculate step size from noise schedule."""
        return (sigma_i ** 2 - sigma_prev ** 2)
    
# Add parameter
    def sample(self, y, disorder_estimator=None, operator=None, x_init=None):
        
        # Choose operator
        A = operator if operator is not None else disorder_estimator.A
        
        # Then use A.forward/adjoint for data gradient

        
        if x_init is None:
            x_template = disorder_estimator.get_A_eff_adjoint()(y)
            x = torch.randn_like(x_template) * self.sigmas[0]
        else:
            x = x_init.clone()
        
        print(f"\nStarting ALSS (alpha={self.alpha})")
        
        with torch.no_grad():
            for i in range(self.N - 1, 0, -1):
                t_i = torch.tensor(i / self.N, device=self.device)
                sigma_i = self.sigmas[i]
                sigma_prev = self.sigmas[i-1]
                
                # Prepare batch time embedding
                if isinstance(t_i, torch.Tensor) and t_i.dim() == 0:
                    t_batch = t_i.unsqueeze(0).expand(x.shape[0])
                else:
                    t_batch = t_i
                
                # ===== 1. Compute Drift =====
                
                # Prior score
                grad_prior = self.score_model(x, t_batch)
                
                # Remove batch for operator (expects 1D)
                x_flat = x.squeeze(0) if x.dim() > 1 else x    # (16384,)
                Ax = A.forward(x_flat)                          # (8192,)

                # Same for y
                y_flat = y.squeeze(0) if y.dim() > 1 else y    # (8192,)
                residual = Ax - y_flat

                # Compute gradient
                grad_data_flat = A.adjoint(residual)            # (16384,)
                grad_data = grad_data_flat.unsqueeze(0)         # Add batch back

                # Composite drift
                drift = grad_prior - self.data_weight * grad_data
                
                # FIX: GRADIENT CLIPPING
                drift_norm = torch.norm(drift)
                if drift_norm > self.max_grad_norm:
                    drift = drift * (self.max_grad_norm / (drift_norm + 1e-6))
                
                # ===== 2. Sample Noise =====
                
                eta_i = self.get_step_size(sigma_i, sigma_prev)
                z = sample_isotropic_levy(
                    shape=x.shape,
                    alpha=self.alpha,
                    device=self.device,
                    truncation_threshold=10.0
                )
                
                # ===== 3. Update =====
                
                noise_scale = torch.pow(eta_i, 1.0 / self.alpha)
                x = x + eta_i * drift + noise_scale * z
                
                # FIX: NaN detection and early stopping
                if torch.isnan(x).any():
                    print(f"  [ERROR] NaN detected at step {i}. Stopping.")
                    break
                
                # Progress
                if (self.N - i) % max(1, self.N // 10) == 0:
                    step_count = self.N - i
                    print(f"  Step {step_count}/{self.N}: sigma={sigma_i.item():.4f}, "
                          f"drift_norm={torch.min(drift_norm, torch.tensor(self.max_grad_norm)).item():.4f}")
        
        print(f"Sampling complete")
        return x