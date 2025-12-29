"""
FIXED: disorder.py - Disorder-Induced Effective Noise and Effective Medium Estimation
Implements Phase 1 of DA-ALSS: Computing A_eff and Sigma_eff

KEY FIXES:
- Added current_sigma parameter to prevent gradient explosion
- Implements ANNEALED PRECISION MATRIX
- Uses Sigma_eff + sigma_t^2 to scale down likelihood when noise is high
"""

import torch
import torch.nn.functional as F


class EffectiveMediumEstimator:
    """
    Handles the Transport-and-Mean-over-Disorder framework.
    Computes A_eff and Sigma_eff from an ensemble of operators {A_omega}.
    
    CRITICAL FIX: Uses annealed precision to prevent NaN divergence.
    """
    
    def __init__(self, operator_ensemble, measurement_noise_std=0.01):
        """
        Initialize the Effective Medium Estimator.
        
        Args:
            operator_ensemble: List of measurement operators
            measurement_noise_std: sigma_y >= 0.01 (IMPORTANT for stability!)
        """
        self.operators = operator_ensemble
        self.sigma_y = measurement_noise_std
        self.K = len(operator_ensemble)
        
        self._Sigma_eff_diag = None
        
        print(f"[EffectiveMediumEstimator] K={self.K} operators, sigma_y={self.sigma_y}")
    
    def calibrate_residual_covariance(self, prior_sampler, num_probes=50):
        """
        Phase 1: Estimate Sigma_model via probe signals.
        """
        print(f"\n[Disorder] Estimating residual covariance from {num_probes} probes...")
        
        probes = prior_sampler(num_probes)
        device = probes.device
        
        # Forward pass all operators
        A_eff_predictions = []
        for k, op in enumerate(self.operators):
            y_k_list = [op.forward(probes[j:j+1]) for j in range(num_probes)]
            y_k_all = torch.cat(y_k_list, dim=0)
            A_eff_predictions.append(y_k_all.unsqueeze(0))
        
        # Stack: [K, num_probes, ...]
        A_eff_predictions = torch.cat(A_eff_predictions, dim=0)
        A_eff_mean = torch.mean(A_eff_predictions, dim=0)
        
        # Residuals and variance
        residuals_list = []
        for k in range(self.K):
            residual_k = A_eff_predictions[k] - A_eff_mean
            residuals_list.append(residual_k)
        
        residuals = torch.cat(residuals_list, dim=0)
        
        # Sigma_eff = variance + sigma_y^2
        self._Sigma_eff_diag = torch.var(residuals, dim=0) + (self.sigma_y ** 2)
        
        print(f"  Sigma_eff_diag mean: {torch.mean(self._Sigma_eff_diag).item():.6f}")
    
    def get_likelihood_score_fn(self, y, x_current, current_sigma=0.0):
        """
        FIX: Computes data-fitting term with ANNEALED PRECISION.
        
        The key insight: when diffusion noise sigma_t is large (early steps),
        we should downweight the likelihood constraint to avoid explosion.
        
        Annealed Precision:
            Lambda(t) = (Sigma_eff + sigma_t^2)^(-1)
        
        This prevents NaN by naturally scaling down the gradient magnitude.
        
        Args:
            y: Measurements
            x_current: Current signal estimate
            current_sigma: Current diffusion noise level (IMPORTANT for stability)
            
        Returns:
            Data score term (safe from explosion)
        """
        
        # Residual: y - A_eff(x)
        preds_list = [op.forward(x_current) for op in self.operators]
        preds_stack = torch.stack(preds_list, dim=0)
        A_eff_x = torch.mean(preds_stack, dim=0)
        residual = y - A_eff_x
        
        # FIX: Annealed precision matrix
        # When current_sigma is large, the effective variance grows,
        # and the precision (inverse) shrinks, preventing explosion
        if self._Sigma_eff_diag is None:
            effective_variance = self.sigma_y**2 + current_sigma**2
            weighted_residual = residual / (effective_variance + 1e-6)
        else:
            # Sigma_total(t) = Sigma_eff + sigma_t^2 * I
            effective_variance = self._Sigma_eff_diag + (current_sigma ** 2)
            weighted_residual = residual / (effective_variance + 1e-6)
        
        # Apply adjoint: A_eff^T(weighted_residual)
        backproj_list = [op.adjoint(weighted_residual) for op in self.operators]
        backproj_stack = torch.stack(backproj_list, dim=0)
        score = torch.mean(backproj_stack, dim=0)
        
        return score
    
    def get_A_eff_adjoint(self):
        """Returns callable for A_eff adjoint operation."""
        def A_eff_adjoint(y):
            adjoints = [op.adjoint(y) for op in self.operators]
            return torch.mean(torch.stack(adjoints, dim=0), dim=0)
        return A_eff_adjoint
