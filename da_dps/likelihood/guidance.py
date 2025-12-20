import torch
from typing import Optional


class LikelihoodGuidance:
    """Standard likelihood guidance."""
    
    def __init__(self, measurement_operator, noise_std: float = 0.01):
        """Initialize likelihood guidance."""
        self.measurement_operator = measurement_operator
        self.noise_std = noise_std
    
    def compute_gradient(self, x: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor:
        """Compute likelihood gradient."""
        Ax = self.measurement_operator(x)
        residual = Ax - y
        grad = 2.0 * self.measurement_operator.adjoint(residual)
        return grad
    
    def compute_log_likelihood(self, x: torch.Tensor,
                               y: torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood."""
        Ax = self.measurement_operator(x)
        error = (Ax - y).pow(2).mean()
        log_likelihood = -error / (2 * self.noise_std ** 2)
        return log_likelihood


class DisorderedLikelihoodGuidance:
    """Likelihood guidance with disorder weighting."""
    
    def __init__(self, measurement_operator, disorder_ensemble,
                 noise_std: float = 0.01):
        """
        Initialize disordered likelihood guidance.
        
        Args:
            measurement_operator: Measurement operator
            disorder_ensemble: DisorderEnsemble instance
            noise_std: Measurement noise standard deviation
        """
        self.measurement_operator = measurement_operator
        self.disorder_ensemble = disorder_ensemble
        self.noise_std = noise_std
        
        from da_dps.disorder.averaging import DisorderAveragingEMA
        self.averaging = DisorderAveragingEMA()
    
    def compute_gradient(self, x: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor:
        """
        Compute disorder-averaged likelihood gradient.
        
        Args:
            x: Current sample
            y: Measurements
            
        Returns:
            torch.Tensor: ∇_x ||A(disorder)*x - y||^2
        """
        gradients = []
        
        for disorder_sample in self.disorder_ensemble.disorder_samples:
            # Apply forward operator with disorder weighting
            Ax = self.measurement_operator(x)
            Ax_disorder = disorder_sample * Ax
            
            # Compute residual
            residual = Ax_disorder - y
            
            # Compute gradient
            grad = 2.0 * self.measurement_operator.adjoint(residual)
            gradients.append(grad)
        
        # Average gradients over disorder
        grad_avg = self.averaging.average(gradients)
        
        return grad_avg
    
    def compute_log_likelihood(self, x: torch.Tensor,
                              y: torch.Tensor) -> torch.Tensor:
        """Compute disorder-averaged log-likelihood."""
        log_likelihoods = []
        
        for disorder_sample in self.disorder_ensemble.disorder_samples:
            Ax = self.measurement_operator(x)
            Ax_disorder = disorder_sample * Ax
            error = (Ax_disorder - y).pow(2).mean()
            log_lik = -error / (2 * self.noise_std ** 2)
            log_likelihoods.append(log_lik)
        
        return self.averaging.average(log_likelihoods)