import torch
import numpy as np

def sample_isotropic_levy(shape, alpha, device='cuda', truncation_threshold=None):
    """
    Samples from an isotropic symmetric alpha-stable distribution S_alpha(gamma=1, 0, 0).
    
    Implements the Chambers-Mallows-Stuck method.
    Ref: Caceres Chapter 1.6 (Generalized CLT and Power Laws)
    
    Args:
        shape: Tensor shape.
        alpha: Stability parameter (1 < alpha <= 2). 
               alpha=2 is Gaussian, alpha<2 is Heavy-tailed.
        truncation_threshold: If set, implements Tempered Levy Flights 
                              by clipping ||z|| > delta.
    """
    if alpha == 2.0:
        return torch.randn(shape, device=device)

    # 1. Generate auxiliary variables
    # V ~ Uniform(-pi/2, pi/2)
    V = (torch.rand(shape, device=device) * np.pi) - (np.pi / 2.0)
    # W ~ Exponential(1)
    W = -torch.log(torch.rand(shape, device=device))

    # 2. Chambers-Mallows-Stuck transformation
    # Term 1: sin(alpha * V) / (cos(V))^(1/alpha)
    term1 = torch.sin(alpha * V) / torch.pow(torch.cos(V), 1.0 / alpha)
    
    # Term 2: (cos((1-alpha) * V) / W)^((1-alpha)/alpha)
    term2 = torch.pow(
        torch.cos((1.0 - alpha) * V) / W, 
        (1.0 - alpha) / alpha
    )
    
    z = term1 * term2

    # 3. Optional: Tempered Levy Flight (Algorithm 2, Line 15)
    if truncation_threshold is not None:
        norm = torch.norm(z.view(z.shape[0], -1), dim=1, keepdim=True)
        mask = norm > truncation_threshold
        # Re-normalize large jumps to the threshold (soft truncation)
        # or reject/resample (hard truncation). Here we clamp magnitude.
        scale_factor = torch.ones_like(norm)
        scale_factor[mask] = truncation_threshold / norm[mask]
        z = z * scale_factor.view(*z.shape[:1], *([1] * (len(z.shape)-1)))
        
    return z
