"""Evaluation metrics for DA-DPS."""

import torch
import numpy as np
from typing import Dict, Optional
import time


class PerformanceMetrics:
    """Performance monitoring utilities."""
    
    @staticmethod
    def measure_score_computation_time(sampler, x: torch.Tensor,
                                       t: torch.Tensor,
                                       n_warmup: int = 1,
                                       n_trials: int = 10) -> float:
        """
        Measure score computation time.
        
        Args:
            sampler: DA_DPS_Sampler instance
            x: Input sample
            t: Timestep
            n_warmup: Number of warmup iterations
            n_trials: Number of timing trials
            
        Returns:
            Average time in milliseconds
        """
        # Warmup
        for _ in range(n_warmup):
            _ = sampler.compute_disorder_averaged_score(x, t)
        
        # Measure
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = sampler.compute_disorder_averaged_score(x, t)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return float(np.mean(times))
    
    @staticmethod
    def measure_memory_scaling(sampler, x: torch.Tensor,
                               t: torch.Tensor,
                               device: str = 'cuda') -> Dict[int, float]:
        """
        Measure GPU memory scaling with disorder samples.
        
        Args:
            sampler: DA_DPS_Sampler instance
            x: Input sample
            t: Timestep
            device: Device to measure on
            
        Returns:
            Dict mapping n_disorder to peak memory (MB)
        """
        memory_usage = {}
        
        if not torch.cuda.is_available():
            return memory_usage
        
        for n_disorder in [1, 5, 10]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
            # Allocate and compute
            _ = sampler.compute_disorder_averaged_score(x, t)
            
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6  # Convert to MB
            memory_usage[n_disorder] = peak_mem
        
        return memory_usage


class EvaluationMetrics:
    """Evaluation metrics for DA-DPS."""
    
    @staticmethod
    def measurement_fidelity(x_recon: torch.Tensor, y: torch.Tensor,
                             measurement_operator) -> float:
        """
        Compute measurement fidelity.
        
        Args:
            x_recon: Reconstructed image
            y: Measurements
            measurement_operator: Measurement operator
            
        Returns:
            Relative measurement error ||A(x_recon) - y|| / ||y||
        """
        Ax = measurement_operator(x_recon)
        error = (Ax - y).norm() / (y.norm() + 1e-8)
        return float(error)
    
    @staticmethod
    def psnr(x_true: torch.Tensor, x_recon: torch.Tensor,
             max_val: float = 1.0) -> float:
        """
        Compute PSNR.
        
        Args:
            x_true: True image
            x_recon: Reconstructed image
            max_val: Maximum value (typically 1.0 for normalized images)
            
        Returns:
            PSNR in dB
        """
        mse = ((x_true - x_recon) ** 2).mean()
        psnr = 10 * torch.log10(max_val ** 2 / mse)
        return float(psnr)
    
    @staticmethod
    def ssim(x_true: torch.Tensor, x_recon: torch.Tensor,
             window_size: int = 11) -> float:
        """
        Compute SSIM.
        
        Args:
            x_true: True image
            x_recon: Reconstructed image
            window_size: Window size
            
        Returns:
            SSIM score
        """
        # Simple SSIM implementation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        mean_true = x_true.mean()
        mean_recon = x_recon.mean()
        var_true = ((x_true - mean_true) ** 2).mean()
        var_recon = ((x_recon - mean_recon) ** 2).mean()
        cov = ((x_true - mean_true) * (x_recon - mean_recon)).mean()
        
        ssim = ((2 * mean_true * mean_recon + c1) *
                (2 * cov + c2)) / (
            (mean_true ** 2 + mean_recon ** 2 + c1) *
            (var_true + var_recon + c2)
        )
        
        return float(ssim)