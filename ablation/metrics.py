#!/usr/bin/env python3
"""
Metrics module for ablation studies.

Computes evaluation metrics for reconstruction tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter


class EvaluationMetrics:
    """Compute evaluation metrics for image reconstruction."""
    
    @staticmethod
    def psnr(x_true: torch.Tensor, x_recon: torch.Tensor) -> float:
        """
        Peak Signal-to-Noise Ratio.
        
        Args:
            x_true: Ground truth image [batch, channels, height, width] or [batch, size]
            x_recon: Reconstructed image, same shape as x_true
        
        Returns:
            PSNR in dB
        """
        x_true = x_true.detach().cpu().numpy()
        x_recon = x_recon.detach().cpu().numpy()
        
        # Ensure values are in valid range
        data_range = max(x_true.max() - x_true.min(), 1.0)
        
        return peak_signal_noise_ratio(x_true, x_recon, data_range=data_range)
    
    @staticmethod
    def ssim(x_true: torch.Tensor, x_recon: torch.Tensor) -> float:
        """
        Structural Similarity Index.
        
        Args:
            x_true: Ground truth image
            x_recon: Reconstructed image
        
        Returns:
            SSIM value [-1, 1]
        """
        x_true = x_true.detach().cpu().numpy()
        x_recon = x_recon.detach().cpu().numpy()
        
        # Flatten if needed
        if x_true.ndim > 2:
            # For multi-channel, compute per-channel and average
            if x_true.shape[0] == 1 or x_true.ndim == 4:
                x_true = x_true.squeeze()
                x_recon = x_recon.squeeze()
        
        data_range = max(x_true.max() - x_true.min(), 1.0)
        
        return structural_similarity(
            x_true, x_recon,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5
        )
    
    @staticmethod
    def mse(x_true: torch.Tensor, x_recon: torch.Tensor) -> float:
        """
        Mean Squared Error.
        
        Args:
            x_true: Ground truth image
            x_recon: Reconstructed image
        
        Returns:
            MSE value
        """
        x_true = x_true.detach().cpu().numpy()
        x_recon = x_recon.detach().cpu().numpy()
        
        return float(np.mean((x_true - x_recon) ** 2))
    
    @staticmethod
    def measurement_fidelity(
        y_true: torch.Tensor,
        y_recon: torch.Tensor
    ) -> float:
        """
        Measurement fidelity (MSE in measurement space).
        
        Args:
            y_true: True measurements [batch, n_measurements]
            y_recon: Reconstructed measurements
        
        Returns:
            Measurement MSE
        """
        y_true = y_true.detach().cpu().numpy()
        y_recon = y_recon.detach().cpu().numpy()
        
        return float(np.mean((y_true - y_recon) ** 2))
    
    @staticmethod
    def compute_all_metrics(
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        y_true: torch.Tensor,
        measurement_operator,
        metrics_list: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute all requested metrics.
        
        Args:
            x_true: Ground truth image [batch, channels, height, width]
            x_recon: Reconstructed image, same shape
            y_true: True measurements [batch, n_measurements]
            measurement_operator: Operator to compute y_recon from x_recon
            metrics_list: List of metrics to compute. Default: all available
        
        Returns:
            Dictionary of metric names to values
        """
        if metrics_list is None:
            metrics_list = ['psnr', 'ssim', 'mse', 'measurement_fidelity']
        
        results = {}
        
        try:
            if 'psnr' in metrics_list:
                results['psnr'] = EvaluationMetrics.psnr(x_true, x_recon)
        except Exception as e:
            print(f"Warning: PSNR computation failed: {e}")
            results['psnr'] = None
        
        try:
            if 'ssim' in metrics_list:
                results['ssim'] = EvaluationMetrics.ssim(x_true, x_recon)
        except Exception as e:
            print(f"Warning: SSIM computation failed: {e}")
            results['ssim'] = None
        
        try:
            if 'mse' in metrics_list:
                results['mse'] = EvaluationMetrics.mse(x_true, x_recon)
        except Exception as e:
            print(f"Warning: MSE computation failed: {e}")
            results['mse'] = None
        
        try:
            if 'measurement_fidelity' in metrics_list:
                with torch.no_grad():
                    y_recon = measurement_operator(x_recon)
                results['measurement_fidelity'] = EvaluationMetrics.measurement_fidelity(
                    y_true, y_recon
                )
        except Exception as e:
            print(f"Warning: Measurement fidelity computation failed: {e}")
            results['measurement_fidelity'] = None
        
        return results
    
    @staticmethod
    def compute_statistics(
        metric_values: List[float]
    ) -> Dict[str, float]:
        """
        Compute statistics over multiple samples.
        
        Args:
            metric_values: List of metric values
        
        Returns:
            Dictionary with mean, std, min, max
        """
        values = np.array(metric_values)
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
