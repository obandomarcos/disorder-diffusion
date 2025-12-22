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


"""Extended evaluation metrics for UQ assessment on CIFAR-10."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Tuple, Optional


class EvaluationMetrics:
    """Comprehensive UQ evaluation metrics."""
    
    # -------------------------------------------------------------------------
    # Image Quality Metrics
    # -------------------------------------------------------------------------
    
    @staticmethod
    def psnr(x_true: torch.Tensor, x_pred: torch.Tensor, max_val: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(x_true, x_pred)
        psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def ssim(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
        """Simplified SSIM (correlation-based approximation)."""
        xt = x_true.reshape(-1)
        xp = x_pred.reshape(-1)
        mt, mp = xt.mean(), xp.mean()
        cov = ((xt - mt) * (xp - mp)).mean()
        vt = (xt - mt).pow(2).mean()
        vp = (xp - mp).pow(2).mean()
        ssim = (2 * cov) / (vt + vp + 1e-8)
        return ssim.item()
    
    @staticmethod
    def mse(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
        """Mean Squared Error."""
        return F.mse_loss(x_true, x_pred).item()
    
    @staticmethod
    def mae(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
        """Mean Absolute Error."""
        return F.l1_loss(x_true, x_pred).item()
    
    # -------------------------------------------------------------------------
    # Classification Metrics (if downstream task exists)
    # -------------------------------------------------------------------------
    
    @staticmethod
    def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Classification accuracy."""
        correct = (y_pred == y_true).float().sum()
        return (correct / y_true.numel()).item()
    
    @staticmethod
    def precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor, 
                            num_classes: int = 10) -> Tuple[float, float, float]:
        """Precision, recall, F1-score (macro-averaged)."""
        precision_list, recall_list, f1_list = [], [], []
        
        for c in range(num_classes):
            tp = ((y_pred == c) & (y_true == c)).float().sum()
            fp = ((y_pred == c) & (y_true != c)).float().sum()
            fn = ((y_pred != c) & (y_true == c)).float().sum()
            
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            
            precision_list.append(prec.item())
            recall_list.append(rec.item())
            f1_list.append(f1.item())
        
        return (np.mean(precision_list), np.mean(recall_list), np.mean(f1_list))
    
    # -------------------------------------------------------------------------
    # Uncertainty Calibration Metrics
    # -------------------------------------------------------------------------
    
    @staticmethod
    def coverage_probability(samples: torch.Tensor, x_true: torch.Tensor,
                            confidence_level: float = 0.95) -> float:
        """
        Compute empirical coverage probability.
        
        Args:
            samples: (n_samples, batch, channels, h, w)
            x_true: (batch, channels, h, w)
            confidence_level: Target coverage (e.g., 0.95)
            
        Returns:
            Empirical coverage (should be ≈ confidence_level if calibrated)
        """
        n_samples = samples.shape[0]
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        # z-score for confidence level (normal approximation)
        from scipy.stats import norm
        z = norm.ppf((1 + confidence_level) / 2)
        
        lower = mean - z * std
        upper = mean + z * std
        
        # Check if true value is within interval
        covered = ((x_true >= lower) & (x_true <= upper)).float()
        coverage = covered.mean().item()
        
        return coverage
    
    @staticmethod
    def expected_calibration_error(samples: torch.Tensor, x_true: torch.Tensor,
                                   n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE) for regression.
        
        Bins predictions by confidence (inverse of std) and checks if
        empirical error matches predicted uncertainty.
        
        Args:
            samples: (n_samples, batch, channels, h, w)
            x_true: (batch, channels, h, w)
            n_bins: Number of calibration bins
            
        Returns:
            ECE score (lower is better; 0 = perfect calibration)
        """
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        # Flatten
        mean_flat = mean.reshape(-1)
        std_flat = std.reshape(-1)
        true_flat = x_true.reshape(-1)
        
        # Compute squared errors
        sq_errors = (mean_flat - true_flat).pow(2)
        
        # Bin by predicted variance
        var_flat = std_flat.pow(2)
        var_bins = torch.linspace(var_flat.min(), var_flat.max(), n_bins + 1)
        
        ece = 0.0
        total = 0
        
        for i in range(n_bins):
            mask = (var_flat >= var_bins[i]) & (var_flat < var_bins[i + 1])
            if mask.sum() == 0:
                continue
            
            # Predicted variance (average in bin)
            pred_var = var_flat[mask].mean()
            
            # Empirical variance (MSE in bin)
            emp_var = sq_errors[mask].mean()
            
            # Weight by bin size
            weight = mask.float().sum() / len(mask)
            ece += weight * torch.abs(pred_var - emp_var)
            total += weight
        
        return (ece / total).item() if total > 0 else 0.0
    
    @staticmethod
    def reliability_diagram_data(samples: torch.Tensor, x_true: torch.Tensor,
                                 n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for reliability diagram.
        
        Returns:
            predicted_confidence: Bin centers (predicted std)
            empirical_error: Actual RMSE in each bin
        """
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        mean_flat = mean.reshape(-1)
        std_flat = std.reshape(-1)
        true_flat = x_true.reshape(-1)
        
        sq_errors = (mean_flat - true_flat).pow(2)
        
        std_bins = torch.linspace(std_flat.min(), std_flat.max(), n_bins + 1)
        pred_conf = []
        emp_err = []
        
        for i in range(n_bins):
            mask = (std_flat >= std_bins[i]) & (std_flat < std_bins[i + 1])
            if mask.sum() == 0:
                continue
            
            pred_conf.append(std_flat[mask].mean().item())
            emp_err.append(torch.sqrt(sq_errors[mask].mean()).item())
        
        return np.array(pred_conf), np.array(emp_err)
    
    # -------------------------------------------------------------------------
    # Posterior Quality Metrics
    # -------------------------------------------------------------------------
    
    @staticmethod
    def wasserstein_distance_images(samples1: torch.Tensor, samples2: torch.Tensor) -> float:
        """
        1-Wasserstein distance between two sample sets (pixel-wise).
        
        Args:
            samples1: (n_samples, batch, C, H, W)
            samples2: (n_samples, batch, C, H, W)
            
        Returns:
            Average Wasserstein distance across pixels
        """
        s1_flat = samples1.reshape(samples1.shape[0], -1).cpu().numpy()
        s2_flat = samples2.reshape(samples2.shape[0], -1).cpu().numpy()
        
        n_pixels = s1_flat.shape[1]
        w_dists = []
        
        # Sample subset of pixels (computing for all 3072 pixels is expensive)
        n_samples = min(n_pixels, 500)
        indices = np.random.choice(n_pixels, n_samples, replace=False)
        
        for idx in indices:
            w_dist = wasserstein_distance(s1_flat[:, idx], s2_flat[:, idx])
            w_dists.append(w_dist)
        
        return np.mean(w_dists)
    
    @staticmethod
    def energy_distance(samples1: torch.Tensor, samples2: torch.Tensor) -> float:
        """
        Energy distance between two sample distributions.
        
        E(X, Y) = 2 E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
        
        Args:
            samples1: (n_samples, batch, C, H, W)
            samples2: (n_samples, batch, C, H, W)
            
        Returns:
            Energy distance (≥0; 0 iff distributions are identical)
        """
        n1, n2 = samples1.shape[0], samples2.shape[0]
        
        # Flatten
        s1 = samples1.reshape(n1, -1)
        s2 = samples2.reshape(n2, -1)
        
        # Cross term: E[||X - Y||]
        cross_term = 0.0
        for i in range(n1):
            for j in range(n2):
                cross_term += torch.norm(s1[i] - s2[j], p=2)
        cross_term /= (n1 * n2)
        
        # Within-distribution terms
        within1 = 0.0
        for i in range(n1):
            for j in range(i + 1, n1):
                within1 += torch.norm(s1[i] - s1[j], p=2)
        within1 /= (n1 * (n1 - 1) / 2)
        
        within2 = 0.0
        for i in range(n2):
            for j in range(i + 1, n2):
                within2 += torch.norm(s2[i] - s2[j], p=2)
        within2 /= (n2 * (n2 - 1) / 2)
        
        energy = 2 * cross_term - within1 - within2
        return energy.item()
    
    @staticmethod
    def maximum_mean_discrepancy(samples1: torch.Tensor, samples2: torch.Tensor,
                                 kernel: str = "rbf", bandwidth: float = 1.0) -> float:
        """
        Maximum Mean Discrepancy (MMD) with RBF kernel.
        
        Args:
            samples1: (n_samples, batch, C, H, W)
            samples2: (n_samples, batch, C, H, W)
            kernel: "rbf" or "linear"
            bandwidth: RBF kernel bandwidth
            
        Returns:
            MMD estimate (≥0; 0 iff distributions match)
        """
        n1, n2 = samples1.shape[0], samples2.shape[0]
        
        # Flatten
        s1 = samples1.reshape(n1, -1)
        s2 = samples2.reshape(n2, -1)
        
        def rbf_kernel(x, y, bw):
            return torch.exp(-torch.norm(x - y, p=2).pow(2) / (2 * bw ** 2))
        
        # K(X, X)
        k_xx = 0.0
        for i in range(n1):
            for j in range(n1):
                k_xx += rbf_kernel(s1[i], s1[j], bandwidth)
        k_xx /= (n1 ** 2)
        
        # K(Y, Y)
        k_yy = 0.0
        for i in range(n2):
            for j in range(n2):
                k_yy += rbf_kernel(s2[i], s2[j], bandwidth)
        k_yy /= (n2 ** 2)
        
        # K(X, Y)
        k_xy = 0.0
        for i in range(n1):
            for j in range(n2):
                k_xy += rbf_kernel(s1[i], s2[j], bandwidth)
        k_xy /= (n1 * n2)
        
        mmd = k_xx + k_yy - 2 * k_xy
        return mmd.item()


# -----------------------------------------------------------------------------
# Helper: Compute all metrics at once
# -----------------------------------------------------------------------------

def compute_all_uq_metrics(
    dps_samples: torch.Tensor,
    da_dps_samples: torch.Tensor,
    x_true: torch.Tensor,
    y_meas: torch.Tensor,
    verbose: bool = True
) -> dict:
    """
    Compute all UQ metrics for DPS vs DA-DPS comparison.
    
    Args:
        dps_samples: (n_samples, batch, C, H, W)
        da_dps_samples: (n_samples, batch, C, H, W)
        x_true: (batch, C, H, W)
        y_meas: Measurement (for reference)
        verbose: Print results
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Means for reconstruction quality
    dps_mean = dps_samples.mean(dim=0)
    da_dps_mean = da_dps_samples.mean(dim=0)
    
    # Normalize to [0, 1] for PSNR/SSIM
    x_true_norm = (x_true + 1) / 2
    dps_mean_norm = (dps_mean + 1) / 2
    da_dps_mean_norm = (da_dps_mean + 1) / 2
    
    # Image quality
    metrics["dps_psnr"] = EvaluationMetrics.psnr(x_true_norm, dps_mean_norm)
    metrics["da_dps_psnr"] = EvaluationMetrics.psnr(x_true_norm, da_dps_mean_norm)
    metrics["dps_ssim"] = Eva
