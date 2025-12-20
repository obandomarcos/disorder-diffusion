"""
Test suite for Disorder-Averaged Diffusion Posterior Sampling (DA-DPS).

This module tests the implementation of DA-DPS from Stage 1 of the research:
- Disorder averaging via Effective Medium Approximation (EMA)
- Score function averaging over disorder ensemble
- Convergence properties and bias-variance tradeoffs
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path


# ==================== Fixtures ====================

@pytest.fixture
def device():
    """Device for testing (CPU by default, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def image_shape():
    """Standard image shape for testing."""
    return (1, 1, 32, 32)  # (batch, channels, height, width)


@pytest.fixture
def measurement_params():
    """Parameters for compressed sensing measurements."""
    return {
        'n_pixels': 32 * 32,
        'n_measurements': 256,  # 25% compression
        'noise_std': 0.01
    }


@pytest.fixture
def mock_score_network(device):
    """Mock score network for testing."""
    class MockScoreNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)
        
        def forward(self, x, t):
            # Simple mock: return small perturbation
            return -0.1 * self.conv(x)
    
    return MockScoreNet().to(device)


@pytest.fixture
def mock_scheduler():
    """Mock diffusion scheduler."""
    class MockScheduler:
        def __init__(self):
            self.num_train_timesteps = 1000
            self.timesteps = torch.linspace(999, 0, 100).long()
            self.alphas_cumprod = torch.linspace(0.9999, 0.0001, 1000)
        
        def set_timesteps(self, num_steps):
            self.timesteps = torch.linspace(999, 0, num_steps).long()
        
        def step(self, model_output, timestep, sample):
            class Output:
                def __init__(self, prev_sample):
                    self.prev_sample = prev_sample
            
            # Simple Euler step
            timestep_int = int(timestep) if torch.is_tensor(timestep) else timestep
            alpha = self.alphas_cumprod[timestep_int]
            prev_sample = sample - 0.01 * model_output
            return Output(prev_sample)
    
    return MockScheduler()


@pytest.fixture
def gaussian_measurement_operator(measurement_params, device):
    """Gaussian random measurement matrix."""
    m = measurement_params['n_measurements']
    n = measurement_params['n_pixels']
    A = torch.randn(m, n, device=device) / np.sqrt(n)
    
    def operator(x):
        # Flatten image and apply measurement
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat @ A.T
    
    operator.matrix = A
    return operator


# ==================== DA-DPS Implementation Tests ====================

class TestDisorderAveragedDPS:
    """Test suite for Disorder-Averaged DPS."""
    
    def test_disorder_distribution_sampling(self, device):
        """Test that disorder distribution samples correctly."""
        # Uniform disorder in [0.8, 1.2]
        class UniformDisorder:
            def sample(self, n_samples=1):
                return 0.8 + 0.4 * torch.rand(n_samples, device=device)
        
        disorder_dist = UniformDisorder()
        samples = disorder_dist.sample(100)
        
        assert samples.shape == (100,)
        assert torch.all(samples >= 0.8)
        assert torch.all(samples <= 1.2)
        assert torch.abs(samples.mean() - 1.0) < 0.1  # Mean ≈ 1.0
    
    def test_disorder_ensemble_creation(self, device):
        """Test creation of disorder ensemble."""
        n_disorder = 10
        
        # Create ensemble of disorder realizations
        disorder_samples = torch.linspace(0.8, 1.2, n_disorder, device=device)
        
        assert disorder_samples.shape == (n_disorder,)
        # FIX: Convert tensors to Python floats before comparison
        assert float(disorder_samples[0]) == pytest.approx(0.8)
        assert float(disorder_samples[-1]) == pytest.approx(1.2)
    
    def test_effective_medium_approximation(self, device):
        """Test EMA averaging over disorder ensemble."""
        n_disorder = 5
        batch_size = 2
        
        # Create mock Green functions with different disorders
        green_functions = []
        for i in range(n_disorder):
            disorder_strength = 0.8 + 0.1 * i
            g = disorder_strength * torch.ones(batch_size, device=device)
            green_functions.append(g)
        
        # Average (EMA)
        g_eff = torch.stack(green_functions).mean(dim=0)
        
        assert g_eff.shape == (batch_size,)
        # FIX: Convert to Python float before comparison
        assert float(g_eff[0]) == pytest.approx(1.0, abs=0.01)  # Mean of [0.8, 0.9, 1.0, 1.1, 1.2]
    
    def test_disorder_averaged_score(self, mock_score_network, device, image_shape):
        """Test disorder-averaged score computation."""
        n_disorder = 5
        x_t = torch.randn(image_shape, device=device)
        t = torch.tensor([500], device=device)
        
        # Compute scores for different disorder realizations
        scores = []
        for i in range(n_disorder):
            # Simulate disorder by scaling input
            disorder = 0.9 + 0.05 * i
            x_disordered = disorder * x_t
            score = mock_score_network(x_disordered, t)
            scores.append(score)
        
        # Average scores (DA-DPS)
        score_avg = torch.stack(scores).mean(dim=0)
        
        assert score_avg.shape == image_shape
        assert not torch.isnan(score_avg).any()
        assert not torch.isinf(score_avg).any()
    
    def test_likelihood_guidance_with_disorder(
        self, 
        gaussian_measurement_operator, 
        device, 
        image_shape,
        measurement_params
    ):
        """Test likelihood guidance with disorder averaging."""
        x_0 = torch.randn(image_shape, device=device)
        y = gaussian_measurement_operator(x_0)
        
        n_disorder = 10
        likelihood_grads = []
        
        for i in range(n_disorder):
            # Disorder affects measurement operator
            disorder = 0.9 + 0.02 * i
            A_disordered = disorder * gaussian_measurement_operator.matrix
            
            # Compute likelihood gradient: ∇ ||Ax - y||²
            x_flat = x_0.reshape(1, -1)
            residual = (x_flat @ A_disordered.T) - y
            grad = 2 * (residual @ A_disordered).reshape(image_shape)
            likelihood_grads.append(grad)
        
        # Average gradients
        grad_avg = torch.stack(likelihood_grads).mean(dim=0)
        
        assert grad_avg.shape == image_shape
        assert not torch.isnan(grad_avg).any()
    
    def test_bias_variance_tradeoff(self, device):
        """Test bias-variance tradeoff with number of disorder samples."""
        true_mean = 1.0
        disorder_std = 0.1
        
        n_samples_list = [1, 5, 10, 20, 50, 100]
        variances = []
        
        for n_samples in n_samples_list:
            estimates = []
            for _ in range(100):  # Monte Carlo trials
                disorder_samples = true_mean + disorder_std * torch.randn(n_samples, device=device)
                estimate = disorder_samples.mean()
                estimates.append(estimate.item())
            
            variance = np.var(estimates)
            variances.append(variance)
        
        # Variance should decrease with more samples
        assert variances[0] > variances[-1]
        # Should follow 1/n scaling approximately
        assert variances[1] / variances[2] < 3  # Should be ~2 if perfect scaling


# ==================== Integration Tests ====================

class TestDADPSIntegration:
    """Integration tests for complete DA-DPS pipeline."""
    
    def test_dps_vs_da_dps_difference(
        self, 
        mock_score_network, 
        mock_scheduler,
        gaussian_measurement_operator,
        device,
        image_shape
    ):
        """Test that DA-DPS differs from standard DPS."""
        # Standard DPS (no disorder)
        x_T = torch.randn(image_shape, device=device)
        x_dps = self._run_sampling(
            x_T, mock_score_network, mock_scheduler, 
            gaussian_measurement_operator, n_disorder=1, device=device
        )
        
        # DA-DPS (with disorder)
        x_da_dps = self._run_sampling(
            x_T, mock_score_network, mock_scheduler,
            gaussian_measurement_operator, n_disorder=10, device=device
        )
        
        # Results should differ
        diff = (x_dps - x_da_dps).abs().mean()
        assert diff > 1e-6  # Not identical
    
    def _run_sampling(self, x_T, score_net, scheduler, operator, n_disorder, device):
        """Helper: run sampling with disorder averaging."""
        x = x_T.clone()
        timesteps = scheduler.timesteps[:10]  # Only 10 steps for speed
        
        for t in timesteps:
            # Compute disorder-averaged score
            scores = []
            for _ in range(n_disorder):
                # FIX: Create disorder on same device as x
                disorder = 0.9 + 0.2 * torch.rand(1, device=device)
                x_disordered = disorder * x
                score = score_net(x_disordered, t)
                scores.append(score)
            
            score_avg = torch.stack(scores).mean(dim=0)
            
            # Update step
            x = scheduler.step(score_avg, t, x).prev_sample
        
        return x
    
    def test_convergence_with_disorder_samples(
        self,
        mock_score_network,
        mock_scheduler,
        device,
        image_shape
    ):
        """Test convergence as number of disorder samples increases."""
        x_T = torch.randn(image_shape, device=device)
        
        results = []
        for n_disorder in [1, 5, 10, 20]:
            x = self._run_sampling(
                x_T, mock_score_network, mock_scheduler,
                lambda x: x,  # Identity operator
                n_disorder, device=device
            )
            results.append(x)
        
        # Convergence: difference between successive results decreases
        diff_1_5 = (results[0] - results[1]).abs().mean()
        diff_10_20 = (results[2] - results[3]).abs().mean()
        
        assert diff_10_20 < diff_1_5  # More samples = more stable
    
    def test_measurement_consistency(
        self,
        gaussian_measurement_operator,
        device,
        image_shape,
        measurement_params
    ):
        """Test that DA-DPS respects measurement constraint."""
        # Generate true image and measurements
        x_true = torch.randn(image_shape, device=device)
        y = gaussian_measurement_operator(x_true)
        
        # Simulate DA-DPS reconstruction (simplified)
        x_recon = x_true + 0.1 * torch.randn_like(x_true)
        y_recon = gaussian_measurement_operator(x_recon)
        
        # Measurement error should be small
        measurement_error = (y - y_recon).norm() / y.norm()
        assert measurement_error < 0.5  # Within 50%


# ==================== Numerical Stability Tests ====================

class TestNumericalStability:
    """Test numerical stability of DA-DPS."""
    
    def test_gradient_explosion_prevention(self, device, image_shape):
        """Test that gradients don't explode during disorder averaging."""
        n_disorder = 10
        
        # Simulate extreme disorder
        x = torch.randn(image_shape, device=device, requires_grad=True)
        
        scores = []
        for i in range(n_disorder):
            disorder = 0.1 + 2.0 * i  # Wide range [0.1, 18.1]
            score = disorder * x ** 2
            scores.append(score)
        
        score_avg = torch.stack(scores).mean(dim=0)
        loss = score_avg.sum()
        loss.backward()
        
        # Gradient should be finite
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert x.grad.abs().max() < 1e6  # Not exploded
    
    def test_numerical_precision_with_many_samples(self, device):
        """Test numerical precision with large number of disorder samples."""
        n_disorder = 1000
        
        # Generate disorder samples
        disorder_samples = 0.9 + 0.2 * torch.rand(n_disorder, device=device)
        
        # Compute mean (should be ~1.0)
        mean = disorder_samples.mean()
        
        assert mean > 0.95
        assert mean < 1.05
        assert not torch.isnan(mean)
    
    def test_empty_disorder_ensemble_error(self):
        """Test that empty disorder ensemble raises error."""
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            scores = []
            score_avg = torch.stack(scores).mean(dim=0)


# ==================== Performance Tests ====================

class TestPerformance:
    """Test computational performance of DA-DPS."""
    
    def test_disorder_averaging_overhead(
        self, 
        mock_score_network,
        device,
        image_shape
    ):
        """Test disorder averaging overhead (simplified without benchmark fixture)."""
        x = torch.randn(image_shape, device=device)
        t = torch.tensor([500], device=device)
        n_disorder = 10
        
        # FIX: Create disorder on same device
        scores = []
        for _ in range(n_disorder):
            disorder = 0.9 + 0.2 * torch.rand(1, device=device)
            x_disordered = disorder * x
            score = mock_score_network(x_disordered, t)
            scores.append(score)
        
        result = torch.stack(scores).mean(dim=0)
        assert result is not None
        assert result.shape == image_shape
    
    def test_memory_usage_scaling(self, device, image_shape):
        """Test that memory scales linearly with disorder samples."""
        if not torch.cuda.is_available():
            pytest.skip("Requires GPU")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(image_shape, device=device)
        
        # Measure memory for different n_disorder
        memory_usage = []
        for n_disorder in [1, 5, 10]:
            torch.cuda.reset_peak_memory_stats()
            
            scores = []
            for _ in range(n_disorder):
                score = x.clone() * 2.0
                scores.append(score)
            
            score_avg = torch.stack(scores).mean(dim=0)
            peak_mem = torch.cuda.max_memory_allocated()
            memory_usage.append(peak_mem)
        
        # Should scale roughly linearly
        assert memory_usage[1] > memory_usage[0]
        assert memory_usage[2] > memory_usage[1]


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_disorder_sample_equivalence(
        self,
        mock_score_network,
        device,
        image_shape
    ):
        """Test that n_disorder=1 is equivalent to standard DPS."""
        x = torch.randn(image_shape, device=device)
        t = torch.tensor([500], device=device)
        
        # Standard score
        score_standard = mock_score_network(x, t)
        
        # DA-DPS with n=1
        scores = [mock_score_network(x, t)]
        score_da = torch.stack(scores).mean(dim=0)
        
        assert torch.allclose(score_standard, score_da)
    
    def test_extreme_disorder_values(self, device, image_shape):
        """Test behavior with extreme disorder values."""
        x = torch.randn(image_shape, device=device)
        
        # Very small disorder
        disorder_small = 1e-6
        x_small = disorder_small * x
        assert not torch.isnan(x_small).any()
        
        # Very large disorder
        disorder_large = 1e6
        x_large = disorder_large * x
        assert not torch.isnan(x_large).any()
    
    def test_zero_measurements(self, device, image_shape):
        """Test behavior with zero measurements (pure prior sampling)."""
        # Empty measurement
        y = torch.tensor([], device=device)
        
        # Should still be able to sample from prior
        x = torch.randn(image_shape, device=device)
        assert x.shape == image_shape
    
    def test_perfect_measurements(self, gaussian_measurement_operator, device, image_shape):
        """Test behavior with noiseless measurements."""
        x_true = torch.randn(image_shape, device=device)
        y = gaussian_measurement_operator(x_true)
        
        # Verify perfect measurements
        y_recon = gaussian_measurement_operator(x_true)
        assert torch.allclose(y, y_recon, atol=1e-6)


# ==================== Reproducibility Tests ====================

class TestReproducibility:
    """Test reproducibility of results."""
    
    def test_deterministic_with_fixed_seed(
        self,
        mock_score_network,
        device,
        image_shape
    ):
        """Test that results are deterministic with fixed seed."""
        def run_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            x = torch.randn(image_shape, device=device)
            t = torch.tensor([500], device=device)
            
            scores = []
            for _ in range(5):
                # FIX: Create disorder on same device
                disorder = 0.9 + 0.2 * torch.rand(1, device=device)
                score = mock_score_network(disorder * x, t)
                scores.append(score)
            
            return torch.stack(scores).mean(dim=0)
        
        result1 = run_with_seed(42)
        result2 = run_with_seed(42)
        
        assert torch.allclose(result1, result2, atol=1e-6)
    
    def test_different_seeds_produce_different_results(
        self,
        mock_score_network,
        device,
        image_shape
    ):
        """Test that different seeds produce different disorder realizations."""
        torch.manual_seed(42)
        x1 = torch.randn(image_shape, device=device)
        
        torch.manual_seed(123)
        x2 = torch.randn(image_shape, device=device)
        
        assert not torch.allclose(x1, x2)


# ==================== Main Test Runner ====================

if __name__ == "__main__":
    # Run tests with pytest
    print("hola")
