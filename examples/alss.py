"""
COMPLETE PIPELINE: Train score model + reconstruct with disorder-aware sampler
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from da_dps.physics.disorder import EffectiveMediumEstimator
from da_dps.solvers.alss import AnomalousLangevinSampler
from da_dps.operators.measurement import GaussianRandomMeasurementOperator


# ============================================================================
# PART 1: SCORE MODEL (Simple Denoising Network)
# ============================================================================

class SimpleScoreNet(nn.Module):
    """Simple CNN-based score network for image denoising."""
    
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 16, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, sigma):
        """
        Args:
            x: image (batch, 1, H, W) or flattened (batch, N)
            sigma: noise level (batch,) or scalar
        """
        # Handle flattened input
        if x.dim() == 2:
            batch = x.shape[0]
            x_img = x.reshape(batch, 1, 128, 128)
        else:
            batch = x.shape[0]
            x_img = x.view(batch, 1, 128, 128)
        
        # Time embedding
        if isinstance(sigma, torch.Tensor):
            if sigma.dim() == 0:
                sigma_scalar = sigma.unsqueeze(0).expand(batch)
            else:
                sigma_scalar = sigma
        else:
            sigma_scalar = torch.full((batch,), sigma, device=x_img.device)
        
        t_emb = self.time_embed(torch.log(sigma_scalar).unsqueeze(-1))  # (batch, 16)
        t_emb = t_emb.reshape(batch, 16, 1, 1).expand(-1, -1, 128, 128)
        
        # Concatenate and process
        h = torch.cat([x_img, t_emb], dim=1)
        score = self.net(h)
        
        # Return as batch format
        return score.reshape(batch, -1)  # (batch, 16384)


def train_score_model(num_epochs=50, batch_size=8, device='cuda'):
    """Train score model on synthetic noisy/clean image pairs."""
    
    print("\n" + "="*70)
    print("PART 1: TRAINING SCORE MODEL")
    print("="*70)
    
    # Create model
    model = SimpleScoreNet(in_channels=1, hidden_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Generate training data: clean cameraman image
    try:
        from skimage import data
        img_clean = data.camera().astype(np.float32)
    except:
        x = np.linspace(-3, 3, 128)
        y = np.linspace(-3, 3, 128)
        X, Y = np.meshgrid(x, y)
        img_clean = 128 + 50 * np.sin(X) * np.cos(Y)
        img_clean = img_clean.astype(np.float32)
    
    img_clean = (img_clean - img_clean.min()) / (img_clean.max() - img_clean.min() + 1e-8)
    
    # Resize to 128x128 if needed
    if img_clean.shape[0] != 128 or img_clean.shape[1] != 128:
        scale = 128.0 / img_clean.shape[0]
        img_clean = ndimage.zoom(img_clean, scale, order=1).astype(np.float32)
    
    img_tensor = torch.from_numpy(img_clean).to(device)  # (128, 128)
    
    print(f"Training on {img_clean.shape} image")
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for _ in range(10):  # 10 mini-batches per epoch
            # Sample noise levels
            sigmas = (torch.rand(batch_size) * 10.0 + 0.01).to(device)
            
            # Add noise to clean image
            noise = torch.randn(batch_size, 1, 128, 128, device=device)
            # Expand: (128, 128) -> (1, 1, 128, 128) -> (batch, 1, 128, 128)
            img_expanded = img_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 128, 128)
            x_noisy = img_expanded + sigmas.reshape(batch_size, 1, 1, 1) * noise
            x_noisy_flat = x_noisy.reshape(batch_size, -1)
            
            # Target: score = -noise / sigma^2
            target_score = -noise.reshape(batch_size, -1) / (sigmas.reshape(batch_size, 1) ** 2)
            
            # Predict score
            pred_score = model(x_noisy_flat, sigmas)
            
            # Loss: MSE on score
            loss = torch.mean((pred_score - target_score) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / 10
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}")
    
    print("✓ Training complete")
    
    # Save model
    model_path = "./cameraman_score_model_trained.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved to {model_path}")
    
    return model


# ============================================================================
# PART 2: RECONSTRUCTION WITH TRAINED SCORE
# ============================================================================

def load_cameraman_image(size=128, device='cuda'):
    """Load and resize cameraman image"""
    try:
        from skimage import data
        img = data.camera().astype(np.float32)
    except ImportError:
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        img = 128 + 50 * np.sin(X) * np.cos(Y)
        img = img.astype(np.float32)
    
    if img.shape[0] != size or img.shape[1] != size:
        scale_factor = size / img.shape[0]
        img = ndimage.zoom(img, scale_factor, order=1)
    
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return torch.from_numpy(img).to(device)


def create_ensemble(ensemble_size=5, device='cuda'):
    """Create ensemble of measurement operators"""
    operators = []
    for _ in range(ensemble_size):
        op = GaussianRandomMeasurementOperator(
            n_measurements=8192, 
            n_pixels=16384,
            device=device, 
            normalize=True
        )
        operators.append(op)
    return operators


def visualize_results(ground_truth, y_measurement, x_recon, A_true, output_dir='./results'):
    """Visualize reconstruction"""
    os.makedirs(output_dir, exist_ok=True)
    
    gt_np = ground_truth.cpu().detach().numpy()
    recon_np = x_recon.cpu().detach().numpy()
    
    x_recon_tensor = x_recon.reshape(-1)
    y_recon = A_true.forward(x_recon_tensor)
    
    measurement_residual = y_measurement - y_recon
    backproj_residual = A_true.adjoint(measurement_residual)
    backproj_np = backproj_residual.cpu().detach().numpy().reshape(128, 128)
    
    recon_error = torch.norm(x_recon - ground_truth) / (torch.norm(ground_truth) + 1e-8)
    meas_error = torch.norm(y_recon - y_measurement) / (torch.norm(y_measurement) + 1e-8)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    
    vmin, vmax = np.percentile(recon_np, [1, 99])
    axes[0, 1].imshow(recon_np, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Reconstruction (L2: {recon_error:.4f})')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(backproj_np, cmap='hot')
    axes[1, 0].set_title(f'Backproj Residual (Meas: {meas_error:.4f})')
    axes[1, 0].axis('off')
    
    axes[1, 1].axis('off')
    info_text = f"Image L2: {recon_error:.4f}\nMeas L2: {meas_error:.4f}"
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=14, fontfamily='monospace', fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, 'reconstruction_results.png'), bbox_inches='tight', dpi=150)
    plt.close()
    return recon_error, meas_error


def reconstruct_with_score(score_model, device='cuda'):
    """Reconstruct using trained score model."""
    
    print("\n" + "="*70)
    print("PART 2: RECONSTRUCTION WITH TRAINED SCORE")
    print("="*70)
    
    # Config
    config = {
        'alpha': 2.0,
        'N': 2000,
        'sigma_min': 0.01,
        'sigma_max': 20.0,
        'device': device,
        'max_grad_norm': 10000.0,
        'data_weight': 100.0,
    }
    
    # Load
    ground_truth = load_cameraman_image(size=128, device=device)
    print(f"Image shape: {ground_truth.shape}")
    
    # Create ensemble (K=5) for disorder estimation
    operators = create_ensemble(ensemble_size=5, device=device)
    A_true = operators[0]
    
    # Measurement
    ground_truth_flat = ground_truth.reshape(-1)
    y_measurement = A_true.forward(ground_truth_flat)
    noise = 0.01 * torch.randn_like(y_measurement)
    y_measurement = y_measurement + noise
    y_measurement_batch = y_measurement.unsqueeze(0)
    
    print(f"Measurement SNR: {(torch.norm(y_measurement) / torch.norm(noise)):.2f}")
    
    # Disorder estimation
    print("\nEstimating disorder...")
    disorder_est = EffectiveMediumEstimator(operators, measurement_noise_std=0.01)
    
    def prior_sampler(n): 
        return torch.randn(n, 16384).to(device)
    
    disorder_est.calibrate_residual_covariance(prior_sampler, num_probes=10)
    
    # Sample with trained score
    print("\nSampling with trained score + disorder + A_true operator...")
    sampler = AnomalousLangevinSampler(score_model, config)
    
    x_recon_batch = sampler.sample(
        y=y_measurement_batch, 
        disorder_estimator=disorder_est,
        operator=A_true
    )
    
    x_recon = x_recon_batch[0].reshape(ground_truth.shape)
    
    # Visualize
    img_error, meas_error = visualize_results(ground_truth, y_measurement, x_recon, A_true)
    
    print("\n" + "="*70)
    print(f"✅ RECONSTRUCTION COMPLETE")
    print(f"Image L2 Error:       {img_error:.4f}")
    print(f"Measurement L2 Error: {meas_error:.4f}")
    print("="*70)
    
    if meas_error < 0.2:
        print("✓ Measurement consistency: EXCELLENT")
        if img_error < 0.5:
            print("✓ Image quality: EXCELLENT (score prior working!)")
        elif img_error < 2.0:
            print("✓ Image quality: GOOD (regularization working)")
        else:
            print("⚠ Image quality: Fair (need more training or better prior)")
    
    print("="*70)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Train score model
    score_model = train_score_model(num_epochs=500, batch_size=8, device=device)
    
    # Step 2: Reconstruct with trained score
    reconstruct_with_score(score_model, device=device)


if __name__ == '__main__':
    main()
