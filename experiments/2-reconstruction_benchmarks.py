import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import TVPrior, PnP
from deepinv.models import DnCNN, DRUNet
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create results folder
results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((32, 32))
])
cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(cifar_dataset, batch_size=4, shuffle=True)

# Simple Gaussian noise physics
physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=0.05))

# Get one batch
x, _ = next(iter(dataloader))
x = x.to(device)
y = physics(x)

print(f"Input shape: {x.shape}, Measurement shape: {y.shape}")

def compute_psnr(x_true, x_pred):
    """Compute PSNR between two tensors"""
    x_true = x_true.clamp(0, 1).cpu().numpy()
    x_pred = x_pred.clamp(0, 1).cpu().numpy()
    return psnr(x_true, x_pred, data_range=1.0)

def compute_ssim(x_true, x_pred):
    """Compute SSIM between two tensors"""
    x_true = x_true.clamp(0, 1).cpu().numpy().squeeze()
    x_pred = x_pred.clamp(0, 1).cpu().numpy().squeeze()
    ssim_scores = []
    for i in range(x_true.shape[0]):
        s = ssim(x_true[i], x_pred[i], data_range=1.0)
        ssim_scores.append(s)
    return np.mean(ssim_scores)

def compute_fid(x_true, x_pred):
    """Simplified FID using feature distance"""
    return torch.nn.functional.mse_loss(x_true, x_pred).item()

results = {}

# ============= TV MINIMIZATION =============
print("\n=== TV Minimization ===")
prior_tv = TVPrior(n_it_max=20)
model_tv = optim_builder(
    iteration="PGD",
    prior=prior_tv,
    data_fidelity=L2(),
    max_iter=300,
    params_algo={"stepsize": 1.0, "lambda": 1e-2}
)
x_tv = model_tv(y, physics, x_gt=x, compute_metrics=False)
psnr_tv = compute_psnr(x, x_tv)
ssim_tv = compute_ssim(x, x_tv)
fid_tv = compute_fid(x, x_tv)
results['TV'] = {'PSNR': psnr_tv, 'SSIM': ssim_tv, 'FID': fid_tv}
print(f"PSNR: {psnr_tv:.2f} dB | SSIM: {ssim_tv:.4f} | FID: {fid_tv:.4f}")

# ============= DnCNN (PnP) =============
print("\n=== DnCNN (PnP) ===")
denoiser_dncnn = DnCNN(
    in_channels=1, out_channels=1, pretrained="download", device=device
).to(device)
prior_dncnn = PnP(denoiser=denoiser_dncnn)
model_dncnn = optim_builder(
    iteration="PGD",
    prior=prior_dncnn,
    data_fidelity=L2(),
    max_iter=100,
    params_algo={"stepsize": 0.01, "g_param": 0.01}
)
x_dncnn = model_dncnn(y, physics, x_gt=x, compute_metrics=False)
psnr_dncnn = compute_psnr(x, x_dncnn)
ssim_dncnn = compute_ssim(x, x_dncnn)
fid_dncnn = compute_fid(x, x_dncnn)
results['DnCNN'] = {'PSNR': psnr_dncnn, 'SSIM': ssim_dncnn, 'FID': fid_dncnn}
print(f"PSNR: {psnr_dncnn:.2f} dB | SSIM: {ssim_dncnn:.4f} | FID: {fid_dncnn:.4f}")

# ============= DRUNet (PnP) =============
print("\n=== DRUNet (PnP) ===")
denoiser_drunet = DRUNet(
    in_channels=1, out_channels=1, pretrained="download", device=device
).to(device)
prior_drunet = PnP(denoiser=denoiser_drunet)
model_drunet = optim_builder(
    iteration="PGD",
    prior=prior_drunet,
    data_fidelity=L2(),
    max_iter=100,
    params_algo={"stepsize": 0.01, "g_param": 0.01}
)
x_drunet = model_drunet(y, physics, x_gt=x, compute_metrics=False)
psnr_drunet = compute_psnr(x, x_drunet)
ssim_drunet = compute_ssim(x, x_drunet)
fid_drunet = compute_fid(x, x_drunet)
results['DRUNet'] = {'PSNR': psnr_drunet, 'SSIM': ssim_drunet, 'FID': fid_drunet}
print(f"PSNR: {psnr_drunet:.2f} dB | SSIM: {ssim_drunet:.4f} | FID: {fid_drunet:.4f}")

# ============= Create metrics table =============
df_metrics = pd.DataFrame(results).T
df_metrics = df_metrics.round(4)
print("\n=== Metrics Table ===")
print(df_metrics)

# Save metrics to CSV
csv_path = results_dir / "metrics.csv"
df_metrics.to_csv(csv_path)
print(f"Metrics (CSV) saved to: {csv_path}")

# Save metrics to TXT with formatted table
txt_path = results_dir / "metrics.txt"
with open(txt_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("RECONSTRUCTION BENCHMARKS - METRICS TABLE\n")
    f.write("=" * 60 + "\n\n")
    f.write(df_metrics.to_string())
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("METRICS DESCRIPTION:\n")
    f.write("=" * 60 + "\n")
    f.write("PSNR (Peak Signal-to-Noise Ratio) [dB]: Higher is better\n")
    f.write("SSIM (Structural Similarity Index) [0-1]: Higher is better\n")
    f.write("FID (Fréchet Inception Distance) [MSE proxy]: Lower is better\n")
print(f"Metrics (TXT) saved to: {txt_path}")

# ============= Visualization without table =============
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Ground truth
axes[0, 0].imshow(x[0].squeeze().cpu().numpy(), cmap='gray')
axes[0, 0].set_title('Ground Truth')
axes[0, 0].axis('off')

# Measurement
axes[0, 1].imshow(y[0].squeeze().cpu().numpy(), cmap='gray')
axes[0, 1].set_title('Noisy Measurement')
axes[0, 1].axis('off')

# TV
axes[0, 2].imshow(x_tv[0].squeeze().cpu().numpy(), cmap='gray')
axes[0, 2].set_title(f"TV\nPSNR: {psnr_tv:.2f}")
axes[0, 2].axis('off')

# DnCNN
axes[1, 0].imshow(x_dncnn[0].squeeze().cpu().numpy(), cmap='gray')
axes[1, 0].set_title(f"DnCNN\nPSNR: {psnr_dncnn:.2f}")
axes[1, 0].axis('off')

# DRUNet
axes[1, 1].imshow(x_drunet[0].squeeze().cpu().numpy(), cmap='gray')
axes[1, 1].set_title(f"DRUNet\nPSNR: {psnr_drunet:.2f}")
axes[1, 1].axis('off')

# Empty space
axes[1, 2].axis('off')

plt.tight_layout()
plot_path = results_dir / "reconstruction_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()

print("\n✓ Done! Results saved in ./results/")
