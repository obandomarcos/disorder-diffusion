import sys
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import wasserstein_distance, norm
from torchvision import datasets, transforms
from diffusers.schedulers import DDPMScheduler
import os
import tarfile
from pathlib import Path
import urllib.request


@dataclass
class UNet2DOutput:
    sample: torch.FloatTensor


class ScoreNetwork(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, n_layers=4):
        super().__init__()
        layers = []
        in_ch = in_channels
        for _ in range(n_layers - 1):
            layers.append(nn.Conv2d(in_ch, hidden_dim, 3, padding=1))
            layers.append(nn.ReLU())
            in_ch = hidden_dim
        layers.append(nn.Conv2d(in_ch, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, sample, timestep=None, return_dict=True):
        score = self.net(sample)
        return UNet2DOutput(sample=score) if return_dict else score


class CompressedSensingOperator:
    def __init__(self, img_size=32, n_channels=3, compression_ratio=0.25, device=torch.device("cpu")):
        self.img_size, self.n_channels = img_size, n_channels
        self.n_pixels = img_size * img_size * n_channels
        self.n_measurements = int(self.n_pixels * compression_ratio)
        self.A = torch.randn(self.n_measurements, self.n_pixels, device=device) / np.sqrt(self.n_measurements)
        self.AT = self.A.t()

    def forward(self, x):
        return x.view(x.shape[0], -1) @ self.AT

    def adjoint(self, y):
        return (y @ self.A).view(-1, self.n_channels, self.img_size, self.img_size)

    def __call__(self, x):
        return self.forward(x)


class BlurOperator:
    def __init__(self, kernel_size=5, sigma=1.5, device=torch.device("cpu")):
        self.kernel_size = kernel_size
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        self.kernel = ((g[:, None] * g[None, :]) / (g[:, None] * g[None, :]).sum()).to(device)

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        out = torch.zeros_like(x)
        for ch in range(c):
            out[:, ch:ch+1] = F.conv2d(x[:, ch:ch+1], k, padding=self.kernel_size // 2)
        return out

    def __call__(self, x):
        return self.forward(x)


class InpaintingOperator:
    def __init__(self, mask_ratio=0.5, img_size=32, device=torch.device("cpu")):
        self.mask = torch.ones(1, 1, img_size, img_size, dtype=torch.bool, device=device)
        hole_size = int(img_size * np.sqrt(mask_ratio))
        start = (img_size - hole_size) // 2
        self.mask[..., start:start + hole_size, start:start + hole_size] = False

    def forward(self, x):
        return x * self.mask.expand(x.shape[0], x.shape[1], -1, -1).float()

    def __call__(self, x):
        return self.forward(x)


class DPS_Sampler:
    def __init__(self, score_network, scheduler, measurement_op, device):
        self.score_network = score_network
        self.scheduler = scheduler
        self.measurement_op = measurement_op
        self.device = device

    def sample(self, x_T, y, num_steps=50):
        self.scheduler.set_timesteps(num_steps, device=self.device)
        x = x_T.to(self.device)
        self.score_network.eval()
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                eps = self.score_network(x, t, return_dict=False)
                x = self.scheduler.step(eps, t, x).prev_sample
        return x


class DA_DPS_Sampler:
    def __init__(self, score_network, scheduler, measurement_op, device, n_disorder_samples=5):
        self.score_network = score_network
        self.scheduler = scheduler
        self.measurement_op = measurement_op
        self.device = device
        self.n_disorder_samples = n_disorder_samples

    def sample(self, x_T, y, num_steps=50):
        self.scheduler.set_timesteps(num_steps, device=self.device)
        x = x_T.to(self.device)
        self.score_network.eval()
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                score_avg = torch.zeros_like(x)
                for _ in range(self.n_disorder_samples):
                    s = self.score_network(x + torch.randn_like(x) * 0.05, t, return_dict=False)
                    score_avg += s
                x = self.scheduler.step(score_avg / self.n_disorder_samples, t, x).prev_sample
        return x


class EvaluationMetrics:
    @staticmethod
    def psnr(x_true, x_pred, max_val=1.0):
        mse = F.mse_loss(x_true, x_pred)
        return (20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))).item()

    @staticmethod
    def ssim(x_true, x_pred):
        xt, xp = x_true.reshape(-1), x_pred.reshape(-1)
        mt, mp = xt.mean(), xp.mean()
        cov = ((xt - mt) * (xp - mp)).mean()
        vt, vp = (xt - mt).pow(2).mean(), (xp - mp).pow(2).mean()
        return ((2 * cov) / (vt + vp + 1e-8)).item()

    @staticmethod
    def mse(x_true, x_pred):
        return F.mse_loss(x_true, x_pred).item()

    @staticmethod
    def mae(x_true, x_pred):
        return F.l1_loss(x_true, x_pred).item()

    @staticmethod
    def coverage_probability(samples, x_true, confidence_level=0.95):
        mean, std = samples.mean(dim=0), samples.std(dim=0)
        z = norm.ppf((1 + confidence_level) / 2)
        lower, upper = mean - z * std, mean + z * std
        return ((x_true >= lower) & (x_true <= upper)).float().mean().item()

    @staticmethod
    def expected_calibration_error(samples, x_true, n_bins=10):
        mean, std = samples.mean(dim=0), samples.std(dim=0)
        mean_flat, std_flat, true_flat = mean.reshape(-1), std.reshape(-1), x_true.reshape(-1)
        var_flat = std_flat.pow(2)
        sq_errors = (mean_flat - true_flat).pow(2)
        var_min, var_max = var_flat.min(), var_flat.max()
        if var_max - var_min < 1e-8:
            return 0.0
        var_bins = torch.linspace(var_min, var_max, n_bins + 1)
        ece, total = 0.0, 0
        for i in range(n_bins):
            mask = (var_flat >= var_bins[i]) & (var_flat < var_bins[i + 1])
            if mask.sum() == 0:
                continue
            weight = mask.float().sum()
            ece += weight * torch.abs(var_flat[mask].mean() - sq_errors[mask].mean())
            total += weight
        return (ece / total).item() if total > 0 else 0.0

    @staticmethod
    def wasserstein_distance_images(samples1, samples2, n_pixel_samples=500):
        s1 = samples1.reshape(samples1.shape[0], -1).cpu().numpy()
        s2 = samples2.reshape(samples2.shape[0], -1).cpu().numpy()
        indices = np.random.choice(s1.shape[1], min(s1.shape[1], n_pixel_samples), replace=False)
        return np.mean([wasserstein_distance(s1[:, i], s2[:, i]) for i in indices])

    @staticmethod
    def energy_distance(samples1, samples2, max_samples=50):
        n1 = min(samples1.shape[0], max_samples)
        n2 = min(samples2.shape[0], max_samples)
        s1 = samples1[torch.randperm(samples1.shape[0])[:n1]].reshape(n1, -1)
        s2 = samples2[torch.randperm(samples2.shape[0])[:n2]].reshape(n2, -1)
        cross = sum(torch.norm(s1[i] - s2[j], p=2) for i in range(n1) for j in range(n2)) / (n1 * n2)
        within1 = sum(torch.norm(s1[i] - s1[j], p=2) for i in range(n1) for j in range(i+1, n1)) / (n1 * (n1-1) / 2) if n1 > 1 else 0
        within2 = sum(torch.norm(s2[i] - s2[j], p=2) for i in range(n2) for j in range(i+1, n2)) / (n2 * (n2-1) / 2) if n2 > 1 else 0
        return (2 * cross - within1 - within2).item()

    @staticmethod
    def maximum_mean_discrepancy(samples1, samples2, bandwidth=1.0, max_samples=50):
        n1 = min(samples1.shape[0], max_samples)
        n2 = min(samples2.shape[0], max_samples)
        s1 = samples1[torch.randperm(samples1.shape[0])[:n1]].reshape(n1, -1)
        s2 = samples2[torch.randperm(samples2.shape[0])[:n2]].reshape(n2, -1)
        rbf = lambda x, y, bw: torch.exp(-torch.norm(x - y, p=2).pow(2) / (2 * bw ** 2))
        k_xx = sum(rbf(s1[i], s1[j], bandwidth) for i in range(n1) for j in range(n1)) / (n1 ** 2)
        k_yy = sum(rbf(s2[i], s2[j], bandwidth) for i in range(n2) for j in range(n2)) / (n2 ** 2)
        k_xy = sum(rbf(s1[i], s2[j], bandwidth) for i in range(n1) for j in range(n2)) / (n1 * n2)
        return (k_xx + k_yy - 2 * k_xy).item()


def run_posterior_sampling(sampler, y_meas, x_true, n_samples, desc):
    samples = []
    for _ in tqdm(range(n_samples), desc=desc):
        samples.append(sampler.sample(torch.randn_like(x_true), y=y_meas).detach().cpu())
    return torch.cat(samples, dim=0)


def compute_all_metrics(dps_samples, da_dps_samples, x_true_cpu):
    """Compute comprehensive TPAMI metrics comparing DPS vs DA-DPS"""
    m = {}
    dps_mean = dps_samples.mean(dim=0, keepdim=True)
    da_dps_mean = da_dps_samples.mean(dim=0, keepdim=True)
    x_norm = (x_true_cpu + 1) / 2
    dps_norm = (dps_mean + 1) / 2
    da_dps_norm = (da_dps_mean + 1) / 2
    
    # Reconstruction quality metrics
    m['dps_psnr'] = EvaluationMetrics.psnr(x_norm, dps_norm)
    m['da_dps_psnr'] = EvaluationMetrics.psnr(x_norm, da_dps_norm)
    m['dps_ssim'] = EvaluationMetrics.ssim(x_norm, dps_norm)
    m['da_dps_ssim'] = EvaluationMetrics.ssim(x_norm, da_dps_norm)
    m['dps_mse'] = EvaluationMetrics.mse(x_norm, dps_norm)
    m['da_dps_mse'] = EvaluationMetrics.mse(x_norm, da_dps_norm)
    m['dps_mae'] = EvaluationMetrics.mae(x_norm, dps_norm)
    m['da_dps_mae'] = EvaluationMetrics.mae(x_norm, da_dps_norm)
    
    # Uncertainty quantification metrics
    m['dps_coverage'] = EvaluationMetrics.coverage_probability(dps_samples, x_true_cpu)
    m['da_dps_coverage'] = EvaluationMetrics.coverage_probability(da_dps_samples, x_true_cpu)
    m['dps_ece'] = EvaluationMetrics.expected_calibration_error(dps_samples, x_true_cpu)
    m['da_dps_ece'] = EvaluationMetrics.expected_calibration_error(da_dps_samples, x_true_cpu)
    
    # Distribution comparison metrics
    m['wasserstein'] = EvaluationMetrics.wasserstein_distance_images(dps_samples, da_dps_samples)
    m['energy'] = EvaluationMetrics.energy_distance(dps_samples, da_dps_samples)
    m['mmd'] = EvaluationMetrics.maximum_mean_discrepancy(dps_samples, da_dps_samples)
    
    return m


def tensor_to_image(t):
    """Convert tensor (C, H, W) to (H, W, C) numpy for display"""
    if t.dim() == 3:
        return (t.permute(1, 2, 0).numpy() + 1) / 2
    elif t.dim() == 2:
        return t.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {t.shape}")


def run_experiment_for_operator(op_name, operator, x_true, dps_sampler, da_dps_sampler, 
                                 output_dir, n_samples=10, num_steps=50):
    """Run full comparison experiment for a single operator"""
    print(f"\n{'='*80}\nRUNNING: {op_name.upper()}\n{'='*80}")
    
    device = x_true.device
    x_true_cpu = x_true.cpu()
    
    # Generate measurement
    y_meas = operator(x_true)
    
    # Run DPS sampling
    print(f"\n[{op_name}] Running DPS sampling...")
    dps_samples = run_posterior_sampling(dps_sampler, y_meas, x_true, n_samples, 
                                        desc=f"DPS {op_name}")
    
    # Run DA-DPS sampling
    print(f"\n[{op_name}] Running DA-DPS sampling...")
    da_dps_samples = run_posterior_sampling(da_dps_sampler, y_meas, x_true, n_samples,
                                           desc=f"DA-DPS {op_name}")
    
    # Compute metrics
    print(f"\n[{op_name}] Computing metrics...")
    metrics = compute_all_metrics(dps_samples, da_dps_samples, x_true_cpu)
    
    # Ensure samples have correct shape: (n_samples, C, H, W)
    if dps_samples.dim() == 2:
        dps_samples = dps_samples.view(dps_samples.shape[0], *x_true_cpu.shape[1:])
    if da_dps_samples.dim() == 2:
        da_dps_samples = da_dps_samples.view(da_dps_samples.shape[0], *x_true_cpu.shape[1:])
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'UQ Comparison: DPS vs DA-DPS ({op_name})', fontsize=16, fontweight='bold')
    
    # Ground truth
    gt_img = tensor_to_image(x_true_cpu[0])
    axes[0, 0].imshow(np.clip(gt_img, 0, 1))
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    
    # DPS mean
    dps_mean = dps_samples.mean(dim=0)
    dps_mean_img = tensor_to_image(dps_mean)
    axes[0, 1].imshow(np.clip(dps_mean_img, 0, 1))
    axes[0, 1].set_title('DPS Mean Reconstruction')
    axes[0, 1].axis('off')
    
    # DA-DPS mean
    da_dps_mean = da_dps_samples.mean(dim=0)
    da_dps_mean_img = tensor_to_image(da_dps_mean)
    axes[0, 2].imshow(np.clip(da_dps_mean_img, 0, 1))
    axes[0, 2].set_title('DA-DPS Mean Reconstruction')
    axes[0, 2].axis('off')
    
    # Uncertainty: DPS
    dps_std = dps_samples.std(dim=0)
    if dps_std.dim() == 3:
        dps_std_vis = dps_std.mean(dim=0).numpy()
    else:
        dps_std_vis = dps_std.numpy()
    im0 = axes[0, 3].imshow(dps_std_vis, cmap='viridis')
    axes[0, 3].set_title('DPS Uncertainty')
    axes[0, 3].axis('off')
    plt.colorbar(im0, ax=axes[0, 3])
    
    # Uncertainty: DA-DPS
    da_dps_std = da_dps_samples.std(dim=0)
    if da_dps_std.dim() == 3:
        da_dps_std_vis = da_dps_std.mean(dim=0).numpy()
    else:
        da_dps_std_vis = da_dps_std.numpy()
    im1 = axes[1, 0].imshow(da_dps_std_vis, cmap='viridis')
    axes[1, 0].set_title('DA-DPS Uncertainty')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Error: DPS
    dps_error = torch.abs(dps_mean - x_true_cpu[0])
    if dps_error.dim() == 3:
        dps_error_vis = dps_error.mean(dim=0).numpy()
    else:
        dps_error_vis = dps_error.numpy()
    im2 = axes[1, 1].imshow(dps_error_vis, cmap='hot')
    axes[1, 1].set_title(f"DPS Error (MSE: {metrics['dps_mse']:.4f})")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Error: DA-DPS
    da_dps_error = torch.abs(da_dps_mean - x_true_cpu[0])
    if da_dps_error.dim() == 3:
        da_dps_error_vis = da_dps_error.mean(dim=0).numpy()
    else:
        da_dps_error_vis = da_dps_error.numpy()
    im3 = axes[1, 2].imshow(da_dps_error_vis, cmap='hot')
    axes[1, 2].set_title(f"DA-DPS Error (MSE: {metrics['da_dps_mse']:.4f})")
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2])
    
    # Metrics comparison
    ax_metrics = axes[1, 3]
    ax_metrics.axis('off')
    metrics_text = (
        f"RECONSTRUCTION QUALITY\n"
        f"DPS PSNR: {metrics['dps_psnr']:.2f} dB\n"
        f"DA-DPS PSNR: {metrics['da_dps_psnr']:.2f} dB\n"
        f"DPS SSIM: {metrics['dps_ssim']:.4f}\n"
        f"DA-DPS SSIM: {metrics['da_dps_ssim']:.4f}\n\n"
        f"UNCERTAINTY QUANTIFICATION\n"
        f"DPS Coverage: {metrics['dps_coverage']:.4f}\n"
        f"DA-DPS Coverage: {metrics['da_dps_coverage']:.4f}\n"
        f"DPS ECE: {metrics['dps_ece']:.4f}\n"
        f"DA-DPS ECE: {metrics['da_dps_ece']:.4f}\n\n"
        f"DISTRIBUTION METRICS\n"
        f"Wasserstein: {metrics['wasserstein']:.4f}\n"
        f"Energy Distance: {metrics['energy']:.4f}\n"
        f"MMD: {metrics['mmd']:.6f}"
    )
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_{op_name}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_dir / f'comparison_{op_name}.png'}")
    plt.close()
    
    return metrics, dps_samples, da_dps_samples


def save_metrics_table(all_results, output_dir):
    """Save comprehensive metrics table"""
    import json
    
    # Create summary table
    summary = {}
    for op_name, (metrics, _, _) in all_results.items():
        summary[op_name] = metrics
    
    # Save as JSON
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save as text table
    with open(output_dir / 'metrics_table.txt', 'w') as f:
        f.write("="*120 + "\n")
        f.write("COMPREHENSIVE UQ COMPARISON: DPS vs DA-DPS on CIFAR-10\n")
        f.write("="*120 + "\n\n")
        
        for op_name, (metrics, _, _) in all_results.items():
            f.write(f"\n{op_name.upper()}\n")
            f.write("-"*80 + "\n")
            
            f.write("\nRECONSTRUCTION QUALITY METRICS:\n")
            f.write(f"  PSNR (dB)      | DPS: {metrics['dps_psnr']:8.4f} | DA-DPS: {metrics['da_dps_psnr']:8.4f} | Δ: {metrics['da_dps_psnr'] - metrics['dps_psnr']:+8.4f}\n")
            f.write(f"  SSIM           | DPS: {metrics['dps_ssim']:8.4f} | DA-DPS: {metrics['da_dps_ssim']:8.4f} | Δ: {metrics['da_dps_ssim'] - metrics['dps_ssim']:+8.4f}\n")
            f.write(f"  MSE            | DPS: {metrics['dps_mse']:8.4f} | DA-DPS: {metrics['da_dps_mse']:8.4f} | Δ: {metrics['da_dps_mse'] - metrics['dps_mse']:+8.4f}\n")
            f.write(f"  MAE            | DPS: {metrics['dps_mae']:8.4f} | DA-DPS: {metrics['da_dps_mae']:8.4f} | Δ: {metrics['da_dps_mae'] - metrics['dps_mae']:+8.4f}\n")
            
            f.write("\nUNCERTAINTY QUANTIFICATION METRICS:\n")
            f.write(f"  Coverage (95%) | DPS: {metrics['dps_coverage']:8.4f} | DA-DPS: {metrics['da_dps_coverage']:8.4f} | Δ: {metrics['da_dps_coverage'] - metrics['dps_coverage']:+8.4f}\n")
            f.write(f"  ECE            | DPS: {metrics['dps_ece']:8.4f} | DA-DPS: {metrics['da_dps_ece']:8.4f} | Δ: {metrics['da_dps_ece'] - metrics['dps_ece']:+8.4f}\n")
            
            f.write("\nDISTRIBUTION COMPARISON METRICS:\n")
            f.write(f"  Wasserstein    | {metrics['wasserstein']:12.4f}\n")
            f.write(f"  Energy Distance| {metrics['energy']:12.4f}\n")
            f.write(f"  MMD            | {metrics['mmd']:12.6f}\n")
    
    print(f"✓ Metrics saved to {output_dir / 'metrics_table.txt'}")


def download_and_prepare_ffhq(root: str):
    """
    Download and extract FFHQ if not already present.
    Expects NVLabs .tar file and arranges images into a single ImageFolder class.
    """
    root = Path(root)
    images_dir = root / "images"
    if images_dir.exists() and any(images_dir.glob("*.png")):
        print(f"FFHQ already prepared at {images_dir}")
        return images_dir

    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "ffhq.tar"

    if not tar_path.exists():
        # URL is an example; point to your actual FFHQ tarball location
        url = "https://archive.org/download/ffhq-dataset/thumbnails128x128.tar"  # replace with your desired res
        print(f"Downloading FFHQ from {url} to {tar_path} ...")
        urllib.request.urlretrieve(url, tar_path)  # [web:1][web:9]
        print("Download complete.")

    print("Extracting FFHQ tar...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=root)  # [web:1]

    # Assume extraction produced a flat directory of images; move into images_dir
    images_dir.mkdir(exist_ok=True)
    for p in root.glob("**/*.png"):
        if p.parent == images_dir:
            continue
        target = images_dir / p.name
        if not target.exists():
            p.rename(target)

    print(f"FFHQ images prepared at {images_dir}")
    return images_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DA-DPS vs DPS UQ Comparison')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        '--dataset', type=str, default='cifar10',
        choices=['cifar10', 'ffhq'],
        help='Dataset to use: cifar10 or ffhq'
    )
    parser.add_argument(
        '--ffhq-root', type=str, default='data/ffhq',
        help='Root directory to store FFHQ (tar + extracted images)'
    )
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--n-disorder-samples', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./unc_comparison_results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DA-DPS vs DPS: Comprehensive UQ Comparison on {args.dataset.upper()}")
    print(f"{'='*80}")

    score_network = ScoreNetwork(in_channels=3, hidden_dim=128, n_layers=4).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Transform: NO resize, only ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    print("Loading test image...")
    if args.dataset == 'cifar10':
        cifar_dataset = datasets.CIFAR10(
            root='/tmp/cifar10',
            train=False,
            download=True,
            transform=transform
        )
        x_true, _ = cifar_dataset[0]
    else:  # ffhq
        ffhq_images_dir = download_and_prepare_ffhq(args.ffhq_root)
        # Put all images in a single dummy class for ImageFolder
        dummy_root = Path(args.ffhq_root) / "imagefolder_ffhq"
        class_dir = dummy_root / "faces"
        if not class_dir.exists():
            class_dir.mkdir(parents=True, exist_ok=True)
            for img_path in ffhq_images_dir.glob("*.png"):
                target = class_dir / img_path.name
                if not target.exists():
                    img_path.link_to(target) if hasattr(img_path, "link_to") else img_path.rename(target)

        ffhq_dataset = datasets.ImageFolder(
            root=str(dummy_root),
            transform=transform
        )
        x_true, _ = ffhq_dataset[0]

    x_true = x_true.unsqueeze(0).to(device)
    print(f"✓ Loaded image from {args.dataset.upper()} with shape {x_true.shape}")

    # You must ensure operators use x_true.shape[-2:] instead of hard-coded 32
    H, W = x_true.shape[-2:]
    operators = {
        'compressed_sensing': CompressedSensingOperator(
            img_size=H, n_channels=3,
            compression_ratio=0.25, device=device
        ),
        'blur': BlurOperator(kernel_size=5, sigma=1.5, device=device),
        'inpainting': InpaintingOperator(mask_ratio=0.5, img_size=H, device=device),
    }
    
    for op_name in operators.keys():
        print(f"✓ Initialized {op_name}")
    
    # Run experiments
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*80}")
    
    all_results = {}
    
    for op_name, operator in operators.items():
        # Initialize samplers for this operator
        dps_sampler = DPS_Sampler(score_network, scheduler, operator, device)
        da_dps_sampler = DA_DPS_Sampler(score_network, scheduler, operator, device, 
                                        n_disorder_samples=args.n_disorder_samples)
        
        # Run experiment
        metrics, dps_samples, da_dps_samples = run_experiment_for_operator(
            op_name, operator, x_true, dps_sampler, da_dps_sampler,
            output_dir, n_samples=args.n_samples, num_steps=args.num_steps
        )
        
        all_results[op_name] = (metrics, dps_samples, da_dps_samples)
        
        # Print summary for this operator
        print(f"\n[{op_name.upper()}] SUMMARY:")
        print(f"  DPS PSNR: {metrics['dps_psnr']:.4f} dB → DA-DPS PSNR: {metrics['da_dps_psnr']:.4f} dB")
        print(f"  DPS Coverage: {metrics['dps_coverage']:.4f} → DA-DPS Coverage: {metrics['da_dps_coverage']:.4f}")
        print(f"  Wasserstein Distance: {metrics['wasserstein']:.6f}")
        print(f"  Energy Distance: {metrics['energy']:.6f}")
        print(f"  MMD: {metrics['mmd']:.8f}")
    
    # Save comprehensive results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    save_metrics_table(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Visualizations: comparison_*.png")
    print(f"  - Metrics table: metrics_table.txt")
    print(f"  - Metrics JSON: metrics_summary.json")


if __name__ == '__main__':
    main()
