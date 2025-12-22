#!/usr/bin/env python3
"""
Simplified CIFAR-10 Ablation Study Script - FIXED VERSION
=========================================================

Uses CIFAR-10 dataset and properly generates measurements.
"""

import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ablation.config import AblationConfig
from ablation.runner import AblationStudyRunner
from ablation.visualization import AblationVisualizer
from da_dps.disorder.ensemble import DisorderEnsemble  
from da_dps.disorder.distribution import DisorderDistribution
from da_dps.operators.measurement import GaussianRandomMeasurementOperator
from da_dps.diffusion.score_network import ScoreNetwork
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class SimpleDataset(Dataset):
    """Simple CIFAR-10 wrapper that returns only images."""
    
    def __init__(self, root='./data', train=True, download=True):
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
    
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, _ = self.cifar10[idx]
        # Convert RGB to grayscale
        grayscale = image.mean(dim=0, keepdim=True)
        # Normalize to [-1, 1]
        grayscale = 2.0 * grayscale - 1.0
        return grayscale


def main():
    """Complete ablation study pipeline with CIFAR-10."""
    
    print("="*80)
    print("ABLATION STUDY: Disorder-Inspired DA-DPS with CIFAR-10")
    print("="*80)
    
    # ========================================================================
    # 1. Setup configuration
    # ========================================================================
    
    config = AblationConfig(
        experiment_name="dadps_disorder_ablation_cifar10",
        n_test_samples=50,  # Reduced for faster testing
        num_steps=50,
        guidance_scale=1.0,
        disorder_type="uniform",
        disorder_low=0.8,
        disorder_high=1.2,
        n_disorder=10,
        reg_type="consistency",
        reg_weight=0.1,
        seed=42,
    )
    
    output_dir = "./ablation_results_cifar10"
    
    print("\nConfiguration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Test samples: {config.n_test_samples}")
    print(f"  Output: {output_dir}")
    print(f"  Disorder ensemble size: {config.n_disorder}")
    
    # ========================================================================
    # 2. Initialize components
    # ========================================================================
    
    print("\n" + "-"*80)
    print("Initializing Components")
    print("-"*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Score network
    print("\n1. Score Network...")
    score_network = ScoreNetwork(
        in_channels=1,
        hidden_dim=64,  # Reduced for faster testing
        n_layers=3
    ).to(device)
    print(f"   Parameters: {sum(p.numel() for p in score_network.parameters()):,}")
    
    # Measurement operator
    print("2. Measurement Operator...")
    measurement_op = GaussianRandomMeasurementOperator(
        n_measurements=256,
        n_pixels=32 * 32,
        device=device,
        normalize=True
    )
    print(f"   Compression ratio: {measurement_op.compression_ratio:.1%}")
    
    # Disorder ensemble
    print("3. Disorder Ensemble...")
    disorder_dist = DisorderDistribution(
        distribution_type=config.disorder_type,
        low=config.disorder_low,
        high=config.disorder_high
    )
    
    disorder_ensemble = DisorderEnsemble(
        n_disorder=config.n_disorder,
        disorder_dist=disorder_dist,
        device=device
    )
    print(f"   Ensemble size: {config.n_disorder}")
    
    # ========================================================================
    # 3. Load CIFAR-10 dataset (ONLY returns images, NOT measurements)
    # ========================================================================
    
    print("\n4. CIFAR-10 Test Dataset...")
    test_dataset = SimpleDataset(
        root='./data',
        train=False,
        download=True
    )
    
    # Create dataloader that returns only images
    # Measurements will be generated in the runner
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"   Dataset size: {len(test_dataset)}")
    print(f"   Using first {config.n_test_samples} samples")
    
    # ========================================================================
    # 4. Run ablation study
    # ========================================================================
    
    print("\n" + "-"*80)
    print("Running Ablation Study")
    print("-"*80)
    
    runner = AblationStudyRunner(
        base_config=config,
        output_dir=output_dir,
        n_seeds=2,  # Reduced for faster testing
    )
    
    runner.run_all_ablations(
        test_dataloader=test_dataloader,
        score_network=score_network,
        measurement_operator=measurement_op,
        disorder_ensemble=disorder_ensemble,
    )
    
    # ========================================================================
    # 5. Generate visualizations
    # ========================================================================
    
    print("\n" + "-"*80)
    print("Generating Visualizations")
    print("-"*80)
    
    visualizer = AblationVisualizer(
        results_path=f"{output_dir}/ablation_results.json",
        output_dir=output_dir
    )
    visualizer.plot_all()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  ✓ ablation_results.json")
    print("  ✓ ablation_summary.csv")
    print("  ✓ ablation_modes.csv")
    print("  ✓ ablation_report.md")
    print("  ✓ ablation_*.png")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
