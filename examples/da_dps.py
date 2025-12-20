import os
import sys

# Add project root (directory containing 'da_dps') to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from da_dps.config import DADPSConfig
import torch
from da_dps.config import DADPSConfig
from da_dps.disorder.distribution import DisorderDistribution
from da_dps.disorder.ensemble import DisorderEnsemble
from da_dps.diffusion.score_network import ScoreNetwork
from da_dps.diffusion.generator import DiffusionGenerator
from da_dps.operators.measurement import GaussianRandomMeasurementOperator
from da_dps.sampling.da_dps import DA_DPS_Sampler
from da_dps.sampling.dps import DPS_Sampler
from da_dps.utils.device import DeviceUtils



def main():
    """Run DA-DPS example."""
    
    # Set seed for reproducibility
    config = DADPSConfig()
    DeviceUtils.set_seed(config.seed)
    
    print("=" * 80)
    print("Disorder-Averaged Diffusion Posterior Sampling (DA-DPS)")
    print("=" * 80)
    
    # Initialize components
    print("\n1. Initializing disorder distribution...")
    disorder_dist = DisorderDistribution(
        distribution_type=config.disorder_type,
        low=config.disorder_low,
        high=config.disorder_high
    )
    print(f"   Distribution statistics: {disorder_dist.get_statistics()}")
    
    print("\n2. Creating disorder ensemble...")
    disorder_ensemble = DisorderEnsemble(
        n_disorder=config.n_disorder,
        disorder_dist=disorder_dist,
        device=config.device
    )
    print(f"   Ensemble statistics: {disorder_ensemble.get_statistics()}")
    
    print("\n3. Initializing measurement operator...")
    measurement_op = GaussianRandomMeasurementOperator(
        n_measurements=config.n_measurements,
        n_pixels=config.n_pixels,
        device=config.device,
        normalize=True
    )
    print(f"   Compression ratio: {measurement_op.compression_ratio:.1%}")
    
    print("\n4. Initializing score network...")
    score_network = ScoreNetwork(
        in_channels=1,
        hidden_dim=config.score_network_hidden_dim,
        n_layers=config.score_network_layers
    ).to(config.device)
    print(f"   Network parameters: {sum(p.numel() for p in score_network.parameters()):,}")
    
    print("\n5. Initializing diffusion generator...")
    diffusion_gen = DiffusionGenerator(
        score_network=score_network,
        num_timesteps=1000,
        device=config.device
    )
    
    print("\n6. Initializing samplers...")
    
    # Standard DPS sampler
    dps_sampler = DPS_Sampler(
        score_network=score_network,
        measurement_operator=measurement_op,
        scheduler=diffusion_gen.scheduler,
        guidance_scale=config.guidance_scale,
        device=config.device
    )
    
    # DA-DPS sampler
    da_dps_sampler = DA_DPS_Sampler(
        score_network=score_network,
        measurement_operator=measurement_op,
        scheduler=diffusion_gen.scheduler,
        disorder_ensemble=disorder_ensemble,
        guidance_scale=config.guidance_scale,
        device=config.device
    )
    
    print("   Both samplers initialized")
    
    # Generate test data
    print("\n7. Generating test data...")
    x_true = torch.randn(1, 1, 32, 32, device=config.device)
    y = measurement_op(x_true)
    print(f"   True image shape: {x_true.shape}")
    print(f"   Measurements shape: {y.shape}")
    
    # Run sampling
    print("\n8. Running DPS sampling...")
    x_T = torch.randn_like(x_true)
    x_recon_dps = dps_sampler.sample(x_T, y, num_steps=50)
    print(f"   DPS reconstruction shape: {x_recon_dps.shape}")
    
    print("\n9. Running DA-DPS sampling...")
    x_T = torch.randn_like(x_true)
    x_recon_da_dps = da_dps_sampler.sample(x_T, y, num_steps=50)
    print(f"   DA-DPS reconstruction shape: {x_recon_da_dps.shape}")
    
    # Evaluate
    print("\n10. Evaluating results...")
    
    metrics_dps = {
        'measurement_fidelity': EvaluationMetrics.measurement_fidelity(x_recon_dps, y, measurement_op),
        'psnr': EvaluationMetrics.psnr(x_true, x_recon_dps),
        'ssim': EvaluationMetrics.ssim(x_true, x_recon_dps),
    }
    
    metrics_da_dps = {
        'measurement_fidelity': EvaluationMetrics.measurement_fidelity(x_recon_da_dps, y, measurement_op),
        'psnr': EvaluationMetrics.psnr(x_true, x_recon_da_dps),
        'ssim': EvaluationMetrics.ssim(x_true, x_recon_da_dps),
    }
    
    print("\n    DPS Metrics:")
    for metric, value in metrics_dps.items():
        print(f"      {metric}: {value:.6f}")
    
    print("\n    DA-DPS Metrics:")
    for metric, value in metrics_da_dps.items():
        print(f"      {metric}: {value:.6f}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()