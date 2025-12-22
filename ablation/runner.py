import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import List, Dict
import pandas as pd
from ablation.config import AblationConfig
import torch.nn as nn
from ablation.components import DisorderAwareScheduler
from ablation.components import AblationDA_DPS_Sampler

class AblationStudyRunner:
    """Orchestrates ablation experiments across all component configurations."""
    
    def __init__(
        self,
        base_config: AblationConfig,
        output_dir: str = "./ablation_results",
        n_seeds: int = 3,
    ):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_seeds = n_seeds
        
        # All ablation modes to test
        self.ablation_modes = [
            "baseline",
            "disorder_strength",
            "noise_schedule",
            "regularization",
            "strength_schedule",
            "strength_reg",
            "schedule_reg",
            "full",
        ]
        
        self.results = []
    
    def run_all_ablations(
        self,
        test_dataloader: DataLoader,
        score_network: nn.Module,
        measurement_operator: nn.Module,
        disorder_ensemble: 'DisorderEnsemble',
    ):
        """Run complete ablation study across all configurations."""
        
        print("=" * 80)
        print("ABLATION STUDY: Disorder-Inspired Framework Components")
        print("=" * 80)
        
        for mode in self.ablation_modes:
            print(f"\n{'='*80}")
            print(f"Testing Configuration: {mode.upper()}")
            print(f"{'='*80}")
            
            mode_results = []
            
            for seed in range(self.n_seeds):
                print(f"\n  Seed {seed + 1}/{self.n_seeds}")
                
                # Create configuration for this ablation mode
                config = AblationConfig(
                    ablation_mode=mode,
                    seed=self.base_config.seed + seed,
                    num_steps=self.base_config.num_steps,
                    guidance_scale=self.base_config.guidance_scale,
                    disorder_type=self.base_config.disorder_type,
                    disorder_low=self.base_config.disorder_low,
                    disorder_high=self.base_config.disorder_high,
                    n_disorder=self.base_config.n_disorder,
                    reg_weight=self.base_config.reg_weight,
                )
                
                # Set seed
                torch.manual_seed(config.seed)
                np.random.seed(config.seed)
                
                # Initialize sampler
                scheduler = DisorderAwareScheduler(
                    num_timesteps=1000,
                    schedule_type=config.schedule_type,
                    disorder_ensemble=disorder_ensemble if config.enable_disorder_schedule else None,
                    disorder_weight=config.disorder_schedule_weight,
                    enable_disorder_schedule=config.enable_disorder_schedule,
                )
                
                sampler = AblationDA_DPS_Sampler(
                    score_network=score_network,
                    measurement_operator=measurement_operator,
                    scheduler=scheduler,
                    disorder_ensemble=disorder_ensemble if config.enable_disorder_strength else None,
                    config=config,
                    device=self.base_config.device if hasattr(self.base_config, 'device') else 'cuda',
                )
                
                # Run evaluation
                metrics = self._evaluate(sampler, test_dataloader, config)
                metrics['seed'] = seed
                metrics['ablation_mode'] = mode
                
                mode_results.append(metrics)
            
            # Aggregate results for this mode
            aggregated = self._aggregate_results(mode_results)
            aggregated['ablation_mode'] = mode
            self.results.append(aggregated)
            
            # Print summary
            self._print_summary(aggregated, mode)
        
        # Save final results
        self._save_results()
        self._generate_report()
        
        print(f"\n{'='*80}")
        print("Ablation study complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")
    
    def _evaluate(self, sampler, test_dataloader, config):
        """Evaluate sampler on test dataset."""
        import time
        import numpy as np
        from ablation.metrics import EvaluationMetrics
        import torch
        
        all_metrics = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'measurement_fidelity': [],
            'inference_time': [],
        }
        
        # Get device from sampler instead of config
        device = sampler.device  # ← KEY FIX: Use sampler.device
        sampler.score_network.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx >= config.n_test_samples:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x_true, y_true = batch
                else:
                    x_true = batch
                    y_true = None
                
                # Move to device (use 'device' variable, not config.device)
                x_true = x_true.to(device)
                if y_true is not None:
                    y_true = y_true.to(device)
                
                # FIX: If y_true has wrong shape, regenerate measurements
                if y_true is not None:
                    if y_true.dim() == 1:
                        y_true = y_true.unsqueeze(0)
                    
                    if y_true.shape[-1] == x_true.numel() // x_true.shape[0]:
                        y_true = sampler.measurement_operator(x_true)
                
                # Generate measurements if not provided
                if y_true is None:
                    y_true = sampler.measurement_operator(x_true)
                
                # Ensure correct shape
                if y_true.dim() > 2:
                    y_true = y_true.view(y_true.shape[0], -1)
                
                # Generate initial noise
                x_T = torch.randn_like(x_true)
                
                # Sample
                start_time = time.time()
                x_recon, sampling_metrics = sampler.sample(
                    x_T, y_true,
                    num_steps=config.num_steps
                )
                inference_time = time.time() - start_time
                
                # Compute metrics
                batch_metrics = EvaluationMetrics.compute_all_metrics(
                    x_true, x_recon, y_true,
                    sampler.measurement_operator,
                    metrics_list=['psnr', 'ssim', 'mse', 'measurement_fidelity']
                )
                
                for metric, value in batch_metrics.items():
                    if value is not None:
                        all_metrics[metric].append(value)
                
                all_metrics['inference_time'].append(inference_time)
        
        # Compute statistics
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
            for metric, values in all_metrics.items()
            if len(values) > 0
        }

    
    def _aggregate_results(self, mode_results: List[Dict]) -> Dict:
        """Aggregate results across multiple seeds."""
        aggregated = {}
        
        # Get all metric names from first result
        sample_result = mode_results[0]
        metrics = [k for k in sample_result.keys() if k not in ['seed', 'ablation_mode']]
        
        for metric in metrics:
            means = [r[metric]['mean'] for r in mode_results]
            stds = [r[metric]['std'] for r in mode_results]
            
            aggregated[metric] = {
                'mean': np.mean(means),
                'std_of_means': np.std(means),
                'mean_of_stds': np.mean(stds),
            }
        
        return aggregated
    
    def _print_summary(self, aggregated: Dict, mode: str):
        """Print summary statistics for a mode."""
        print(f"\n  Results for {mode}:")
        print(f"  {'-' * 60}")
        for metric, stats in aggregated.items():
            if metric != 'ablation_mode':
                print(f"    {metric:20s}: {stats['mean']:.4f} ± {stats['std_of_means']:.4f}")
    
    def _save_results(self):
        """Save results to JSON and CSV."""
        # Save detailed JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary CSV
        df_data = []
        for result in self.results:
            row = {'ablation_mode': result['ablation_mode']}
            for metric, stats in result.items():
                if metric != 'ablation_mode':
                    row[f"{metric}_mean"] = stats['mean']
                    row[f"{metric}_std"] = stats['std_of_means']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "ablation_summary.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n  Results saved:")
        print(f"    JSON: {json_path}")
        print(f"    CSV:  {csv_path}")
    
    def _generate_report(self):
        """Generate markdown report with analysis."""
        report_path = self.output_dir / "ablation_report.md"
        
        # Create DataFrame for easier manipulation
        df_data = []
        for result in self.results:
            row = {'Mode': result['ablation_mode']}
            for metric, stats in result.items():
                if metric != 'ablation_mode':
                    row[metric] = f"{stats['mean']:.4f} ± {stats['std_of_means']:.4f}"
            df_data.append(row)
        df = pd.DataFrame(df_data)
        
        # Calculate improvements over baseline
        baseline_results = next(r for r in self.results if r['ablation_mode'] == 'baseline')
        full_results = next(r for r in self.results if r['ablation_mode'] == 'full')
        
        with open(report_path, 'w') as f:
            f.write("# Ablation Study Report: Disorder-Inspired Framework\n\n")
            f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Seeds**: {self.n_seeds}\n\n")
            
            f.write("## Configuration Components\n\n")
            f.write("| Component | Description |\n")
            f.write("|-----------|-------------|\n")
            f.write("| Disorder Strength | Ensemble of disorder realizations with varying strength parameters |\n")
            f.write("| Disorder-Aware Scheduling | Adaptive noise schedule modulated by disorder level |\n")
            f.write("| Disorder Regularization | Consistency/variance/entropy-based regularization terms |\n\n")
            
            f.write("## Component Activation Matrix\n\n")
            f.write("| Mode | Disorder Strength | Noise Schedule | Regularization |\n")
            f.write("|------|-------------------|----------------|----------------|\n")
            mode_config_map = {
                "baseline": ("❌", "❌", "❌"),
                "disorder_strength": ("✅", "❌", "❌"),
                "noise_schedule": ("❌", "✅", "❌"),
                "regularization": ("❌", "❌", "✅"),
                "strength_schedule": ("✅", "✅", "❌"),
                "strength_reg": ("✅", "❌", "✅"),
                "schedule_reg": ("❌", "✅", "✅"),
                "full": ("✅", "✅", "✅"),
            }
            for mode, (s, n, r) in mode_config_map.items():
                f.write(f"| {mode} | {s} | {n} | {r} |\n")
            
            f.write("\n## Summary Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Compute improvements
            for metric in ['psnr', 'ssim', 'measurement_fidelity']:
                baseline_val = baseline_results[metric]['mean']
                full_val = full_results[metric]['mean']
                
                if 'fidelity' in metric.lower():  # Lower is better
                    improvement = (baseline_val - full_val) / baseline_val * 100
                    direction = "reduction"
                else:  # Higher is better
                    improvement = (full_val - baseline_val) / baseline_val * 100
                    direction = "improvement"
                
                f.write(f"- **{metric.upper()}**: {abs(improvement):.2f}% {direction} (baseline→full)\n")
            
            f.write("\n## Component Contributions\n\n")
            f.write("Individual component contributions measured by comparing single-component ablations to baseline:\n\n")
            
            for component_mode in ["disorder_strength", "noise_schedule", "regularization"]:
                comp_results = next(r for r in self.results if r['ablation_mode'] == component_mode)
                psnr_improvement = (comp_results['psnr']['mean'] - baseline_results['psnr']['mean']) / baseline_results['psnr']['mean'] * 100
                
                f.write(f"- **{component_mode.replace('_', ' ').title()}**: {psnr_improvement:.2f}% PSNR improvement\n")
            
            f.write("\n## Visualization\n\n")
            f.write("See `ablation_plots.png` for visual comparison of component contributions.\n\n")
            
            f.write("## Conclusions\n\n")
            f.write("1. **Component Importance Ranking**: [To be filled based on results]\n")
            f.write("2. **Interaction Effects**: [Analyze pairwise vs individual contributions]\n")
            f.write("3. **Recommendations**: [Based on cost-benefit analysis]\n\n")
        
        print(f"    Report: {report_path}")
