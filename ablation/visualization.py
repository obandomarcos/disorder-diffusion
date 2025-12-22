#!/usr/bin/env python3
"""
Visualization utilities for ablation studies.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class AblationVisualizer:
    """Visualization utilities for ablation results."""
    
    def __init__(self, results_path: str, output_dir: str = "./"):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to ablation results JSON
            output_dir: Output directory for plots
        """
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = None
        
        # Load results
        self._load_results()
    
    def _load_results(self):
        """Load results from JSON file."""
        if not self.results_path.exists():
            print(f"Warning: Results file not found: {self.results_path}")
            return
        
        try:
            with open(self.results_path, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded results from {self.results_path}")
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def plot_all(self):
        """Generate all visualizations."""
        if self.results is None:
            print("No results to visualize")
            return
        
        try:
            self.plot_metric_comparison()
            self.plot_metric_heatmap()
            self.plot_metric_progression()
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def plot_metric_comparison(self):
        """Plot metric comparison across ablations."""
        if self.results is None or 'ablations' not in self.results:
            return
        
        try:
            ablations = self.results.get('ablations', {})
            
            if not ablations:
                return
            
            # Extract metrics
            configs = list(ablations.keys())
            metrics = {}
            
            for config, data in ablations.items():
                if isinstance(data, dict) and 'metrics' in data:
                    for metric, value in data['metrics'].items():
                        if metric not in metrics:
                            metrics[metric] = []
                        metrics[metric].append(value)
            
            if not metrics:
                return
            
            # Plot
            fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
            if len(metrics) == 1:
                axes = [axes]
            
            for ax, (metric, values) in zip(axes, metrics.items()):
                ax.bar(range(len(configs)), values)
                ax.set_xticks(range(len(configs)))
                ax.set_xticklabels(configs, rotation=45, ha='right')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} Comparison')
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ablation_bars.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Saved: ablation_bars.png")
        except Exception as e:
            print(f"Error plotting metric comparison: {e}")
    
    def plot_metric_heatmap(self):
        """Plot metric heatmap."""
        if self.results is None or 'ablations' not in self.results:
            return
        
        try:
            ablations = self.results.get('ablations', {})
            
            if not ablations:
                return
            
            # Build matrix
            configs = list(ablations.keys())
            all_metrics = set()
            
            for config, data in ablations.items():
                if isinstance(data, dict) and 'metrics' in data:
                    all_metrics.update(data['metrics'].keys())
            
            if not all_metrics:
                return
            
            metrics_list = sorted(list(all_metrics))
            matrix = np.zeros((len(metrics_list), len(configs)))
            
            for j, config in enumerate(configs):
                data = ablations[config]
                if isinstance(data, dict) and 'metrics' in data:
                    for i, metric in enumerate(metrics_list):
                        matrix[i, j] = data['metrics'].get(metric, 0)
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            
            ax.set_xticks(range(len(configs)))
            ax.set_yticks(range(len(metrics_list)))
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.set_yticklabels(metrics_list)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Metric Value')
            
            plt.title('Ablation Study Metrics Heatmap')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ablation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Saved: ablation_heatmap.png")
        except Exception as e:
            print(f"Error plotting heatmap: {e}")
    
    def plot_metric_progression(self):
        """Plot metric progression across components."""
        if self.results is None or 'ablations' not in self.results:
            return
        
        try:
            ablations = self.results.get('ablations', {})
            
            if not ablations:
                return
            
            # Extract primary metrics (PSNR, SSIM, MSE)
            configs = list(ablations.keys())
            psnr_vals = []
            ssim_vals = []
            mse_vals = []
            
            for config in configs:
                data = ablations[config]
                if isinstance(data, dict) and 'metrics' in data:
                    metrics = data['metrics']
                    psnr_vals.append(metrics.get('psnr', 0))
                    ssim_vals.append(metrics.get('ssim', 0))
                    mse_vals.append(metrics.get('mse', 0))
            
            # Plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            x = np.arange(len(configs))
            width = 0.6
            
            # PSNR
            ax1.bar(x, psnr_vals, width, color='steelblue')
            ax1.set_ylabel('PSNR (dB)', fontsize=12)
            ax1.set_title('PSNR Progression', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # SSIM
            ax2.bar(x, ssim_vals, width, color='seagreen')
            ax2.set_ylabel('SSIM', fontsize=12)
            ax2.set_title('SSIM Progression', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # MSE
            ax3.bar(x, mse_vals, width, color='coral')
            ax3.set_ylabel('MSE', fontsize=12)
            ax3.set_xlabel('Configuration', fontsize=12)
            ax3.set_title('MSE Progression', fontsize=14, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            # Set x-labels only on bottom
            ax3.set_xticks(x)
            ax3.set_xticklabels(configs, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'ablation_progression.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Saved: ablation_progression.png")
        except Exception as e:
            print(f"Error plotting progression: {e}")
    
    def save_summary_csv(self):
        """Save ablation results to CSV."""
        if self.results is None or 'ablations' not in self.results:
            return
        
        try:
            ablations = self.results.get('ablations', {})
            
            rows = []
            for config, data in ablations.items():
                row = {'Configuration': config}
                if isinstance(data, dict) and 'metrics' in data:
                    row.update(data['metrics'])
                rows.append(row)
            
            df = pd.DataFrame(rows)
            csv_path = self.output_dir / 'ablation_summary.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"Saved: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
