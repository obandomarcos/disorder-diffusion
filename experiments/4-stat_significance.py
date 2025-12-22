from scipy import stats

def compute_statistical_significance(results: List[Dict], baseline_mode: str = "baseline"):
    """Perform statistical tests to validate component contributions."""
    
    baseline = next(r for r in results if r['ablation_mode'] == baseline_mode)
    
    significance_results = []
    
    for result in results:
        if result['ablation_mode'] == baseline_mode:
            continue
        
        mode = result['ablation_mode']
        
        # Perform t-test for each metric
        for metric in ['psnr', 'ssim', 'measurement_fidelity']:
            baseline_mean = baseline[metric]['mean']
            baseline_std = baseline[metric]['mean_of_stds']
            
            test_mean = result[metric]['mean']
            test_std = result[metric]['mean_of_stds']
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind_from_stats(
                baseline_mean, baseline_std, n_seeds,
                test_mean, test_std, n_seeds,
                equal_var=False
            )
            
            significance_results.append({
                'mode': mode,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': (test_mean - baseline_mean) / baseline_std,
            })
    
    return pd.DataFrame(significance_results)
