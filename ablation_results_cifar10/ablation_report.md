# Ablation Study Report: Disorder-Inspired Framework

**Date**: 2025-12-22 17:09:03

**Seeds**: 2

## Configuration Components

| Component | Description |
|-----------|-------------|
| Disorder Strength | Ensemble of disorder realizations with varying strength parameters |
| Disorder-Aware Scheduling | Adaptive noise schedule modulated by disorder level |
| Disorder Regularization | Consistency/variance/entropy-based regularization terms |

## Component Activation Matrix

| Mode | Disorder Strength | Noise Schedule | Regularization |
|------|-------------------|----------------|----------------|
| baseline | ❌ | ❌ | ❌ |
| disorder_strength | ✅ | ❌ | ❌ |
| noise_schedule | ❌ | ✅ | ❌ |
| regularization | ❌ | ❌ | ✅ |
| strength_schedule | ✅ | ✅ | ❌ |
| strength_reg | ✅ | ❌ | ✅ |
| schedule_reg | ❌ | ✅ | ✅ |
| full | ✅ | ✅ | ✅ |

## Summary Results

| Mode              | psnr            | ssim            | mse             | measurement_fidelity   | inference_time   |
|:------------------|:----------------|:----------------|:----------------|:-----------------------|:-----------------|
| baseline          | 3.5190 ± 0.0104 | 0.0046 ± 0.0005 | 1.2064 ± 0.0028 | 1.0653 ± 0.0138        | 0.0605 ± 0.0007  |
| disorder_strength | 1.7841 ± 0.2110 | 0.0047 ± 0.0004 | 4.3554 ± 0.5724 | 4.1827 ± 0.6037        | 0.0597 ± 0.0000  |
| noise_schedule    | 3.5190 ± 0.0104 | 0.0046 ± 0.0005 | 1.2064 ± 0.0028 | 1.0653 ± 0.0138        | 0.0596 ± 0.0000  |
| regularization    | 3.5190 ± 0.0104 | 0.0046 ± 0.0005 | 1.2064 ± 0.0028 | 1.0653 ± 0.0138        | 0.0597 ± 0.0000  |
| strength_schedule | 1.7841 ± 0.2110 | 0.0047 ± 0.0004 | 4.3554 ± 0.5724 | 4.1827 ± 0.6037        | 0.0599 ± 0.0002  |
| strength_reg      | 1.7841 ± 0.2110 | 0.0047 ± 0.0004 | 4.3554 ± 0.5724 | 4.1827 ± 0.6037        | 0.0597 ± 0.0001  |
| schedule_reg      | 3.5190 ± 0.0104 | 0.0046 ± 0.0005 | 1.2064 ± 0.0028 | 1.0653 ± 0.0138        | 0.0596 ± 0.0000  |
| full              | 1.7841 ± 0.2110 | 0.0047 ± 0.0004 | 4.3554 ± 0.5724 | 4.1827 ± 0.6037        | 0.0600 ± 0.0000  |

## Key Findings

- **PSNR**: 49.30% improvement (baseline→full)
- **SSIM**: 0.22% improvement (baseline→full)
- **MEASUREMENT_FIDELITY**: 292.62% reduction (baseline→full)

## Component Contributions

Individual component contributions measured by comparing single-component ablations to baseline:

- **Disorder Strength**: -49.30% PSNR improvement
- **Noise Schedule**: 0.00% PSNR improvement
- **Regularization**: 0.00% PSNR improvement

## Visualization

See `ablation_plots.png` for visual comparison of component contributions.

## Conclusions

1. **Component Importance Ranking**: [To be filled based on results]
2. **Interaction Effects**: [Analyze pairwise vs individual contributions]
3. **Recommendations**: [Based on cost-benefit analysis]

