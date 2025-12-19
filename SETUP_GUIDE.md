# Policy Training Experiments

Orchestrates large-scale policy training experiments comparing approximation strategies, noise robustness, and curriculum learning schedules for Bayesian optimal experimental design.

## Quick Start

```bash
bash a_policy_and_approximations.sh
```

Enable/disable experiments by setting flags at the script start:
- `RUN_STATE_APPROX_REWARD_EXACT`, `RUN_STATE_EXACT_REWARD_APPROX`, `RUN_STATE_APPROX_REWARD_APPROX`
- `RUN_NOISY_SIGMA`, `RUN_EXACT_CATEGORICAL`, `RUN_EXACT_BERNOULLI`
- `RUN_EXACT_CONSTRAINED_REWARD`, `RUN_EXACT_CONDITIONAL_BERNOULLI`
- `RUN_CURRICULUM_LEARNING_*` (CONSTANT, LINEAR, EXPONENTIAL, STEPWISE)
- `RUN_DETERMINISTIC` (runs optimal policy baseline)

## Key Configurations

**General Parameters:**
```bash
num_episodes=20000000
agents_per_setting=100
log_every=10000
```

**Device Assignment:**
```bash
DEVICE_0="cuda:0"
DEVICE_1="cuda:1"
```

## Experiment Types

| Experiment | Sweep Variable | Values | Output Pattern |
|---|---|---|---|
| State Approximation | `state_num_samples` | {1, 2, 4, 8, 16} | `STATE_APPROX_S{N}_EXACT_POSTFORREWARD.out` |
| Reward Approximation | `reward_num_samples` | {1, 2, 4, 8} | `REWARD_APPROX_R{N}_EXACT_STATE.out` |
| Noise Robustness | `noise_sigma` | {0.05, 0.1, 0.2, 0.4, 0.8} | `NOISE_SIGMA_TRAINING_{σ}.out` |
| Episode Budgets | `episode_budget` | {1-8} | `EXACT_CONDITIONAL_BERNOULLI_BUDGET_{B}.out` |
| Constrained Reward | `reward_objective_lambda` | {0, 0.01, ..., 0.4} | `EXACT_CONSTRAINED_REWARD_LAMBDA_{λ}.out` |
| Curriculum (Linear) | `sigma_noise_evaluation` | {0.05, 0.1, 0.2, 0.4, 0.8} | `CURRICULUM_LEARNING_LINEAR_NOISE_SIGMA_TRAINING_{σ}.out` |
| Curriculum (Exponential) | `cl_alpha` | {0.1, 0.2, 0.5, 1, 2, 5, 10} | `CURRICULUM_LEARNING_EXPONENTIAL_{α}_NOISE_SIGMA_TRAINING_{σ}.out` |
| Curriculum (Stepwise) | `cl_num_partitions` | {2, 5, 10, 20} | `CURRICULUM_LEARNING_STEPWISE_{p}_NOISE_SIGMA_TRAINING_{σ}.out` |
| Image Scaling | `image_dims` | {4, 8, 12, 16} | `EXACTPOST_REWARD_EXACT_STATE_CATEGORICAL_IMG_{D}.out` |

## Policy Types

- **`categorical`**: Single-pixel selection (main policy)
- **`element_wise`**: Per-pixel decisions (Bernoulli)
- **`conditional_bernoulli`**: Budget-aware pixel selection
- **`optimal`**: Deterministic oracle baseline

## Core Arguments

| Argument | Options | Purpose |
|---|---|---|
| `--policy_type` | categorical, element_wise, conditional_bernoulli, optimal | Policy architecture |
| `--state_mode` | exact, sample_based | State representation |
| `--reward_mode` | exact, sample_based | Reward computation |
| `--state_num_samples` | int | MC samples for state (approximation) |
| `--reward_num_samples` | int | MC samples for reward (approximation) |
| `--image_dims` | int | Problem size (NxN pixels) |
| `--episode_budget` | int | Max measurements per episode |
| `--noise_sigma_training` | float | Training noise std |
| `--noise_sigma_evaluation` | float | Eval noise std |
| `--curriculum_learning_function` | constant, linear, exponential, stepwise | Noise schedule |
| `--curriculum_learning_alpha` | float | Exponential schedule rate |
| `--curriculum_learning_partitions` | int | Stepwise schedule steps |
| `--reward_objective_mode` | unconstrained, budget_constrained | Constraint type |
| `--reward_objective_lambda` | float | Lagrange penalty weight |
| `--device` | cuda:0, cuda:1, cpu | Compute device |
| `--name` | string | Experiment ID |

## Monitoring

```bash
# Real-time progress
tail -f EXPERIMENT_NAME.out

# Active jobs
ps aux | grep 3_policy_training_nonoise.py

# GPU usage
nvidia-smi
```

## Output Files

Named as: `{EXPERIMENT_TYPE}_{PARAMETERS}[_OPTIMAL].out`

Each `.out` file contains training logs for one experiment. Append `_OPTIMAL` indicates the deterministic baseline.

## Notes

- Experiments run in parallel across GPUs
- Adjust `agents_per_setting` to fit memory constraints
- Test with reduced `num_episodes` (~100k) before full runs
- Verify CUDA with `nvidia-smi` before execution
