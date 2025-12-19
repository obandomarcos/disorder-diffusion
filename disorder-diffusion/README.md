# Setup Guide for Disorder-Diffusion Single-Pixel Imaging Research

## Quick Start

### Option 1: Using Conda (Recommended)

```bash
# Clone your repository (or create new project directory)
cd /path/to/disorder-diffusion

# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate disorder-diffusion

# Verify installation
python -c "import torch, diffusers, jax; print('All imports successful!')"

# Launch Jupyter Lab
jupyter lab
```

### Option 2: Using pip (Alternative)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch, diffusers, jax; print('All imports successful!')"
```

---

## Project Structure

Create this directory structure for your research:

```
disorder-diffusion/
├── environment.yml              # Conda environment specification
├── requirements.txt             # Pip requirements
├── setup.py                     # Package setup (if needed)
├── README.md                    # Project overview
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── diffusion/               # Diffusion model code
│   │   ├── __init__.py
│   │   ├── base.py              # Base DPS class
│   │   ├── dps.py               # Standard DPS
│   │   ├── da_dps.py            # Disorder-averaged DPS (Stage 1)
│   │   ├── ctrw_dps.py          # CTRW-based DPS (Stage 2)
│   │   └── da_boed.py           # DA-BOED framework (Stage 3)
│   │
│   ├── disorder/                # Disorder theory implementations
│   │   ├── __init__.py
│   │   ├── ema.py               # Effective Medium Approximation
│   │   ├── ctrw.py              # CTRW formalism
│   │   ├── anomalous.py         # Anomalous diffusion utilities
│   │   └── green_functions.py   # Green function calculations
│   │
│   ├── imaging/                 # Imaging-specific code
│   │   ├── __init__.py
│   │   ├── single_pixel.py      # Single-pixel imaging simulation
│   │   ├── measurement.py       # Measurement matrices (Gaussian, Hadamard, DMD)
│   │   ├── forward_models.py    # Forward imaging models
│   │   └── noise_models.py      # Photon noise, readout noise
│   │
│   ├── bayesian/                # Bayesian experimental design
│   │   ├── __init__.py
│   │   ├── eig.py               # Expected Information Gain computation
│   │   ├── pooled_posterior.py  # Pooled posterior sampling
│   │   ├── adaptive_design.py   # Adaptive measurement design
│   │   └── optimization.py      # Design optimization algorithms
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── metrics.py           # PSNR, SSIM, FID, LPIPS
│       ├── plotting.py          # Visualization helpers
│       ├── io.py                # Data I/O functions
│       ├── config.py            # Configuration management
│       └── logging.py           # Logging utilities
│
├── experiments/                 # Experimental scripts
│   ├── stage1/                  # Stage 1: DA-DPS
│   │   ├── __init__.py
│   │   ├── train_diffusion.py   # Train score network
│   │   ├── evaluate_dps.py      # Evaluate standard DPS
│   │   ├── evaluate_da_dps.py   # Evaluate disorder-averaged DPS
│   │   ├── compare_methods.py   # Comparison experiments
│   │   └── config.yaml          # Experiment config
│   │
│   ├── stage2/                  # Stage 2: CTRW-DPS
│   │   ├── __init__.py
│   │   ├── optimal_alpha.py     # Compute optimal α
│   │   ├── ctrw_scheduling.py   # Test CTRW schedules
│   │   ├── convergence_analysis.py  # Convergence rate experiments
│   │   ├── ablation_study.py    # Ablation on α parameter
│   │   └── config.yaml
│   │
│   └── stage3/                  # Stage 3: DA-BOED
│       ├── __init__.py
│       ├── boed_optimization.py # BOED with DA-DPS
│       ├── sequential_design.py # Sequential measurement selection
│       ├── domain_generalization.py  # Cross-domain testing
│       ├── real_data_validation.py   # Real imaging data
│       └── config.yaml
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_disorder_theory_intro.ipynb
│   ├── 02_ema_validation.ipynb
│   ├── 03_ctrw_scheduling.ipynb
│   ├── 04_da_boed_demo.ipynb
│   └── 05_results_visualization.ipynb
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_diffusion.py
│   ├── test_disorder.py
│   ├── test_imaging.py
│   ├── test_bayesian.py
│   └── test_utils.py
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed datasets
│   ├── pretrained/              # Pretrained diffusion models
│   └── results/                 # Experiment results
│
├── outputs/                     # Experiment outputs (gitignored)
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
│
├── docs/                        # Documentation
│   ├── api.md
│   ├── methods.md
│   ├── experiments.md
│   └── faq.md
│
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions CI
```

---

## Installation Verification

After installation, verify everything works:

```python
# test_imports.py
import sys

print("Testing imports...")

# Core packages
import numpy as np
print(f"✓ NumPy {np.__version__}")

import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
print(f"  - Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

import scipy
print(f"✓ SciPy {scipy.__version__}")

# Deep Learning
from diffusers import DDIMScheduler, UNet2DModel
print("✓ Diffusers imported successfully")

import pytorch_lightning as pl
print(f"✓ PyTorch Lightning {pl.__version__}")

# Bayesian
import pymc as pm
print(f"✓ PyMC {pm.__version__}")

try:
    import jax
    print(f"✓ JAX {jax.__version__}")
except ImportError:
    print("⚠ JAX not available (optional)")

# Visualization
import matplotlib.pyplot as plt
print(f"✓ Matplotlib {plt.matplotlib.__version__}")

import seaborn as sns
print(f"✓ Seaborn {sns.__version__}")

# Optimization
import cvxpy as cp
print(f"✓ CVXPY {cp.__version__}")

print("\n✅ All critical imports successful!")
print(f"Python: {sys.version}")
```

Run with:
```bash
python test_imports.py
```

---

## GPU Setup (Optional but Recommended)

### Check GPU:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
```

### Install CUDA Toolkit (if not using conda):
- Visit: https://developer.nvidia.com/cuda-downloads
- Follow installation instructions for your OS

### Install cuDNN (optional, for speedup):
- Download from: https://developer.nvidia.com/cudnn
- Follow Nvidia's installation guide

---

## Development Workflow

### 1. Code Quality Tools

```bash
# Format code with Black
black src/ experiments/ tests/

# Check code style
flake8 src/ --max-line-length=100

# Type checking
mypy src/

# Run linter
pylint src/
```

### 2. Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_diffusion.py::test_dps_initialization -v

# Benchmark specific functions
pytest tests/test_diffusion.py --benchmark-only
```

### 3. Version Control

```bash
# Create feature branch
git checkout -b feature/stage1-da-dps

# Commit with conventional commits
git commit -m "feat(diffusion): implement disorder-averaged DPS"

# Push and create PR
git push origin feature/stage1-da-dps
```

---

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

```python
# Solution: Reduce batch size in config.yaml
batch_size: 4  # Instead of 32

# Or enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Issue 2: Import Errors

```bash
# Reinstall in editable mode
pip install -e .

# Or reinstall environment
conda env remove -n disorder-diffusion
conda env create -f environment.yml
```

### Issue 3: Dependency Conflicts

```bash
# Check dependency tree
pip tree

# Update all packages
conda update -n disorder-diffusion --all

# Or for pip
pip install --upgrade -r requirements.txt
```

### Issue 4: JAX/GPU Issues

```bash
# JAX might conflict with PyTorch on GPU
# Solution: Use CPU for JAX if needed
import jax
jax.config.update('jax_platform_name', 'cpu')
```

---

## Environment Variables

Create `.env` file in project root:

```bash
# .env
CUDA_VISIBLE_DEVICES=0  # GPU to use
PYTHONPATH=/path/to/disorder-diffusion:$PYTHONPATH
WANDB_PROJECT=disorder-diffusion-research
WANDB_ENTITY=your-username
LOG_LEVEL=INFO
```

Load with:
```bash
source .env
```

Or in Python:
```python
import os
from dotenv import load_dotenv
load_dotenv('.env')
```

---

## Computing Resources

### Recommended Hardware

**Stage 1 (DA-DPS)**:
- GPU: RTX 3070 or better (8GB VRAM sufficient)
- RAM: 32GB
- Storage: 100GB (models + datasets)
- Time: 1-2 weeks

**Stage 2 (CTRW-DPS)**:
- GPU: A100 or RTX 4090 (optimal)
- GPU: RTX 3080+ acceptable
- RAM: 64GB
- Storage: 200GB
- Time: 3-4 weeks

**Stage 3 (DA-BOED)**:
- GPU: Multi-GPU (2x A100 or 4x RTX 3090)
- RAM: 128GB
- Storage: 500GB
- Time: 8-12 weeks

### Cloud Options

**Google Colab** (Free/Paid):
```bash
# Install in Colab
!pip install -r requirements.txt
```

**AWS SageMaker** (Recommended for scale):
- Pre-configured PyTorch environments
- Easy multi-GPU/TPU scaling
- Integrated with notebooks

**Lambda Labs**:
- On-demand GPU rental
- Pay-per-hour
- Good for quick experiments

---

## Troubleshooting Conda

```bash
# Clear conda cache
conda clean --all

# Repair environment
conda install --force-reinstall -y -q --name disorder-diffusion --file requirements.txt

# See what's installed
conda list -n disorder-diffusion

# Export current environment
conda env export -n disorder-diffusion > environment_current.yml

# Remove and recreate
conda env remove -n disorder-diffusion
conda env create -f environment.yml
```

---

## Next Steps

1. **Install environment**: `conda env create -f environment.yml`
2. **Verify installation**: `python test_imports.py`
3. **Start with tutorials**: `jupyter lab` and open notebooks in `notebooks/`
4. **Run Stage 1 baseline**: `python experiments/stage1/evaluate_dps.py`

---

## Reference Documentation

- PyTorch: https://pytorch.org/docs/stable/
- Diffusers: https://huggingface.co/docs/diffusers/
- JAX: https://jax.readthedocs.io/
- PyMC: https://www.pymc.io/
- PyTorch Lightning: https://lightning.ai/

---

## Support and Questions

- Create GitHub issues for bugs
- Discussions tab for questions
- Email: your-email@institution.edu

Happy researching! 🚀
