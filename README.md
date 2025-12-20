# Environment Setup

## Quick Start

```bash
# 1. Create conda environment
conda create -n da-dps python=3.10 -y
conda activate da-dps

# 2. Install PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Requirements

**Python:** 3.9–3.11  
**PyTorch:** ≥2.0.0  
**CUDA:** 11.8+ (optional, CPU works)

## File Structure

```
requirements.txt              # Core dependencies
requirements-datasets.txt     # Optional dataset packages
```

## Troubleshooting

### CUDA Mismatch
```bash
# Check installed CUDA
nvcc --version

# Install matching PyTorch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Out of Memory
```bash
# Reduce batch size in config
batch_size: 1
# Or reduce n_disorder
n_disorder: 5
```

### Import Errors
```bash
# Reinstall with editable mode
pip install -e .
```

## GPU Info
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Deactivate
```bash
conda deactivate
```
