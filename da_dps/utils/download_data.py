"""
Dataset Downloader for DA-DPS Benchmarks (FFHQ & ImageNet).
Requires: pip install datasets huggingface_hub torchvision
"""

import os
import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

def download_and_prepare_benchmark(
    dataset_name: str = "ffhq",
    save_dir: str = "./data",
    num_samples: int = 100,
    resolution: int = 256
):
    """
    Downloads standard benchmark datasets and saves them as a PyTorch tensor.
    
    Args:
        dataset_name: 'ffhq' or 'imagenet'
        save_dir: Directory to save the output tensor
        num_samples: Number of evaluation images (TPAMI standard is often 1k or 10k)
        resolution: Target image resolution
    """
    print(f"⬇️  Preparing {dataset_name.upper()} dataset...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Define Preprocessing (matches standard Diffusion input [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scale to [-1, 1]
    ])

    # 2. Load Dataset Stream (Streaming mode avoids downloading terabytes)
    images = []
    
    if dataset_name == "ffhq":
        # Uses standard FFHQ (requires accepting terms on HF usually, or use substitute)
        # Using a reliable mirror for 256x256 faces if official is gated
        try:
            ds = load_dataset("krea/ffhq-256", split="train", streaming=True)
        except:
            print("⚠️  Official FFHQ might be gated. Trying alternate source...")
            ds = load_dataset("huggan/ffhq", split="train", streaming=True)
            
    elif dataset_name == "imagenet":
        # Use ImageNet-1k validation set
        ds = load_dataset("imagenet-1k", split="validation", streaming=True)
    
    else:
        raise ValueError("Dataset must be 'ffhq' or 'imagenet'")

    # 3. Process Samples
    count = 0
    print(f"🔄 Processing {num_samples} images...")
    
    for item in ds:
        if count >= num_samples:
            break
            
        # Handle different column names
        img = item['image'] if 'image' in item else item['file']
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Transform and collect
        tensor = transform(img)
        images.append(tensor)
        count += 1

    # 4. Stack and Save
    batch = torch.stack(images) # (N, 3, 256, 256)
    
    save_path = os.path.join(save_dir, f"{dataset_name}_val_{num_samples}.pt")
    torch.save(batch, save_path)
    
    print(f"✅ Saved {batch.shape} to {save_path}")
    print(f"   Range: [{batch.min():.2f}, {batch.max():.2f}]")
    return save_path

if __name__ == "__main__":
    # Example: Create standard benchmark sets
    # 1. FFHQ - 100 samples for quick dev, 1000 for paper
    download_and_prepare_benchmark("ffhq", num_samples=100)
    
    # 2. ImageNet - 100 samples
    download_and_prepare_benchmark("imagenet", num_samples=100)
