"""
MINIMAL TEST: Does data gradient actually reduce measurement error?
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from da_dps.operators.measurement import GaussianRandomMeasurementOperator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*70)
print("MINIMAL TEST: Data gradient descent")
print("="*70)

# Setup
A = GaussianRandomMeasurementOperator(
    n_measurements=8192,
    n_pixels=16384,
    device=device,
    normalize=True
)

# Generate synthetic data
x_true = torch.randn(16384, device=device)
y = A.forward(x_true)
print(f"\nGenerated measurement: {y.shape}")

# Initialize random x
x = torch.randn(16384, device=device)
print(f"Initial measurement error: {torch.norm(A.forward(x) - y) / torch.norm(y):.4f}")

# Gradient descent on data term: min 0.5 * ||A(x) - y||^2
print("\nRunning gradient descent...")
lr = 0.001
for step in range(100):
    Ax = A.forward(x)
    residual = Ax - y
    
    # Gradient of 0.5 * ||residual||^2
    grad = A.adjoint(residual)
    grad_norm = torch.norm(grad).item()
    
    # Update
    x = x - lr * grad
    
    # Error
    meas_error = torch.norm(residual) / torch.norm(y)
    
    if step % 20 == 0:
        print(f"  Step {step}: meas_error={meas_error.item():.4f}, grad_norm={grad_norm:.2f}")

print("\n" + "="*70)
print("RESULT:")
print("="*70)
final_meas_error = torch.norm(A.forward(x) - y) / torch.norm(y)
print(f"Final measurement error: {final_meas_error:.4f}")

if final_meas_error < 0.1:
    print("✓ Gradient descent works! A and adjoint are correct.")
    print("\nThen the problem in ALSS is:")
    print("  - Data gradient computed but not being used")
    print("  - Data gradient computed with WRONG operator")
    print("  - Data weight parameter not being applied correctly")
else:
    print("✗ Gradient descent doesn't work - fundamental issue with A/adjoint")

print("="*70)
