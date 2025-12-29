"""
TEST: Does normalization kill the gradient?
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
print("TEST: Normalization effect on gradient descent")
print("="*70)

for normalize in [True, False]:
    print(f"\n{'='*70}")
    print(f"Test with normalize={normalize}")
    print(f"{'='*70}")
    
    # Setup
    A = GaussianRandomMeasurementOperator(
        n_measurements=8192,
        n_pixels=16384,
        device=device,
        normalize=normalize
    )

    # Generate synthetic data
    x_true = torch.randn(16384, device=device)
    y = A.forward(x_true)
    
    # Initialize random x
    x = torch.randn(16384, device=device)
    initial_error = torch.norm(A.forward(x) - y) / torch.norm(y)
    print(f"Initial measurement error: {initial_error:.4f}")

    # Gradient descent
    lr = 0.001
    for step in range(100):
        Ax = A.forward(x)
        residual = Ax - y
        
        grad = A.adjoint(residual)
        grad_norm = torch.norm(grad).item()
        
        x = x - lr * grad
        
        meas_error = torch.norm(residual) / torch.norm(y)
        
        if step % 20 == 0:
            print(f"  Step {step}: meas_error={meas_error.item():.4f}, grad_norm={grad_norm:.2f}")

    final_error = torch.norm(A.forward(x) - y) / torch.norm(y)
    print(f"Final measurement error: {final_error:.4f}")
    print(f"Improvement: {(initial_error - final_error) / initial_error * 100:.1f}%")
    
    if final_error < 0.1:
        print("✓ WORKS")
    else:
        print("✗ FAILS")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("""
If normalize=False gives good results but normalize=True doesn't:
→ The normalization factor is incompatible with the learning rate
→ Solution: Increase lr significantly when normalize=True, or disable normalize
""")
