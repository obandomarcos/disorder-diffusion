"""
DIAGNOSTIC: Check if batch dimension fix is actually being used
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from da_dps.operators.measurement import GaussianRandomMeasurementOperator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*70)
print("DIAGNOSTIC: Operator batch dimension handling")
print("="*70)

# Create operator
A = GaussianRandomMeasurementOperator(
    n_measurements=8192,
    n_pixels=16384,
    device=device,
    normalize=True
)

# Test 1D input (correct)
print("\n1. Testing 1D input (CORRECT):")
x_1d = torch.randn(16384, device=device)
try:
    Ax_1d = A.forward(x_1d)
    print(f"   Input shape: {x_1d.shape}")
    print(f"   Output shape: {Ax_1d.shape}")
    print(f"   ✓ Works correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2D input (batch, incorrect for this operator)
print("\n2. Testing 2D input (WRONG - what the bug does):")
x_2d = torch.randn(1, 16384, device=device)
try:
    Ax_2d = A.forward(x_2d)
    print(f"   Input shape: {x_2d.shape}")
    print(f"   Output shape: {Ax_2d.shape}")
    print(f"   ⚠ Produces: {Ax_2d.shape} instead of (8192,)")
    if Ax_2d.shape != torch.Size([8192]):
        print(f"   ✗ WRONG SHAPE! This is the bug!")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test residual computation
print("\n3. Testing residual computation:")
y = torch.randn(8192, device=device)

print("\n   a) With 1D input (correct):")
x_1d = torch.randn(16384, device=device)
Ax_1d = A.forward(x_1d)
residual_1d = Ax_1d - y
grad_1d = A.adjoint(residual_1d)
print(f"      x shape: {x_1d.shape}, grad shape: {grad_1d.shape}")
print(f"      ✓ Correct gradient computation")

print("\n   b) With 2D input (buggy):")
x_2d = torch.randn(1, 16384, device=device)
Ax_2d = A.forward(x_2d)
print(f"      A.forward({x_2d.shape}) → {Ax_2d.shape}")
if Ax_2d.shape != torch.Size([8192]):
    print(f"      ⚠ Output is wrong shape! Cannot compute residual properly")
    print(f"      This explains measurement error = 607!")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("""
If test 2a shows gradient shape (16384,) and test 2b shows wrong output shape:
→ The batch dimension fix needs to be in alss.py

Check that this code is in the sampling loop:

    x_flat = x[0].clone()  # Extract from batch
    Ax = A.forward(x_flat)  # Now (8192,)
    y_flat = y[0]
    residual = Ax - y_flat
    grad_data_flat = A.adjoint(residual)
    grad_data = grad_data_flat.unsqueeze(0)  # Add batch back

NOT:
    x_flat = x.reshape(x.shape[0], -1)  # WRONG!
    Ax = A.forward(x_flat)  # Produces wrong shape
""")
print("="*70)