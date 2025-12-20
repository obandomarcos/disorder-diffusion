"""DeepInv measurement operator wrapper."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable


class LinearMeasurementOperator:
    """Linear measurement operator A."""
    
    def __init__(self, A: torch.Tensor, device: str = 'cpu'):
        """
        Initialize linear measurement operator.
        
        Args:
            A: Measurement matrix of shape (m, n)
            device: Torch device
        """
        self.A = A.to(device)
        self.device = device
        self.m, self.n = A.shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward operator.
        
        Args:
            x: Image of shape (batch, channels, height, width) or flattened
            
        Returns:
            torch.Tensor: Measurements
        """
        # Handle different input shapes
        if x.dim() == 4:
            batch, channels, h, w = x.shape
            x_flat = x.reshape(batch, -1)
            y = torch.matmul(x_flat, self.A.T)
            return y
        else:
            return torch.matmul(x, self.A.T)
    
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint operator (transpose).
        
        Args:
            y: Measurements
            
        Returns:
            torch.Tensor: Adjoint result
        """
        return torch.matmul(y, self.A)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.forward(x)


class GaussianRandomMeasurementOperator(LinearMeasurementOperator):
    """Gaussian random measurement operator."""
    
    def __init__(self, n_measurements: int, n_pixels: int,
                 device: str = 'cpu', normalize: bool = True):
        """
        Initialize Gaussian random measurement operator.
        
        Args:
            n_measurements: Number of measurements (m)
            n_pixels: Signal dimension (n)
            device: Torch device
            normalize: Whether to normalize rows
        """
        # Create random Gaussian matrix
        A = torch.randn(n_measurements, n_pixels, device=device)
        
        # Normalize if requested
        if normalize:
            A = A / np.sqrt(n_pixels)
        
        super().__init__(A, device)
        self.compression_ratio = n_measurements / n_pixels


class CompositeMeasurementOperator:
    """Composite measurement operator combining multiple operators."""
    
    def __init__(self, operators: list, device: str = 'cpu'):
        """
        Initialize composite operator.
        
        Args:
            operators: List of measurement operators
            device: Torch device
        """
        self.operators = operators
        self.device = device
    
    def forward(self, x: torch.Tensor) -> list:
        """Apply all operators."""
        return [op(x) for op in self.operators]
    
    def adjoint(self, y_list: list) -> torch.Tensor:
        """Apply adjoints and sum."""
        result = None
        for op, y in zip(self.operators, y_list):
            adj = op.adjoint(y)
            if result is None:
                result = adj
            else:
                result = result + adj
        return result


class DeepInvMeasurementOperator:
    """Wrapper for DeepInv measurement operators."""
    
    def __init__(self, operator_type: str = 'gaussian_cs',
                 n_measurements: int = 256, n_pixels: int = 1024,
                 device: str = 'cpu', **kwargs):
        """
        Initialize DeepInv measurement operator.
        
        Args:
            operator_type: 'gaussian_cs', 'blur', 'inpainting', 'compose'
            n_measurements: Number of measurements
            n_pixels: Signal dimension
            device: Torch device
            **kwargs: Additional operator-specific parameters
        """
        self.operator_type = operator_type
        self.device = device
        
        if operator_type == 'gaussian_cs':
            self.operator = GaussianRandomMeasurementOperator(
                n_measurements=n_measurements,
                n_pixels=n_pixels,
                device=device,
                normalize=kwargs.get('normalize', True)
            )
        
        elif operator_type == 'blur':
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', 1.5)
            self.operator = BlurOperator(
                kernel_size=kernel_size,
                sigma=sigma,
                device=device
            )
        
        elif operator_type == 'inpainting':
            mask = kwargs.get('mask', None)
            if mask is None:
                mask = torch.ones(1, 1, 32, 32, device=device)
                mask[..., 10:20, 10:20] = 0
            self.operator = InpaintingOperator(mask=mask, device=device)
        
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator."""
        return self.operator.forward(x)
    
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint operator."""
        return self.operator.adjoint(y)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class BlurOperator:
    """Blur measurement operator."""
    
    def __init__(self, kernel_size: int = 5, sigma: float = 1.5,
                 device: str = 'cpu'):
        """Initialize blur operator."""
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device
        
        # Create Gaussian kernel
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        x = torch.linspace(-size // 2, size // 2, size)
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply blur."""
        # Simple implementation using conv2d
        kernel_2d = self.kernel.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d @ kernel_2d.T
        
        return torch.nn.functional.conv2d(
            x, kernel_2d, padding=self.kernel_size // 2
        )
    
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint blur (same as forward for symmetric operator)."""
        return self.forward(y)


class InpaintingOperator:
    """Inpainting measurement operator."""
    
    def __init__(self, mask: torch.Tensor, device: str = 'cpu'):
        """
        Initialize inpainting operator.
        
        Args:
            mask: Binary mask (1 = observed, 0 = missing)
        """
        self.mask = mask.to(device).bool()
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inpainting operator (extract observed region)."""
        return x * self.mask
    
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint inpainting."""
        return y * self.mask