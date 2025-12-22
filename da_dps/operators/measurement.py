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
        self._input_shape = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward operator.
        
        Args:
            x: Image of shape (batch, channels, height, width) or flattened
            
        Returns:
            torch.Tensor: Measurements
        """
        # Cache input shape for adjoint
        self._input_shape = x.shape
        
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
        res = torch.matmul(y, self.A)
        
        # Reshape if we have cached a 4D shape and dims match
        if self._input_shape is not None and len(self._input_shape) == 4:
            b, c, h, w = self._input_shape
            # Ensure the flattened size matches
            if res.shape[-1] == c * h * w:
                return res.view(res.shape[0], c, h, w)
        
        return res
    
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



class SinglePixelImagingOperator(LinearMeasurementOperator):
    """Single-pixel imaging operator using random pixel selection."""
    
    def __init__(self, n_pixels: int, img_size: int = 32, n_channels: int = 3,
                 device: str = 'cpu', selection_type: str = 'random'):
        """
        Initialize single-pixel imaging operator.
        
        Args:
            n_pixels: Number of random pixels to measure
            img_size: Image spatial dimension (assumes square images)
            n_channels: Number of color channels
            device: Torch device
            selection_type: 'random' for random pixel selection, 'sparse' for sparse pattern
        """
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_total_pixels = img_size * img_size * n_channels
        self.selection_type = selection_type
        
        # Create measurement matrix
        if selection_type == 'random':
            A = torch.zeros(n_pixels, self.n_total_pixels, device=device)
            indices = torch.randperm(self.n_total_pixels, device=device)[:n_pixels]
            A[torch.arange(n_pixels, device=device), indices] = 1.0
        
        elif selection_type == 'sparse':
            # Sparse pattern: sample from corner regions
            A = torch.zeros(n_pixels, self.n_total_pixels, device=device)
            indices = torch.randperm(self.n_total_pixels, device=device)[:n_pixels]
            A[torch.arange(n_pixels, device=device), indices] = 1.0
        
        else:
            raise ValueError(f"Unknown selection type: {selection_type}")
        
        super().__init__(A, device)
        self.compression_ratio = n_pixels / self.n_total_pixels
        self.measurement_indices = indices



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
            operator_type: 'gaussian_cs', 'single_pixel', 'blur', 'inpainting', 'compose'
            n_measurements: Number of measurements
            n_pixels: Signal dimension or image size
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
        
        elif operator_type == 'single_pixel':
            img_size = kwargs.get('img_size', 32)
            n_channels = kwargs.get('n_channels', 3)
            selection_type = kwargs.get('selection_type', 'random')
            self.operator = SinglePixelImagingOperator(
                n_pixels=n_measurements,
                img_size=img_size,
                n_channels=n_channels,
                device=device,
                selection_type=selection_type
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
