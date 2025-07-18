"""
Tests for Swin-UNETR Architecture
"""

import pytest
import torch
import torch.nn as nn
from gazimed.models.swin_unetr import (
    WindowAttention3D,
    SwinTransformerBlock3D,
    BasicLayer3D,
    SwinUNETR,
    PatchMerging3D,
    window_partition,
    window_reverse,
    compute_mask
)


class TestWindowAttention3D:
    """Test cases for WindowAttention3D module."""
    
    def test_init(self):
        """Test initialization."""
        attn = WindowAttention3D(
            dim=96,
            window_size=(7, 7, 7),
            num_heads=3
        )
        
        assert attn.dim == 96
        assert attn.window_size == (7, 7, 7)
        assert attn.num_heads == 3
        assert attn.scale == (96 // 3) ** -0.5
        
    def test_forward_pass(self):
        """Test forward pass."""
        attn = WindowAttention3D(
            dim=96,
            window_size=(4, 4, 4),
            num_heads=3
        )
        
        # Create test input (num_windows*B, window_size^3, C)
        B_windows = 8
        window_volume = 4 * 4 * 4  # 64
        x = torch.randn(B_windows, window_volume, 96)
        
        output = attn(x)
        
        assert output.shape == (B_windows, window_volume, 96)


class TestSwinTransformerBlock3D:
    """Test cases for SwinTransformerBlock3D."""
    
    def test_init(self):
        """Test initialization."""
        block = SwinTransformerBlock3D(
            dim=96,
            num_heads=3,
            window_size=(7, 7, 7),
            shift_size=(3, 3, 3)
        )
        
        assert block.dim == 96
        assert block.num_heads == 3
        assert block.window_size == (7, 7, 7)
        assert block.shift_size == (3, 3, 3)
        
    def test_forward_pass(self):
        """Test forward pass."""
        block = SwinTransformerBlock3D(
            dim=96,
            num_heads=3,
            window_size=(4, 4, 4),
            shift_size=(0, 0, 0)  # No shift for simplicity
        )
        
        # Create test input (B, D, H, W, C)
        x = torch.randn(2, 8, 8, 8, 96)
        
        output = block(x)
        
        assert output.shape == (2, 8, 8, 8, 96)


class TestBasicLayer3D:
    """Test cases for BasicLayer3D."""
    
    def test_init_without_downsample(self):
        """Test initialization without downsampling."""
        layer = BasicLayer3D(
            dim=96,
            depth=2,
            num_heads=3,
            window_size=(7, 7, 7)
        )
        
        assert len(layer.blocks) == 2
        assert layer.downsample is None
        
    def test_init_with_downsample(self):
        """Test initialization with downsampling."""
        layer = BasicLayer3D(
            dim=96,
            depth=2,
            num_heads=3,
            window_size=(7, 7, 7),
            downsample=PatchMerging3D
        )
        
        assert len(layer.blocks) == 2
        assert layer.downsample is not None
        
    def test_forward_pass_without_downsample(self):
        """Test forward pass without downsampling."""
        layer = BasicLayer3D(
            dim=96,
            depth=2,
            num_heads=3,
            window_size=(4, 4, 4)
        )
        
        x = torch.randn(2, 8, 8, 8, 96)
        
        x_out, x_down = layer(x)
        
        assert x_out.shape == (2, 8, 8, 8, 96)
        assert x_down.shape == (2, 8, 8, 8, 96)  # Same as input when no downsample
        
    def test_forward_pass_with_downsample(self):
        """Test forward pass with downsampling."""
        layer = BasicLayer3D(
            dim=96,
            depth=2,
            num_heads=3,
            window_size=(4, 4, 4),
            downsample=PatchMerging3D
        )
        
        x = torch.randn(2, 8, 8, 8, 96)
        
        x_out, x_down = layer(x)
        
        assert x_out.shape == (2, 8, 8, 8, 96)
        assert x_down.shape == (2, 4, 4, 4, 192)  # Downsampled


class TestPatchMerging3D:
    """Test cases for PatchMerging3D."""
    
    def test_init(self):
        """Test initialization."""
        merge = PatchMerging3D(dim=96)
        
        assert merge.dim == 96
        
    def test_forward_pass_even_dimensions(self):
        """Test forward pass with even dimensions."""
        merge = PatchMerging3D(dim=96)
        
        # Even dimensions
        x = torch.randn(2, 8, 8, 8, 96)
        
        output = merge(x)
        
        assert output.shape == (2, 4, 4, 4, 192)  # Halved spatial, doubled channels
        
    def test_forward_pass_odd_dimensions(self):
        """Test forward pass with odd dimensions."""
        merge = PatchMerging3D(dim=96)
        
        # Odd dimensions
        x = torch.randn(2, 7, 7, 7, 96)
        
        output = merge(x)
        
        assert output.shape == (2, 4, 4, 4, 192)  # Padded then halved


class TestSwinUNETR:
    """Test cases for SwinUNETR model."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        model = SwinUNETR()
        
        assert model.embed_dim == 96
        assert model.num_layers == 4
        assert len(model.layers) == 4
        
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        model = SwinUNETR(
            img_size=(64, 64, 64),
            patch_size=(4, 4, 4),
            embed_dim=48,
            depths=[2, 2, 2, 2],
            num_heads=[2, 4, 8, 16]
        )
        
        assert model.embed_dim == 48
        assert model.num_layers == 4
        
    def test_forward_pass_small(self):
        """Test forward pass with small input."""
        model = SwinUNETR(
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            embed_dim=48,
            depths=[1, 1, 1, 1],  # Reduced depth for faster testing
            num_heads=[2, 4, 8, 16],
            window_size=(4, 4, 4)  # Smaller window size
        )
        
        # Create test input
        x = torch.randn(1, 1, 32, 32, 32)
        
        features = model(x)
        
        # Should return 4 feature maps (one per stage)
        assert len(features) == 4
        
        # Check feature map shapes
        # Grid size is (32/4, 32/4, 32/4) = (8, 8, 8)
        expected_shapes = [
            (1, 8, 8, 8, 48),      # Stage 0: embed_dim
            (1, 4, 4, 4, 96),      # Stage 1: embed_dim * 2
            (1, 2, 2, 2, 192),     # Stage 2: embed_dim * 4
            (1, 1, 1, 1, 384),     # Stage 3: embed_dim * 8
        ]
        
        for i, (feature, expected_shape) in enumerate(zip(features, expected_shapes)):
            assert feature.shape == expected_shape, f"Stage {i}: expected {expected_shape}, got {feature.shape}"
            
    def test_forward_pass_standard(self):
        """Test forward pass with standard configuration."""
        model = SwinUNETR(
            img_size=(96, 96, 96),
            patch_size=(4, 4, 4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(7, 7, 7)
        )
        
        # Create test input
        x = torch.randn(1, 1, 96, 96, 96)
        
        features = model(x)
        
        # Should return 4 feature maps
        assert len(features) == 4
        
        # Grid size is (96/4, 96/4, 96/4) = (24, 24, 24)
        expected_shapes = [
            (1, 24, 24, 24, 96),    # Stage 0
            (1, 12, 12, 12, 192),   # Stage 1
            (1, 6, 6, 6, 384),      # Stage 2
            (1, 3, 3, 3, 768),      # Stage 3
        ]
        
        for i, (feature, expected_shape) in enumerate(zip(features, expected_shapes)):
            assert feature.shape == expected_shape, f"Stage {i}: expected {expected_shape}, got {feature.shape}"


class TestHelperFunctions:
    """Test cases for helper functions."""
    
    def test_window_partition_and_reverse(self):
        """Test window partition and reverse operations."""
        # Create test tensor
        B, D, H, W, C = 2, 8, 8, 8, 96
        x = torch.randn(B, D, H, W, C)
        window_size = (4, 4, 4)
        
        # Partition into windows
        windows = window_partition(x, window_size)
        
        # Check window shape
        expected_num_windows = B * (D // window_size[0]) * (H // window_size[1]) * (W // window_size[2])
        assert windows.shape == (expected_num_windows, window_size[0], window_size[1], window_size[2], C)
        
        # Reverse partition
        x_restored = window_reverse(windows, window_size, D, H, W)
        
        # Should match original
        assert x_restored.shape == (B, D, H, W, C)
        assert torch.allclose(x, x_restored, atol=1e-6)
        
    def test_compute_mask(self):
        """Test attention mask computation."""
        D, H, W = 8, 8, 8
        window_size = (4, 4, 4)
        shift_size = (2, 2, 2)
        device = torch.device('cpu')
        
        mask = compute_mask(D, H, W, window_size, shift_size, device)
        
        # Check mask shape
        num_windows = (D // window_size[0]) * (H // window_size[1]) * (W // window_size[2])
        window_volume = window_size[0] * window_size[1] * window_size[2]
        
        assert mask.shape == (num_windows, window_volume, window_volume)


@pytest.fixture
def sample_3d_input():
    """Create a sample 3D input for testing."""
    return torch.randn(1, 1, 32, 32, 32)


def test_integration_swin_unetr(sample_3d_input):
    """Test integration of Swin-UNETR with sample data."""
    model = SwinUNETR(
        img_size=(32, 32, 32),
        patch_size=(4, 4, 4),
        embed_dim=48,
        depths=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 16],
        window_size=(4, 4, 4)
    )
    
    # Set to evaluation mode
    model.eval()
    
    with torch.no_grad():
        features = model(sample_3d_input)
        
    # Verify we get features from all stages
    assert len(features) == 4
    
    # Verify features are not all zeros
    for i, feature in enumerate(features):
        assert not torch.allclose(feature, torch.zeros_like(feature)), f"Stage {i} features are all zeros"


def test_model_parameter_count():
    """Test that model has reasonable parameter count."""
    model = SwinUNETR(
        img_size=(96, 96, 96),
        embed_dim=96,
        depths=[2, 2, 6, 2]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Should have reasonable number of parameters (not too small, not too large)
    assert 1_000_000 < total_params < 100_000_000, f"Parameter count {total_params} seems unreasonable"


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    model = SwinUNETR(
        img_size=(32, 32, 32),
        patch_size=(4, 4, 4),
        embed_dim=48,
        depths=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 16],
        window_size=(4, 4, 4)
    )
    
    x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
    
    features = model(x)
    
    # Create a dummy loss from the last feature
    loss = features[-1].sum()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"