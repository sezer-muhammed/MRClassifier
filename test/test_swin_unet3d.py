"""
Test script for SwinUNet3D Backbone module
"""

import torch
import sys
import os

# Add the parent directory to the path to import gazimed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gazimed.models.swin_unet3d import SwinUNet3DBackbone


def test_swin_unet3d_backbone():
    """Test SwinUNet3D backbone with specified configuration"""
    
    print("Testing SwinUNet3D Backbone...")
    
    # Create model with specified configuration
    model = SwinUNet3DBackbone(
        in_channels=2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        dropout_path_rate=0.1,
        feature_size=96
    )
    
    print(f"Model created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with sample input
    batch_size = 2
    # Using smaller input size for testing (can be scaled up)
    H, W, D = 64, 64, 64
    
    # Create sample input (B, C, H, W, D)
    sample_input = torch.randn(batch_size, 2, H, W, D)
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        features = model(sample_input)
    
    print(f"Output shape: {features.shape}")
    print(f"Expected output shape: ({batch_size}, 96)")
    
    # Verify output shape
    assert features.shape == (batch_size, 96), f"Expected shape ({batch_size}, 96), got {features.shape}"
    
    # Verify output is not all zeros or NaN
    assert not torch.isnan(features).any(), "Output contains NaN values"
    assert not torch.all(features == 0), "Output is all zeros"
    
    print("✓ All tests passed!")
    print(f"Feature statistics - Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
    
    return True


def test_different_input_sizes():
    """Test with different input sizes"""
    print("\nTesting different input sizes...")
    
    model = SwinUNet3DBackbone(
        in_channels=2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        dropout_path_rate=0.1,
        feature_size=96
    )
    
    # Test different input sizes
    test_sizes = [
        (32, 32, 32),
        (48, 48, 48),
        (64, 64, 64)
    ]
    
    model.eval()
    for H, W, D in test_sizes:
        sample_input = torch.randn(1, 2, H, W, D)
        with torch.no_grad():
            features = model(sample_input)
        
        assert features.shape == (1, 96), f"Failed for size {(H, W, D)}: got shape {features.shape}"
        print(f"✓ Size {(H, W, D)} -> Output shape: {features.shape}")


def test_model_configuration():
    """Test that model has correct configuration"""
    print("\nTesting model configuration...")
    
    model = SwinUNet3DBackbone(
        in_channels=2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        dropout_path_rate=0.1,
        feature_size=96
    )
    
    # Verify configuration
    assert model.in_channels == 2, f"Expected in_channels=2, got {model.in_channels}"
    assert model.depths == [2, 2, 6, 2], f"Expected depths=[2,2,6,2], got {model.depths}"
    assert model.num_heads == [3, 6, 12, 24], f"Expected num_heads=[3,6,12,24], got {model.num_heads}"
    assert model.dropout_path_rate == 0.1, f"Expected dropout_path_rate=0.1, got {model.dropout_path_rate}"
    assert model.feature_size == 96, f"Expected feature_size=96, got {model.feature_size}"
    
    print("✓ Model configuration is correct")


if __name__ == "__main__":
    test_swin_unet3d_backbone()
    test_different_input_sizes()
    test_model_configuration()
    print("\n🎉 All tests completed successfully!")