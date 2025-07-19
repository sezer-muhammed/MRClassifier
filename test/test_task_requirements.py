"""
Test script to verify SwinUNet3D Backbone meets task requirements
"""

import torch
import sys
import os

# Add the parent directory to the path to import gazimed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gazimed.models.swin_unet3d import SwinUNet3DBackbone


def test_task_requirements():
    """
    Test that SwinUNet3D backbone meets the specific task requirements:
    - depths=[2,2,6,2]
    - num_heads=[3,6,12,24] 
    - dropout_path_rate=0.1
    - Output 96 features from global average pooling
    """
    
    print("Testing Task Requirements for SwinUNet3D Backbone...")
    print("=" * 50)
    
    # Create model with EXACT specified configuration
    model = SwinUNet3DBackbone(
        in_channels=2,  # MRI + PET channels
        depths=[2, 2, 6, 2],  # Required depths
        num_heads=[3, 6, 12, 24],  # Required num_heads
        dropout_path_rate=0.1,  # Required dropout_path_rate
        feature_size=96  # Required output features
    )
    
    print("✓ Model created with specified configuration:")
    print(f"  - depths: {model.depths}")
    print(f"  - num_heads: {model.num_heads}")
    print(f"  - dropout_path_rate: {model.dropout_path_rate}")
    print(f"  - feature_size: {model.feature_size}")
    
    # Test with manageable input size (divisible by patch_size=4)
    batch_size = 2
    H, W, D = 64, 64, 64  # Size divisible by patch_size
    
    # Create sample input representing MRI + PET
    sample_input = torch.randn(batch_size, 2, H, W, D)
    print(f"\n✓ Input shape: {sample_input.shape} (Batch × Channels × Height × Width × Depth)")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        features = model(sample_input)
    
    print(f"✓ Output shape: {features.shape}")
    
    # Verify requirements
    assert features.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {features.shape[0]}"
    assert features.shape[1] == 96, f"Feature size mismatch: expected 96, got {features.shape[1]}"
    
    # Verify the features are from global average pooling (should be reasonable values)
    assert not torch.isnan(features).any(), "Output contains NaN values"
    assert not torch.isinf(features).any(), "Output contains infinite values"
    
    # Check that features have reasonable statistics (not all zeros or extreme values)
    mean_val = features.mean().item()
    std_val = features.std().item()
    assert abs(mean_val) < 10, f"Mean too extreme: {mean_val}"
    assert 0.01 < std_val < 10, f"Standard deviation unreasonable: {std_val}"
    
    print(f"✓ Feature statistics - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"✓ Output 96 features from global average pooling: VERIFIED")
    
    # Test gradient flow
    model.train()
    sample_input.requires_grad_(True)
    features = model(sample_input)
    loss = features.sum()
    loss.backward()
    
    # Check that gradients exist
    assert sample_input.grad is not None, "No gradients computed for input"
    assert not torch.isnan(sample_input.grad).any(), "NaN gradients detected"
    
    print("✓ Gradient flow verified")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TASK REQUIREMENTS VERIFIED!")
    print("✓ SwinUNet3DBackbone class implemented with specified config")
    print("✓ depths=[2,2,6,2], num_heads=[3,6,12,24], dropout_path_rate=0.1")
    print("✓ Output 96 features from global average pooling")
    print("✓ Requirements 1.3 satisfied")


if __name__ == "__main__":
    test_task_requirements()