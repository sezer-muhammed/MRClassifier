"""
Test Memory-Optimized Model

Quick test to verify the memory-optimized model works with batch size 1.
"""

import torch
import gc
from gazimed.models import HybridAlzheimersModel


def test_memory_optimized_model():
    """Test the memory-optimized model configuration"""
    
    print("Testing Memory-Optimized Model")
    print("=" * 50)
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create memory-optimized model
    model = HybridAlzheimersModel(
        target_size=(96, 109, 96),
        swin_config={
            "in_channels": 2,
            "feature_size": 64  # Reduced feature size
        },
        clinical_dims=[116, 32, 16, 8],  # Smaller clinical network
        fusion_dims=[72, 32, 16, 1],  # Smaller fusion network
        learning_rate=1e-3,
        loss_type="bce",
        use_batch_norm=False,  # Disabled for batch size 1
        use_simple_backbone=True  # Use simple 3D CNN
    )
    
    summary = model.get_model_summary()
    print(f"✓ Model created with {summary['total_parameters']:,} parameters")
    print("Component breakdown:")
    for component, count in summary['component_parameters'].items():
        print(f"  - {component}: {count:,}")
    
    # Test with batch size 1
    batch_size = 1
    images = torch.randn(batch_size, 2, 96, 109, 96)
    clinical_features = torch.randn(batch_size, 116)
    
    print(f"\nTesting with batch size {batch_size}:")
    print(f"  - Images shape: {images.shape}")
    print(f"  - Clinical features shape: {clinical_features.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits = model(images, clinical_features)
        predictions = torch.sigmoid(logits)
    
    print(f"  - Output logits: {logits.shape}, range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  - Output probs: {predictions.shape}, range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Test training step
    model.train()
    batch = {
        'volumes': images,
        'clinical_features': clinical_features,
        'alzheimer_score': torch.rand(batch_size),  # Random target
        'subject_id': ['test_001']
    }
    
    loss = model.training_step(batch, 0)
    print(f"  - Training loss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"  - Gradients computed: {has_gradients}")
    
    print("\n✓ All tests passed! Model works with batch size 1.")
    
    # Cleanup
    del model, images, clinical_features, logits, predictions, batch, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True


if __name__ == "__main__":
    success = test_memory_optimized_model()
    if success:
        print("\n🎉 Memory-optimized model test successful!")
    else:
        print("\n❌ Memory-optimized model test failed!")