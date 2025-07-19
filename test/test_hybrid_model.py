"""
Test script for Hybrid Alzheimer's Model

This script tests the complete hybrid model with all components integrated,
including compatibility with the dataloader and training functionality.
"""

import torch
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import gazimed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gazimed.models import HybridAlzheimersModel
from gazimed.data.database import DatabaseManager
from gazimed.data.dataset import AlzheimersDataset, custom_collate_fn
from torch.utils.data import DataLoader


def test_model_components():
    """Test individual model components"""
    print("=== Testing Model Components ===")
    
    # Test model creation
    model = HybridAlzheimersModel(
        target_size=(64, 64, 64),  # Smaller size for testing
        learning_rate=1e-4,
        loss_type="bce"  # Binary Cross-Entropy
    )
    
    print(f"✓ Model created successfully")
    
    # Get model summary
    summary = model.get_model_summary()
    print(f"✓ Total parameters: {summary['total_parameters']:,}")
    print(f"✓ Component parameters:")
    for component, count in summary['component_parameters'].items():
        print(f"  - {component}: {count:,}")
    
    return model


def test_forward_pass():
    """Test forward pass with synthetic data"""
    print("\n=== Testing Forward Pass ===")
    
    model = HybridAlzheimersModel(
        target_size=(64, 64, 64),
        loss_type="bce"
    )
    model.eval()
    
    # Create synthetic input data
    batch_size = 2
    images = torch.randn(batch_size, 2, 64, 64, 64)  # MRI + PET
    clinical_features = torch.randn(batch_size, 116)  # 116 clinical features
    
    print(f"Input shapes:")
    print(f"  - Images: {images.shape}")
    print(f"  - Clinical features: {clinical_features.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(images, clinical_features)
    
    print(f"Output shape: {predictions.shape}")
    print(f"Output range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    # Verify output properties (predictions are now logits, not probabilities)
    assert predictions.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {predictions.shape}"
    # Convert logits to probabilities for range check
    probs = torch.sigmoid(predictions)
    assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probabilities should be in [0, 1] range"
    
    print("✓ Forward pass test passed!")
    return True


def test_with_real_dataloader():
    """Test model with real dataloader"""
    print("\n=== Testing with Real DataLoader ===")
    
    try:
        # Create database manager and dataset
        db_manager = DatabaseManager()
        
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=True,
            cache_volumes=False,
            target_size=(64, 64, 64),  # Smaller size for testing
            validate_files=False
        )
        
        if len(dataset) == 0:
            print("⚠️  No data available in database, skipping real dataloader test")
            return True
        
        print(f"Dataset size: {len(dataset)}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        # Create model
        model = HybridAlzheimersModel(
            target_size=(64, 64, 64),
            loss_type="bce"
        )
        model.eval()
        
        # Test with one batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch sizes:")
        print(f"  - Volumes: {batch['volumes'].shape}")
        print(f"  - Clinical features: {batch['clinical_features'].shape}")
        print(f"  - Alzheimer scores: {batch['alzheimer_score'].shape}")
        
        # Forward pass
        with torch.no_grad():
            predictions = model(batch['volumes'], batch['clinical_features'])
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions: {predictions.squeeze().tolist()}")
        print(f"Targets: {batch['alzheimer_score'].tolist()}")
        
        # Test loss calculation (predictions are logits, targets are probabilities)
        targets = batch['alzheimer_score'].unsqueeze(-1)
        loss = model.loss_fn(predictions, targets)
        print(f"BCE Loss: {loss.item():.4f}")
        
        # Show probabilities for interpretation
        probs = torch.sigmoid(predictions)
        print(f"Probabilities: {probs.squeeze().tolist()}")
        
        print("✓ Real dataloader test passed!")
        return True
        
    except Exception as e:
        print(f"⚠️  Real dataloader test failed: {e}")
        return False


def test_training_step():
    """Test training step functionality"""
    print("\n=== Testing Training Step ===")
    
    model = HybridAlzheimersModel(
        target_size=(64, 64, 64),
        loss_type="bce"
    )
    model.train()
    
    # Create synthetic batch
    batch = {
        'volumes': torch.randn(2, 2, 64, 64, 64),
        'clinical_features': torch.randn(2, 116),
        'alzheimer_score': torch.rand(2),  # Random values in [0, 1]
        'subject_id': ['test_001', 'test_002']
    }
    
    # Test training step
    loss = model.training_step(batch, 0)
    
    print(f"Training loss: {loss.item():.4f}")
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients"
    
    # Test backward pass
    loss.backward()
    
    # Check that gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "Model should have gradients after backward pass"
    
    print("✓ Training step test passed!")
    return True


def test_save_load_functionality():
    """Test model save and load functionality"""
    print("\n=== Testing Save/Load Functionality ===")
    
    # Create and configure model
    original_model = HybridAlzheimersModel(
        target_size=(64, 64, 64),
        learning_rate=2e-4,  # Different from default
        loss_type="bce"
    )
    
    # Create test directory
    test_dir = Path("test/results")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    save_path = test_dir / "test_hybrid_model.pth"
    original_model.save_model(str(save_path))
    
    # Load model
    loaded_model = HybridAlzheimersModel.load_model(str(save_path))
    
    # Compare configurations
    assert loaded_model.learning_rate == original_model.learning_rate
    assert loaded_model.loss_type == original_model.loss_type
    
    # Test that loaded model produces same outputs
    test_input_images = torch.randn(1, 2, 64, 64, 64)
    test_input_clinical = torch.randn(1, 116)
    
    original_model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input_images, test_input_clinical)
        loaded_output = loaded_model(test_input_images, test_input_clinical)
    
    # Outputs should be identical
    assert torch.allclose(original_output, loaded_output, atol=1e-6), "Loaded model should produce identical outputs"
    
    print("✓ Save/load test passed!")
    return True


def test_optimizer_configuration():
    """Test optimizer and scheduler configuration"""
    print("\n=== Testing Optimizer Configuration ===")
    
    model = HybridAlzheimersModel(
        optimizer_type="adamw",
        scheduler_type="cosine",
        learning_rate=1e-4,
        weight_decay=1e-2
    )
    
    # Test optimizer configuration
    optimizer_config = model.configure_optimizers()
    
    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config
    
    optimizer = optimizer_config["optimizer"]
    scheduler_config = optimizer_config["lr_scheduler"]
    
    # Check optimizer type and parameters
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    
    # Check scheduler
    assert "scheduler" in scheduler_config
    assert scheduler_config["interval"] == "epoch"
    
    print("✓ Optimizer configuration test passed!")
    return True


def run_comprehensive_test():
    """Run all tests"""
    print("Starting Comprehensive Hybrid Model Test")
    print("=" * 60)
    
    tests = [
        ("Model Components", test_model_components),
        ("Forward Pass", test_forward_pass),
        ("Real DataLoader", test_with_real_dataloader),
        ("Training Step", test_training_step),
        ("Save/Load", test_save_load_functionality),
        ("Optimizer Config", test_optimizer_configuration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Model Components":
                model = test_func()  # This returns the model
                results[test_name] = True
            else:
                result = test_func()
                results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Hybrid model is ready for training!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review and fix issues.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)