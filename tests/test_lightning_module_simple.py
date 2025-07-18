"""
Simple tests for the Alzheimer's Detection Lightning Module

This module contains basic unit tests for the PyTorch Lightning module
without heavy dependencies.
"""

import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_lightning_module_imports():
    """Test that the Lightning module can be imported."""
    try:
        from gazimed.models.lightning_module import AlzheimersLightningModule
        print("✓ Successfully imported AlzheimersLightningModule")
        return True
    except ImportError as e:
        print(f"✗ Failed to import AlzheimersLightningModule: {e}")
        return False

def test_model_components():
    """Test that model components can be created."""
    try:
        from gazimed.models.swin_unetr import SwinUNETR
        from gazimed.models.clinical_encoder import ClinicalFeaturesEncoder
        from gazimed.models.cross_attention import HierarchicalFusion
        
        print("✓ Successfully imported model components")
        return True
    except ImportError as e:
        print(f"✗ Failed to import model components: {e}")
        return False

def test_basic_model_creation():
    """Test basic model creation without PyTorch Lightning."""
    try:
        # Test individual components
        from gazimed.models.clinical_encoder import ClinicalFeaturesEncoder
        
        # Create clinical encoder
        clinical_encoder = ClinicalFeaturesEncoder(
            input_dim=118,
            hidden_dims=[64, 128, 64],
            output_dim=32,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 2
        clinical_features = torch.randn(batch_size, 118)
        
        with torch.no_grad():
            output = clinical_encoder(clinical_features)
        
        assert output.shape == (batch_size, 32)
        print("✓ Clinical encoder works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Basic model creation failed: {e}")
        return False

def test_training_config():
    """Test training configuration."""
    try:
        from gazimed.training.trainer_config import TrainingConfig, DEFAULT_TRAINING_CONFIG
        
        # Test default config
        config = DEFAULT_TRAINING_CONFIG
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-2
        assert config.batch_size == 4
        assert config.accumulate_grad_batches == 8
        
        # Test config conversion
        trainer_kwargs = config.to_trainer_kwargs()
        assert 'max_epochs' in trainer_kwargs
        assert 'precision' in trainer_kwargs
        assert 'gradient_clip_val' in trainer_kwargs
        
        print("✓ Training configuration works correctly")
        return True
    except Exception as e:
        print(f"✗ Training configuration test failed: {e}")
        return False

def test_model_architecture_parameters():
    """Test that model architecture parameters are reasonable."""
    try:
        # Test parameter calculations
        img_size = (91, 120, 91)
        patch_size = (4, 4, 4)
        
        # Calculate expected number of patches
        expected_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        # Should be reasonable number of patches
        assert expected_patches > 0
        assert expected_patches < 100000  # Not too many patches
        
        print(f"✓ Model will create {expected_patches} patches from input size {img_size}")
        
        # Test embedding dimensions
        embed_dim = 96
        depths = [2, 2, 6, 2]
        
        # Calculate feature dimensions for each stage
        feature_dims = [int(embed_dim * 2 ** i) for i in range(len(depths))]
        print(f"✓ Feature dimensions across stages: {feature_dims}")
        
        return True
    except Exception as e:
        print(f"✗ Architecture parameter test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Import Lightning Module", test_lightning_module_imports),
        ("Import Model Components", test_model_components),
        ("Basic Model Creation", test_basic_model_creation),
        ("Training Configuration", test_training_config),
        ("Architecture Parameters", test_model_architecture_parameters),
    ]
    
    results = []
    print("Running Lightning Module Tests")
    print("=" * 50)
    
    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)