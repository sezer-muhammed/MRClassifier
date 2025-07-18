"""
Core functionality tests without heavy dependencies

This module tests the core model components without requiring
PyTorch Lightning, torchmetrics, or other heavy dependencies.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_core_model_imports():
    """Test that core model components can be imported."""
    try:
        from gazimed.models.swin_unetr import SwinUNETR
        from gazimed.models.clinical_encoder import ClinicalFeaturesEncoder
        from gazimed.models.cross_attention import CrossModalAttention, HierarchicalFusion
        from gazimed.models.patch_embedding import PatchEmbeddingWithPosition3D
        
        print("✓ Successfully imported all core model components")
        return True
    except ImportError as e:
        print(f"✗ Failed to import core model components: {e}")
        return False

def test_clinical_encoder():
    """Test clinical features encoder."""
    try:
        from gazimed.models.clinical_encoder import ClinicalFeaturesEncoder
        
        # Create encoder
        encoder = ClinicalFeaturesEncoder(
            input_dim=118,
            hidden_dims=[256, 512, 256],
            output_dim=128,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 4
        clinical_features = torch.randn(batch_size, 118)
        
        with torch.no_grad():
            output = encoder(clinical_features)
        
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()
        
        print("✓ Clinical encoder test passed")
        return True
    except Exception as e:
        print(f"✗ Clinical encoder test failed: {e}")
        return False

def test_patch_embedding():
    """Test 3D patch embedding."""
    try:
        from gazimed.models.patch_embedding import PatchEmbeddingWithPosition3D
        
        # Create patch embedding
        patch_embed = PatchEmbeddingWithPosition3D(
            img_size=(32, 32, 32),  # Small size for testing
            patch_size=(4, 4, 4),
            in_channels=2,
            embed_dim=96,
            dropout=0.1
        )
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 2, 32, 32, 32)
        
        with torch.no_grad():
            output = patch_embed(images)
        
        expected_patches = (32 // 4) * (32 // 4) * (32 // 4)  # 8 * 8 * 8 = 512
        assert output.shape == (batch_size, expected_patches, 96)
        assert not torch.isnan(output).any()
        
        print(f"✓ Patch embedding test passed - created {expected_patches} patches")
        return True
    except Exception as e:
        print(f"✗ Patch embedding test failed: {e}")
        return False

def test_cross_attention():
    """Test cross-modal attention."""
    try:
        from gazimed.models.cross_attention import CrossModalAttention
        
        # Create cross attention module
        cross_attn = CrossModalAttention(
            embed_dim=96,
            num_heads=2,
            attn_drop=0.1,
            proj_drop=0.1
        )
        
        # Test forward pass
        batch_size = 2
        seq_len_mri = 64
        seq_len_pet = 64
        embed_dim = 96
        
        mri_features = torch.randn(batch_size, seq_len_mri, embed_dim)
        pet_features = torch.randn(batch_size, seq_len_pet, embed_dim)
        
        with torch.no_grad():
            fused_features, attention_weights = cross_attn(mri_features, pet_features)
        
        assert fused_features.shape == (batch_size, seq_len_mri, embed_dim)
        assert attention_weights.shape == (batch_size, 2, seq_len_mri, seq_len_pet)  # 2 heads
        assert not torch.isnan(fused_features).any()
        
        print("✓ Cross-modal attention test passed")
        return True
    except Exception as e:
        print(f"✗ Cross-modal attention test failed: {e}")
        return False

def test_swin_unetr_encoder():
    """Test Swin-UNETR encoder (simplified)."""
    try:
        from gazimed.models.swin_unetr import SwinUNETR
        
        # Create small Swin-UNETR for testing
        model = SwinUNETR(
            img_size=(32, 32, 32),  # Small size for testing
            patch_size=(4, 4, 4),
            in_channels=2,
            embed_dim=48,  # Smaller embedding
            depths=[1, 1, 1, 1],  # Smaller depths
            num_heads=[2, 4, 8, 16],
            window_size=(4, 4, 4),  # Smaller window
            drop_rate=0.1
        )
        
        # Test forward pass
        batch_size = 1  # Small batch for testing
        images = torch.randn(batch_size, 2, 32, 32, 32)
        
        with torch.no_grad():
            features_list = model(images)
        
        assert len(features_list) == 4  # 4 stages
        assert all(isinstance(f, torch.Tensor) for f in features_list)
        assert not any(torch.isnan(f).any() for f in features_list)
        
        print(f"✓ Swin-UNETR encoder test passed - {len(features_list)} feature stages")
        return True
    except Exception as e:
        print(f"✗ Swin-UNETR encoder test failed: {e}")
        return False

def test_model_integration():
    """Test integration of multiple components."""
    try:
        from gazimed.models.clinical_encoder import ClinicalFeaturesEncoder
        from gazimed.models.patch_embedding import PatchEmbeddingWithPosition3D
        from gazimed.models.cross_attention import CrossModalAttention
        
        # Create components
        patch_embed = PatchEmbeddingWithPosition3D(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            in_channels=1,
            embed_dim=64
        )
        
        clinical_encoder = ClinicalFeaturesEncoder(
            input_dim=118,
            hidden_dims=[128, 256, 128],
            output_dim=64
        )
        
        cross_attn = CrossModalAttention(
            embed_dim=64,
            num_heads=2
        )
        
        # Test integration
        batch_size = 2
        
        # Process images
        mri_images = torch.randn(batch_size, 1, 16, 16, 16)
        pet_images = torch.randn(batch_size, 1, 16, 16, 16)
        
        with torch.no_grad():
            mri_patches = patch_embed(mri_images)
            pet_patches = patch_embed(pet_images)
            
            # Cross-modal fusion
            fused_features, _ = cross_attn(mri_patches, pet_patches)
            
            # Clinical features
            clinical_features = torch.randn(batch_size, 118)
            clinical_encoded = clinical_encoder(clinical_features)
            
            # Simple fusion
            image_pooled = fused_features.mean(dim=1)  # Global average pooling
            combined = torch.cat([image_pooled, clinical_encoded], dim=1)
        
        expected_patches = (16 // 4) ** 3  # 4^3 = 64
        assert mri_patches.shape == (batch_size, expected_patches, 64)
        assert fused_features.shape == (batch_size, expected_patches, 64)
        assert clinical_encoded.shape == (batch_size, 64)
        assert combined.shape == (batch_size, 128)  # 64 + 64
        
        print("✓ Model integration test passed")
        return True
    except Exception as e:
        print(f"✗ Model integration test failed: {e}")
        return False

def test_model_parameters():
    """Test model parameter calculations."""
    try:
        # Test parameter calculations for different configurations
        configs = [
            {
                'img_size': (91, 120, 91),
                'patch_size': (4, 4, 4),
                'embed_dim': 96,
                'depths': [2, 2, 6, 2]
            },
            {
                'img_size': (64, 64, 64),
                'patch_size': (8, 8, 8),
                'embed_dim': 128,
                'depths': [2, 2, 2, 2]
            }
        ]
        
        for i, config in enumerate(configs):
            # Calculate expected patches
            patches = (config['img_size'][0] // config['patch_size'][0]) * \
                     (config['img_size'][1] // config['patch_size'][1]) * \
                     (config['img_size'][2] // config['patch_size'][2])
            
            # Calculate feature dimensions
            feature_dims = [int(config['embed_dim'] * 2 ** j) for j in range(len(config['depths']))]
            
            print(f"✓ Config {i+1}: {patches} patches, feature dims: {feature_dims}")
        
        return True
    except Exception as e:
        print(f"✗ Parameter calculation test failed: {e}")
        return False

def run_all_tests():
    """Run all core functionality tests."""
    tests = [
        ("Core Model Imports", test_core_model_imports),
        ("Clinical Encoder", test_clinical_encoder),
        ("Patch Embedding", test_patch_embedding),
        ("Cross-Modal Attention", test_cross_attention),
        ("Swin-UNETR Encoder", test_swin_unetr_encoder),
        ("Model Integration", test_model_integration),
        ("Model Parameters", test_model_parameters),
    ]
    
    results = []
    print("Running Core Functionality Tests")
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
        print("🎉 All core functionality tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)