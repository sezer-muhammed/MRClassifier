"""
Tests for Cross-Modal Attention Fusion Module
"""

import pytest
import torch
import torch.nn as nn
from gazimed.models.cross_attention import (
    CrossModalAttention,
    MultiModalFusionBlock,
    HierarchicalFusion,
    MLP,
    AttentionPooling
)


class TestCrossModalAttention:
    """Test cases for CrossModalAttention module."""
    
    def test_init(self):
        """Test initialization."""
        attn = CrossModalAttention(
            embed_dim=96,
            num_heads=2,
            attn_drop=0.1,
            proj_drop=0.1
        )
        
        assert attn.embed_dim == 96
        assert attn.num_heads == 2
        assert attn.head_dim == 48  # 96 // 2
        
    def test_init_invalid_heads(self):
        """Test initialization with invalid number of heads."""
        with pytest.raises(AssertionError):
            CrossModalAttention(embed_dim=97, num_heads=2)  # 97 not divisible by 2
            
    def test_forward_pass_same_length(self):
        """Test forward pass with same sequence lengths."""
        attn = CrossModalAttention(embed_dim=96, num_heads=2)
        
        batch_size = 2
        seq_len = 64
        mri_features = torch.randn(batch_size, seq_len, 96)
        pet_features = torch.randn(batch_size, seq_len, 96)
        
        fused_features, attn_weights = attn(mri_features, pet_features)
        
        # Check output shapes
        assert fused_features.shape == (batch_size, seq_len, 96)
        assert attn_weights.shape == (batch_size, 2, seq_len, seq_len)  # num_heads=2
        
    def test_forward_pass_different_lengths(self):
        """Test forward pass with different sequence lengths."""
        attn = CrossModalAttention(embed_dim=96, num_heads=2)
        
        batch_size = 2
        mri_len = 64
        pet_len = 32
        mri_features = torch.randn(batch_size, mri_len, 96)
        pet_features = torch.randn(batch_size, pet_len, 96)
        
        fused_features, attn_weights = attn(mri_features, pet_features)
        
        # Check output shapes
        assert fused_features.shape == (batch_size, mri_len, 96)
        assert attn_weights.shape == (batch_size, 2, mri_len, pet_len)
        
    def test_forward_pass_with_mask(self):
        """Test forward pass with attention mask."""
        attn = CrossModalAttention(embed_dim=96, num_heads=2)
        
        batch_size = 2
        mri_len = 8
        pet_len = 4
        mri_features = torch.randn(batch_size, mri_len, 96)
        pet_features = torch.randn(batch_size, pet_len, 96)
        
        # Create attention mask (mask out last 2 positions)
        attention_mask = torch.ones(batch_size, mri_len, pet_len)
        attention_mask[:, :, -2:] = 0
        
        fused_features, attn_weights = attn(mri_features, pet_features, attention_mask)
        
        # Check output shapes
        assert fused_features.shape == (batch_size, mri_len, 96)
        assert attn_weights.shape == (batch_size, 2, mri_len, pet_len)
        
        # Check that masked positions have zero attention (approximately)
        assert torch.allclose(attn_weights[:, :, :, -2:], torch.zeros_like(attn_weights[:, :, :, -2:]), atol=1e-6)


class TestMultiModalFusionBlock:
    """Test cases for MultiModalFusionBlock."""
    
    def test_init(self):
        """Test initialization."""
        block = MultiModalFusionBlock(
            embed_dim=96,
            num_heads=2,
            mlp_ratio=4.0,
            drop=0.1
        )
        
        assert block.embed_dim == 96
        assert block.num_heads == 2
        
    def test_forward_pass(self):
        """Test forward pass."""
        block = MultiModalFusionBlock(embed_dim=96, num_heads=2)
        
        batch_size = 2
        seq_len = 32
        mri_features = torch.randn(batch_size, seq_len, 96)
        pet_features = torch.randn(batch_size, seq_len, 96)
        
        fused_features, attn_weights = block(mri_features, pet_features)
        
        # Check output shapes
        assert fused_features.shape == (batch_size, seq_len, 96)
        assert attn_weights.shape == (batch_size, 2, seq_len, seq_len)


class TestHierarchicalFusion:
    """Test cases for HierarchicalFusion."""
    
    def test_init(self):
        """Test initialization."""
        fusion = HierarchicalFusion(
            feature_dims=[96, 192, 384, 768],
            target_dim=512,
            num_heads=2,
            num_layers=2
        )
        
        assert fusion.target_dim == 512
        assert fusion.num_layers == 2
        assert len(fusion.feature_projections) == 4
        assert len(fusion.fusion_blocks) == 2
        
    def test_forward_pass_3d_features(self):
        """Test forward pass with 3D features."""
        fusion = HierarchicalFusion(
            feature_dims=[96, 192, 384, 768],
            target_dim=256,
            num_heads=2,
            num_layers=1
        )
        
        batch_size = 2
        
        # Create features with different spatial dimensions (simulating multi-scale)
        mri_features = [
            torch.randn(batch_size, 8, 8, 8, 96),    # Stage 0
            torch.randn(batch_size, 4, 4, 4, 192),   # Stage 1
            torch.randn(batch_size, 2, 2, 2, 384),   # Stage 2
            torch.randn(batch_size, 1, 1, 1, 768),   # Stage 3
        ]
        
        pet_features = [
            torch.randn(batch_size, 8, 8, 8, 96),    # Stage 0
            torch.randn(batch_size, 4, 4, 4, 192),   # Stage 1
            torch.randn(batch_size, 2, 2, 2, 384),   # Stage 2
            torch.randn(batch_size, 1, 1, 1, 768),   # Stage 3
        ]
        
        fused_features, attn_weights_list = fusion(mri_features, pet_features)
        
        # Check output shapes
        assert fused_features.shape == (batch_size, 256)
        assert len(attn_weights_list) == 1  # num_layers=1
        
        # Total patches = 8*8*8 + 4*4*4 + 2*2*2 + 1*1*1 = 512 + 64 + 8 + 1 = 585
        expected_total_patches = 585
        assert attn_weights_list[0].shape == (batch_size, 2, expected_total_patches, expected_total_patches)
        
    def test_forward_pass_2d_features(self):
        """Test forward pass with already flattened 2D features."""
        fusion = HierarchicalFusion(
            feature_dims=[96, 192],
            target_dim=128,
            num_heads=1,
            num_layers=1
        )
        
        batch_size = 2
        
        # Create 2D features (already flattened)
        mri_features = [
            torch.randn(batch_size, 64, 96),   # 64 patches
            torch.randn(batch_size, 16, 192),  # 16 patches
        ]
        
        pet_features = [
            torch.randn(batch_size, 64, 96),   # 64 patches
            torch.randn(batch_size, 16, 192),  # 16 patches
        ]
        
        fused_features, attn_weights_list = fusion(mri_features, pet_features)
        
        # Check output shapes
        assert fused_features.shape == (batch_size, 128)
        assert len(attn_weights_list) == 1
        
        # Total patches = 64 + 16 = 80
        expected_total_patches = 80
        assert attn_weights_list[0].shape == (batch_size, 1, expected_total_patches, expected_total_patches)


class TestMLP:
    """Test cases for MLP module."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        mlp = MLP(in_features=256)
        
        assert mlp.fc1.in_features == 256
        assert mlp.fc1.out_features == 256  # hidden_features defaults to in_features
        assert mlp.fc2.in_features == 256
        assert mlp.fc2.out_features == 256  # out_features defaults to in_features
        
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        mlp = MLP(
            in_features=256,
            hidden_features=512,
            out_features=128,
            activation='relu'
        )
        
        assert mlp.fc1.in_features == 256
        assert mlp.fc1.out_features == 512
        assert mlp.fc2.in_features == 512
        assert mlp.fc2.out_features == 128
        assert isinstance(mlp.act, nn.ReLU)
        
    def test_forward_pass(self):
        """Test forward pass."""
        mlp = MLP(in_features=256, hidden_features=512, out_features=128)
        
        x = torch.randn(2, 64, 256)
        output = mlp(x)
        
        assert output.shape == (2, 64, 128)
        
    def test_invalid_activation(self):
        """Test initialization with invalid activation."""
        with pytest.raises(ValueError):
            MLP(in_features=256, activation='invalid')


class TestAttentionPooling:
    """Test cases for AttentionPooling module."""
    
    def test_init(self):
        """Test initialization."""
        pool = AttentionPooling(embed_dim=256, num_heads=4)
        
        assert pool.embed_dim == 256
        assert pool.num_heads == 4
        assert pool.head_dim == 64  # 256 // 4
        assert pool.query.shape == (1, 1, 256)
        
    def test_forward_pass(self):
        """Test forward pass."""
        pool = AttentionPooling(embed_dim=256, num_heads=4)
        
        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, 256)
        
        output = pool(x)
        
        assert output.shape == (batch_size, 256)
        
    def test_forward_pass_single_head(self):
        """Test forward pass with single head."""
        pool = AttentionPooling(embed_dim=128, num_heads=1)
        
        batch_size = 3
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 128)
        
        output = pool(x)
        
        assert output.shape == (batch_size, 128)


@pytest.fixture
def sample_multimodal_features():
    """Create sample multimodal features for testing."""
    batch_size = 2
    seq_len = 32
    embed_dim = 96
    
    mri_features = torch.randn(batch_size, seq_len, embed_dim)
    pet_features = torch.randn(batch_size, seq_len, embed_dim)
    
    return mri_features, pet_features


def test_integration_cross_modal_attention(sample_multimodal_features):
    """Test integration of cross-modal attention with sample data."""
    mri_features, pet_features = sample_multimodal_features
    
    # Create cross-modal attention module
    attn = CrossModalAttention(embed_dim=96, num_heads=2)
    
    # Set to evaluation mode
    attn.eval()
    
    with torch.no_grad():
        fused_features, attn_weights = attn(mri_features, pet_features)
        
    # Verify outputs
    assert fused_features.shape == mri_features.shape
    assert attn_weights.shape == (2, 2, 32, 32)  # (batch, heads, mri_len, pet_len)
    
    # Check that attention weights sum to 1 along last dimension
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-6)


def test_integration_fusion_block(sample_multimodal_features):
    """Test integration of fusion block with sample data."""
    mri_features, pet_features = sample_multimodal_features
    
    # Create fusion block
    fusion_block = MultiModalFusionBlock(embed_dim=96, num_heads=2)
    
    # Set to evaluation mode
    fusion_block.eval()
    
    with torch.no_grad():
        fused_features, attn_weights = fusion_block(mri_features, pet_features)
        
    # Verify outputs
    assert fused_features.shape == mri_features.shape
    assert attn_weights.shape == (2, 2, 32, 32)
    
    # Check that features are not all zeros (indicating proper processing)
    assert not torch.allclose(fused_features, torch.zeros_like(fused_features))


def test_gradient_flow_cross_attention():
    """Test that gradients flow properly through cross-attention."""
    attn = CrossModalAttention(embed_dim=96, num_heads=2)
    
    mri_features = torch.randn(2, 32, 96, requires_grad=True)
    pet_features = torch.randn(2, 32, 96, requires_grad=True)
    
    fused_features, _ = attn(mri_features, pet_features)
    
    # Create dummy loss
    loss = fused_features.sum()
    loss.backward()
    
    # Check that gradients exist
    assert mri_features.grad is not None
    assert pet_features.grad is not None
    assert not torch.allclose(mri_features.grad, torch.zeros_like(mri_features.grad))
    assert not torch.allclose(pet_features.grad, torch.zeros_like(pet_features.grad))
    
    # Check that model parameters have gradients
    for name, param in attn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_attention_mask_effectiveness():
    """Test that attention mask effectively blocks attention."""
    attn = CrossModalAttention(embed_dim=96, num_heads=2)
    
    batch_size = 1
    seq_len = 8
    mri_features = torch.randn(batch_size, seq_len, 96)
    pet_features = torch.randn(batch_size, seq_len, 96)
    
    # Create mask that blocks half the positions
    attention_mask = torch.ones(batch_size, seq_len, seq_len)
    attention_mask[:, :, seq_len//2:] = 0  # Block second half
    
    with torch.no_grad():
        _, attn_weights = attn(mri_features, pet_features, attention_mask)
        
    # Check that masked positions have zero attention
    assert torch.allclose(
        attn_weights[:, :, :, seq_len//2:], 
        torch.zeros_like(attn_weights[:, :, :, seq_len//2:]), 
        atol=1e-6
    )
    
    # Check that unmasked positions have non-zero attention
    assert not torch.allclose(
        attn_weights[:, :, :, :seq_len//2], 
        torch.zeros_like(attn_weights[:, :, :, :seq_len//2])
    )