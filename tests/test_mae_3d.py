"""
Unit tests for MAE-3D (Masked Autoencoder for 3D volumes) implementation.

Tests the core functionality of 3D patch masking, MAE decoder,
and reconstruction loss computation.
"""

import pytest
import torch
import torch.nn as nn
from gazimed.models.mae_3d import (
    PatchMasking3D, MAEDecoder3D, MAE3D, MAELoss3D, TransformerBlock3D
)


class TestPatchMasking3D:
    """Test 3D patch masking functionality."""
    
    def test_patch_masking_initialization(self):
        """Test patch masking module initialization."""
        masking = PatchMasking3D(mask_ratio=0.75)
        assert masking.mask_ratio == 0.75
        
    def test_patch_masking_forward(self):
        """Test patch masking forward pass."""
        B, N, D = 2, 64, 96  # batch_size, num_patches, embed_dim
        masking = PatchMasking3D(mask_ratio=0.75)
        
        # Create dummy patch embeddings
        x = torch.randn(B, N, D)
        grid_size = (4, 4, 4)  # 4x4x4 = 64 patches
        
        # Apply masking
        x_masked, mask, ids_restore = masking(x, grid_size)
        
        # Check output shapes
        len_keep = int(N * (1 - 0.75))  # 25% kept
        assert x_masked.shape == (B, len_keep, D)
        assert mask.shape == (B, N)
        assert ids_restore.shape == (B, N)
        
        # Check mask values (0 = keep, 1 = remove)
        assert torch.all((mask == 0) | (mask == 1))
        assert torch.sum(mask, dim=1).float().mean().item() == pytest.approx(N * 0.75, abs=1)
        
    def test_different_mask_ratios(self):
        """Test different mask ratios."""
        B, N, D = 2, 64, 96
        grid_size = (4, 4, 4)
        x = torch.randn(B, N, D)
        
        for mask_ratio in [0.5, 0.75, 0.9]:
            masking = PatchMasking3D(mask_ratio=mask_ratio)
            x_masked, mask, ids_restore = masking(x, grid_size)
            
            len_keep = int(N * (1 - mask_ratio))
            assert x_masked.shape == (B, len_keep, D)
            
            # Check that correct number of patches are masked
            num_masked = torch.sum(mask, dim=1).float().mean().item()
            expected_masked = N * mask_ratio
            assert num_masked == pytest.approx(expected_masked, abs=1)


class TestTransformerBlock3D:
    """Test 3D transformer block for MAE decoder."""
    
    def test_transformer_block_initialization(self):
        """Test transformer block initialization."""
        block = TransformerBlock3D(
            dim=512,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True
        )
        
        assert block.norm1.normalized_shape == (512,)
        assert block.attn.embed_dim == 512
        assert block.attn.num_heads == 8
        
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        B, N, D = 2, 64, 512
        block = TransformerBlock3D(dim=D, num_heads=8)
        
        x = torch.randn(B, N, D)
        output = block(x)
        
        assert output.shape == (B, N, D)
        assert not torch.allclose(output, x)  # Should modify input
        
    def test_transformer_block_residual_connection(self):
        """Test that residual connections work properly."""
        B, N, D = 2, 64, 512
        block = TransformerBlock3D(dim=D, num_heads=8)
        
        # Create input
        x = torch.randn(B, N, D)
        
        # Forward pass
        output = block(x)
        
        # Output should be different from input due to transformations
        assert not torch.allclose(output, x)
        assert output.shape == x.shape


class TestMAEDecoder3D:
    """Test MAE 3D decoder functionality."""
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = MAEDecoder3D(
            embed_dim=96,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            patch_size=(4, 4, 4),
            in_channels=1
        )
        
        assert decoder.embed_dim == 96
        assert decoder.decoder_embed_dim == 512
        assert len(decoder.decoder_blocks) == 8
        assert decoder.mask_token.shape == (1, 1, 512)
        
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = MAEDecoder3D(
            embed_dim=96,
            decoder_embed_dim=512,
            decoder_depth=4,  # Smaller for testing
            decoder_num_heads=8,
            patch_size=(4, 4, 4),
            in_channels=1
        )
        
        B = 2
        len_keep = 16  # Number of unmasked patches
        embed_dim = 96
        grid_size = (4, 4, 4)  # 64 total patches
        num_patches = 64
        
        # Create dummy inputs
        x = torch.randn(B, len_keep, embed_dim)
        ids_restore = torch.randperm(num_patches).unsqueeze(0).repeat(B, 1)
        
        # Forward pass
        output = decoder(x, ids_restore, grid_size)
        
        # Check output shape
        patch_dim = 4 * 4 * 4 * 1  # patch_size * in_channels
        assert output.shape == (B, num_patches, patch_dim)
        
    def test_decoder_mask_token_expansion(self):
        """Test that mask tokens are properly expanded."""
        decoder = MAEDecoder3D(
            embed_dim=96,
            decoder_embed_dim=512,
            decoder_depth=2,
            decoder_num_heads=8,
            patch_size=(4, 4, 4),
            in_channels=1
        )
        
        B = 2
        len_keep = 16
        embed_dim = 96
        grid_size = (4, 4, 4)
        num_patches = 64
        
        x = torch.randn(B, len_keep, embed_dim)
        ids_restore = torch.arange(num_patches).unsqueeze(0).repeat(B, 1)
        
        output = decoder(x, ids_restore, grid_size)
        
        # Should reconstruct all patches
        patch_dim = 4 * 4 * 4 * 1
        assert output.shape == (B, num_patches, patch_dim)


class TestMAE3D:
    """Test complete MAE-3D model."""
    
    def test_mae3d_initialization(self):
        """Test MAE-3D model initialization."""
        model = MAE3D(
            img_size=(32, 32, 32),  # Small size for testing
            patch_size=(4, 4, 4),
            in_channels=1,
            embed_dim=96,
            encoder_depth=4,  # Smaller for testing
            decoder_depth=4,
            mask_ratio=0.75
        )
        
        assert model.mask_ratio == 0.75
        assert len(model.encoder_blocks) == 4
        assert model.patch_embed.num_patches == 8 * 8 * 8  # (32/4)^3
        
    def test_mae3d_forward(self):
        """Test MAE-3D forward pass."""
        model = MAE3D(
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_channels=1,
            embed_dim=96,
            encoder_depth=2,  # Small for testing
            decoder_depth=2,
            mask_ratio=0.75
        )
        
        B, C, D, H, W = 2, 1, 32, 32, 32
        imgs = torch.randn(B, C, D, H, W)
        
        # Forward pass
        output = model(imgs)
        
        # Check output dictionary
        assert 'loss' in output
        assert 'pred' in output
        assert 'mask' in output
        assert 'latent' in output
        assert 'ids_restore' in output
        
        # Check shapes
        num_patches = 8 * 8 * 8  # (32/4)^3
        patch_dim = 4 * 4 * 4 * 1  # patch_size * in_channels
        
        assert output['pred'].shape == (B, num_patches, patch_dim)
        assert output['mask'].shape == (B, num_patches)
        assert output['loss'].dim() == 0  # Scalar loss
        
    def test_mae3d_patchify_unpatchify(self):
        """Test patchify and unpatchify operations."""
        model = MAE3D(
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_channels=1
        )
        
        B, C, D, H, W = 2, 1, 32, 32, 32
        imgs = torch.randn(B, C, D, H, W)
        
        # Patchify
        patches = model.patchify(imgs)
        num_patches = 8 * 8 * 8
        patch_dim = 4 * 4 * 4 * 1
        assert patches.shape == (B, num_patches, patch_dim)
        
        # Unpatchify
        reconstructed = model.unpatchify(patches)
        assert reconstructed.shape == (B, C, D, H, W)
        
        # Should be identical (within floating point precision)
        assert torch.allclose(imgs, reconstructed, atol=1e-6)
        
    def test_mae3d_loss_computation(self):
        """Test loss computation."""
        model = MAE3D(
            img_size=(32, 32, 32),
            patch_size=(4, 4, 4),
            in_channels=1,
            encoder_depth=2,
            decoder_depth=2,
            mask_ratio=0.75
        )
        
        B, C, D, H, W = 2, 1, 32, 32, 32
        imgs = torch.randn(B, C, D, H, W)
        
        # Forward pass
        output = model(imgs)
        loss = output['loss']
        
        # Loss should be positive scalar
        assert loss.dim() == 0
        assert loss.item() >= 0
        
        # Loss should be differentiable
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMAELoss3D:
    """Test MAE-3D loss module."""
    
    def test_mae_loss_initialization(self):
        """Test MAE loss initialization."""
        loss_fn = MAELoss3D(norm_pix_loss=False)
        assert not loss_fn.norm_pix_loss
        
        loss_fn_norm = MAELoss3D(norm_pix_loss=True)
        assert loss_fn_norm.norm_pix_loss
        
    def test_mae_loss_forward(self):
        """Test MAE loss forward pass."""
        loss_fn = MAELoss3D(norm_pix_loss=False)
        
        B, C, D, H, W = 2, 1, 32, 32, 32
        patch_size = (4, 4, 4)
        num_patches = 8 * 8 * 8
        patch_dim = 4 * 4 * 4 * 1
        
        imgs = torch.randn(B, C, D, H, W)
        pred = torch.randn(B, num_patches, patch_dim)
        mask = torch.randint(0, 2, (B, num_patches)).float()
        
        loss = loss_fn(imgs, pred, mask, patch_size, in_channels=1)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        
    def test_mae_loss_normalized(self):
        """Test MAE loss with pixel normalization."""
        loss_fn = MAELoss3D(norm_pix_loss=True)
        
        B, C, D, H, W = 2, 1, 32, 32, 32
        patch_size = (4, 4, 4)
        num_patches = 8 * 8 * 8
        patch_dim = 4 * 4 * 4 * 1
        
        imgs = torch.randn(B, C, D, H, W)
        pred = torch.randn(B, num_patches, patch_dim)
        mask = torch.ones(B, num_patches)  # All patches masked
        
        loss = loss_fn(imgs, pred, mask, patch_size, in_channels=1)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        
    def test_mae_loss_mask_effect(self):
        """Test that mask properly affects loss computation."""
        loss_fn = MAELoss3D(norm_pix_loss=False)
        
        B, C, D, H, W = 2, 1, 32, 32, 32
        patch_size = (4, 4, 4)
        num_patches = 8 * 8 * 8
        patch_dim = 4 * 4 * 4 * 1
        
        imgs = torch.randn(B, C, D, H, W)
        pred = torch.randn(B, num_patches, patch_dim)
        
        # All patches masked
        mask_all = torch.ones(B, num_patches)
        loss_all = loss_fn(imgs, pred, mask_all, patch_size, in_channels=1)
        
        # No patches masked
        mask_none = torch.zeros(B, num_patches)
        loss_none = loss_fn(imgs, pred, mask_none, patch_size, in_channels=1)
        
        # Loss should be 0 when no patches are masked
        assert loss_none.item() == 0.0
        assert loss_all.item() > 0.0


@pytest.fixture
def sample_mae_model():
    """Fixture providing a sample MAE-3D model for testing."""
    return MAE3D(
        img_size=(32, 32, 32),
        patch_size=(4, 4, 4),
        in_channels=1,
        embed_dim=96,
        encoder_depth=2,
        decoder_depth=2,
        mask_ratio=0.75
    )


@pytest.fixture
def sample_input():
    """Fixture providing sample input data."""
    return torch.randn(2, 1, 32, 32, 32)


def test_mae3d_integration(sample_mae_model, sample_input):
    """Integration test for complete MAE-3D pipeline."""
    model = sample_mae_model
    imgs = sample_input
    
    # Test training mode
    model.train()
    output_train = model(imgs)
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        output_eval = model(imgs)
    
    # Both should produce valid outputs
    for output in [output_train, output_eval]:
        assert 'loss' in output
        assert 'pred' in output
        assert 'mask' in output
        assert output['loss'].item() >= 0
        
    # Training and eval outputs should be different due to random masking
    assert not torch.allclose(output_train['mask'], output_eval['mask'])


def test_mae3d_gradient_flow(sample_mae_model, sample_input):
    """Test gradient flow through MAE-3D model."""
    model = sample_mae_model
    imgs = sample_input
    
    # Forward pass
    output = model(imgs)
    loss = output['loss']
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Zero gradient for parameter: {name}"


if __name__ == "__main__":
    pytest.main([__file__])