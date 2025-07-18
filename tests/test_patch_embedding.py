"""
Tests for 3D Patch Embedding Module
"""

import pytest
import torch
import torch.nn as nn
from gazimed.models.patch_embedding import (
    PatchEmbedding3D,
    PositionalEncoding3D,
    PatchEmbeddingWithPosition3D
)


class TestPatchEmbedding3D:
    """Test cases for PatchEmbedding3D module."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        patch_embed = PatchEmbedding3D()
        
        assert patch_embed.img_size == (128, 128, 128)
        assert patch_embed.patch_size == (16, 16, 16)
        assert patch_embed.in_channels == 1
        assert patch_embed.embed_dim == 768
        assert patch_embed.grid_size == (8, 8, 8)
        assert patch_embed.num_patches == 512
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        patch_embed = PatchEmbedding3D(
            img_size=(64, 64, 64),
            patch_size=(8, 8, 8),
            in_channels=2,
            embed_dim=384
        )
        
        assert patch_embed.img_size == (64, 64, 64)
        assert patch_embed.patch_size == (8, 8, 8)
        assert patch_embed.in_channels == 2
        assert patch_embed.embed_dim == 384
        assert patch_embed.grid_size == (8, 8, 8)
        assert patch_embed.num_patches == 512
        
    def test_forward_pass(self):
        """Test forward pass with valid input."""
        patch_embed = PatchEmbedding3D(
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            embed_dim=256
        )
        
        # Create test input
        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 32, 32)
        
        # Forward pass
        output = patch_embed(x)
        
        # Check output shape
        expected_num_patches = 4 * 4 * 4  # (32/8)^3
        assert output.shape == (batch_size, expected_num_patches, 256)
        
    def test_forward_pass_invalid_size(self):
        """Test forward pass with invalid input size."""
        patch_embed = PatchEmbedding3D(
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8)
        )
        
        # Create test input with wrong size
        x = torch.randn(2, 1, 24, 24, 24)  # Wrong size
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            patch_embed(x)
            
    def test_forward_pass_invalid_channels(self):
        """Test forward pass with invalid number of channels."""
        patch_embed = PatchEmbedding3D(
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1
        )
        
        # Create test input with wrong channels
        x = torch.randn(2, 3, 32, 32, 32)  # Wrong channels
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            patch_embed(x)


class TestPositionalEncoding3D:
    """Test cases for PositionalEncoding3D module."""
    
    def test_init(self):
        """Test initialization."""
        pos_enc = PositionalEncoding3D(
            embed_dim=768,
            grid_size=(8, 8, 8),
            dropout=0.1
        )
        
        assert pos_enc.embed_dim == 768
        assert pos_enc.grid_size == (8, 8, 8)
        assert pos_enc.pos_embed.shape == (1, 512, 768)  # 8*8*8 = 512
        
    def test_forward_pass(self):
        """Test forward pass."""
        pos_enc = PositionalEncoding3D(
            embed_dim=256,
            grid_size=(4, 4, 4)
        )
        
        # Create test input
        batch_size = 2
        num_patches = 64  # 4*4*4
        x = torch.randn(batch_size, num_patches, 256)
        
        # Forward pass
        output = pos_enc(x)
        
        # Check output shape (should be same as input)
        assert output.shape == (batch_size, num_patches, 256)


class TestPatchEmbeddingWithPosition3D:
    """Test cases for combined PatchEmbeddingWithPosition3D module."""
    
    def test_init(self):
        """Test initialization."""
        combined = PatchEmbeddingWithPosition3D(
            img_size=(64, 64, 64),
            patch_size=(16, 16, 16),
            embed_dim=512
        )
        
        assert combined.num_patches == 64  # (64/16)^3 = 4^3 = 64
        assert combined.grid_size == (4, 4, 4)
        
    def test_forward_pass(self):
        """Test forward pass of combined module."""
        combined = PatchEmbeddingWithPosition3D(
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            embed_dim=256
        )
        
        # Create test input
        batch_size = 3
        x = torch.randn(batch_size, 1, 32, 32, 32)
        
        # Forward pass
        output = combined(x)
        
        # Check output shape
        expected_num_patches = 64  # (32/8)^3 = 4^3 = 64
        assert output.shape == (batch_size, expected_num_patches, 256)
        
    def test_properties(self):
        """Test module properties."""
        combined = PatchEmbeddingWithPosition3D(
            img_size=(48, 48, 48),
            patch_size=(12, 12, 12),
            embed_dim=384
        )
        
        assert combined.num_patches == 64  # (48/12)^3 = 4^3 = 64
        assert combined.grid_size == (4, 4, 4)


@pytest.fixture
def sample_mri_batch():
    """Create a sample MRI batch for testing."""
    return torch.randn(2, 1, 64, 64, 64)


def test_integration_with_sample_data(sample_mri_batch):
    """Test integration with sample MRI data."""
    # Create patch embedding module
    patch_embed = PatchEmbeddingWithPosition3D(
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        embed_dim=768
    )
    
    # Process sample data
    embeddings = patch_embed(sample_mri_batch)
    
    # Verify output
    batch_size = sample_mri_batch.shape[0]
    expected_patches = 64  # (64/16)^3
    assert embeddings.shape == (batch_size, expected_patches, 768)
    
    # Check that embeddings are not all zeros (indicating proper initialization)
    assert not torch.allclose(embeddings, torch.zeros_like(embeddings))