"""
3D Patch Embedding Module for MRI Analysis

This module implements 3D patch embedding for converting MRI volume patches
into embeddings suitable for transformer processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding module that converts 3D MRI patches into embeddings.
    
    This module takes 3D MRI volumes and converts them into a sequence of patch embeddings
    that can be processed by a transformer architecture.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None
    ):
        """
        Initialize the 3D Patch Embedding module.
        
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Size of each patch (D, H, W)
            in_channels: Number of input channels (typically 1 for MRI)
            embed_dim: Embedding dimension
            norm_layer: Optional normalization layer
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches in each dimension
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1], 
            img_size[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        # 3D Convolution to create patch embeddings
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Optional normalization
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, D, H, W = x.shape
        
        # Validate input dimensions
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input size ({D}, {H}, {W}) doesn't match expected size {self.img_size}"
        assert C == self.in_channels, \
            f"Input channels ({C}) doesn't match expected channels ({self.in_channels})"
        
        # Apply 3D convolution to create patches
        # Output shape: (B, embed_dim, grid_d, grid_h, grid_w)
        x = self.proj(x)
        
        # Flatten spatial dimensions and transpose
        # Shape: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Apply normalization
        x = self.norm(x)
        
        return x


class PositionalEncoding3D(nn.Module):
    """
    3D Positional Encoding for patch embeddings.
    
    Adds learnable positional embeddings to patch embeddings to preserve
    spatial relationships in 3D space.
    """
    
    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int, int],
        dropout: float = 0.1
    ):
        """
        Initialize 3D positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            grid_size: Grid size (num_patches_d, num_patches_h, num_patches_w)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize positional embeddings
        self._init_weights()
        
    def _init_weights(self):
        """Initialize positional embedding weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to patch embeddings.
        
        Args:
            x: Patch embeddings of shape (B, num_patches, embed_dim)
            
        Returns:
            Position-encoded embeddings of shape (B, num_patches, embed_dim)
        """
        x = x + self.pos_embed
        return self.dropout(x)


class PatchEmbeddingWithPosition3D(nn.Module):
    """
    Combined 3D Patch Embedding with Positional Encoding.
    
    This module combines patch embedding and positional encoding into a single
    component for convenience.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        dropout: float = 0.1,
        norm_layer: Optional[nn.Module] = None
    ):
        """
        Initialize combined patch embedding with positional encoding.
        
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Size of each patch (D, H, W)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            dropout: Dropout rate for positional encoding
            norm_layer: Optional normalization layer
        """
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(
            embed_dim=embed_dim,
            grid_size=self.patch_embed.grid_size,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with patch embedding and positional encoding.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Position-encoded patch embeddings of shape (B, num_patches, embed_dim)
        """
        # Create patch embeddings
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        return x
    
    @property
    def num_patches(self) -> int:
        """Get number of patches."""
        return self.patch_embed.num_patches
    
    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Get grid size."""
        return self.patch_embed.grid_size