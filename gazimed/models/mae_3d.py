"""
Masked Autoencoder for 3D Volumes (MAE-3D)

This module implements a 3D Masked Autoencoder for self-supervised pretraining
on unlabeled T1-weighted brain MRI volumes. The MAE-3D uses patch masking
with 75% mask ratio and reconstruction loss for pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
from .patch_embedding import PatchEmbeddingWithPosition3D
from .swin_unetr import SwinUNETR


class PatchMasking3D(nn.Module):
    """
    3D Patch Masking module for MAE pretraining.
    
    Randomly masks patches from 3D volumes with configurable mask ratio.
    """
    
    def __init__(self, mask_ratio: float = 0.75):
        """
        Initialize patch masking.
        
        Args:
            mask_ratio: Ratio of patches to mask (default: 0.75)
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        
    def forward(self, x: torch.Tensor, 
                grid_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply random masking to patch embeddings.
        
        Args:
            x: Patch embeddings of shape (B, num_patches, embed_dim)
            grid_size: Grid size (D, H, W) for patches
            
        Returns:
            Tuple of (masked_embeddings, mask, ids_restore)
        """
        B, N, D = x.shape  # batch, num_patches, embed_dim
        
        # Calculate number of patches to keep
        len_keep = int(N * (1 - self.mask_ratio))
        
        # Generate random noise for each patch
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get random permutation
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only the first subset (unmasked patches)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore


class MAEDecoder3D(nn.Module):
    """
    3D MAE Decoder for reconstructing masked patches.
    
    Lightweight decoder that reconstructs the original patches from
    the encoded representations and mask tokens.
    """
    
    def __init__(
        self,
        embed_dim: int = 96,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1
    ):
        """
        Initialize MAE decoder.
        
        Args:
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            patch_size: Size of each patch
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Decoder embedding projection
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding (learned)
        # Will be initialized based on input size
        self.decoder_pos_embed = None
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock3D(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0
            )
            for _ in range(decoder_depth)
        ])
        
        # Decoder normalization
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Decoder prediction head
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize decoder weights."""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights for linear layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _init_decoder_pos_embed(self, grid_size: Tuple[int, int, int]):
        """Initialize decoder positional embedding based on grid size."""
        if self.decoder_pos_embed is None:
            num_patches = grid_size[0] * grid_size[1] * grid_size[2]
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.decoder_embed_dim),
                requires_grad=True
            )
            torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
            
    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor,
                grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Forward pass of MAE decoder.
        
        Args:
            x: Encoded patch embeddings (B, len_keep, embed_dim)
            ids_restore: Indices to restore original order
            grid_size: Grid size for positional embedding
            
        Returns:
            Reconstructed patches (B, num_patches, patch_dim)
        """
        # Initialize positional embedding if needed
        self._init_decoder_pos_embed(grid_size)
        
        # Embed tokens to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        B, len_keep, D = x.shape
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        
        mask_tokens = self.mask_token.repeat(B, num_patches - len_keep, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, num_patches, decoder_embed_dim)
        
        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
            
        x_full = self.decoder_norm(x_full)
        
        # Predictor projection
        x_full = self.decoder_pred(x_full)
        
        return x_full


class TransformerBlock3D(nn.Module):
    """
    Simple transformer block for MAE decoder.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias to qkv projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class MAE3D(nn.Module):
    """
    Complete 3D Masked Autoencoder model.
    
    Combines encoder, masking, decoder, and reconstruction loss
    for self-supervised pretraining on 3D volumes.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False
    ):
        """
        Initialize MAE-3D model.
        
        Args:
            img_size: Input image size
            patch_size: Patch size for embedding
            in_channels: Number of input channels
            embed_dim: Encoder embedding dimension
            encoder_depth: Encoder depth (number of transformer blocks)
            encoder_num_heads: Number of encoder attention heads
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Decoder depth
            decoder_num_heads: Number of decoder attention heads
            mask_ratio: Ratio of patches to mask
            norm_pix_loss: Whether to normalize pixel values in loss
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Patch embedding
        self.patch_embed = PatchEmbeddingWithPosition3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Encoder (simplified transformer blocks)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock3D(
                dim=embed_dim,
                num_heads=encoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0
            )
            for _ in range(encoder_depth)
        ])
        
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Masking module
        self.masking = PatchMasking3D(mask_ratio=mask_ratio)
        
        # Decoder
        self.decoder = MAEDecoder3D(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            patch_size=patch_size,
            in_channels=in_channels
        )
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize model weights."""
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights for different layer types."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: Input images (B, C, D, H, W)
            
        Returns:
            Patches (B, num_patches, patch_dim)
        """
        B, C, D, H, W = imgs.shape
        pd, ph, pw = self.patch_size
        
        # Ensure dimensions are divisible by patch size
        assert D % pd == 0 and H % ph == 0 and W % pw == 0
        
        # Reshape to patches
        x = imgs.reshape(B, C, D // pd, pd, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, (D // pd) * (H // ph) * (W // pw), C * pd * ph * pw)
        
        return x
        
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.
        
        Args:
            x: Patches (B, num_patches, patch_dim)
            
        Returns:
            Images (B, C, D, H, W)
        """
        B, num_patches, patch_dim = x.shape
        pd, ph, pw = self.patch_size
        C = self.in_channels
        
        # Calculate grid dimensions
        D, H, W = self.img_size
        grid_d, grid_h, grid_w = D // pd, H // ph, W // pw
        
        assert num_patches == grid_d * grid_h * grid_w
        
        # Reshape patches to image
        x = x.reshape(B, grid_d, grid_h, grid_w, C, pd, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.reshape(B, C, D, H, W)
        
        return x
        
    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking.
        
        Args:
            x: Input images (B, C, D, H, W)
            
        Returns:
            Tuple of (encoded_features, mask, ids_restore)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Apply masking
        x, mask, ids_restore = self.masking(x, self.patch_embed.grid_size)
        
        # Apply encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)
            
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
        
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Encoded features (B, len_keep, embed_dim)
            ids_restore: Indices to restore patch order
            
        Returns:
            Reconstructed patches (B, num_patches, patch_dim)
        """
        return self.decoder(x, ids_restore, self.patch_embed.grid_size)
        
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, 
                    mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            imgs: Original images (B, C, D, H, W)
            pred: Predicted patches (B, num_patches, patch_dim)
            mask: Binary mask (B, num_patches), 0 is keep, 1 is remove
            
        Returns:
            Reconstruction loss
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch
        
        # Only compute loss on masked patches
        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = (loss * mask).sum() / mask_sum
        else:
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        
        return loss
        
    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MAE-3D.
        
        Args:
            imgs: Input images (B, C, D, H, W)
            
        Returns:
            Dictionary containing loss, predictions, and mask
        """
        # Encoder
        latent, mask, ids_restore = self.forward_encoder(imgs)
        
        # Decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Loss
        loss = self.forward_loss(imgs, pred, mask)
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'latent': latent,
            'ids_restore': ids_restore
        }


class MAELoss3D(nn.Module):
    """
    MAE-3D reconstruction loss module.
    
    Computes mean squared error loss between original and reconstructed
    patches, focusing only on masked regions.
    """
    
    def __init__(self, norm_pix_loss: bool = False):
        """
        Initialize MAE loss.
        
        Args:
            norm_pix_loss: Whether to normalize pixel values
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        
    def forward(self, imgs: torch.Tensor, pred: torch.Tensor, 
                mask: torch.Tensor, patch_size: Tuple[int, int, int],
                in_channels: int = 1) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            imgs: Original images (B, C, D, H, W)
            pred: Predicted patches (B, num_patches, patch_dim)
            mask: Binary mask (B, num_patches)
            patch_size: Size of patches
            in_channels: Number of input channels
            
        Returns:
            Reconstruction loss
        """
        # Convert images to patches
        target = self._patchify(imgs, patch_size, in_channels)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        # Compute MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch
        
        # Only compute loss on masked patches
        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = (loss * mask).sum() / mask_sum
        else:
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        
        return loss
        
    def _patchify(self, imgs: torch.Tensor, patch_size: Tuple[int, int, int],
                  in_channels: int) -> torch.Tensor:
        """Convert images to patches."""
        B, C, D, H, W = imgs.shape
        pd, ph, pw = patch_size
        
        # Reshape to patches
        x = imgs.reshape(B, C, D // pd, pd, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, (D // pd) * (H // ph) * (W // pw), C * pd * ph * pw)
        
        return x