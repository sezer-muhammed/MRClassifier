"""
SwinUNet3D Backbone Module for 3D Medical Image Processing

This module implements a 3D Swin Transformer-based U-Net architecture
for processing 3D medical images (MRI + PET) in Alzheimer's detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for Swin Transformer"""
    
    def __init__(self, patch_size=4, in_chans=2, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, H, W, D)
        B, C, H, W, D = x.shape
        
        # Patch embedding
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size, D//patch_size)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        
        return x


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-head Self Attention"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D data"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x, H, W, D):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Window attention
        x = self.attn(x)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Mlp(nn.Module):
    """MLP module"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class BasicLayer3D(nn.Module):
    """Basic Swin Transformer layer for 3D data"""
    
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None
    
    def forward(self, x, H, W, D):
        for blk in self.blocks:
            x = blk(x, H, W, D)
        
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, D)
            Wh, Ww, Wd = (H + 1) // 2, (W + 1) // 2, (D + 1) // 2
            return x, H, W, D, x_down, Wh, Ww, Wd
        else:
            return x, H, W, D, x, H, W, D


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for 3D data"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)
    
    def forward(self, x, H, W, D):
        B, L, C = x.shape
        
        # Ensure L matches H*W*D
        assert L == H * W * D, f"Sequence length {L} doesn't match spatial dimensions {H}x{W}x{D}={H*W*D}"
        
        x = x.view(B, H, W, D, C)
        
        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))
            H_pad, W_pad, D_pad = x.shape[1:4]
        else:
            H_pad, W_pad, D_pad = H, W, D
        
        # Merge patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x4 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*D/2 8*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class SwinUNet3DBackbone(nn.Module):
    """
    SwinUNet3D Backbone for 3D medical image processing
    
    This backbone processes 3D medical images (MRI + PET) using Swin Transformer
    architecture and outputs a 96-dimensional feature vector through global average pooling.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        dropout_path_rate: float = 0.1,
        feature_size: int = 96,
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0
    ):
        """
        Initialize SwinUNet3D Backbone
        
        Args:
            in_channels: Number of input channels (2 for MRI + PET)
            depths: Number of blocks in each stage [2,2,6,2]
            num_heads: Number of attention heads in each stage [3,6,12,24]
            dropout_path_rate: Stochastic depth rate (0.1)
            feature_size: Output feature dimension (96)
            patch_size: Patch size for patch embedding
            window_size: Window size for attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: If True, add a learnable bias to query, key, value
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.depths = depths
        self.num_heads = num_heads
        self.dropout_path_rate = dropout_path_rate
        self.feature_size = feature_size
        self.num_layers = len(depths)
        
        # Patch embedding - use smaller embedding dimension to save RAM
        embed_dim = 96
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=embed_dim
        )
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, dropout_path_rate, sum(depths))]
        
        # Build layers - use dynamic embedding dimension
        base_dim = embed_dim
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer3D(
                dim=int(base_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)
        
        # Final norm layer - use dynamic dimension
        final_dim = int(base_dim * 2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        
        # Global average pooling and feature projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_proj = nn.Linear(final_dim, feature_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwinUNet3D Backbone
        
        Args:
            x: Input tensor of shape (B, 2, H, W, D)
            
        Returns:
            features: Output features of shape (B, 96)
        """
        B, C, H_orig, W_orig, D_orig = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, 96)
        
        # Calculate spatial dimensions after patch embedding
        patch_size = self.patch_embed.patch_size
        H = H_orig // patch_size
        W = W_orig // patch_size  
        D = D_orig // patch_size
        
        # Forward through Swin layers
        for layer in self.layers:
            x, H, W, D, x, H, W, D = layer(x, H, W, D)
        
        # Final normalization
        x = self.norm(x)  # (B, num_patches, final_dim)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (B, final_dim, num_patches)
        x = self.global_pool(x)  # (B, final_dim, 1)
        x = x.squeeze(-1)  # (B, final_dim)
        
        # Project to target feature size
        features = self.feature_proj(x)  # (B, 96)
        
        return features
    
    def get_num_layers(self):
        """Get number of layers"""
        return len(self.layers)
    
    def no_weight_decay(self):
        """Parameters that should not have weight decay"""
        return {'absolute_pos_embed'}
    
    def no_weight_decay_keywords(self):
        """Keywords for parameters that should not have weight decay"""
        return {'relative_position_bias_table'}