"""
Swin-UNETR Architecture for 3D Medical Image Analysis

This module implements the Swin-UNETR (Swin Transformer + UNETR) architecture
specifically designed for 3D medical image analysis tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np
# from einops import rearrange  # Unused import removed to avoid dependency conflicts
from .patch_embedding import PatchEmbeddingWithPosition3D


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self Attention (W-MSA) module with relative position bias.
    
    This implements the core attention mechanism used in Swin Transformer for 3D data.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Initialize 3D Window Attention.
        
        Args:
            dim: Input feature dimension
            window_size: Window size (D, H, W)
            num_heads: Number of attention heads
            qkv_bias: Whether to add bias to qkv projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        # Get pair-wise relative position indices
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of window attention.
        
        Args:
            x: Input tensor of shape (num_windows*B, window_size*window_size*window_size, C)
            mask: Attention mask (optional)
            
        Returns:
            Output tensor of same shape as input
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
            
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    Swin Transformer Block for 3D data.
    
    This block implements the core Swin Transformer functionality including
    window-based attention and shifted window attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        """
        Initialize Swin Transformer Block.
        
        Args:
            dim: Input feature dimension
            num_heads: Number of attention heads
            window_size: Window size for attention
            shift_size: Shift size for shifted window attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias to qkv projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must be smaller than window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Swin Transformer Block.
        
        Args:
            x: Input tensor of shape (B, D, H, W, C)
            mask_matrix: Attention mask matrix
            
        Returns:
            Output tensor of same shape as input
        """
        B, D, H, W, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        
        shortcut = x
        x = self.norm1(x)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        # Partition windows
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        shifted_x = window_reverse(attn_windows, window_size, Dp, Hp, Wp)
        
        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
            
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
            
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class BasicLayer3D(nn.Module):
    """
    A basic Swin Transformer layer for one stage in 3D.
    
    This layer contains multiple Swin Transformer blocks and handles
    downsampling between stages.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize Basic Layer.
        
        Args:
            dim: Input feature dimension
            depth: Number of blocks in this layer
            num_heads: Number of attention heads
            window_size: Window size for attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias to qkv projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rates for each block
            norm_layer: Normalization layer
            downsample: Downsample layer at the end of the layer
        """
        super().__init__()
        
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if drop_path is not None else 0.0,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Basic Layer.
        
        Args:
            x: Input tensor of shape (B, D, H, W, C)
            
        Returns:
            Tuple of (output_tensor, downsampled_tensor)
        """
        # Calculate attention mask for SW-MSA
        Dp = int(np.ceil(x.size(1) / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(x.size(2) / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(x.size(3) / self.window_size[2])) * self.window_size[2]
        
        attn_mask = compute_mask(Dp, Hp, Wp, self.window_size, self.shift_size, x.device)
        
        for blk in self.blocks:
            x = blk(x, attn_mask)
            
        x_down = x
        if self.downsample is not None:
            x_down = self.downsample(x)
            
        return x, x_down


class SwinUNETR(nn.Module):
    """
    Swin-UNETR: Swin Transformer based U-Net for medical image analysis.
    
    This model combines the Swin Transformer encoder with a U-Net style decoder
    for dense prediction tasks in 3D medical imaging.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        feature_size: int = 24,
        use_checkpoint: bool = False
    ):
        """
        Initialize Swin-UNETR model.
        
        Args:
            img_size: Input image size
            patch_size: Patch size for patch embedding
            in_channels: Number of input channels
            embed_dim: Patch embedding dimension
            depths: Depths of each Swin Transformer layer
            num_heads: Number of attention heads in different layers
            window_size: Window size for attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add bias to qkv projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
            patch_norm: Whether to add normalization after patch embedding
            feature_size: Feature size for decoder
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.feature_size = feature_size
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbeddingWithPosition3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)
            
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of Swin-UNETR encoder.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            List of feature maps from different stages
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Reshape to 3D format
        B, num_patches, C = x.shape
        grid_size = self.patch_embed.grid_size
        x = x.view(B, grid_size[0], grid_size[1], grid_size[2], C)
        
        # Forward through layers
        features = []
        for layer in self.layers:
            x_out, x = layer(x)
            features.append(x_out)
            
        # Final normalization
        x = self.norm(x)
        features[-1] = x
        
        return features


# Helper functions and classes

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
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


class Mlp(nn.Module):
    """MLP module."""
    
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


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for 3D data."""
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """Forward pass of patch merging."""
        B, D, H, W, C = x.shape

        # Padding
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = self.norm(x)
        x = self.reduction(x)  # B D/2 H/2 W/2 2*C

        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def compute_mask(D, H, W, window_size, shift_size, device):
    """Compute attention mask for SW-MSA."""
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask