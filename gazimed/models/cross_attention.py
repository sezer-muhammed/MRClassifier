"""
Cross-Modal Attention Fusion Module

This module implements cross-attention mechanisms for fusing MRI and PET
modalities in the Alzheimer's detection model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module for fusing MRI and PET features.
    
    This module uses MRI features as queries and PET features as keys/values
    to create attention-weighted fusion of multimodal information.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize Cross-Modal Attention module.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            qkv_bias: Whether to add bias to qkv projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            temperature: Temperature scaling for attention scores
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = (self.head_dim ** -0.5) * temperature
        
        # Query projection (for MRI features)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Key and Value projections (for PET features)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Layer normalization
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(
        self, 
        mri_features: torch.Tensor, 
        pet_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-modal attention.
        
        Args:
            mri_features: MRI features as queries (B, N, C)
            pet_features: PET features as keys/values (B, M, C)
            attention_mask: Optional attention mask (B, N, M)
            
        Returns:
            Tuple of (fused_features, attention_weights)
            - fused_features: Attention-weighted fusion (B, N, C)
            - attention_weights: Attention weights (B, num_heads, N, M)
        """
        B, N, C = mri_features.shape
        _, M, _ = pet_features.shape
        
        # Apply layer normalization
        mri_norm = self.norm_q(mri_features)
        pet_norm_k = self.norm_k(pet_features)
        pet_norm_v = self.norm_v(pet_features)
        
        # Project to queries, keys, and values
        q = self.q_proj(mri_norm)  # (B, N, C)
        k = self.k_proj(pet_norm_k)  # (B, M, C)
        v = self.v_proj(pet_norm_v)  # (B, M, C)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, M, head_dim)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, M, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multiple heads
            if attention_mask.dim() == 3:  # (B, N, M)
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, N, M)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, M)
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        fused_features = self.out_proj(attn_output)
        fused_features = self.proj_drop(fused_features)
        
        # Add residual connection
        fused_features = fused_features + mri_features
        
        return fused_features, attn_weights


class MultiModalFusionBlock(nn.Module):
    """
    Multi-modal fusion block that combines cross-attention with feed-forward processing.
    
    This block implements a complete fusion layer including cross-attention,
    normalization, and feed-forward processing.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.1,
        attn_drop: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize Multi-modal Fusion Block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            qkv_bias: Whether to add bias to qkv projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            activation: Activation function ('gelu', 'relu', 'swish')
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embed_dim,
            activation=activation,
            drop=drop
        )
        
    def forward(
        self, 
        mri_features: torch.Tensor, 
        pet_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-modal fusion block.
        
        Args:
            mri_features: MRI features (B, N, C)
            pet_features: PET features (B, M, C)
            attention_mask: Optional attention mask (B, N, M)
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        # Cross-modal attention with residual connection
        attn_out, attn_weights = self.cross_attn(mri_features, pet_features, attention_mask)
        x = self.norm1(attn_out)
        
        # Feed-forward with residual connection
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x, attn_weights


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion module that combines features from multiple scales.
    
    This module processes features from different stages of the Swin-UNETR
    encoder and fuses them hierarchically.
    """
    
    def __init__(
        self,
        feature_dims: list,
        target_dim: int = 768,
        num_heads: int = 2,
        num_layers: int = 2,
        drop: float = 0.1
    ):
        """
        Initialize Hierarchical Fusion module.
        
        Args:
            feature_dims: List of feature dimensions from different stages
            target_dim: Target dimension for all features
            num_heads: Number of attention heads
            num_layers: Number of fusion layers
            drop: Dropout rate
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.target_dim = target_dim
        self.num_layers = num_layers
        
        # Feature projection layers to unify dimensions
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, target_dim) if dim != target_dim else nn.Identity()
            for dim in feature_dims
        ])
        
        # Fusion blocks for each layer
        self.fusion_blocks = nn.ModuleList([
            MultiModalFusionBlock(
                embed_dim=target_dim,
                num_heads=num_heads,
                drop=drop
            )
            for _ in range(num_layers)
        ])
        
        # Feature aggregation
        self.aggregation = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(target_dim, target_dim)
        )
        
    def forward(
        self, 
        mri_features_list: list, 
        pet_features_list: list
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass of hierarchical fusion.
        
        Args:
            mri_features_list: List of MRI features from different stages
            pet_features_list: List of PET features from different stages
            
        Returns:
            Tuple of (fused_features, attention_weights_list)
        """
        assert len(mri_features_list) == len(pet_features_list) == len(self.feature_dims)
        
        # Project all features to target dimension
        mri_projected = []
        pet_projected = []
        
        for i, (mri_feat, pet_feat) in enumerate(zip(mri_features_list, pet_features_list)):
            # Flatten spatial dimensions if needed
            if mri_feat.dim() == 5:  # (B, D, H, W, C)
                B, D, H, W, C = mri_feat.shape
                mri_feat = mri_feat.view(B, D * H * W, C)
            if pet_feat.dim() == 5:  # (B, D, H, W, C)
                B, D, H, W, C = pet_feat.shape
                pet_feat = pet_feat.view(B, D * H * W, C)
                
            mri_proj = self.feature_projections[i](mri_feat)
            pet_proj = self.feature_projections[i](pet_feat)
            
            mri_projected.append(mri_proj)
            pet_projected.append(pet_proj)
        
        # Concatenate features from all stages
        mri_concat = torch.cat(mri_projected, dim=1)  # (B, total_patches, target_dim)
        pet_concat = torch.cat(pet_projected, dim=1)  # (B, total_patches, target_dim)
        
        # Apply fusion blocks
        fused_features = mri_concat
        attention_weights_list = []
        
        for fusion_block in self.fusion_blocks:
            fused_features, attn_weights = fusion_block(fused_features, pet_concat)
            attention_weights_list.append(attn_weights)
        
        # Global average pooling and aggregation
        pooled_features = fused_features.mean(dim=1)  # (B, target_dim)
        final_features = self.aggregation(pooled_features)
        
        return final_features, attention_weights_list


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable activation.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: str = 'gelu',
        drop: float = 0.0
    ):
        """
        Initialize MLP.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            activation: Activation function name
            drop: Dropout rate
        """
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # Activation function
        if activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'swish':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating sequence features.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 1):
        """
        Initialize Attention Pooling.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        
        nn.init.xavier_uniform_(self.query)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention pooling.
        
        Args:
            x: Input features (B, N, C)
            
        Returns:
            Pooled features (B, C)
        """
        B, N, C = x.shape
        
        # Expand query for batch
        q = self.query.expand(B, -1, -1)  # (B, 1, C)
        k = self.key_proj(x)  # (B, N, C)
        v = self.value_proj(x)  # (B, N, C)
        
        # Reshape for multi-head attention
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, C)  # (B, 1, C)
        
        # Project output
        output = self.out_proj(attn_output.squeeze(1))  # (B, C)
        
        return output