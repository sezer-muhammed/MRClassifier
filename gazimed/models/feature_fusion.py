"""
Feature Fusion Module

This module implements the feature fusion component that combines image features
from SwinUNet3D backbone with clinical features for final prediction.
"""

import torch
import torch.nn as nn
from typing import List


class FeatureFusion(nn.Module):
    """
    Feature Fusion Module for combining image and clinical features
    
    This module combines 96 image features from SwinUNet3D backbone with
    16 clinical features from the clinical processor, then processes them
    through final MLP layers: 112 → 64 → 32 → 1 for Alzheimer score prediction.
    """
    
    def __init__(
        self,
        image_dim: int = 96,
        clinical_dim: int = 16,
        fusion_dims: List[int] = [112, 64, 32, 1],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        fusion_strategy: str = "concatenate"
    ):
        """
        Initialize Feature Fusion Module
        
        Args:
            image_dim: Dimension of image features (96)
            clinical_dim: Dimension of clinical features (16)
            fusion_dims: List of fusion layer dimensions [112, 64, 32, 1]
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            fusion_strategy: Strategy for combining features ("concatenate", "add", "attention")
        """
        super().__init__()
        
        self.image_dim = image_dim
        self.clinical_dim = clinical_dim
        self.fusion_dims = fusion_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.fusion_strategy = fusion_strategy
        
        # Validate fusion dimensions
        if fusion_strategy == "concatenate":
            expected_input_dim = image_dim + clinical_dim
        elif fusion_strategy == "add":
            if image_dim != clinical_dim:
                raise ValueError(f"For 'add' fusion, image_dim ({image_dim}) must equal clinical_dim ({clinical_dim})")
            expected_input_dim = image_dim
        else:
            expected_input_dim = image_dim + clinical_dim  # Default to concatenate
        
        if fusion_dims[0] != expected_input_dim:
            raise ValueError(f"First fusion dim ({fusion_dims[0]}) must match expected input dim ({expected_input_dim})")
        
        # Build fusion layers
        if fusion_strategy == "attention":
            self.attention_fusion = self._build_attention_fusion()
        
        # Build MLP layers for final prediction
        layers = []
        for i in range(len(fusion_dims) - 1):
            input_dim = fusion_dims[i]
            output_dim = fusion_dims[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(input_dim, output_dim))
            
            # Don't add activation/normalization/dropout for the final output layer
            if i < len(fusion_dims) - 2:
                # Batch normalization (optional)
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim))
                
                # ReLU activation
                layers.append(nn.ReLU(inplace=True))
                
                # Dropout
                layers.append(nn.Dropout(dropout_rate))
        
        self.fusion_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_attention_fusion(self):
        """Build attention-based fusion mechanism"""
        return nn.MultiheadAttention(
            embed_dim=self.image_dim,
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
    
    def _init_weights(self, m):
        """Initialize weights using Xavier initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, image_features: torch.Tensor, clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Feature Fusion Module
        
        Args:
            image_features: Image features tensor of shape (B, 96)
            clinical_features: Clinical features tensor of shape (B, 16)
            
        Returns:
            alzheimer_score: Predicted Alzheimer score of shape (B, 1)
        """
        # Validate input shapes
        if image_features.dim() != 2:
            raise ValueError(f"Expected 2D image features (B, 96), got {image_features.dim()}D")
        if clinical_features.dim() != 2:
            raise ValueError(f"Expected 2D clinical features (B, 16), got {clinical_features.dim()}D")
        
        if image_features.shape[1] != self.image_dim:
            raise ValueError(f"Expected {self.image_dim} image features, got {image_features.shape[1]}")
        if clinical_features.shape[1] != self.clinical_dim:
            raise ValueError(f"Expected {self.clinical_dim} clinical features, got {clinical_features.shape[1]}")
        
        batch_size = image_features.shape[0]
        if clinical_features.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: image {batch_size} vs clinical {clinical_features.shape[0]}")
        
        # Fuse features based on strategy
        if self.fusion_strategy == "concatenate":
            fused_features = torch.cat([image_features, clinical_features], dim=1)
        elif self.fusion_strategy == "add":
            fused_features = image_features + clinical_features
        elif self.fusion_strategy == "attention":
            # Use attention to combine features
            # Reshape for attention: (B, seq_len, embed_dim)
            image_seq = image_features.unsqueeze(1)  # (B, 1, 96)
            clinical_seq = clinical_features.unsqueeze(1)  # (B, 1, 16)
            
            # Pad clinical features to match image dimension for attention
            if self.clinical_dim < self.image_dim:
                padding = torch.zeros(batch_size, 1, self.image_dim - self.clinical_dim, 
                                    device=clinical_features.device, dtype=clinical_features.dtype)
                clinical_seq = torch.cat([clinical_seq, padding], dim=2)
            
            # Apply attention
            attended_features, _ = self.attention_fusion(image_seq, clinical_seq, clinical_seq)
            fused_features = attended_features.squeeze(1)  # (B, 96)
            
            # Concatenate with original clinical features
            fused_features = torch.cat([fused_features, clinical_features], dim=1)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Process through fusion MLP
        alzheimer_score = self.fusion_mlp(fused_features)
        
        # Note: No sigmoid here - BCEWithLogitsLoss will apply it internally
        return alzheimer_score
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.fusion_dims[-1]
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())