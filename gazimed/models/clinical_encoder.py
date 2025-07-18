"""
Clinical Features Encoder Module

This module implements encoders for processing clinical features in the
Alzheimer's detection model, including numerical features, categorical
features, and multimodal fusion with imaging features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import numpy as np


class ClinicalFeaturesEncoder(nn.Module):
    """
    Encoder for processing numerical clinical features.
    
    This encoder processes the 118 numerical clinical features through
    multiple MLP layers with normalization and dropout for regularization.
    """
    
    def __init__(
        self,
        input_dim: int = 118,
        hidden_dims: List[int] = [256, 512, 256],
        output_dim: int = 128,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = 'relu',
        use_residual: bool = True
    ):
        """
        Initialize Clinical Features Encoder.
        
        Args:
            input_dim: Number of input clinical features (default: 118)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'swish')
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim) if batch_norm else nn.Identity()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            elif activation.lower() == 'swish':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Residual projection if needed
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of clinical features encoder.
        
        Args:
            x: Clinical features tensor of shape (B, input_dim)
            
        Returns:
            Encoded features of shape (B, output_dim)
        """
        # Input validation
        assert x.shape[-1] == self.input_dim, f"Expected {self.input_dim} features, got {x.shape[-1]}"
        
        # Input normalization
        x_norm = self.input_norm(x)
        
        # Forward through MLP
        output = self.mlp(x_norm)
        
        # Residual connection if applicable
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(x)
            output = output + residual
        elif self.use_residual and x.shape[-1] == self.output_dim:
            output = output + x
            
        return output


class CategoricalEncoder(nn.Module):
    """
    Encoder for categorical clinical features.
    
    This encoder handles categorical features using embedding layers,
    useful for features like gender, education level, etc.
    """
    
    def __init__(
        self,
        categorical_dims: Dict[str, int],
        embedding_dims: Optional[Dict[str, int]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize Categorical Encoder.
        
        Args:
            categorical_dims: Dictionary mapping feature names to number of categories
            embedding_dims: Dictionary mapping feature names to embedding dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.categorical_dims = categorical_dims
        self.feature_names = list(categorical_dims.keys())
        
        # Default embedding dimensions (rule of thumb: min(50, (num_categories + 1) // 2))
        if embedding_dims is None:
            embedding_dims = {
                name: min(50, (num_cats + 1) // 2) 
                for name, num_cats in categorical_dims.items()
            }
        
        self.embedding_dims = embedding_dims
        
        # Create embedding layers
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_cats, embedding_dims[name])
            for name, num_cats in categorical_dims.items()
        })
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate total output dimension
        self.output_dim = sum(embedding_dims.values())
        
    def forward(self, categorical_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of categorical encoder.
        
        Args:
            categorical_features: Dictionary mapping feature names to category indices
            
        Returns:
            Concatenated embeddings of shape (B, total_embedding_dim)
        """
        embeddings = []
        
        for name in self.feature_names:
            if name in categorical_features:
                emb = self.embeddings[name](categorical_features[name])
                embeddings.append(emb)
            else:
                # Handle missing features with zeros
                batch_size = next(iter(categorical_features.values())).shape[0]
                zero_emb = torch.zeros(batch_size, self.embedding_dims[name], 
                                     device=next(iter(categorical_features.values())).device)
                embeddings.append(zero_emb)
        
        # Concatenate all embeddings
        output = torch.cat(embeddings, dim=-1)
        output = self.dropout(output)
        
        return output


class MultiModalClinicalEncoder(nn.Module):
    """
    Multi-modal encoder that combines numerical and categorical clinical features.
    
    This encoder processes both types of clinical features and combines them
    into a unified representation.
    """
    
    def __init__(
        self,
        numerical_dim: int = 118,
        categorical_dims: Optional[Dict[str, int]] = None,
        numerical_hidden_dims: List[int] = [256, 512, 256],
        numerical_output_dim: int = 128,
        categorical_embedding_dims: Optional[Dict[str, int]] = None,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize Multi-modal Clinical Encoder.
        
        Args:
            numerical_dim: Number of numerical features
            categorical_dims: Dictionary of categorical feature dimensions
            numerical_hidden_dims: Hidden dimensions for numerical encoder
            numerical_output_dim: Output dimension for numerical encoder
            categorical_embedding_dims: Embedding dimensions for categorical features
            fusion_dim: Final fusion dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.numerical_dim = numerical_dim
        self.categorical_dims = categorical_dims or {}
        self.fusion_dim = fusion_dim
        
        # Numerical features encoder
        self.numerical_encoder = ClinicalFeaturesEncoder(
            input_dim=numerical_dim,
            hidden_dims=numerical_hidden_dims,
            output_dim=numerical_output_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Categorical features encoder
        if categorical_dims:
            self.categorical_encoder = CategoricalEncoder(
                categorical_dims=categorical_dims,
                embedding_dims=categorical_embedding_dims,
                dropout=dropout
            )
            total_categorical_dim = self.categorical_encoder.output_dim
        else:
            self.categorical_encoder = None
            total_categorical_dim = 0
        
        # Fusion layer
        fusion_input_dim = numerical_output_dim + total_categorical_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True) if activation.lower() == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(
        self, 
        numerical_features: torch.Tensor,
        categorical_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of multi-modal clinical encoder.
        
        Args:
            numerical_features: Numerical features (B, numerical_dim)
            categorical_features: Dictionary of categorical features
            
        Returns:
            Fused clinical features (B, fusion_dim)
        """
        # Encode numerical features
        numerical_encoded = self.numerical_encoder(numerical_features)
        
        # Encode categorical features if available
        if self.categorical_encoder is not None and categorical_features is not None:
            categorical_encoded = self.categorical_encoder(categorical_features)
            # Concatenate numerical and categorical features
            combined_features = torch.cat([numerical_encoded, categorical_encoded], dim=-1)
        else:
            combined_features = numerical_encoded
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        return fused_features


class ClinicalAttentionEncoder(nn.Module):
    """
    Clinical encoder with self-attention mechanism.
    
    This encoder treats clinical features as a sequence and applies
    self-attention to capture feature interactions.
    """
    
    def __init__(
        self,
        input_dim: int = 118,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        """
        Initialize Clinical Attention Encoder.
        
        Args:
            input_dim: Number of input clinical features
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Feature embedding (treat each feature as a token)
        self.feature_embedding = nn.Linear(1, embed_dim)
        
        # Positional encoding for features
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.feature_embedding.weight)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of clinical attention encoder.
        
        Args:
            x: Clinical features (B, input_dim)
            
        Returns:
            Encoded features (B, embed_dim)
        """
        B, N = x.shape
        assert N == self.input_dim, f"Expected {self.input_dim} features, got {N}"
        
        # Reshape for feature embedding: (B, N) -> (B, N, 1)
        x = x.unsqueeze(-1)
        
        # Feature embedding: (B, N, 1) -> (B, N, embed_dim)
        x = self.feature_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer
        x = self.transformer(x)  # (B, N, embed_dim)
        
        # Output projection
        x = self.output_proj(x)
        
        # Global pooling: (B, N, embed_dim) -> (B, embed_dim)
        x = x.transpose(1, 2)  # (B, embed_dim, N)
        x = self.global_pool(x).squeeze(-1)  # (B, embed_dim)
        
        return x


class ClinicalFeatureNormalizer(nn.Module):
    """
    Normalizer for clinical features with learned statistics.
    
    This module learns to normalize clinical features and can handle
    missing values and outliers.
    """
    
    def __init__(
        self,
        num_features: int,
        normalization_type: str = 'batch',
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True
    ):
        """
        Initialize Clinical Feature Normalizer.
        
        Args:
            num_features: Number of clinical features
            normalization_type: Type of normalization ('batch', 'layer', 'instance')
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        
        self.num_features = num_features
        self.normalization_type = normalization_type.lower()
        
        if self.normalization_type == 'batch':
            self.norm = nn.BatchNorm1d(
                num_features, 
                eps=eps, 
                momentum=momentum, 
                track_running_stats=track_running_stats
            )
        elif self.normalization_type == 'layer':
            self.norm = nn.LayerNorm(num_features, eps=eps)
        elif self.normalization_type == 'instance':
            self.norm = nn.InstanceNorm1d(
                num_features, 
                eps=eps, 
                momentum=momentum, 
                track_running_stats=track_running_stats
            )
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")
        
        # Learnable feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(num_features))
        
        # Missing value handling
        self.missing_value_embedding = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of clinical feature normalizer.
        
        Args:
            x: Clinical features (B, num_features)
            mask: Binary mask indicating valid features (B, num_features)
            
        Returns:
            Normalized features (B, num_features)
        """
        # Handle missing values
        if mask is not None:
            # Replace missing values with learned embeddings
            x = x * mask + self.missing_value_embedding * (1 - mask)
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply feature importance weights
        x = x * self.feature_weights
        
        return x


# Utility functions for clinical feature processing

def create_clinical_feature_mask(features: torch.Tensor, missing_value: float = -999.0) -> torch.Tensor:
    """
    Create a binary mask indicating valid (non-missing) clinical features.
    
    Args:
        features: Clinical features tensor (B, num_features)
        missing_value: Value used to indicate missing features
        
    Returns:
        Binary mask (B, num_features) where 1 indicates valid features
    """
    return (features != missing_value).float()


def impute_missing_features(
    features: torch.Tensor, 
    mask: torch.Tensor, 
    strategy: str = 'mean'
) -> torch.Tensor:
    """
    Impute missing clinical features.
    
    Args:
        features: Clinical features tensor (B, num_features)
        mask: Binary mask indicating valid features (B, num_features)
        strategy: Imputation strategy ('mean', 'median', 'zero')
        
    Returns:
        Features with imputed values (B, num_features)
    """
    if strategy == 'zero':
        return features * mask
    elif strategy == 'mean':
        # Compute mean of valid features for each feature dimension
        valid_features = features * mask
        feature_sums = valid_features.sum(dim=0)
        feature_counts = mask.sum(dim=0).clamp(min=1)
        feature_means = feature_sums / feature_counts
        
        # Replace missing values with means
        imputed = features * mask + feature_means.unsqueeze(0) * (1 - mask)
        return imputed
    elif strategy == 'median':
        # Compute median for each feature (more complex, simplified here)
        return impute_missing_features(features, mask, strategy='mean')  # Fallback to mean
    else:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")


def standardize_clinical_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize clinical features to zero mean and unit variance.
    
    Args:
        features: Clinical features tensor (B, num_features)
        eps: Small constant for numerical stability
        
    Returns:
        Standardized features (B, num_features)
    """
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp(min=eps)
    return (features - mean) / std