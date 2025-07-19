"""
Clinical Feature Processor Module

This module implements the clinical feature processing component of the hybrid
Alzheimer's model, which processes 116 clinical features through an MLP.
"""

import torch
import torch.nn as nn
from typing import List


class ClinicalFeatureProcessor(nn.Module):
    """
    Clinical Feature Processor for processing 116 clinical features
    
    This module processes clinical features through a compact MLP with the
    architecture: 116 → 64 → 32 → 32 → 16, using ReLU activations and
    dropout for regularization.
    """
    
    def __init__(
        self,
        input_dim: int = 116,
        hidden_dims: List[int] = [64, 32, 32, 16],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Clinical Feature Processor
        
        Args:
            input_dim: Number of input clinical features (116)
            hidden_dims: List of hidden layer dimensions [64, 32, 32, 16]
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # ReLU activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout (except for the last layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using Xavier initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Clinical Feature Processor
        
        Args:
            clinical_features: Input tensor of shape (B, 116)
            
        Returns:
            processed_features: Output tensor of shape (B, 16)
        """
        # Validate input shape
        if clinical_features.dim() != 2:
            raise ValueError(f"Expected 2D input (B, 116), got {clinical_features.dim()}D")
        
        if clinical_features.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {clinical_features.shape[1]}")
        
        # Process through MLP
        processed_features = self.mlp(clinical_features)
        
        return processed_features
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.hidden_dims[-1]
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())