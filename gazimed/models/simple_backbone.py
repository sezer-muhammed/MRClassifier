"""
Simple 3D CNN Backbone for Memory-Constrained Training

This is a simplified 3D CNN backbone that replaces the complex SwinUNet3D
for memory-constrained environments.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Simple3DBackbone(nn.Module):
    """
    Simple 3D CNN Backbone for processing MRI + PET volumes
    
    This is a lightweight alternative to SwinUNet3D for memory-constrained training.
    Uses standard 3D convolutions with global average pooling.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        feature_size: int = 64,
        base_channels: int = 32
    ):
        """
        Initialize Simple 3D Backbone
        
        Args:
            in_channels: Number of input channels (2 for MRI + PET)
            feature_size: Output feature dimension
            base_channels: Base number of channels for convolutions
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.base_channels = base_channels
        
        # 3D CNN layers with progressive downsampling
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))  # Reduce to fixed small size
        )
        
        # Final feature extraction
        final_conv_features = base_channels * 4 * 8  # 4 channels * 2*2*2 spatial
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_conv_features, feature_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_size * 2, feature_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 2, H, W, D)
            
        Returns:
            features: Output features of shape (B, feature_size)
        """
        # Progressive downsampling
        x = self.conv1(x)  # Reduce spatial dimensions
        x = self.conv2(x)  # Further reduce
        x = self.conv3(x)  # Final reduction to small fixed size
        
        # Extract features
        features = self.feature_extractor(x)
        
        return features
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())