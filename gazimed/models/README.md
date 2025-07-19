# Gazimed Models

This directory contains deep learning models for medical image analysis, specifically for Alzheimer's disease detection using multimodal data.

## SwinUNet3D Backbone

The `SwinUNet3DBackbone` is a 3D Swin Transformer-based backbone for processing 3D medical images (MRI + PET) in Alzheimer's detection.

### Features

- **3D Swin Transformer Architecture**: Utilizes hierarchical vision transformers with shifted windows for efficient 3D processing
- **Configurable Architecture**: Supports customizable depths, attention heads, and dropout rates
- **Global Average Pooling**: Outputs a fixed-size feature vector regardless of input dimensions
- **Medical Image Optimized**: Designed specifically for dual-channel medical imaging (MRI + PET)

### Configuration

The backbone is configured with the following default parameters as specified in the requirements:

- `depths=[2, 2, 6, 2]`: Number of transformer blocks in each stage
- `num_heads=[3, 6, 12, 24]`: Number of attention heads in each stage  
- `dropout_path_rate=0.1`: Stochastic depth rate for regularization
- `feature_size=96`: Output feature dimension

### Usage

```python
from gazimed.models import SwinUNet3DBackbone

# Create model with default configuration
model = SwinUNet3DBackbone(
    in_channels=2,  # MRI + PET
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24], 
    dropout_path_rate=0.1,
    feature_size=96
)

# Forward pass
# Input: (batch_size, 2, height, width, depth)
# Output: (batch_size, 96)
features = model(input_tensor)
```

### Input/Output

- **Input**: `(B, 2, H, W, D)` - Batch of dual-channel 3D medical images
- **Output**: `(B, 96)` - Fixed-size feature vectors from global average pooling

### Architecture Details

1. **Patch Embedding**: Converts 3D input into patch tokens
2. **Hierarchical Stages**: Four stages with increasing feature dimensions
3. **Window Attention**: Efficient self-attention within local windows
4. **Patch Merging**: Downsampling between stages
5. **Global Pooling**: Aggregates spatial features into fixed-size output
6. **Feature Projection**: Projects to target feature dimension (96)

### Testing

Run the test suite to verify the implementation:

```bash
python test/test_swin_unet3d.py
python test/test_task_requirements.py
```

The tests verify:
- Correct output shapes for various input sizes
- Proper configuration parameters
- Gradient flow for training
- Feature statistics and numerical stability