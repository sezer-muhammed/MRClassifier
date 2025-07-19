# Design Document

## Overview

The Hybrid Alzheimer's Model is a multimodal deep learning architecture that combines 3D brain imaging (MRI + PET) with clinical features for Alzheimer's disease detection. The design follows a two-stream approach with late fusion, where imaging data is processed through a SwinUNet3D backbone and clinical features through a compact MLP, before combining both representations for final prediction.

## Architecture

### High-Level Architecture

```
Input: Images (B×2×H×W×D) + Clinical Features (B×116)
    ↓                           ↓
SwinUNet3D Backbone         Clinical MLP
    ↓                           ↓
Image Features (B×96)      Clinical Features (B×16)
    ↓                           ↓
        Feature Fusion (B×112)
                ↓
        Final MLP Layers
                ↓
        Alzheimer Score (B×1)
```

### Component Architecture

#### 1. Image Processing Stream
- **Input**: Batch × 2 × H × W × D (MRI + PET channels)
- **Backbone**: SwinUNet3D with specified configuration
- **Output**: 96-dimensional feature vector per sample

#### 2. Clinical Processing Stream  
- **Input**: Batch × 116 (clinical features)
- **Architecture**: 116 → 64 → 32 → 32 → 16
- **Output**: 16-dimensional feature vector per sample

#### 3. Fusion and Prediction
- **Fusion**: Concatenation of image (96) + clinical (16) features
- **Final Layers**: 112 → 64 → 32 → 1
- **Output**: Single Alzheimer's score per sample

## Components and Interfaces

### Core Components

#### HybridAlzheimersModel
```python
class HybridAlzheimersModel(pl.LightningModule):
    def __init__(
        self,
        target_size: Tuple[int, int, int] = (91, 109, 91),
        swin_config: Dict = None,
        clinical_dims: List[int] = [116, 64, 32, 32, 16],
        fusion_dims: List[int] = [112, 64, 32, 1],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2
    )
```

#### SwinUNet3DBackbone
```python
class SwinUNet3DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        dropout_path_rate: float = 0.1,
        feature_size: int = 96
    )
```

#### ClinicalFeatureProcessor
```python
class ClinicalFeatureProcessor(nn.Module):
    def __init__(
        self,
        input_dim: int = 116,
        hidden_dims: List[int] = [64, 32, 32, 16],
        dropout_rate: float = 0.1
    )
```

#### FeatureFusion
```python
class FeatureFusion(nn.Module):
    def __init__(
        self,
        image_dim: int = 96,
        clinical_dim: int = 16,
        fusion_dims: List[int] = [112, 64, 32, 1],
        dropout_rate: float = 0.1
    )
```

### Interface Specifications

#### Model Input Interface
```python
def forward(
    self,
    images: torch.Tensor,      # Shape: (B, 2, H, W, D)
    clinical: torch.Tensor     # Shape: (B, 116)
) -> torch.Tensor:            # Shape: (B, 1)
```

#### Configuration Interface
```python
@dataclass
class ModelConfig:
    target_size: Tuple[int, int, int] = (91, 109, 91)
    swin_depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    swin_num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    swin_dropout_path_rate: float = 0.1
    clinical_dims: List[int] = field(default_factory=lambda: [116, 64, 32, 32, 16])
    fusion_dims: List[int] = field(default_factory=lambda: [112, 64, 32, 1])
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
```

## Data Models

### Input Data Structure
```python
class ModelInput:
    images: torch.Tensor       # (B, 2, H, W, D) - MRI + PET
    clinical_features: torch.Tensor  # (B, 116) - Clinical features
    alzheimer_score: torch.Tensor    # (B, 1) - Target scores
```

### Model State Structure
```python
class ModelState:
    model_state_dict: Dict
    optimizer_state_dict: Dict
    scheduler_state_dict: Dict
    epoch: int
    best_val_loss: float
    config: ModelConfig
    metadata: Dict
```

### Feature Representations
```python
class FeatureRepresentations:
    image_features: torch.Tensor     # (B, 96) - From SwinUNet3D
    clinical_features: torch.Tensor  # (B, 16) - From Clinical MLP
    fused_features: torch.Tensor     # (B, 112) - Concatenated
    predictions: torch.Tensor        # (B, 1) - Final scores
```

## Error Handling

### Input Validation
- **Shape Validation**: Verify input tensors match expected dimensions
- **Range Validation**: Ensure clinical features are in expected range [-1, 1]
- **Device Consistency**: Check all tensors are on the same device
- **Batch Size Consistency**: Verify images and clinical features have same batch size

### Training Error Handling
- **Memory Management**: Handle CUDA out-of-memory errors gracefully
- **Gradient Issues**: Detect and handle gradient explosion/vanishing
- **NaN Detection**: Monitor for NaN values in loss and gradients
- **Checkpoint Recovery**: Automatic recovery from training interruptions

### Model Loading/Saving Errors
- **Version Compatibility**: Handle model version mismatches
- **Missing Components**: Graceful handling of missing model components
- **Corrupted Files**: Validation of saved model integrity
- **Device Mismatch**: Handle CPU/GPU device mismatches during loading

## Testing Strategy

### Unit Testing
1. **Component Testing**
   - Test SwinUNet3D backbone with various input shapes
   - Test Clinical MLP with different feature dimensions
   - Test Feature Fusion with various input combinations
   - Test Model save/load functionality

2. **Integration Testing**
   - Test complete forward pass with real data
   - Test backward pass and gradient computation
   - Test PyTorch Lightning integration
   - Test FP16 mixed precision training

3. **Shape Testing**
   - Test configurable input shapes
   - Test batch size variations
   - Test memory usage with different configurations
   - Test model scaling with different parameters

### Performance Testing
1. **Memory Profiling**
   - Test GPU memory usage with different batch sizes
   - Test memory efficiency of caching strategies
   - Profile memory leaks during training

2. **Speed Benchmarking**
   - Measure forward pass inference time
   - Measure training step time
   - Compare FP16 vs FP32 performance
   - Profile bottlenecks in the architecture

### Validation Testing
1. **Data Compatibility**
   - Test with real dataloader output
   - Test with various image resolutions
   - Test with missing clinical features
   - Test with edge cases in data

2. **Training Validation**
   - Test convergence on synthetic data
   - Test overfitting detection
   - Test learning rate scheduling
   - Test checkpoint resumption

## Implementation Notes

### SwinUNet3D Integration
- Use existing SwinUNet3D implementation or adapt from 2D version
- Ensure proper 3D convolution and attention mechanisms
- Implement efficient patch embedding for 3D volumes
- Add global average pooling for feature extraction

### Clinical Feature Processing
- Simple MLP architecture with ReLU activations
- Dropout for regularization between layers
- Batch normalization for stable training
- Skip connections for better gradient flow

### Feature Fusion Strategy
- Late fusion approach with concatenation
- Alternative fusion methods (attention, gating) as future extensions
- Learnable fusion weights for balancing modalities
- Dropout for preventing overfitting in fusion layers

### PyTorch Lightning Integration
- Implement training_step, validation_step, test_step
- Configure optimizers and learning rate schedulers
- Add logging for metrics and visualizations
- Support for distributed training and checkpointing

### Mixed Precision Training
- Use PyTorch Lightning's automatic mixed precision
- Ensure all operations are FP16 compatible
- Handle gradient scaling automatically
- Monitor for numerical instabilities