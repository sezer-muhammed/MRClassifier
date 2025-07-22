# Design Document

## Overview

The universal model structure provides a standardized interface for AI models in the Gazimed system through Pydantic objects for input/output validation and a base PyTorch Lightning class for consistent model implementation. The design focuses on parametric initialization allowing dynamic specification of input shapes, channels, and feature dimensions.

## Architecture

### Core Components

1. **Pydantic Input Objects**: Validated data containers for model inputs
2. **Pydantic Output Objects**: Structured prediction results with metadata
3. **Base Model Class**: PyTorch Lightning module with parametric initialization
4. **Validation System**: Shape and type checking for tensors

### Component Relationships

```
ModelInput (Pydantic) → BaseAIModel (PyTorch Lightning) → ModelOutput (Pydantic)
                              ↓
                    Parametric Initialization
                    (shape, channels, features)
```

## Components and Interfaces

### 1. Pydantic Input Objects

#### ModelInput Base Class
```python
class ModelInput(BaseModel):
    """Base input class for all AI models"""
    
    # Core tensor fields
    image: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    
    # Shape metadata
    batch_size: int
    input_shape: Tuple[int, int, int]  # H, W, D
    in_channels: int
    clinical_features_size: Optional[int] = None
    
    # Optional metadata
    subject_id: Optional[str] = None
    device: str = "cpu"
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('image')
    def validate_image_shape(cls, v, values):
        if v is not None and 'batch_size' in values and 'in_channels' in values and 'input_shape' in values:
            expected_shape = (values['batch_size'], values['in_channels']) + values['input_shape']
            if v.shape != expected_shape:
                raise ValueError(f"Image tensor shape {v.shape} doesn't match expected {expected_shape}")
        return v
    
    @validator('features')
    def validate_features_shape(cls, v, values):
        if v is not None and 'batch_size' in values and 'clinical_features_size' in values:
            expected_shape = (values['batch_size'], values['clinical_features_size'])
            if v.shape != expected_shape:
                raise ValueError(f"Features tensor shape {v.shape} doesn't match expected {expected_shape}")
        return v
```


### 2. Pydantic Output Objects

#### ModelOutput Base Class
```python
class ModelOutput(BaseModel):
    """Base output class for all AI models"""
    
    # Core prediction fields
    predictions: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    
    # Optional explanation fields
    attention_maps: Optional[torch.Tensor] = None
    feature_importance: Optional[torch.Tensor] = None
    
    # Shape metadata
    input_shape: Tuple[int, int, int]  # H, W, D
    in_channels: int
    clinical_features_size: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
```

### 3. Base AI Model Class

#### BaseAIModel Class
```python
class BaseAIModel(pl.LightningModule):
    """Base class for all AI models with parametric initialization"""
    
    def __init__(
        self,
        # Required shape parameters
        input_shape: Tuple[int, int, int],
        in_channels: int,
        output_size: int = 1,
        
        # Optional parameters
        clinical_features_size: Optional[int] = None,
        model_name: str = "BaseAIModel",
        model_version: str = "1.0",
        
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        
        # Architecture parameters
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation_type: str = "relu",
        
        # Loss parameters
        loss_type: str = "bce",
        label_smoothing: float = 0.0,
        
        # Regularization parameters
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.output_size = output_size
        self.clinical_features_size = clinical_features_size
        self.model_name = model_name
        self.model_version = model_version
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_type = activation_type
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize architecture (to be implemented by subclasses)
        self._build_architecture()
    
    def _build_architecture(self):
        """Build model architecture - must be implemented by subclasses"""
        raise NotImplementedError
    
    def _validate_input(self, model_input: ModelInput):
        """Validate input tensor shapes match model configuration"""
        if model_input.image is not None:
            expected_shape = (model_input.batch_size, self.in_channels) + self.input_shape
            if model_input.image.shape != expected_shape:
                raise ShapeMismatchError(expected_shape, model_input.image.shape)
        
        if model_input.features is not None and self.clinical_features_size is not None:
            expected_shape = (model_input.batch_size, self.clinical_features_size)
            if model_input.features.shape != expected_shape:
                raise ShapeMismatchError(expected_shape, model_input.features.shape)
    
    def forward(self, model_input: ModelInput) -> torch.Tensor:
        """Forward pass with Pydantic input validation"""
        # Validate input
        self._validate_input(model_input)
        
        # Extract tensors
        imaging_data = model_input.image
        clinical_features = model_input.features
        
        # Forward pass (to be implemented by subclasses)
        return self._forward_impl(imaging_data, clinical_features)
    
    def _forward_impl(self, imaging_data, clinical_features):
        """Actual forward implementation - must be implemented by subclasses"""
        raise NotImplementedError
    
    def predict(self, model_input: ModelInput) -> ModelOutput:
        """Make prediction and return structured output"""
        # Forward pass
        with torch.no_grad():
            logits = self.forward(model_input)
            probabilities = torch.sigmoid(logits)  # or softmax for multi-class
        
        # Create output object
        return ModelOutput(
            predictions=probabilities,
            probabilities=probabilities,
            logits=logits,
            attention_maps=None,
            feature_importance=None,
            input_shape=model_input.input_shape,
            in_channels=model_input.in_channels,
            clinical_features_size=model_input.clinical_features_size
        )
```

## Data Models

### Input Validation Schema

```python
# Shape validation functions
def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> bool:
    """Validate tensor matches expected shape (ignoring batch dimension)"""
    return tensor.shape[1:] == expected_shape

def validate_batch_consistency(tensors: List[torch.Tensor]) -> bool:
    """Validate all tensors have same batch size"""
    batch_sizes = [t.shape[0] for t in tensors if t is not None]
    return len(set(batch_sizes)) <= 1
```

### Configuration Schema

```python
class ModelConfig(BaseModel):
    """Configuration for model initialization"""
    
    # Architecture parameters
    input_shape: Tuple[int, int, int]
    in_channels: int
    output_size: int = 1
    clinical_features_size: Optional[int] = None
    
    # Model metadata
    model_name: str
    model_version: str = "1.0"
    architecture_type: str
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"
    
    # Validation
    @validator('input_shape')
    def validate_input_shape(cls, v):
        if len(v) != 3 or any(dim <= 0 for dim in v):
            raise ValueError("input_shape must be 3 positive integers (H, W, D)")
        return v
    
    @validator('in_channels')
    def validate_channels(cls, v):
        if v <= 0:
            raise ValueError("in_channels must be positive")
        return v
```

## Error Handling

### Validation Errors

```python
class ModelValidationError(Exception):
    """Raised when model input validation fails"""
    pass

class ShapeMismatchError(ModelValidationError):
    """Raised when tensor shapes don't match expected dimensions"""
    def __init__(self, expected_shape, actual_shape):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(f"Expected shape {expected_shape}, got {actual_shape}")
```

### Error Handling Strategy

1. **Input Validation**: Pydantic validators catch shape/type mismatches early
2. **Runtime Checks**: Additional validation in forward pass
3. **Clear Messages**: Detailed error messages with expected vs actual dimensions
4. **Graceful Degradation**: Optional fields allow partial functionality

## Testing Strategy

### Unit Tests

1. **Pydantic Validation Tests**
   - Test valid input creation
   - Test invalid shape rejection
   - Test optional field handling

2. **Base Model Tests**
   - Test parametric initialization
   - Test input validation
   - Test output structure

3. **Integration Tests**
   - Test full input → model → output pipeline
   - Test different model architectures
   - Test batch processing

### Test Data

```python
# Test fixtures
@pytest.fixture
def sample_imaging_input():
    return ImagingInput(
        imaging_data=torch.randn(2, 3, 96, 96, 96),
        batch_size=2,
        input_shape=(96, 96, 96),
        in_channels=3
    )

@pytest.fixture
def sample_multimodal_input():
    return MultimodalInput(
        imaging_data=torch.randn(2, 2, 96, 96, 96),
        clinical_features=torch.randn(2, 116),
        batch_size=2,
        input_shape=(96, 96, 96),
        in_channels=2,
        clinical_features_size=116
    )
```

## Implementation Notes

### Key Design Decisions

1. **Pydantic for Validation**: Provides automatic type checking and clear error messages
2. **PyTorch Lightning Base**: Ensures compatibility with existing training infrastructure
3. **Parametric Initialization**: Allows dynamic model configuration based on data
4. **Optional Fields**: Supports both imaging-only and multimodal models
5. **Metadata Inclusion**: Tracks model version and processing information

### Performance Considerations

1. **Lazy Validation**: Only validate shapes when needed
2. **Tensor Sharing**: Avoid unnecessary tensor copying
3. **Device Management**: Handle GPU/CPU transfers efficiently
4. **Batch Processing**: Support variable batch sizes

### Extensibility

1. **Inheritance**: Easy to create specialized input/output classes
2. **Custom Validators**: Add domain-specific validation rules
3. **Plugin Architecture**: Register new model types
4. **Configuration Files**: Support external configuration loading