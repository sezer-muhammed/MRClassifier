"""Pydantic input/output objects for AI models with validation."""

from typing import Optional, Tuple
import torch
from pydantic import BaseModel, validator, Field


class ModelInput(BaseModel):
    """Base input class for all AI models with tensor validation."""
    
    # Core tensor fields
    image: Optional[torch.Tensor] = Field(None, description="Imaging tensor data")
    features: Optional[torch.Tensor] = Field(None, description="Clinical features tensor")
    
    # Shape metadata
    batch_size: int = Field(..., description="Batch size for input tensors")
    input_shape: Tuple[int, int, int] = Field(..., description="Input shape (H, W, D)")
    in_channels: int = Field(..., description="Number of input channels")
    clinical_features_size: Optional[int] = Field(None, description="Size of clinical features vector")
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v
    
    @validator('input_shape')
    def validate_input_shape(cls, v):
        """Validate input shape has 3 positive dimensions."""
        if len(v) != 3:
            raise ValueError("input_shape must have exactly 3 dimensions (H, W, D)")
        if any(dim <= 0 for dim in v):
            raise ValueError("All dimensions in input_shape must be positive")
        return v
    
    @validator('in_channels')
    def validate_in_channels(cls, v):
        """Validate number of channels is positive."""
        if v <= 0:
            raise ValueError("in_channels must be positive")
        return v
    
    @validator('clinical_features_size')
    def validate_clinical_features_size(cls, v):
        """Validate clinical features size is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("clinical_features_size must be positive if provided")
        return v
    
    @validator('image')
    def validate_image_shape(cls, v, values):
        """Validate image tensor shape matches expected dimensions."""
        if v is not None:
            # Check if required fields are available
            if 'batch_size' not in values or 'in_channels' not in values or 'input_shape' not in values:
                return v  # Skip validation if dependencies not available
            
            expected_shape = (values['batch_size'], values['in_channels']) + values['input_shape']
            if v.shape != expected_shape:
                raise ValueError(
                    f"Image tensor shape {v.shape} doesn't match expected {expected_shape}. "
                    f"Expected: (batch_size={values['batch_size']}, "
                    f"in_channels={values['in_channels']}, "
                    f"H={values['input_shape'][0]}, "
                    f"W={values['input_shape'][1]}, "
                    f"D={values['input_shape'][2]})"
                )
            
            # Validate tensor type
            if not isinstance(v, torch.Tensor):
                raise ValueError("image must be a torch.Tensor")
        
        return v
    
    @validator('features')
    def validate_features_shape(cls, v, values):
        """Validate features tensor shape matches expected dimensions."""
        if v is not None:
            # Check if required fields are available
            if 'batch_size' not in values:
                return v  # Skip validation if dependencies not available
            
            # If clinical_features_size is specified, validate against it
            if 'clinical_features_size' in values and values['clinical_features_size'] is not None:
                expected_shape = (values['batch_size'], values['clinical_features_size'])
                if v.shape != expected_shape:
                    raise ValueError(
                        f"Features tensor shape {v.shape} doesn't match expected {expected_shape}. "
                        f"Expected: (batch_size={values['batch_size']}, "
                        f"clinical_features_size={values['clinical_features_size']})"
                    )
            else:
                # If no clinical_features_size specified, just check batch dimension
                if len(v.shape) != 2 or v.shape[0] != values['batch_size']:
                    raise ValueError(
                        f"Features tensor must have shape (batch_size, features) where "
                        f"batch_size={values['batch_size']}, got shape {v.shape}"
                    )
            
            # Validate tensor type
            if not isinstance(v, torch.Tensor):
                raise ValueError("features must be a torch.Tensor")
        
        return v


class ModelOutput(BaseModel):
    """Base output class for all AI models with structured prediction results."""
    
    # Core prediction fields
    predictions: torch.Tensor = Field(..., description="Main prediction tensor")
    probabilities: Optional[torch.Tensor] = Field(None, description="Probability scores")
    logits: Optional[torch.Tensor] = Field(None, description="Raw logit values")
    
    # Optional explanation fields
    attention_maps: Optional[torch.Tensor] = Field(None, description="Attention visualization maps")
    feature_importance: Optional[torch.Tensor] = Field(None, description="Feature importance scores")
    
    # Input shape metadata (for reference)
    input_shape: Tuple[int, int, int] = Field(..., description="Original input shape")
    in_channels: int = Field(..., description="Number of input channels")
    clinical_features_size: Optional[int] = Field(None, description="Size of clinical features")
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @validator('predictions')
    def validate_predictions(cls, v):
        """Validate predictions tensor."""
        if not isinstance(v, torch.Tensor):
            raise ValueError("predictions must be a torch.Tensor")
        if v.numel() == 0:
            raise ValueError("predictions tensor cannot be empty")
        return v
    
    @validator('probabilities')
    def validate_probabilities(cls, v):
        """Validate probabilities are in valid range."""
        if v is not None:
            if not isinstance(v, torch.Tensor):
                raise ValueError("probabilities must be a torch.Tensor")
            if torch.any(v < 0) or torch.any(v > 1):
                raise ValueError("probabilities must be in range [0, 1]")
        return v