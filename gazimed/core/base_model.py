"""Base PyTorch Lightning model class with parametric initialization."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import pytorch_lightning as pl
from .model_io import ModelInput, ModelOutput


class ModelValidationError(Exception):
    """Raised when model input validation fails."""
    pass


class ShapeMismatchError(ModelValidationError):
    """Raised when tensor shapes don't match expected dimensions."""
    
    def __init__(self, expected_shape: Tuple[int, ...], actual_shape: Tuple[int, ...]):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(f"Expected shape {expected_shape}, got {actual_shape}")


class BaseAIModel(pl.LightningModule, ABC):
    """Base class for all AI models with parametric initialization."""
    
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
        """Initialize base AI model with parametric configuration.
        
        Args:
            input_shape: Input tensor shape (H, W, D)
            in_channels: Number of input channels
            output_size: Size of output predictions
            clinical_features_size: Size of clinical features vector
            model_name: Name identifier for the model
            model_version: Version string for the model
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
            scheduler_type: Type of learning rate scheduler
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation_type: Type of activation function
            loss_type: Type of loss function
            label_smoothing: Label smoothing factor
            gradient_clip_val: Gradient clipping value
            early_stopping_patience: Patience for early stopping
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store configuration
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.output_size = output_size
        self.clinical_features_size = clinical_features_size
        self.model_name = model_name
        self.model_version = model_version
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        
        # Architecture parameters
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_type = activation_type
        
        # Loss parameters
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        # Regularization parameters
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize architecture (to be implemented by subclasses)
        self._build_architecture()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must have exactly 3 dimensions (H, W, D)")
        
        if any(dim <= 0 for dim in self.input_shape):
            raise ValueError("All dimensions in input_shape must be positive")
        
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        
        if self.clinical_features_size is not None and self.clinical_features_size <= 0:
            raise ValueError("clinical_features_size must be positive if provided")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
    
    @abstractmethod
    def _build_architecture(self):
        """Build model architecture - must be implemented by subclasses."""
        pass
    
    def _validate_input(self, model_input: ModelInput):
        """Validate input tensor shapes match model configuration.
        
        Args:
            model_input: Pydantic input object to validate
            
        Raises:
            ShapeMismatchError: If tensor shapes don't match expected dimensions
        """
        # Validate image tensor if present
        if model_input.image is not None:
            expected_shape = (model_input.batch_size, self.in_channels) + self.input_shape
            if model_input.image.shape != expected_shape:
                raise ShapeMismatchError(expected_shape, model_input.image.shape)
        
        # Validate features tensor if present
        if model_input.features is not None and self.clinical_features_size is not None:
            expected_shape = (model_input.batch_size, self.clinical_features_size)
            if model_input.features.shape != expected_shape:
                raise ShapeMismatchError(expected_shape, model_input.features.shape)
        
        # Validate configuration consistency
        if model_input.input_shape != self.input_shape:
            raise ModelValidationError(
                f"Input shape mismatch: model expects {self.input_shape}, "
                f"got {model_input.input_shape}"
            )
        
        if model_input.in_channels != self.in_channels:
            raise ModelValidationError(
                f"Channel count mismatch: model expects {self.in_channels}, "
                f"got {model_input.in_channels}"
            )
        
        if model_input.clinical_features_size != self.clinical_features_size:
            raise ModelValidationError(
                f"Clinical features size mismatch: model expects {self.clinical_features_size}, "
                f"got {model_input.clinical_features_size}"
            )
    
    def forward(self, model_input: ModelInput) -> torch.Tensor:
        """Forward pass with Pydantic input validation.
        
        Args:
            model_input: Validated input object containing tensors and metadata
            
        Returns:
            Raw model output tensor (logits)
            
        Raises:
            ShapeMismatchError: If input shapes don't match model configuration
            ModelValidationError: If input configuration is inconsistent
        """
        # Validate input
        self._validate_input(model_input)
        
        # Extract tensors
        imaging_data = model_input.image
        clinical_features = model_input.features
        
        # Forward pass (to be implemented by subclasses)
        return self._forward_impl(imaging_data, clinical_features)
    
    @abstractmethod
    def _forward_impl(self, imaging_data: Optional[torch.Tensor], 
                     clinical_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Actual forward implementation - must be implemented by subclasses.
        
        Args:
            imaging_data: Image tensor or None
            clinical_features: Clinical features tensor or None
            
        Returns:
            Raw model output tensor (logits)
        """
        pass
    
    def predict(self, model_input: ModelInput) -> ModelOutput:
        """Make prediction and return structured output.
        
        Args:
            model_input: Validated input object
            
        Returns:
            Structured output with predictions, probabilities, and metadata
        """
        # Forward pass
        with torch.no_grad():
            logits = self.forward(model_input)
            
            # Apply appropriate activation based on output size
            if self.output_size == 1:
                # Binary classification
                probabilities = torch.sigmoid(logits)
            else:
                # Multi-class classification
                probabilities = torch.softmax(logits, dim=-1)
        
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
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Select optimizer
        if self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        # Select scheduler
        if self.scheduler_type.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        elif self.scheduler_type.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        else:
            return optimizer