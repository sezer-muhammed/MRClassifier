"""
Training Configuration for Alzheimer's Detection Model

This module provides comprehensive training configuration utilities that implement
the requirements for task 6.3: Configure optimizers and mixed precision.

Key features:
- AdamW optimizer with specified learning rate and weight decay
- FP16 mixed precision training for memory efficiency
- Gradient clipping and accumulation support
- Proper trainer configuration for requirements 3.2 and 3.3
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    GradientAccumulationScheduler
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Dict, Any, Optional, List, Union
import os
from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """Configuration for AdamW optimizer as specified in requirements 3.2."""
    
    # Core optimizer parameters from requirements
    learning_rate: float = 1e-4  # 1×10⁻⁴ as per requirement 3.2
    weight_decay: float = 1e-2   # 1×10⁻² as per requirement 3.2
    
    # AdamW specific parameters
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    
    # Learning rate scheduler parameters
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6


@dataclass
class MixedPrecisionConfig:
    """Configuration for FP16 mixed precision training."""
    
    # Mixed precision settings
    precision: Union[int, str] = 16  # Use FP16 for memory efficiency
    amp_backend: str = 'native'      # Use native PyTorch AMP
    amp_level: Optional[str] = None  # For APEX (not used with native)
    
    # Gradient scaling for mixed precision
    gradient_clip_val: Optional[float] = 1.0
    gradient_clip_algorithm: str = 'norm'  # 'norm' or 'value'


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation as specified in requirement 3.3."""
    
    # Gradient accumulation settings from requirements
    accumulate_grad_batches: int = 8  # 8 steps as per requirement 3.3
    
    # Dynamic gradient accumulation (optional)
    gradient_accumulation_schedule: Optional[Dict[int, int]] = None
    
    def __post_init__(self):
        """Set up default gradient accumulation schedule if not provided."""
        if self.gradient_accumulation_schedule is None:
            # Default: accumulate 8 batches throughout training
            self.gradient_accumulation_schedule = {0: self.accumulate_grad_batches}


@dataclass
class TrainingConfig:
    """Comprehensive training configuration for Alzheimer's detection model."""
    
    # Core training parameters
    max_epochs: int = 100
    batch_size: int = 4  # Batch size 4 as per requirement 3.3
    num_workers: int = 4
    
    # Configuration components
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    mixed_precision_config: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    gradient_config: GradientAccumulationConfig = field(default_factory=GradientAccumulationConfig)
    
    # Hardware settings
    accelerator: str = 'gpu'
    devices: Union[int, List[int]] = 1
    strategy: Optional[str] = None  # 'ddp' for multi-GPU
    
    # Logging and checkpointing
    log_every_n_steps: int = 10
    val_check_interval: Union[int, float] = 1.0  # Check validation every epoch
    check_val_every_n_epoch: int = 1
    
    # Early stopping configuration
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_auc'
    early_stopping_mode: str = 'max'
    early_stopping_min_delta: float = 0.001
    
    # Model checkpointing
    checkpoint_monitor: str = 'val_auc'
    checkpoint_mode: str = 'max'
    save_top_k: int = 3
    save_last: bool = True
    
    # Reproducibility
    seed: Optional[int] = 42
    deterministic: bool = True
    benchmark: bool = False  # Set to True for consistent input sizes


class TrainingConfigurationManager:
    """
    Manager class for configuring PyTorch Lightning trainer with all required settings.
    
    This class implements the complete configuration for task 6.3:
    - AdamW optimizer with specified learning rate and weight decay
    - FP16 mixed precision training for memory efficiency  
    - Gradient clipping and accumulation support
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize training configuration manager.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        
    def create_trainer(
        self, 
        logger: Optional[Union[pl.loggers.Logger, List[pl.loggers.Logger]]] = None,
        callbacks: Optional[List[pl.Callback]] = None,
        **trainer_kwargs
    ) -> pl.Trainer:
        """
        Create PyTorch Lightning trainer with all required configurations.
        
        This method implements the complete trainer setup for requirements 3.2 and 3.3:
        - Mixed precision training (FP16)
        - Gradient accumulation (8 steps)
        - Gradient clipping (norm-based)
        - Proper logging and checkpointing
        
        Args:
            logger: Logger(s) for experiment tracking
            callbacks: Additional callbacks
            **trainer_kwargs: Additional trainer arguments
            
        Returns:
            Configured PyTorch Lightning trainer
        """
        # Set up reproducibility
        if self.config.seed is not None:
            pl.seed_everything(self.config.seed, workers=True)
        
        # Create default callbacks if not provided
        if callbacks is None:
            callbacks = self._create_default_callbacks()
        
        # Create default logger if not provided
        if logger is None:
            logger = self._create_default_logger()
        
        # Note: We use trainer's accumulate_grad_batches instead of GradientAccumulationScheduler
        # to avoid conflicts. The gradient accumulation is handled directly by the trainer.
        
        # Create trainer with all configurations
        trainer_args = {
            # Core training settings
            'max_epochs': self.config.max_epochs,
            
            # Hardware configuration
            'accelerator': self.config.accelerator,
            'devices': self.config.devices,
            
            # Mixed precision configuration (FP16 for memory efficiency)
            'precision': self.config.mixed_precision_config.precision,
            
            # Gradient configuration
            'gradient_clip_val': self.config.mixed_precision_config.gradient_clip_val,
            'gradient_clip_algorithm': self.config.mixed_precision_config.gradient_clip_algorithm,
            'accumulate_grad_batches': self.config.gradient_config.accumulate_grad_batches,
            
            # Validation and logging
            'log_every_n_steps': self.config.log_every_n_steps,
            'val_check_interval': self.config.val_check_interval,
            'check_val_every_n_epoch': self.config.check_val_every_n_epoch,
            
            # Reproducibility
            'deterministic': self.config.deterministic,
            'benchmark': self.config.benchmark,
            
            # Callbacks and logging
            'callbacks': callbacks,
            'logger': logger,
        }
        
        # Add strategy only if specified
        if self.config.strategy is not None:
            trainer_args['strategy'] = self.config.strategy
        
        # Add any additional trainer arguments
        trainer_args.update(trainer_kwargs)
        
        trainer = pl.Trainer(**trainer_args)
        
        return trainer
    
    def _create_default_callbacks(self) -> List[pl.Callback]:
        """Create default callbacks for training."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.checkpoint_monitor,
            mode=self.config.checkpoint_mode,
            save_top_k=self.config.save_top_k,
            save_last=self.config.save_last,
            filename='alzheimers-{epoch:02d}-{val_auc:.3f}',
            auto_insert_metric_name=False
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping_callback = EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            verbose=True
        )
        callbacks.append(early_stopping_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def _create_default_logger(self) -> pl.loggers.Logger:
        """Create default logger for training."""
        return TensorBoardLogger(
            save_dir='logs',
            name='alzheimers_detection',
            version=None,
            log_graph=True
        )
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """
        Get model initialization kwargs based on configuration.
        
        Returns:
            Dictionary of model initialization arguments
        """
        return {
            # Optimizer configuration
            'learning_rate': self.config.optimizer_config.learning_rate,
            'weight_decay': self.config.optimizer_config.weight_decay,
            'cosine_t_max': self.config.optimizer_config.cosine_t_max,
            
            # Mixed precision and gradient settings
            'use_mixed_precision': self.config.mixed_precision_config.precision == 16,
            'gradient_clip_val': self.config.mixed_precision_config.gradient_clip_val,
            'gradient_clip_algorithm': self.config.mixed_precision_config.gradient_clip_algorithm,
            'accumulate_grad_batches': self.config.gradient_config.accumulate_grad_batches,
            
            # Optimizer specific settings
            'optimizer_betas': self.config.optimizer_config.betas,
            'optimizer_eps': self.config.optimizer_config.eps,
            'optimizer_amsgrad': self.config.optimizer_config.amsgrad,
        }
    
    def validate_configuration(self) -> bool:
        """
        Validate the training configuration.
        
        Returns:
            True if configuration is valid, raises ValueError otherwise
        """
        # Validate gradient accumulation (flexible batch size allowed)
        if self.config.gradient_config.accumulate_grad_batches != 8:
            raise ValueError(f"Gradient accumulation must be 8 steps as per requirement 3.3, got {self.config.gradient_config.accumulate_grad_batches}")
        
        # Validate optimizer settings
        if self.config.optimizer_config.learning_rate != 1e-4:
            raise ValueError(f"Learning rate must be 1e-4 as per requirement 3.2, got {self.config.optimizer_config.learning_rate}")
        
        if self.config.optimizer_config.weight_decay != 1e-2:
            raise ValueError(f"Weight decay must be 1e-2 as per requirement 3.2, got {self.config.optimizer_config.weight_decay}")
        
        # Validate mixed precision settings
        if self.config.mixed_precision_config.precision not in [16, '16', 'bf16']:
            raise ValueError(f"Mixed precision should use 16-bit precision for memory efficiency")
        
        # Validate hardware settings
        if self.config.accelerator == 'gpu' and not torch.cuda.is_available():
            raise ValueError("GPU accelerator specified but CUDA is not available")
        
        return True


def create_training_setup(
    model_class,
    data_module_class,
    config: Optional[TrainingConfig] = None,
    **model_kwargs
) -> tuple:
    """
    Create complete training setup with model, data module, and trainer.
    
    This function provides a convenient way to set up training with all
    requirements for task 6.3 properly configured.
    
    Args:
        model_class: Lightning module class
        data_module_class: Data module class
        config: Training configuration (uses default if None)
        **model_kwargs: Additional model initialization arguments
        
    Returns:
        Tuple of (model, data_module, trainer, config_manager)
    """
    # Use default configuration if not provided
    if config is None:
        config = TrainingConfig()
    
    # Create configuration manager
    config_manager = TrainingConfigurationManager(config)
    
    # Validate configuration
    config_manager.validate_configuration()
    
    # Get model kwargs from configuration
    model_init_kwargs = config_manager.get_model_kwargs()
    model_init_kwargs.update(model_kwargs)
    
    # Create model
    model = model_class(**model_init_kwargs)
    
    # Create data module
    data_module = data_module_class(batch_size=config.batch_size)
    
    # Create trainer
    trainer = config_manager.create_trainer()
    
    return model, data_module, trainer, config_manager


# Example usage and configuration presets
def get_development_config() -> TrainingConfig:
    """Get configuration for development/debugging."""
    return TrainingConfig(
        max_epochs=10,
        batch_size=4,
        early_stopping_patience=5,
        log_every_n_steps=5
    )


def get_production_config() -> TrainingConfig:
    """Get configuration for production training."""
    return TrainingConfig(
        max_epochs=100,
        batch_size=4,
        early_stopping_patience=10,
        log_every_n_steps=10,
        devices=1,  # Can be increased for multi-GPU
        strategy=None  # Set to 'ddp' for multi-GPU
    )


def get_multi_gpu_config(num_gpus: int = 2) -> TrainingConfig:
    """Get configuration for multi-GPU training."""
    return TrainingConfig(
        max_epochs=100,
        batch_size=4,  # Per GPU batch size
        devices=num_gpus,
        strategy='ddp',
        early_stopping_patience=10,
        log_every_n_steps=10
    )