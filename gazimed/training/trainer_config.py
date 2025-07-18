"""
Training Configuration for Alzheimer's Detection Model

This module provides configuration classes and utilities for setting up
PyTorch Lightning training with mixed precision, gradient accumulation,
and other advanced training features.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    GradientAccumulationScheduler
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


@dataclass
class TrainingConfig:
    """
    Configuration class for training the Alzheimer's detection model.
    
    This class encapsulates all training-related hyperparameters and settings
    as specified in requirements 3.2 and 3.3.
    """
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 4
    accumulate_grad_batches: int = 8  # Effective batch size = 4 * 8 = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    
    # Mixed precision training (FP16)
    precision: str = '16-mixed'  # Enable FP16 mixed precision
    
    # Gradient clipping
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    
    # Learning rate scheduling
    cosine_t_max: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_auc'
    early_stopping_mode: str = 'max'
    early_stopping_min_delta: float = 0.001
    
    # Model checkpointing
    checkpoint_monitor: str = 'val_auc'
    checkpoint_mode: str = 'max'
    save_top_k: int = 3
    save_last: bool = True
    
    # Hardware settings
    accelerator: str = 'gpu'
    devices: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging
    log_every_n_steps: int = 10
    logger_name: str = 'alzheimers_detection'
    
    # Validation
    val_check_interval: float = 1.0  # Check validation every epoch
    check_val_every_n_epoch: int = 1
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def to_trainer_kwargs(self) -> Dict[str, Any]:
        """Convert configuration to PyTorch Lightning Trainer kwargs."""
        return {
            'max_epochs': self.max_epochs,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'precision': self.precision,
            'gradient_clip_val': self.gradient_clip_val,
            'gradient_clip_algorithm': self.gradient_clip_algorithm,
            'accelerator': self.accelerator,
            'devices': self.devices,
            'log_every_n_steps': self.log_every_n_steps,
            'val_check_interval': self.val_check_interval,
            'check_val_every_n_epoch': self.check_val_every_n_epoch,
            'deterministic': self.deterministic,
            'enable_progress_bar': True,
            'enable_model_summary': True,
        }


class TrainerFactory:
    """
    Factory class for creating configured PyTorch Lightning trainers.
    
    This factory creates trainers with all the necessary callbacks and
    configurations for training the Alzheimer's detection model.
    """
    
    @staticmethod
    def create_trainer(
        config: TrainingConfig,
        experiment_name: str = "alzheimers_detection",
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "alzheimers-detection"
    ) -> pl.Trainer:
        """
        Create a configured PyTorch Lightning trainer.
        
        Args:
            config: Training configuration
            experiment_name: Name of the experiment
            log_dir: Directory for logging
            checkpoint_dir: Directory for checkpoints
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            
        Returns:
            Configured PyTorch Lightning trainer
        """
        # Set seed for reproducibility
        pl.seed_everything(config.seed, workers=True)
        
        # Create callbacks
        callbacks = TrainerFactory._create_callbacks(config, checkpoint_dir)
        
        # Create logger
        logger = TrainerFactory._create_logger(
            config, experiment_name, log_dir, use_wandb, wandb_project
        )
        
        # Create trainer
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            **config.to_trainer_kwargs()
        )
        
        return trainer
    
    @staticmethod
    def _create_callbacks(config: TrainingConfig, checkpoint_dir: str) -> List[pl.Callback]:
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{val_auc:.3f}',
            monitor=config.checkpoint_monitor,
            mode=config.checkpoint_mode,
            save_top_k=config.save_top_k,
            save_last=config.save_last,
            auto_insert_metric_name=False,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=config.early_stopping_monitor,
            mode=config.early_stopping_mode,
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        )
        callbacks.append(lr_monitor)
        
        # Gradient accumulation scheduler (if needed for dynamic accumulation)
        if hasattr(config, 'dynamic_accumulation') and config.dynamic_accumulation:
            grad_accum_scheduler = GradientAccumulationScheduler(
                scheduling={0: config.accumulate_grad_batches}
            )
            callbacks.append(grad_accum_scheduler)
        
        return callbacks
    
    @staticmethod
    def _create_logger(
        config: TrainingConfig,
        experiment_name: str,
        log_dir: str,
        use_wandb: bool,
        wandb_project: str
    ) -> pl.loggers.Logger:
        """Create experiment logger."""
        if use_wandb:
            try:
                import wandb
                logger = WandbLogger(
                    project=wandb_project,
                    name=experiment_name,
                    save_dir=log_dir,
                    log_model=True
                )
            except ImportError:
                print("Weights & Biases not installed, falling back to TensorBoard")
                logger = TensorBoardLogger(
                    save_dir=log_dir,
                    name=experiment_name,
                    version=None
                )
        else:
            logger = TensorBoardLogger(
                save_dir=log_dir,
                name=experiment_name,
                version=None
            )
        
        return logger


class MixedPrecisionConfig:
    """
    Configuration for mixed precision training.
    
    This class provides utilities for configuring FP16 mixed precision
    training as specified in requirements 3.2 and 3.3.
    """
    
    @staticmethod
    def get_precision_config(enable_fp16: bool = True) -> Dict[str, Any]:
        """
        Get mixed precision configuration.
        
        Args:
            enable_fp16: Whether to enable FP16 mixed precision
            
        Returns:
            Dictionary with precision configuration
        """
        if enable_fp16 and torch.cuda.is_available():
            return {
                'precision': '16-mixed',
                'enable_checkpointing': True,  # Gradient checkpointing for memory efficiency
            }
        else:
            return {
                'precision': '32-true',
                'enable_checkpointing': False,
            }
    
    @staticmethod
    def configure_model_for_mixed_precision(model: torch.nn.Module) -> torch.nn.Module:
        """
        Configure model for mixed precision training.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model configured for mixed precision
        """
        # Enable mixed precision optimizations
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        return model


class GradientAccumulationConfig:
    """
    Configuration for gradient accumulation.
    
    This class provides utilities for configuring gradient accumulation
    to achieve larger effective batch sizes with limited GPU memory.
    """
    
    @staticmethod
    def calculate_effective_batch_size(
        batch_size: int, 
        accumulate_grad_batches: int, 
        num_gpus: int = 1
    ) -> int:
        """
        Calculate effective batch size with gradient accumulation.
        
        Args:
            batch_size: Batch size per GPU
            accumulate_grad_batches: Number of batches to accumulate
            num_gpus: Number of GPUs
            
        Returns:
            Effective batch size
        """
        return batch_size * accumulate_grad_batches * num_gpus
    
    @staticmethod
    def get_accumulation_schedule(
        total_epochs: int,
        initial_accumulation: int = 8,
        final_accumulation: int = 4
    ) -> Dict[int, int]:
        """
        Get gradient accumulation schedule.
        
        Args:
            total_epochs: Total number of training epochs
            initial_accumulation: Initial accumulation steps
            final_accumulation: Final accumulation steps
            
        Returns:
            Dictionary mapping epoch to accumulation steps
        """
        schedule = {}
        
        # Start with higher accumulation, reduce over time
        transition_epoch = total_epochs // 2
        
        for epoch in range(total_epochs):
            if epoch < transition_epoch:
                schedule[epoch] = initial_accumulation
            else:
                schedule[epoch] = final_accumulation
        
        return schedule


# Example usage and default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    max_epochs=100,
    batch_size=4,
    accumulate_grad_batches=8,
    learning_rate=1e-4,
    weight_decay=1e-2,
    precision='16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    early_stopping_patience=10,
    early_stopping_monitor='val_auc',
    checkpoint_monitor='val_auc',
    accelerator='gpu',
    devices=1,
    seed=42
)

# Configuration for different training scenarios
FAST_DEV_CONFIG = TrainingConfig(
    max_epochs=10,
    batch_size=2,
    accumulate_grad_batches=4,
    learning_rate=1e-3,
    weight_decay=1e-2,
    precision='32-true',  # Faster for development
    gradient_clip_val=1.0,
    early_stopping_patience=5,
    val_check_interval=0.5,  # Check validation twice per epoch
    seed=42
)

PRODUCTION_CONFIG = TrainingConfig(
    max_epochs=200,
    batch_size=4,
    accumulate_grad_batches=8,
    learning_rate=1e-4,
    weight_decay=1e-2,
    precision='16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    early_stopping_patience=15,
    early_stopping_monitor='val_auc',
    checkpoint_monitor='val_auc',
    save_top_k=5,
    accelerator='gpu',
    devices=1,
    seed=42,
    deterministic=True
)