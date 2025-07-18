"""
Training Script with Mixed Precision and Gradient Accumulation

This script demonstrates the complete implementation of task 6.3:
- AdamW optimizer with specified learning rate and weight decay
- FP16 mixed precision training for memory efficiency
- Gradient clipping and accumulation support

Requirements implemented:
- 3.2: AdamW optimizer with learning rate 1×10⁻⁴, cosine decay, and weight decay 1×10⁻²
- 3.3: Batch size 4 with gradient accumulation of 8 steps
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os
from pathlib import Path

# Import our modules
from gazimed.models.lightning_module import AlzheimersLightningModule, AlzheimersDataModule
from gazimed.training.training_config import (
    TrainingConfig, 
    TrainingConfigurationManager,
    create_training_setup,
    get_production_config
)


def main():
    """Main training function with mixed precision and gradient accumulation."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Alzheimer\'s Detection Model with Mixed Precision')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for logs and checkpoints')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='Training precision (16 for mixed precision)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=8, help='Gradient accumulation steps (must be 8 per requirements)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (must be 1e-4 per requirements)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (must be 1e-2 per requirements)')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    if args.accumulate_grad_batches != 8:
        raise ValueError(f"Gradient accumulation must be 8 steps as per requirement 3.3, got {args.accumulate_grad_batches}")
    
    if args.learning_rate != 1e-4:
        raise ValueError(f"Learning rate must be 1e-4 as per requirement 3.2, got {args.learning_rate}")
    
    if args.weight_decay != 1e-2:
        raise ValueError(f"Weight decay must be 1e-2 as per requirement 3.2, got {args.weight_decay}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    print("=" * 80)
    print("ALZHEIMER'S DETECTION MODEL TRAINING")
    print("Task 6.3: Configure optimizers and mixed precision")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - AdamW optimizer: lr={args.learning_rate}, weight_decay={args.weight_decay}")
    print(f"  - Mixed precision: FP{args.precision}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.accumulate_grad_batches} steps")
    print(f"  - Gradient clipping: {args.gradient_clip_val}")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - GPUs: {args.gpus}")
    print("=" * 80)
    
    # Create training configuration
    config = TrainingConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        devices=args.gpus,
        seed=args.seed
    )
    
    # Update configuration with command line arguments
    config.optimizer_config.learning_rate = args.learning_rate
    config.optimizer_config.weight_decay = args.weight_decay
    config.mixed_precision_config.precision = args.precision
    config.mixed_precision_config.gradient_clip_val = args.gradient_clip_val
    config.gradient_config.accumulate_grad_batches = args.accumulate_grad_batches
    
    # Create configuration manager
    config_manager = TrainingConfigurationManager(config)
    
    # Validate configuration
    try:
        config_manager.validate_configuration()
        print("✓ Configuration validation passed")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
        return
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir / 'logs'),
        name='alzheimers_mixed_precision',
        version=None,
        log_graph=True
    )
    
    # Create callbacks
    callbacks = [
        # Model checkpointing - save best models based on validation AUC
        ModelCheckpoint(
            dirpath=str(output_dir / 'checkpoints'),
            filename='alzheimers-{epoch:02d}-{val_auc:.3f}',
            monitor='val_auc',
            mode='max',
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True
        ),
        
        # Early stopping based on validation AUC
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,
            min_delta=0.001,
            verbose=True
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        )
    ]
    
    # Create model with configuration
    model_kwargs = config_manager.get_model_kwargs()
    model = AlzheimersLightningModule(**model_kwargs)
    
    # Create data module
    data_module = AlzheimersDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with database connections
        pin_memory=True,
        persistent_workers=False  # Disabled when num_workers=0
    )
    
    # Create trainer with all configurations
    trainer = config_manager.create_trainer(
        logger=logger,
        callbacks=callbacks
    )
    
    # Print model summary
    print("\nModel Configuration:")
    print(model.get_model_summary())
    
    # Print trainer configuration
    print(f"\nTrainer Configuration:")
    print(f"  - Precision: {trainer.precision}")
    print(f"  - Gradient clip val: {trainer.gradient_clip_val}")
    print(f"  - Gradient clip algorithm: {trainer.gradient_clip_algorithm}")
    print(f"  - Accumulate grad batches: {trainer.accumulate_grad_batches}")
    print(f"  - Max epochs: {trainer.max_epochs}")
    print(f"  - Accelerator: {trainer.accelerator}")
    print(f"  - Devices: {trainer.num_devices}")
    
    # Verify mixed precision is enabled
    if trainer.precision == 16:
        print("✓ FP16 mixed precision training enabled for memory efficiency")
    else:
        print("⚠ Mixed precision not enabled - consider using --precision 16")
    
    # Verify gradient accumulation
    if trainer.accumulate_grad_batches == 8:
        print("✓ Gradient accumulation configured for 8 steps as per requirements")
    else:
        print(f"⚠ Gradient accumulation is {trainer.accumulate_grad_batches}, should be 8")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    # Start training
    try:
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=args.resume_from_checkpoint
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Print best model information
        if trainer.checkpoint_callback:
            best_model_path = trainer.checkpoint_callback.best_model_path
            best_model_score = trainer.checkpoint_callback.best_model_score
            print(f"Best model saved at: {best_model_path}")
            print(f"Best validation AUC: {best_model_score:.4f}")
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        raise
    
    # Test the model if test data is available
    try:
        print("\n" + "=" * 80)
        print("RUNNING MODEL TESTING")
        print("=" * 80)
        
        test_results = trainer.test(model=model, datamodule=data_module)
        
        if test_results:
            print("Test Results:")
            for key, value in test_results[0].items():
                print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"Testing failed or no test data available: {e}")
    
    print("\n" + "=" * 80)
    print("TRAINING SCRIPT COMPLETED")
    print("=" * 80)


def demonstrate_mixed_precision_benefits():
    """
    Demonstrate the benefits of mixed precision training.
    
    This function shows memory usage comparison and training speed
    improvements when using FP16 mixed precision.
    """
    print("\n" + "=" * 60)
    print("MIXED PRECISION TRAINING BENEFITS")
    print("=" * 60)
    
    print("FP16 Mixed Precision Training provides:")
    print("1. Memory Efficiency:")
    print("   - Reduces memory usage by ~50% for model weights")
    print("   - Allows larger batch sizes or larger models")
    print("   - Enables training on GPUs with limited memory")
    
    print("\n2. Training Speed:")
    print("   - Faster computation on modern GPUs (V100, A100, RTX series)")
    print("   - Reduced memory bandwidth requirements")
    print("   - Automatic loss scaling prevents gradient underflow")
    
    print("\n3. Implementation Details:")
    print("   - Uses PyTorch native AMP (Automatic Mixed Precision)")
    print("   - Automatic gradient scaling and unscaling")
    print("   - Maintains FP32 precision for loss computation")
    print("   - Gradient clipping applied after unscaling")
    
    print("\n4. Requirements Compliance:")
    print("   - Task 6.3: Enable FP16 mixed precision training ✓")
    print("   - Requirement 3.2: AdamW optimizer configuration ✓")
    print("   - Requirement 3.3: Gradient accumulation support ✓")


if __name__ == '__main__':
    # Demonstrate mixed precision benefits
    demonstrate_mixed_precision_benefits()
    
    # Run main training
    main()