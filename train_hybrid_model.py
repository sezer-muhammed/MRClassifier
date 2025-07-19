"""
Training Script for Hybrid Alzheimer's Model

This script trains the hybrid deep learning model that combines 3D brain imaging 
data (MRI + PET) with clinical features for Alzheimer's detection.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from gazimed.models import HybridAlzheimersModel
from gazimed.data.database import DatabaseManager
from gazimed.data.dataset import AlzheimersDataset, custom_collate_fn, DataSplitter


def setup_directories(experiment_name: str):
    """Setup experiment directories"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{experiment_name}_{timestamp}")
    
    # Create directories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def create_data_loaders(
    db_manager: DatabaseManager,
    target_size: tuple = (96, 109, 96),
    batch_size: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.2,
    num_workers: int = 0,
    random_state: int = 42
):
    """Create train and validation data loaders"""
    
    print("Creating data loaders...")
    
    # Create data splitter
    splitter = DataSplitter(db_manager, random_state=random_state)
    
    # Get train/val splits
    train_ids, val_ids , _= splitter.train_val_test_split(
        train_size=train_split,
        val_size=val_split,
        test_size=0.0,  # No test set for now
        stratify=True
    )
    
    print(f"Train subjects: {len(train_ids)}")
    print(f"Validation subjects: {len(val_ids)}")
    
    # Create datasets
    train_dataset = AlzheimersDataset(
        db_manager=db_manager,
        subject_ids=train_ids,
        load_volumes=True,
        cache_volumes=False,  # Disable caching to save memory
        target_size=target_size,
        validate_files=False
    )
    
    val_dataset = AlzheimersDataset(
        db_manager=db_manager,
        subject_ids=val_ids,
        load_volumes=True,
        cache_volumes=False,
        target_size=target_size,
        validate_files=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def create_model(target_size: tuple = (96, 109, 96), **kwargs):
    """Create and configure the hybrid model"""
    
    print("Creating hybrid model...")

    # Default model configuration
    model_config = {
        "target_size": target_size,
        "swin_config": {
            "in_channels": 2,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "dropout_path_rate": 0.1,
            "feature_size": 64
        },
        "clinical_dims": [116, 32, 16, 8],
        "fusion_dims": [72, 32, 16, 1],
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "optimizer_type": "adamw",
        "scheduler_type": "cosine",
        "loss_type": "bce",
        "dropout_rate": 0.1,
        "use_batch_norm": False,
        "fusion_strategy": "concatenate"
    }
    
    # Update with any provided kwargs
    model_config.update(kwargs)
    
    # Create model
    model = HybridAlzheimersModel(**model_config)
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"Model created with {summary['total_parameters']:,} parameters")
    print("Component breakdown:")
    for component, count in summary['component_parameters'].items():
        print(f"  - {component}: {count:,} parameters")
    
    return model


def setup_callbacks(exp_dir: Path, monitor_metric: str = "val_loss"):
    """Setup training callbacks"""
    
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename="hybrid-model-{epoch:02d}-{val_loss:.4f}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode="min",
        patience=15,
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_trainer(
    exp_dir: Path,
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    precision: str = "16-mixed"
):
    """Setup PyTorch Lightning trainer"""
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=exp_dir / "logs",
        name="hybrid_alzheimers",
        version=None
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(exp_dir)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def save_experiment_config(exp_dir: Path, config: dict):
    """Save experiment configuration"""
    config_path = exp_dir / "experiment_config.json"
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2, default=str)
    
    print(f"Experiment config saved to: {config_path}")


def train_model(
    experiment_name: str = "hybrid_alzheimers",
    target_size: tuple = (96, 109, 96),
    batch_size: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    train_split: float = 0.8,
    val_split: float = 0.2,
    num_workers: int = 0,
    precision: str = "16-mixed",
    accelerator: str = "auto",
    devices: str = "auto",
    random_state: int = 42
):
    """Main training function"""
    
    print("=" * 80)
    print(f"HYBRID ALZHEIMER'S MODEL TRAINING")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Target size: {target_size}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Precision: {precision}")
    print("=" * 80)
    
    # Setup experiment directory
    exp_dir = setup_directories(experiment_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Setup database and data loaders
    db_manager = DatabaseManager()
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        db_manager=db_manager,
        target_size=target_size,
        batch_size=batch_size,
        train_split=train_split,
        val_split=val_split,
        num_workers=num_workers,
        random_state=random_state
    )
    
    # Create model
    model = create_model(
        target_size=target_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup trainer
    trainer = setup_trainer(
        exp_dir=exp_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision
    )
    
    # Save experiment configuration
    config = {
        "experiment_name": experiment_name,
        "target_size": target_size,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_split": train_split,
        "val_split": val_split,
        "precision": precision,
        "accelerator": accelerator,
        "devices": devices,
        "random_state": random_state,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "model_parameters": model.get_model_summary()["total_parameters"]
    }
    save_experiment_config(exp_dir, config)
    
    # Start training
    print("\nStarting training...")
    print("=" * 80)
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Save final model
        final_model_path = exp_dir / "final_model.pth"
        model.save_model(str(final_model_path))
        
        # Print best metrics
        if trainer.callback_metrics:
            print("Best metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.4f}")
                else:
                    print(f"  {key}: {value}")
        
        return exp_dir, model, trainer
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        return exp_dir, model, trainer
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"TRAINING FAILED: {e}")
        print("=" * 80)
        raise


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Train Hybrid Alzheimer's Model")
    
    # Experiment settings
    parser.add_argument("--experiment-name", type=str, default="hybrid_alzheimers",
                       help="Name of the experiment")
    parser.add_argument("--target-size", type=int, nargs=3, default=[96, 109, 96],
                       help="Target size for input volumes (H W D)")
    
    # Training settings
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for training")
    parser.add_argument("--max-epochs", type=int, default=100,
                       help="Maximum number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                       help="Weight decay for regularization")
    
    # Data settings
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Fraction of data for validation")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loader workers")
    
    # Hardware settings
    parser.add_argument("--precision", type=str, default="16-mixed",
                       choices=["16-mixed", "32", "64"],
                       help="Training precision")
    parser.add_argument("--accelerator", type=str, default="auto",
                       help="Training accelerator")
    parser.add_argument("--devices", type=str, default="auto",
                       help="Training devices")
    
    # Other settings
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Convert target size to tuple
    target_size = tuple(args.target_size)
    
    # Start training
    exp_dir, model, trainer = train_model(
        experiment_name=args.experiment_name,
        target_size=target_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        val_split=args.val_split,
        num_workers=args.num_workers,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        random_state=args.random_state
    )
    
    print(f"\nExperiment completed. Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()