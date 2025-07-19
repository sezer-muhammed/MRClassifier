"""
Training script for Comprehensive Hybrid Model (MR + PET + Clinical)
Compatible with your existing dataloader and experiment setup.
"""
import os
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from gazimed.data.database import DatabaseManager
from gazimed.data.dataset import create_data_loaders, DataSplitter
from gazimed.models_v2.comprehensive_hybrid_model import ComprehensiveHybridModel

def setup_directories(experiment_name: str):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{experiment_name}_{timestamp}")
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    return exp_dir

def setup_callbacks(exp_dir: Path, monitor_metric: str = "val_loss"):
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename="comprehensive-model-{epoch:02d}-{val_loss:.4f}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=50,
            verbose=True,
            min_delta=0.001
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar()
    ]
    return callbacks

def setup_trainer(exp_dir: Path, max_epochs: int = 100, accelerator: str = "auto", devices: str = "auto", precision: str = "16-mixed"):
    logger = TensorBoardLogger(
        save_dir=exp_dir / "logs",
        name="comprehensive_hybrid",
        version=None
    )
    callbacks = setup_callbacks(exp_dir)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=2.0,
        accumulate_grad_batches=2,
        log_every_n_steps=3,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Train Comprehensive Hybrid Model")
    parser.add_argument("--experiment-name", type=str, default="comprehensive_hybrid", help="Experiment name")
    parser.add_argument("--target-size", type=int, nargs=3, default=[96, 109, 96], help="Input volume size (H W D)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--val-split", type=float, default=0.2, help="Val split fraction")
    parser.add_argument("--num-workers", type=int, default=0, help="Num dataloader workers")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-mixed", "32", "64"], help="Precision")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--devices", type=str, default="auto", help="Devices")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--backbone-type", type=str, default="unetr", choices=["unetr", "cnn"], help="Backbone type")
    args = parser.parse_args()

    target_size = tuple(args.target_size)
    exp_dir = setup_directories(args.experiment_name)
    db_manager = DatabaseManager()
    # Use DataSplitter to get train/val IDs
    splitter = DataSplitter(db_manager, random_state=args.random_state)
    train_ids, val_ids, _ = splitter.train_val_test_split(
        train_size=args.train_split,
        val_size=args.val_split,
        test_size=0.0,
        stratify=True
    )
    data_loaders = create_data_loaders(
        db_manager=db_manager,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=target_size
    )
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    model = ComprehensiveHybridModel(
        target_size=target_size,
        backbone_type=args.backbone_type,
        feature_dim=96,
        clinical_hidden=[64, 32, 16],
        fusion_hidden=[128, 64, 32, 1],
        dropout=0.1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    trainer = setup_trainer(
        exp_dir=exp_dir,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision
    )
    print(f"Starting training... Experiment dir: {exp_dir}")
    trainer.fit(model, train_loader, val_loader)
    print(f"Training complete. Best model and logs saved to: {exp_dir}")

if __name__ == "__main__":
    main()
