"""
Memory-Optimized Training Script for Hybrid Alzheimer's Model

This script is optimized for low-RAM systems and supports batch size 1.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import gc
import os

from gazimed.models import HybridAlzheimersModel
from gazimed.data.database import DatabaseManager
from gazimed.data.dataset import AlzheimersDataset, custom_collate_fn, DataSplitter


def memory_optimized_train():
    """Memory-optimized training function"""
    
    print("=" * 60)
    print("MEMORY-OPTIMIZED TRAINING - HYBRID ALZHEIMER'S MODEL")
    print("=" * 60)
    
    # Optimized configuration for 12GB RAM
    target_size = (96, 109, 96)
    batch_size = 4  # Increase batch size since we have 12GB RAM
    max_epochs = 50  # More epochs for better training
    accumulate_grad_batches = 2  # Effective batch size: 8
    
    print(f"Target size: {target_size}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {accumulate_grad_batches} (effective batch size: {batch_size * accumulate_grad_batches})")
    print(f"Max epochs: {max_epochs}")
    print("Using larger model for 12GB RAM system")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Setup database and data
    print("\nSetting up data...")
    db_manager = DatabaseManager()
    
    # Create data splitter
    splitter = DataSplitter(db_manager, random_state=42)
    splits = splitter.train_val_test_split(
        train_size=0.8,
        val_size=0.2,
        test_size=0.0,
        stratify=True
    )
    train_ids, val_ids, _ = splits
    
    print(f"Train subjects: {len(train_ids)}")
    print(f"Validation subjects: {len(val_ids)}")
    
    # Use smaller subset for memory optimization
    train_subset = train_ids[:20] if len(train_ids) > 20 else train_ids
    val_subset = val_ids[:10] if len(val_ids) > 10 else val_ids
    
    print(f"Using train subset: {len(train_subset)}")
    print(f"Using val subset: {len(val_subset)}")
    
    # Create datasets with memory optimization
    train_dataset = AlzheimersDataset(
        db_manager=db_manager,
        subject_ids=train_subset,
        load_volumes=True,
        cache_volumes=False,  # Disable caching to save memory
        target_size=target_size,
        validate_files=False
    )
    
    val_dataset = AlzheimersDataset(
        db_manager=db_manager,
        subject_ids=val_subset,
        load_volumes=True,
        cache_volumes=False,  # Disable caching to save memory
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
        num_workers=0,  # Use 0 workers to save memory
        collate_fn=custom_collate_fn,
        pin_memory=False,  # Disable pin_memory to save memory
        drop_last=False  # Don't drop last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=False,
        drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create memory-optimized model
    print("\nCreating memory-optimized model...")
    model = HybridAlzheimersModel(
        target_size=target_size,
        swin_config={
            "in_channels": 2,
            "feature_size": 64  # Reduced feature size
        },
        clinical_dims=[116, 32, 16, 8],  # Smaller clinical network
        fusion_dims=[72, 32, 16, 1],  # Smaller fusion network
        learning_rate=1e-3,  # Higher learning rate for faster convergence
        loss_type="bce",
        use_batch_norm=False,  # Disable batch norm for batch size 1
        use_simple_backbone=True  # Use simple 3D CNN instead of SwinUNet3D
    )
    
    summary = model.get_model_summary()
    print(f"Model parameters: {summary['total_parameters']:,}")
    print("Component breakdown:")
    for component, count in summary['component_parameters'].items():
        print(f"  - {component}: {count:,}")
    
    # Force garbage collection after model creation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Setup trainer with memory optimization
    print("\nSetting up memory-optimized trainer...")
    
    # Simple checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="memory_optimized_checkpoints",
        filename="memory-opt-model-{epoch:02d}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=False  # Don't save last to save disk space
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",  # Force CPU to avoid GPU memory issues
        devices=1,
        precision="16",  # Use FP32 for stability
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        enable_progress_bar=True,
        enable_model_summary=False,
        accumulate_grad_batches=accumulate_grad_batches,  # Gradient accumulation
        gradient_clip_val=1.0,  # Gradient clipping
        limit_train_batches=0.5,  # Use only 50% of training data to save time/memory
        limit_val_batches=1.0,  # Use all validation data
        check_val_every_n_epoch=2,  # Validate every 2 epochs to save time
        enable_checkpointing=True,
        logger=False  # Disable logging to save memory
    )
    
    # Test one batch first
    print("\nTesting one batch...")
    try:
        batch = next(iter(train_loader))
        print(f"Batch shapes:")
        print(f"  - Volumes: {batch['volumes'].shape}")
        print(f"  - Clinical: {batch['clinical_features'].shape}")
        print(f"  - Targets: {batch['alzheimer_score'].shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits = model(batch['volumes'], batch['clinical_features'])
            predictions = torch.sigmoid(logits)
        print(f"  - Logits: {logits.shape}, range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  - Predictions: {predictions.shape}, range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Test training step
        model.train()
        loss = model.training_step(batch, 0)
        print(f"  - Training loss: {loss.item():.4f}")
        
        print("✓ Batch test successful!")
        
        # Force garbage collection
        del batch, logits, predictions, loss
        gc.collect()
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False
    
    # Start training
    print("\nStarting memory-optimized training...")
    print("=" * 60)
    
    try:
        # Set environment variables for memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        trainer.fit(model, train_loader, val_loader)
        
        print("\n" + "=" * 60)
        print("MEMORY-OPTIMIZED TRAINING COMPLETED!")
        print("=" * 60)
        
        # Print final metrics
        if trainer.callback_metrics:
            print("Final metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.item():.4f}")
        
        # Save final model
        model.save_model("memory_optimized_final_model.pth")
        print("Model saved to: memory_optimized_final_model.pth")
        
        # Final cleanup
        del model, trainer, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Try reducing target_size or using even smaller model configuration.")
        return False


if __name__ == "__main__":
    print("Memory-Optimized Training Script")
    print("This script is designed for low-RAM systems.")
    print("It uses batch size 1 and gradient accumulation.")
    print()
    
    success = memory_optimized_train()
    if success:
        print("\n🎉 Memory-optimized training completed successfully!")
    else:
        print("\n⚠️ Memory-optimized training failed. Check the error messages above.")
        print("Consider further reducing model size or target dimensions.")