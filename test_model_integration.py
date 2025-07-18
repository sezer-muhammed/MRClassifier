#!/usr/bin/env python3
"""
Test script to verify model integration with actual data dimensions.
"""

import sys
sys.path.append('.')

try:
    import torch
    from gazimed.models.lightning_module import AlzheimersLightningModule, AlzheimersDataModule
    
    print("Testing model integration with actual data dimensions...")
    
    # Initialize DataModule
    dm = AlzheimersDataModule(data_dir='gazimed_database.db', batch_size=2, num_workers=0)
    dm.setup()
    
    # Get a sample batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Volumes shape: {batch['volumes'].shape}")
    print(f"Clinical features shape: {batch['clinical_features'].shape}")
    print(f"Alzheimer scores shape: {batch['alzheimer_score'].shape}")
    
    # Initialize model with updated dimensions
    model = AlzheimersLightningModule(
        img_size=(91, 109, 91),  # Updated to match actual data
        clinical_features_dim=116,  # Updated to match actual data
        in_channels=2,
        batch_size=2
    )
    
    print(f"\nModel initialized with:")
    print(f"  Image size: {model.hparams.img_size}")
    print(f"  Clinical features dim: {model.hparams.clinical_features_dim}")
    print(f"  Input channels: {model.hparams.in_channels}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            predictions = model(batch['volumes'], batch['clinical_features'])
            print(f"✓ Forward pass successful!")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            # Test training step
            print("\nTesting training step...")
            model.train()
            loss = model.training_step(batch, 0)
            print(f"✓ Training step successful!")
            print(f"  Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Model integration test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
