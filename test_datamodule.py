#!/usr/bin/env python3
"""
Test script to verify AlzheimersDataModule functionality.
"""

import sys
sys.path.append('.')

try:
    from gazimed.models.lightning_module import AlzheimersDataModule
    print('Testing AlzheimersDataModule dataloaders...')
    
    # Initialize with correct database path
    dm = AlzheimersDataModule(data_dir='gazimed_database.db', batch_size=2, num_workers=0)
    dm.setup()
    
    # Test train dataloader
    print('\n--- Testing train dataloader ---')
    train_loader = dm.train_dataloader()
    if train_loader is not None:
        print('✓ Train dataloader created')
        try:
            batch = next(iter(train_loader))
            print(f'✓ Got batch with keys: {list(batch.keys())}')
            
            # Check expected keys from test requirements
            expected_keys = ['clinical_features', 'volumes', 'alzheimer_score']
            for key in expected_keys:
                if key in batch:
                    print(f'✓ Key "{key}" present in batch')
                    if hasattr(batch[key], 'shape'):
                        print(f'  Shape: {batch[key].shape}')
                else:
                    print(f'✗ Key "{key}" missing from batch')
                    
        except Exception as e:
            print(f'✗ Error getting batch: {e}')
    else:
        print('✗ Train dataloader is None')
    
    # Test validation dataloader
    print('\n--- Testing validation dataloader ---')
    val_loader = dm.val_dataloader()
    if val_loader is not None:
        print('✓ Validation dataloader created')
        try:
            batch = next(iter(val_loader))
            print(f'✓ Got validation batch with keys: {list(batch.keys())}')
        except Exception as e:
            print(f'✗ Error getting validation batch: {e}')
    else:
        print('✗ Validation dataloader is None')
        
    # Test test dataloader
    print('\n--- Testing test dataloader ---')
    test_loader = dm.test_dataloader()
    if test_loader is not None:
        print('✓ Test dataloader created')
        try:
            batch = next(iter(test_loader))
            print(f'✓ Got test batch with keys: {list(batch.keys())}')
        except Exception as e:
            print(f'✗ Error getting test batch: {e}')
    else:
        print('✗ Test dataloader is None')
        
    print('\n--- Testing predict dataloader ---')
    predict_loader = dm.predict_dataloader()
    if predict_loader is not None:
        print('✓ Predict dataloader created')
    else:
        print('✓ Predict dataloader is None (as expected - not implemented)')
        
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
