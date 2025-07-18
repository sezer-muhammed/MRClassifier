#!/usr/bin/env python3
"""
Step-by-step test to identify where the DataModule is hanging.
"""

import sys
sys.path.append('.')

try:
    print("Step 1: Import modules...")
    from gazimed.models.lightning_module import AlzheimersDataModule
    print("✓ AlzheimersDataModule imported")
    
    print("\nStep 2: Initialize DataModule...")
    dm = AlzheimersDataModule(
        data_dir='gazimed_database.db', 
        batch_size=1, 
        num_workers=0,  # Use 0 workers to avoid multiprocessing issues
        pin_memory=False  # Disable pin_memory for debugging
    )
    print("✓ DataModule initialized")
    
    print("\nStep 3: Setup DataModule...")
    dm.setup()
    print("✓ Setup completed")
    
    print("\nStep 4: Get train dataloader...")
    train_loader = dm.train_dataloader()
    print(f"✓ Train loader created: {train_loader is not None}")
    
    if train_loader:
        print(f"  Dataset size: {len(train_loader.dataset)}")
        print(f"  Batch size: {train_loader.batch_size}")
        
        print("\nStep 5: Create iterator...")
        train_iter = iter(train_loader)
        print("✓ Iterator created")
        
        print("\nStep 6: Get first batch (this is where it might hang)...")
        import time
        start_time = time.time()
        
        try:
            batch = next(train_iter)
            elapsed = time.time() - start_time
            print(f"✓ First batch retrieved in {elapsed:.2f} seconds")
            print(f"  Batch keys: {list(batch.keys())}")
            
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")
                    
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Error after {elapsed:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
