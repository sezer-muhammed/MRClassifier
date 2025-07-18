#!/usr/bin/env python3
"""
Check image shapes in the dataset to verify consistency.
"""

import sys
sys.path.append('.')

try:
    from gazimed.data.database import DatabaseManager, Subject
    from gazimed.data.dataset import AlzheimersDataset
    import numpy as np
    
    print("Checking image shapes in dataset...")
    
    # Initialize database
    db_manager = DatabaseManager('gazimed_database.db')
    
    # Get a sample of subjects to check shapes
    with db_manager.get_session() as session:
        subjects = session.query(Subject).limit(10).all()
        print(f"Checking shapes for {len(subjects)} subjects...")
        
        shapes = []
        for i, subject in enumerate(subjects):
            try:
                # Create a minimal dataset to load volumes
                dataset = AlzheimersDataset(
                    db_manager=db_manager,
                    subject_ids=[subject.subject_id],
                    load_volumes=True,
                    cache_volumes=False
                )
                
                # Get the sample
                sample = dataset[0]
                if sample is not None and 'volumes' in sample:
                    volume_shape = sample['volumes'].shape
                    shapes.append(volume_shape)
                    print(f"Subject {subject.subject_id}: {volume_shape}")
                else:
                    print(f"Subject {subject.subject_id}: Failed to load")
                    
            except Exception as e:
                print(f"Subject {subject.subject_id}: Error - {e}")
        
        # Analyze shapes
        if shapes:
            unique_shapes = list(set(shapes))
            print(f"\nFound {len(unique_shapes)} unique shapes:")
            for shape in unique_shapes:
                count = shapes.count(shape)
                print(f"  {shape}: {count} subjects")
            
            if len(unique_shapes) == 1:
                print(f"\n✓ All images have consistent shape: {unique_shapes[0]}")
            else:
                print(f"\n⚠ Images have inconsistent shapes!")
        else:
            print("\n✗ No valid shapes found")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
