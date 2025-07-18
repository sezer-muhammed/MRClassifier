#!/usr/bin/env python3
"""
Simple test to check database connectivity and basic functionality.
"""

import sys
sys.path.append('.')

try:
    print("Testing database connectivity...")
    from gazimed.data.database import DatabaseManager
    
    # Test database connection
    db_manager = DatabaseManager('gazimed_database.db')
    print("✓ Database manager created")
    
    # Test getting subjects
    with db_manager.get_session() as session:
        from gazimed.data.database import Subject
        subjects = session.query(Subject).limit(5).all()
        print(f"✓ Found {len(subjects)} subjects in database")
        
        if subjects:
            subject = subjects[0]
            print(f"  Sample subject: {subject.subject_id}")
            print(f"  Clinical features type: {type(subject.clinical_features)}")
            print(f"  Clinical features is None: {subject.clinical_features is None}")
            if subject.clinical_features is not None:
                print(f"  Clinical features length: {len(subject.clinical_features) if hasattr(subject.clinical_features, '__len__') else 'N/A'}")
            print(f"  Alzheimer score: {subject.alzheimer_score}")
            
            # Test validation
            print(f"  Score validation: {subject.validate_score()}")
            print(f"  Clinical features validation: {subject.validate_clinical_features()}")
            print(f"  Paths validation: {subject.validate_paths()}")
    
    print("\n✓ Database test completed successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
