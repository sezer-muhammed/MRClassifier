"""
Pytest configuration and fixtures for gazimed tests.
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path

from gazimed.data.database import DatabaseManager, insert_subject


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        yield db_path
    finally:
        # Clean up
        try:
            os.unlink(db_path)
        except:
            pass


@pytest.fixture
def db_manager(temp_db):
    """Create a database manager with test data."""
    db_manager = DatabaseManager(temp_db)
    db_manager.create_tables()
    
    # Create sample subjects
    sample_subjects = []
    for i in range(20):
        subject_data = {
            "subject_id": f"TEST_{i:03d}",
            "mri_path": f"/fake/path/mri_{i:03d}.nii.gz",
            "pet_path": f"/fake/path/pet_{i:03d}.nii.gz",
            "age": 60 + np.random.normal(10, 5),
            "sex": "M" if i % 2 == 0 else "F",
            "alzheimer_score": np.random.beta(2, 2),  # Scores between 0-1
            "clinical_features": np.random.randn(116).tolist(),  # 116 features
            "dataset_source": "TEST_ADNI" if i < 10 else "TEST_OASIS"
        }
        sample_subjects.append(subject_data)
    
    # Insert subjects into database
    with db_manager.get_session() as session:
        for subject_data in sample_subjects:
            insert_subject(session, subject_data)
    
    return db_manager


@pytest.fixture
def small_db_manager(temp_db):
    """Create a database manager with minimal test data for faster tests."""
    db_manager = DatabaseManager(temp_db)
    db_manager.create_tables()
    
    # Create minimal sample subjects
    sample_subjects = []
    for i in range(6):
        subject_data = {
            "subject_id": f"SMALL_{i:03d}",
            "mri_path": f"/fake/path/mri_{i:03d}.nii.gz",
            "pet_path": f"/fake/path/pet_{i:03d}.nii.gz",
            "age": 60 + i * 5,
            "sex": "M" if i % 2 == 0 else "F",
            "alzheimer_score": 0.1 + (i * 0.15),  # Evenly distributed scores
            "clinical_features": [0.1 * i] * 116,  # Simple features
            "dataset_source": "TEST_SMALL"
        }
        sample_subjects.append(subject_data)
    
    # Insert subjects into database
    with db_manager.get_session() as session:
        for subject_data in sample_subjects:
            insert_subject(session, subject_data)
    
    return db_manager