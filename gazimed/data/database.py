"""
Database models and connection utilities for the Alzheimer's detection system.

This module defines the SQLite database schema for storing:
- Subject information with MRI/PET paths and clinical features
- Model experiment results and metrics
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Text, Boolean, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()


class Subject(Base):
    """
    Subject table for storing patient/subject information.
    
    Stores paths to MRI and PET NIfTI files, clinical features,
    and Alzheimer's outcome scores.
    """
    __tablename__ = 'subjects'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Subject identification
    subject_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # File paths
    mri_path = Column(String(500), nullable=False)
    pet_path = Column(String(500), nullable=False)
    
    # Demographics
    age = Column(Float, nullable=True)
    sex = Column(String(10), nullable=True)
    
    # Clinical outcome (continuous score 0-1)
    alzheimer_score = Column(Float, nullable=False)
    
    # Target value from ROI data (the actual output value from the data files)
    target = Column(Float, nullable=True)
    
    # Clinical features (116 numerical features stored as JSON)
    clinical_features = Column(JSON, nullable=False)
    
    # Metadata
    acquisition_date = Column(DateTime, nullable=True)
    scanner_info = Column(JSON, nullable=True)
    dataset_source = Column(String(50), nullable=True)  # ADNI, OASIS-3, etc.
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    model_results = relationship("ModelResult", back_populates="subject")
    
    def __repr__(self):
        return f"<Subject(id={self.id}, subject_id='{self.subject_id}', score={self.alzheimer_score})>"
    
    def validate_paths(self) -> bool:
        """Validate that MRI and PET file paths exist."""
        return Path(self.mri_path).exists() and Path(self.pet_path).exists()
    
    def validate_clinical_features(self) -> bool:
        """Validate that clinical features contain exactly 116 numerical values."""
        if not isinstance(self.clinical_features, list):
            return False
        return len(self.clinical_features) == 116 and all(
            isinstance(x, (int, float)) for x in self.clinical_features
        )
    
    def validate_score(self) -> bool:
        """Validate that Alzheimer's score is between 0 and 1."""
        return 0.0 <= self.alzheimer_score <= 1.0


class ModelResult(Base):
    """
    Model results table for storing experiment results and metrics.
    
    Stores training/validation metrics, hyperparameters, and model
    performance for different experiments and cross-validation folds.
    """
    __tablename__ = 'model_results'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Experiment identification
    experiment_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    fold_number = Column(Integer, nullable=True)  # For cross-validation
    
    # Subject reference
    subject_id = Column(Integer, ForeignKey('subjects.id'), nullable=True)
    
    # Model configuration
    hyperparameters = Column(JSON, nullable=False)
    model_architecture = Column(String(100), nullable=False)
    
    # Training metrics
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    train_auc = Column(Float, nullable=True)
    val_auc = Column(Float, nullable=True)
    
    # Performance metrics
    auc_score = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    correlation = Column(Float, nullable=True)
    
    # Prediction results
    predicted_score = Column(Float, nullable=True)
    true_score = Column(Float, nullable=True)
    confidence_interval = Column(JSON, nullable=True)  # [lower, upper]
    
    # Model artifacts
    model_checkpoint_path = Column(String(500), nullable=True)
    attention_maps_path = Column(String(500), nullable=True)
    
    # Training details
    epoch = Column(Integer, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    gpu_memory_used_mb = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    subject = relationship("Subject", back_populates="model_results")
    
    def __repr__(self):
        return f"<ModelResult(id={self.id}, experiment='{self.experiment_name}', auc={self.auc_score})>"


class DatabaseManager:
    """
    Database connection and session management utility.
    
    Provides methods for creating connections, initializing schema,
    and managing database sessions.
    """
    
    def __init__(self, db_path: str = "gazimed_database.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all database tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database."""
        import shutil
        shutil.copy2(self.db_path, backup_path)
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic statistics about the database."""
        with self.get_session() as session:
            subject_count = session.query(Subject).count()
            result_count = session.query(ModelResult).count()
            
            return {
                "total_subjects": subject_count,
                "total_results": result_count,
                "unique_experiments": session.query(ModelResult.experiment_name).distinct().count()
            }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the database."""
        with self.get_session() as session:
            # Basic counts
            subject_count = session.query(Subject).count()
            result_count = session.query(ModelResult).count()
            
            # Subject statistics
            subjects = session.query(Subject).all()
            
            if not subjects:
                return {"error": "No subjects found in database"}
            
            # Demographics
            ages = [s.age for s in subjects if s.age is not None]
            sexes = [s.sex for s in subjects if s.sex is not None]
            scores = [s.alzheimer_score for s in subjects]
            sources = [s.dataset_source for s in subjects if s.dataset_source is not None]
            
            # Calculate statistics
            import numpy as np
            from collections import Counter
            
            stats = {
                "total_subjects": subject_count,
                "total_results": result_count,
                "demographics": {
                    "age_stats": {
                        "count": len(ages),
                        "mean": float(np.mean(ages)) if ages else None,
                        "std": float(np.std(ages)) if ages else None,
                        "min": float(np.min(ages)) if ages else None,
                        "max": float(np.max(ages)) if ages else None
                    },
                    "sex_distribution": dict(Counter(sexes)),
                    "missing_demographics": subject_count - len([s for s in subjects if s.age and s.sex])
                },
                "alzheimer_scores": {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "positive_rate": float(np.mean([s > 0.5 for s in scores]))
                },
                "dataset_sources": dict(Counter(sources)),
                "data_quality": {
                    "complete_demographics": len([s for s in subjects if s.age and s.sex]),
                    "valid_clinical_features": len([s for s in subjects if s.validate_clinical_features()]),
                    "valid_scores": len([s for s in subjects if s.validate_score()]),
                    "accessible_files": len([s for s in subjects if s.validate_paths()])
                }
            }
            
            return stats
    
    def create_data_splits(self, test_size: float = 0.2, val_size: float = 0.1, 
                          stratify_by: str = "alzheimer_score", random_state: int = 42) -> Dict[str, List[int]]:
        """
        Create train/validation/test splits for machine learning.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set  
            stratify_by: Column to stratify by ('alzheimer_score', 'dataset_source', or None)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' lists of subject IDs
        """
        with self.get_session() as session:
            subjects = session.query(Subject).all()
            
            if not subjects:
                return {"error": "No subjects found"}
            
            import numpy as np
            from sklearn.model_selection import train_test_split
            
            # Prepare data
            subject_ids = [s.id for s in subjects]
            
            if stratify_by == "alzheimer_score":
                # Binary stratification
                stratify_labels = [1 if s.alzheimer_score > 0.5 else 0 for s in subjects]
            elif stratify_by == "dataset_source":
                stratify_labels = [s.dataset_source for s in subjects]
            else:
                stratify_labels = None
            
            # Create splits
            if stratify_labels:
                # First split: train+val vs test
                train_val_ids, test_ids = train_test_split(
                    subject_ids, test_size=test_size, 
                    stratify=stratify_labels, random_state=random_state
                )
                
                # Second split: train vs val
                if val_size > 0:
                    train_val_labels = [stratify_labels[subject_ids.index(id)] for id in train_val_ids]
                    train_ids, val_ids = train_test_split(
                        train_val_ids, test_size=val_size/(1-test_size),
                        stratify=train_val_labels, random_state=random_state
                    )
                else:
                    train_ids, val_ids = train_val_ids, []
            else:
                # Random splits without stratification
                train_val_ids, test_ids = train_test_split(
                    subject_ids, test_size=test_size, random_state=random_state
                )
                
                if val_size > 0:
                    train_ids, val_ids = train_test_split(
                        train_val_ids, test_size=val_size/(1-test_size), random_state=random_state
                    )
                else:
                    train_ids, val_ids = train_val_ids, []
            
            return {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "split_info": {
                    "total_subjects": len(subjects),
                    "train_size": len(train_ids),
                    "val_size": len(val_ids),
                    "test_size": len(test_ids),
                    "stratify_by": stratify_by
                }
            }


# Utility functions for database operations
def insert_subject(session: Session, subject_data: Dict[str, Any]) -> Subject:
    """
    Insert a new subject into the database.
    
    Args:
        session: Database session
        subject_data: Dictionary containing subject information
        
    Returns:
        Created Subject instance
    """
    subject = Subject(**subject_data)
    session.add(subject)
    session.commit()
    session.refresh(subject)
    return subject


def insert_model_result(session: Session, result_data: Dict[str, Any]) -> ModelResult:
    """
    Insert a new model result into the database.
    
    Args:
        session: Database session
        result_data: Dictionary containing result information
        
    Returns:
        Created ModelResult instance
    """
    result = ModelResult(**result_data)
    session.add(result)
    session.commit()
    session.refresh(result)
    return result


def query_subjects_by_score_range(session: Session, min_score: float, max_score: float) -> List[Subject]:
    """
    Query subjects by Alzheimer's score range.
    
    Args:
        session: Database session
        min_score: Minimum score (inclusive)
        max_score: Maximum score (inclusive)
        
    Returns:
        List of Subject instances
    """
    return session.query(Subject).filter(
        Subject.alzheimer_score >= min_score,
        Subject.alzheimer_score <= max_score
    ).all()


def get_best_model_results(session: Session, experiment_name: str, metric: str = "auc_score") -> List[ModelResult]:
    """
    Get best model results for an experiment based on a metric.
    
    Args:
        session: Database session
        experiment_name: Name of the experiment
        metric: Metric to optimize (default: auc_score)
        
    Returns:
        List of ModelResult instances sorted by metric (descending)
    """
    return session.query(ModelResult).filter(
        ModelResult.experiment_name == experiment_name
    ).order_by(getattr(ModelResult, metric).desc()).all()


# Additional utility functions for data analysis and management

def get_subjects_by_dataset_source(session: Session, source: str) -> List[Subject]:
    """
    Get subjects from a specific dataset source.
    
    Args:
        session: Database session
        source: Dataset source (e.g., 'ADNI', 'GAZI')
        
    Returns:
        List of Subject instances
    """
    return session.query(Subject).filter(Subject.dataset_source == source).all()


def get_balanced_dataset(session: Session, threshold: float = 0.5) -> Dict[str, List[Subject]]:
    """
    Get balanced positive and negative samples based on Alzheimer's score threshold.
    
    Args:
        session: Database session
        threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        Dictionary with 'positive' and 'negative' subject lists
    """
    positive_subjects = session.query(Subject).filter(Subject.alzheimer_score > threshold).all()
    negative_subjects = session.query(Subject).filter(Subject.alzheimer_score <= threshold).all()
    
    return {
        'positive': positive_subjects,
        'negative': negative_subjects
    }


def get_subjects_with_complete_demographics(session: Session) -> List[Subject]:
    """
    Get subjects that have complete demographic information (age and sex).
    
    Args:
        session: Database session
        
    Returns:
        List of Subject instances with complete demographics
    """
    return session.query(Subject).filter(
        Subject.age.isnot(None),
        Subject.sex.isnot(None)
    ).all()


def get_clinical_features_statistics(session: Session) -> Dict[str, Any]:
    """
    Calculate statistics for clinical features across all subjects.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary containing feature statistics
    """
    subjects = session.query(Subject).all()
    
    if not subjects:
        return {}
    
    # Extract all clinical features
    all_features = []
    for subject in subjects:
        if subject.clinical_features:
            all_features.append(subject.clinical_features)
    
    if not all_features:
        return {}
    
    import numpy as np
    features_array = np.array(all_features)
    
    return {
        'num_subjects': len(subjects),
        'num_features': features_array.shape[1],
        'mean': np.mean(features_array, axis=0).tolist(),
        'std': np.std(features_array, axis=0).tolist(),
        'min': np.min(features_array, axis=0).tolist(),
        'max': np.max(features_array, axis=0).tolist(),
        'overall_mean': float(np.mean(features_array)),
        'overall_std': float(np.std(features_array))
    }


def validate_file_paths(session: Session) -> Dict[str, Any]:
    """
    Validate that all file paths in the database exist.
    
    Args:
        session: Database session
        
    Returns:
        Dictionary with validation results
    """
    subjects = session.query(Subject).all()
    
    mri_exists = 0
    pet_exists = 0
    both_exist = 0
    missing_files = []
    
    for subject in subjects:
        mri_valid = subject.validate_paths() and Path(subject.mri_path).exists()
        pet_valid = subject.validate_paths() and Path(subject.pet_path).exists()
        
        if mri_valid:
            mri_exists += 1
        if pet_valid:
            pet_exists += 1
        if mri_valid and pet_valid:
            both_exist += 1
        
        if not (mri_valid and pet_valid):
            missing_files.append({
                'subject_id': subject.subject_id,
                'mri_exists': mri_valid,
                'pet_exists': pet_valid,
                'mri_path': subject.mri_path,
                'pet_path': subject.pet_path
            })
    
    total_subjects = len(subjects)
    
    return {
        'total_subjects': total_subjects,
        'mri_files_exist': mri_exists,
        'pet_files_exist': pet_exists,
        'both_files_exist': both_exist,
        'mri_success_rate': mri_exists / total_subjects if total_subjects > 0 else 0,
        'pet_success_rate': pet_exists / total_subjects if total_subjects > 0 else 0,
        'complete_pairs_rate': both_exist / total_subjects if total_subjects > 0 else 0,
        'missing_files': missing_files[:10]  # Limit to first 10 for brevity
    }


def export_subjects_to_csv(session: Session, output_path: str, include_clinical_features: bool = False):
    """
    Export subjects data to CSV file.
    
    Args:
        session: Database session
        output_path: Path for output CSV file
        include_clinical_features: Whether to include clinical features columns
    """
    subjects = session.query(Subject).all()
    
    data = []
    for subject in subjects:
        row = {
            'subject_id': subject.subject_id,
            'mri_path': subject.mri_path,
            'pet_path': subject.pet_path,
            'age': subject.age,
            'sex': subject.sex,
            'alzheimer_score': subject.alzheimer_score,
            'target': subject.target,
            'dataset_source': subject.dataset_source,
            'created_at': subject.created_at
        }
        
        if include_clinical_features and subject.clinical_features:
            # Add clinical features as separate columns
            for i, feature_value in enumerate(subject.clinical_features):
                row[f'clinical_feature_{i:03d}'] = feature_value
        
        data.append(row)
    
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return len(data)


# Database initialization function
def initialize_database(db_path: str = "gazimed_database.db") -> DatabaseManager:
    """
    Initialize the database with tables and return manager instance.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        DatabaseManager instance
    """
    db_manager = DatabaseManager(db_path)
    db_manager.create_tables()
    return db_manager


if __name__ == "__main__":
    # Example usage and testing
    db_manager = initialize_database("test_gazimed.db")
    
    # Test database creation
    stats = db_manager.get_database_stats()
    print(f"Database initialized with {stats['total_subjects']} subjects and {stats['total_results']} results")
    
    # Example subject insertion
    with db_manager.get_session() as session:
        sample_subject = {
            "subject_id": "ADNI_001",
            "mri_path": "/data/mri/ADNI_001_T1.nii.gz",
            "pet_path": "/data/pet/ADNI_001_FDG.nii.gz",
            "age": 72.5,
            "sex": "F",
            "alzheimer_score": 0.75,
            "clinical_features": [0.1] * 116,  # 116 dummy features
            "dataset_source": "ADNI"
        }
        
        subject = insert_subject(session, sample_subject)
        print(f"Inserted subject: {subject}")
        
        # Example model result insertion
        sample_result = {
            "experiment_name": "swin_unetr_baseline",
            "model_version": "v1.0",
            "fold_number": 1,
            "subject_id": subject.id,
            "hyperparameters": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "patch_size": [4, 4, 4]
            },
            "model_architecture": "SwinUNETR",
            "auc_score": 0.92,
            "sensitivity": 0.88,
            "specificity": 0.89,
            "predicted_score": 0.73,
            "true_score": 0.75
        }
        
        result = insert_model_result(session, sample_result)
        print(f"Inserted result: {result}")