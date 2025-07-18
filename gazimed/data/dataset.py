"""
Dataset classes for Alzheimer's detection system.

This module provides PyTorch Dataset classes for loading MRI/PET data
with clinical features from the database, along with data splitting utilities
and cross-validation support.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sqlalchemy.orm import Session
from tqdm import tqdm

from gazimed.data.database import DatabaseManager, Subject


def custom_collate_fn(batch):
    """
    Custom collate function that handles None values and ensures proper batch structure.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary with properly collated tensors
    """
    # Filter out None samples
    valid_batch = [sample for sample in batch if sample is not None]
    
    if len(valid_batch) == 0:
        raise ValueError("All samples in batch are None")
    
    # Get the keys from the first valid sample
    keys = valid_batch[0].keys()
    
    collated = {}
    for key in keys:
        if key == 'metadata':
            # Handle metadata separately (don't collate)
            collated[key] = [sample[key] for sample in valid_batch]
        elif key == 'subject_id':
            # Handle subject IDs separately (don't collate)
            collated[key] = [sample[key] for sample in valid_batch]
        else:
            # Collate tensors
            values = [sample[key] for sample in valid_batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
    
    return collated

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. NIfTI loading will be disabled.")


class AlzheimersDataset(Dataset):
    """
    PyTorch Dataset for Alzheimer's detection with MRI/PET and clinical features.
    
    Loads subjects from database and provides MRI/PET volumes with clinical features
    and Alzheimer's scores for training/evaluation. Supports both volume loading
    and clinical-only modes for flexible usage.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        subject_ids: Optional[List[str]] = None,
        subject_filter: Optional[Dict[str, Any]] = None,
        transform=None,
        load_volumes: bool = True,
        cache_volumes: bool = False,
        include_difference_channel: bool = False,
        validate_files: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            db_manager: Database manager instance
            subject_ids: List of subject IDs to include (None for all)
            subject_filter: Additional filters for subject selection
            transform: Optional transforms to apply to volumes
            load_volumes: Whether to load NIfTI volumes (False for clinical-only)
            cache_volumes: Whether to cache loaded volumes in memory
            include_difference_channel: Whether to include PET-MRI difference channel
            validate_files: Whether to validate file paths exist
        """
        self.db_manager = db_manager
        self.transform = transform
        self.load_volumes = load_volumes
        self.cache_volumes = cache_volumes
        self.include_difference_channel = include_difference_channel
        self.validate_files = validate_files
        self.volume_cache = {}
        
        # Load subjects from database with filtering
        self.subjects = self._load_subjects(subject_ids, subject_filter)
        
        if len(self.subjects) == 0:
            raise ValueError("No valid subjects found in database")
    
    def _load_subjects(self, subject_ids: Optional[List[str]], 
                      subject_filter: Optional[Dict[str, Any]]) -> List[Subject]:
        """Load and filter subjects from database."""
        with self.db_manager.get_session() as session:
            query = session.query(Subject)
            
            # Apply subject ID filter
            if subject_ids is not None:
                query = query.filter(Subject.subject_id.in_(subject_ids))
            
            # Apply additional filters
            if subject_filter:
                if 'min_score' in subject_filter:
                    query = query.filter(Subject.alzheimer_score >= subject_filter['min_score'])
                if 'max_score' in subject_filter:
                    query = query.filter(Subject.alzheimer_score <= subject_filter['max_score'])
                if 'dataset_source' in subject_filter:
                    query = query.filter(Subject.dataset_source == subject_filter['dataset_source'])
                if 'min_age' in subject_filter:
                    query = query.filter(Subject.age >= subject_filter['min_age'])
                if 'max_age' in subject_filter:
                    query = query.filter(Subject.age <= subject_filter['max_age'])
                if 'sex' in subject_filter:
                    query = query.filter(Subject.sex == subject_filter['sex'])
            
            subjects = query.all()
        
        # Validate subjects
        valid_subjects = []
        for subject in tqdm(subjects, desc="Validating subjects"):
            if self._validate_subject(subject):
                valid_subjects.append(subject)
        
        return valid_subjects
    
    def _validate_subject(self, subject: Subject) -> bool:
        """Validate that subject has required data."""
        try:
            # Check Alzheimer's score
            if not subject.validate_score():
                return False
            
            # Check clinical features
            if not subject.validate_clinical_features():
                return False
            
            # Check that clinical_features is not None
            if subject.clinical_features is None:
                return False
            
            # Check file paths if loading volumes
            if self.load_volumes and not subject.validate_paths():
                return False
            
            return True
        except Exception as e:
            return False
    
    def _load_nifti_volume(self, file_path: str) -> np.ndarray:
        """Load NIfTI volume and return as numpy array."""
        if not NIBABEL_AVAILABLE:
            raise RuntimeError("nibabel is required for NIfTI loading but not available")
        
        try:
            nii_img = nib.load(file_path)
            volume = nii_img.get_fdata().astype(np.float32)
            return volume
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI file {file_path}: {e}") from e
    
    def _get_volumes(self, subject: Subject) -> np.ndarray:
        """Load and combine MRI and PET volumes for a subject."""
        cache_key = subject.subject_id
        
        if self.cache_volumes and cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        
        # Load individual volumes
        mri_volume = self._load_nifti_volume(subject.mri_path)
        pet_volume = self._load_nifti_volume(subject.pet_path)
        
        # Combine volumes into multi-channel tensor
        if self.include_difference_channel:
            # Create 3-channel volume: [MRI, PET, PET-MRI]
            diff_volume = pet_volume - mri_volume
            combined_volume = np.stack([mri_volume, pet_volume, diff_volume], axis=0)
        else:
            # Create 2-channel volume: [MRI, PET]
            combined_volume = np.stack([mri_volume, pet_volume], axis=0)
        
        if self.cache_volumes:
            self.volume_cache[cache_key] = combined_volume
        
        return combined_volume
    
    def __len__(self) -> int:
        return len(self.subjects)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'volumes': Combined MRI/PET tensor [2/3, H, W, D] (if load_volumes=True)
            - 'clinical_features': Clinical features tensor [116]
            - 'alzheimer_score': Target score [1]
            - 'subject_id': Subject identifier
            - 'metadata': Additional subject metadata
            
            Returns None if sample is invalid
        """
        try:
            subject = self.subjects[idx]
            
            # Double-check validation
            if not self._validate_subject(subject):
                return None
            
            # Ensure clinical_features is valid
            if subject.clinical_features is None:
                return None
            
            # Convert clinical features to numpy array first for validation
            try:
                clinical_features = np.array(subject.clinical_features, dtype=np.float32)
                if clinical_features.shape[0] != 116:
                    return None
            except Exception as e:
                return None
            
            sample = {
                'subject_id': subject.subject_id,
                'alzheimer_score': torch.tensor(subject.alzheimer_score, dtype=torch.float32),
                'clinical_features': torch.tensor(clinical_features, dtype=torch.float32),
                'metadata': {
                    'age': subject.age,
                    'sex': subject.sex,
                    'dataset_source': subject.dataset_source,
                    'acquisition_date': subject.acquisition_date
                }
            }
            
            if self.load_volumes:
                try:
                    # Get combined volumes (already stacked with proper channels)
                    combined_volume = self._get_volumes(subject)
                    sample['volumes'] = torch.tensor(combined_volume, dtype=torch.float32)
                except Exception as e:
                    return None
                
                # Apply transforms if provided
                if self.transform is not None:
                    try:
                        sample = self.transform(sample)
                    except Exception as e:
                        return None
            
            return sample
            
        except Exception as e:
            return None
    
    def get_subject_info(self, idx: int) -> Subject:
        """Get full subject information for given index."""
        return self.subjects[idx]
    
    def get_score_distribution(self) -> Dict[str, float]:
        """Get statistics about Alzheimer's score distribution."""
        scores = [s.alzheimer_score for s in self.subjects]
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }


class DataSplitter:
    """
    Utility class for splitting data into train/validation/test sets.
    
    Supports stratified splitting based on Alzheimer's scores and
    cross-validation fold generation.
    """
    
    def __init__(self, db_manager: DatabaseManager, random_state: int = 42):
        """
        Initialize data splitter.
        
        Args:
            db_manager: Database manager instance
            random_state: Random seed for reproducible splits
        """
        self.db_manager = db_manager
        self.random_state = random_state
    
    def _get_all_subject_ids(self) -> List[str]:
        """Get all subject IDs from database."""
        with self.db_manager.get_session() as session:
            subjects = session.query(Subject).all()
            return [s.subject_id for s in subjects if s.validate_score() and s.validate_clinical_features()]
    
    def _get_stratification_labels(self, subject_ids: List[str]) -> List[int]:
        """Convert continuous scores to discrete labels for stratification."""
        with self.db_manager.get_session() as session:
            subjects = session.query(Subject).filter(
                Subject.subject_id.in_(subject_ids)
            ).all()
            
            # Create binary stratification based on median score
            # This is more robust for small datasets
            scores = [s.alzheimer_score for s in subjects]
            median_score = np.median(scores)
            
            labels = []
            for subject in subjects:
                if subject.alzheimer_score <= median_score:
                    labels.append(0)  # Low risk
                else:
                    labels.append(1)  # High risk
            
            return labels
    
    def train_val_test_split(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: Optional[float] = 0.15,
        stratify: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set
            stratify: Whether to stratify split based on scores
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        if test_size is None:
            test_size = 0.0
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
        
        subject_ids = self._get_all_subject_ids()
        
        if stratify:
            labels = self._get_stratification_labels(subject_ids)
        else:
            labels = None
        
        # First split: separate test set (if test_size > 0)
        if test_size > 0:
            train_val_ids, test_ids = train_test_split(
                subject_ids,
                test_size=test_size,
                stratify=labels,
                random_state=self.random_state
            )
        else:
            train_val_ids = subject_ids
            test_ids = []
        
        # Second split: separate train and validation
        if stratify:
            train_val_labels = self._get_stratification_labels(train_val_ids)
        else:
            train_val_labels = None
        
        val_proportion = val_size / (train_size + val_size)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_proportion,
            stratify=train_val_labels,
            random_state=self.random_state
        )
        
        return train_ids, val_ids, test_ids
    
    def create_cv_folds(
        self,
        n_folds: int = 5,
        stratify: bool = True
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Create cross-validation folds.
        
        Args:
            n_folds: Number of folds
            stratify: Whether to stratify folds based on scores
            
        Returns:
            List of (train_ids, val_ids) tuples for each fold
        """
        subject_ids = self._get_all_subject_ids()
        
        if stratify:
            labels = self._get_stratification_labels(subject_ids)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = skf.split(subject_ids, labels)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            splits = kf.split(subject_ids)
        
        folds = []
        for train_idx, val_idx in splits:
            train_ids = [subject_ids[i] for i in train_idx]
            val_ids = [subject_ids[i] for i in val_idx]
            folds.append((train_ids, val_ids))
        
        return folds


class CrossValidationDataManager:
    """
    Manager for cross-validation data loading and fold management.
    
    Provides utilities for creating and managing cross-validation folds
    with proper data loading and caching strategies.
    """
    
    def __init__(self, db_manager: DatabaseManager, n_folds: int = 5, 
                 random_state: int = 42, stratify: bool = True):
        """
        Initialize cross-validation manager.
        
        Args:
            db_manager: Database manager instance
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducible splits
            stratify: Whether to stratify folds based on scores
        """
        self.db_manager = db_manager
        self.n_folds = n_folds
        self.random_state = random_state
        self.stratify = stratify
        self.splitter = DataSplitter(db_manager, random_state)
        
        # Create folds once during initialization
        self.folds = self.splitter.create_cv_folds(n_folds, stratify)
    
    def get_fold_data_loaders(
        self,
        fold_idx: int,
        batch_size: int = 4,
        num_workers: int = 4,
        train_transform=None,
        val_transform=None,
        load_volumes: bool = True,
        include_difference_channel: bool = False
    ) -> Dict[str, DataLoader]:
        """
        Get data loaders for a specific fold.
        
        Args:
            fold_idx: Index of the fold (0 to n_folds-1)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            train_transform: Transforms for training data
            val_transform: Transforms for validation data
            load_volumes: Whether to load NIfTI volumes
            include_difference_channel: Whether to include PET-MRI difference channel
            
        Returns:
            Dictionary with 'train' and 'val' DataLoaders
        """
        if fold_idx >= len(self.folds):
            raise ValueError(f"Fold index {fold_idx} out of range (0-{len(self.folds)-1})")
        
        train_ids, val_ids = self.folds[fold_idx]
        
        # Create datasets
        train_dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            subject_ids=train_ids,
            transform=train_transform,
            load_volumes=load_volumes,
            cache_volumes=False,
            include_difference_channel=include_difference_channel
        )
        
        val_dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            subject_ids=val_ids,
            transform=val_transform,
            load_volumes=load_volumes,
            cache_volumes=True,
            include_difference_channel=include_difference_channel
        )
        
        # Create data loaders
        return {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        }
    
    def get_fold_statistics(self, fold_idx: int) -> Dict[str, Any]:
        """Get statistics for a specific fold."""
        if fold_idx >= len(self.folds):
            raise ValueError(f"Fold index {fold_idx} out of range (0-{len(self.folds)-1})")
        
        train_ids, val_ids = self.folds[fold_idx]
        
        with self.db_manager.get_session() as session:
            # Get train subjects
            train_subjects = session.query(Subject).filter(
                Subject.subject_id.in_(train_ids)
            ).all()
            
            # Get validation subjects
            val_subjects = session.query(Subject).filter(
                Subject.subject_id.in_(val_ids)
            ).all()
            
            # Calculate statistics
            train_scores = [s.alzheimer_score for s in train_subjects]
            val_scores = [s.alzheimer_score for s in val_subjects]
            
            return {
                'fold_idx': fold_idx,
                'train_size': len(train_subjects),
                'val_size': len(val_subjects),
                'train_score_stats': {
                    'mean': np.mean(train_scores),
                    'std': np.std(train_scores),
                    'min': np.min(train_scores),
                    'max': np.max(train_scores)
                },
                'val_score_stats': {
                    'mean': np.mean(val_scores),
                    'std': np.std(val_scores),
                    'min': np.min(val_scores),
                    'max': np.max(val_scores)
                }
            }
    
    def get_all_fold_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all folds."""
        return [self.get_fold_statistics(i) for i in range(len(self.folds))]


def create_data_loaders(
    db_manager: DatabaseManager,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
    load_volumes: bool = True,
    include_difference_channel: bool = False,
    subject_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/validation/test sets.
    
    Args:
        db_manager: Database manager instance
        train_ids: Training subject IDs
        val_ids: Validation subject IDs
        test_ids: Test subject IDs (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_transform: Transforms for training data
        val_transform: Transforms for validation/test data
        load_volumes: Whether to load NIfTI volumes
        
    Returns:
        Dictionary of DataLoaders
    """
    # Create datasets
    train_dataset = AlzheimersDataset(
        db_manager=db_manager,
        subject_ids=train_ids,
        subject_filter=subject_filter,
        transform=train_transform,
        load_volumes=load_volumes,
        cache_volumes=False,  # Don't cache for training (memory intensive)
        include_difference_channel=include_difference_channel
    )
    
    val_dataset = AlzheimersDataset(
        db_manager=db_manager,
        subject_ids=val_ids,
        subject_filter=subject_filter,
        transform=val_transform,
        load_volumes=load_volumes,
        cache_volumes=True,  # Cache validation data
        include_difference_channel=include_difference_channel
    )
    
    # Create data loaders with custom collate function
    data_loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
    }
    
    # Add test loader if test IDs provided
    if test_ids is not None:
        test_dataset = AlzheimersDataset(
            db_manager=db_manager,
            subject_ids=test_ids,
            subject_filter=subject_filter,
            transform=val_transform,
            load_volumes=load_volumes,
            cache_volumes=True,
            include_difference_channel=include_difference_channel
        )
        
        data_loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
    
    return data_loaders


# Utility functions
def get_dataset_statistics(db_manager: DatabaseManager) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the dataset.
    
    Args:
        db_manager: Database manager instance
        
    Returns:
        Dictionary with dataset statistics
    """
    with db_manager.get_session() as session:
        subjects = session.query(Subject).all()
        
        # Filter valid subjects
        valid_subjects = [s for s in subjects if s.validate_score() and s.validate_clinical_features()]
        
        if not valid_subjects:
            return {"error": "No valid subjects found"}
        
        # Score statistics
        scores = [s.alzheimer_score for s in valid_subjects]
        
        # Age statistics (excluding None values)
        ages = [s.age for s in valid_subjects if s.age is not None]
        
        # Sex distribution
        sex_counts = {}
        for s in valid_subjects:
            if s.sex:
                sex_counts[s.sex] = sex_counts.get(s.sex, 0) + 1
        
        # Dataset source distribution
        source_counts = {}
        for s in valid_subjects:
            if s.dataset_source:
                source_counts[s.dataset_source] = source_counts.get(s.dataset_source, 0) + 1
        
        return {
            "total_subjects": len(valid_subjects),
            "score_stats": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores),
                "quartiles": np.percentile(scores, [25, 50, 75]).tolist()
            },
            "age_stats": {
                "mean": np.mean(ages) if ages else None,
                "std": np.std(ages) if ages else None,
                "min": np.min(ages) if ages else None,
                "max": np.max(ages) if ages else None,
                "count": len(ages)
            },
            "sex_distribution": sex_counts,
            "dataset_sources": source_counts
        }


if __name__ == "__main__":
    # Example usage
    from gazimed.data.database import initialize_database
    
    # Initialize database
    db_manager = initialize_database("test_dataset.db")
    
    # Get dataset statistics
    stats = get_dataset_statistics(db_manager)
    print("Dataset Statistics:")
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"Score range: {stats['score_stats']['min']:.3f} - {stats['score_stats']['max']:.3f}")
    
    # Create data splits
    splitter = DataSplitter(db_manager)
    train_ids, val_ids, test_ids = splitter.train_val_test_split()
    
    print(f"\nData splits:")
    print(f"Train: {len(train_ids)} subjects")
    print(f"Validation: {len(val_ids)} subjects")
    print(f"Test: {len(test_ids)} subjects")
    
    # Create cross-validation folds
    cv_folds = splitter.create_cv_folds(n_folds=5)
    print(f"\nCreated {len(cv_folds)} CV folds")
    
    # Create datasets (without loading volumes for testing)
    try:
        train_dataset = AlzheimersDataset(
            db_manager=db_manager,
            subject_ids=train_ids[:5],  # Just first 5 for testing
            load_volumes=False  # Don't load volumes for testing
        )
        
        print(f"\nTrain dataset created with {len(train_dataset)} samples")
        
        # Test getting a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Clinical features shape: {sample['clinical_features'].shape}")
            print(f"Alzheimer score: {sample['alzheimer_score'].item():.3f}")
            
    except Exception as e:
        print(f"Dataset creation failed (expected if no data): {e}")