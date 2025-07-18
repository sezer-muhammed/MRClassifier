"""
Tests for the dataset implementation (Task 2.3).

Tests the AlzheimersDataset, data splitting utilities, and cross-validation support.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from gazimed.data.dataset import (
    AlzheimersDataset, DataSplitter, CrossValidationDataManager,
    create_data_loaders, get_dataset_statistics
)
from gazimed.data.database import Subject


def custom_collate_fn(batch):
    """Custom collate function that handles None values in metadata."""
    import torch
    from torch.utils.data.dataloader import default_collate
    
    # Handle None values by converting them to a placeholder
    def replace_none_recursive(obj):
        if obj is None:
            return "None"  # Convert None to string for collation
        elif isinstance(obj, dict):
            return {k: replace_none_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_none_recursive(item) for item in obj]
        else:
            return obj
    
    # Process each sample in the batch
    processed_batch = []
    for sample in batch:
        processed_sample = replace_none_recursive(sample)
        processed_batch.append(processed_sample)
    
    return default_collate(processed_batch)


class TestAlzheimersDataset:
    """Test the AlzheimersDataset class."""
    
    def test_dataset_creation(self, db_manager):
        """Test basic dataset creation."""
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=False,
            validate_files=False
        )
        
        assert len(dataset) == 20
        assert len(dataset.subjects) == 20
    
    def test_dataset_sample_structure(self, db_manager):
        """Test the structure of dataset samples."""
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=False,
            validate_files=False
        )
        
        sample = dataset[0]
        
        # Check required keys
        assert 'subject_id' in sample
        assert 'alzheimer_score' in sample
        assert 'clinical_features' in sample
        assert 'metadata' in sample
        
        # Check data types and shapes
        assert isinstance(sample['subject_id'], str)
        assert isinstance(sample['alzheimer_score'], torch.Tensor)
        assert isinstance(sample['clinical_features'], torch.Tensor)
        assert sample['clinical_features'].shape == (116,)
        assert 0.0 <= sample['alzheimer_score'].item() <= 1.0
        
        # Check metadata
        metadata = sample['metadata']
        assert 'age' in metadata
        assert 'sex' in metadata
        assert 'dataset_source' in metadata
    
    def test_dataset_filtering_by_score(self, db_manager):
        """Test dataset filtering by score range."""
        # Test high score filtering
        high_score_dataset = AlzheimersDataset(
            db_manager=db_manager,
            subject_filter={'min_score': 0.7},
            load_volumes=False,
            validate_files=False
        )
        
        # Verify all subjects have high scores
        for i in range(len(high_score_dataset)):
            sample = high_score_dataset[i]
            assert sample['alzheimer_score'].item() >= 0.7
    
    def test_dataset_filtering_by_source(self, db_manager):
        """Test dataset filtering by dataset source."""
        adni_dataset = AlzheimersDataset(
            db_manager=db_manager,
            subject_filter={'dataset_source': 'TEST_ADNI'},
            load_volumes=False,
            validate_files=False
        )
        
        assert len(adni_dataset) == 10
        
        # Verify all subjects are from ADNI
        for i in range(len(adni_dataset)):
            sample = adni_dataset[i]
            assert sample['metadata']['dataset_source'] == 'TEST_ADNI'
    
    def test_dataset_filtering_by_sex(self, db_manager):
        """Test dataset filtering by sex."""
        male_dataset = AlzheimersDataset(
            db_manager=db_manager,
            subject_filter={'sex': 'M'},
            load_volumes=False,
            validate_files=False
        )
        
        assert len(male_dataset) == 10
        
        # Verify all subjects are male
        for i in range(len(male_dataset)):
            sample = male_dataset[i]
            assert sample['metadata']['sex'] == 'M'
    
    def test_dataset_subject_ids_filter(self, db_manager):
        """Test dataset filtering by specific subject IDs."""
        target_ids = ['TEST_000', 'TEST_001', 'TEST_002']
        
        filtered_dataset = AlzheimersDataset(
            db_manager=db_manager,
            subject_ids=target_ids,
            load_volumes=False,
            validate_files=False
        )
        
        assert len(filtered_dataset) == 3
        
        # Verify correct subjects are included
        found_ids = set()
        for i in range(len(filtered_dataset)):
            sample = filtered_dataset[i]
            found_ids.add(sample['subject_id'])
        
        assert found_ids == set(target_ids)
    
    def test_dataset_score_distribution(self, db_manager):
        """Test score distribution calculation."""
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=False,
            validate_files=False
        )
        
        score_dist = dataset.get_score_distribution()
        
        assert 'mean' in score_dist
        assert 'std' in score_dist
        assert 'min' in score_dist
        assert 'max' in score_dist
        assert 'median' in score_dist
        
        assert 0.0 <= score_dist['min'] <= 1.0
        assert 0.0 <= score_dist['max'] <= 1.0
        assert score_dist['min'] <= score_dist['max']
    
    def test_dataset_with_difference_channel(self, db_manager):
        """Test dataset with difference channel option."""
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=False,  # Still don't load actual volumes
            include_difference_channel=True,
            validate_files=False
        )
        
        # Should still work without loading volumes
        assert len(dataset) == 20
    
    def test_dataset_empty_result(self, db_manager):
        """Test dataset behavior with no matching subjects."""
        with pytest.raises(ValueError, match="No valid subjects found"):
            AlzheimersDataset(
                db_manager=db_manager,
                subject_filter={'min_score': 2.0},  # Impossible score
                load_volumes=False,
                validate_files=False
            )


class TestDataSplitter:
    """Test the DataSplitter class."""
    
    def test_data_splitter_creation(self, db_manager):
        """Test data splitter creation."""
        splitter = DataSplitter(db_manager, random_state=42)
        assert splitter.db_manager == db_manager
        assert splitter.random_state == 42
    
    def test_train_val_test_split(self, db_manager):
        """Test train/validation/test split."""
        splitter = DataSplitter(db_manager, random_state=42)
        
        train_ids, val_ids, test_ids = splitter.train_val_test_split(
            train_size=0.6, val_size=0.2, test_size=0.2, stratify=True
        )
        
        # Check sizes
        total_size = len(train_ids) + len(val_ids) + len(test_ids)
        assert total_size == 20
        
        # Check proportions (approximately)
        assert len(train_ids) == 12  # 60% of 20
        assert len(val_ids) == 4     # 20% of 20
        assert len(test_ids) == 4    # 20% of 20
        
        # Check no overlap
        all_ids = set(train_ids + val_ids + test_ids)
        assert len(all_ids) == total_size
    
    def test_train_val_test_split_without_stratify(self, db_manager):
        """Test train/validation/test split without stratification."""
        splitter = DataSplitter(db_manager, random_state=42)
        
        train_ids, val_ids, test_ids = splitter.train_val_test_split(
            train_size=0.7, val_size=0.15, test_size=0.15, stratify=False
        )
        
        # Check sizes
        total_size = len(train_ids) + len(val_ids) + len(test_ids)
        assert total_size == 20
        
        # Check no overlap
        all_ids = set(train_ids + val_ids + test_ids)
        assert len(all_ids) == total_size
    
    def test_create_cv_folds(self, db_manager):
        """Test cross-validation fold creation."""
        splitter = DataSplitter(db_manager, random_state=42)
        
        cv_folds = splitter.create_cv_folds(n_folds=5, stratify=True)
        
        assert len(cv_folds) == 5
        
        # Check each fold
        all_val_ids = set()
        for i, (train_ids, val_ids) in enumerate(cv_folds):
            # Check no overlap within fold
            assert len(set(train_ids) & set(val_ids)) == 0
            
            # Check sizes
            assert len(train_ids) == 16  # 4/5 of 20
            assert len(val_ids) == 4     # 1/5 of 20
            
            # Collect validation IDs
            all_val_ids.update(val_ids)
        
        # Check that all subjects appear in validation exactly once
        assert len(all_val_ids) == 20
    
    def test_create_cv_folds_without_stratify(self, db_manager):
        """Test cross-validation fold creation without stratification."""
        splitter = DataSplitter(db_manager, random_state=42)
        
        cv_folds = splitter.create_cv_folds(n_folds=4, stratify=False)
        
        assert len(cv_folds) == 4
        
        # Check that all subjects are covered
        all_val_ids = set()
        for train_ids, val_ids in cv_folds:
            all_val_ids.update(val_ids)
        
        assert len(all_val_ids) == 20


class TestCrossValidationDataManager:
    """Test the CrossValidationDataManager class."""
    
    def test_cv_manager_creation(self, small_db_manager):
        """Test CV manager creation."""
        cv_manager = CrossValidationDataManager(
            db_manager=small_db_manager,
            n_folds=3,
            random_state=42,
            stratify=True
        )
        
        assert cv_manager.n_folds == 3
        assert cv_manager.random_state == 42
        assert cv_manager.stratify == True
        assert len(cv_manager.folds) == 3
    
    def test_get_fold_data_loaders(self, small_db_manager):
        """Test getting data loaders for a fold."""
        cv_manager = CrossValidationDataManager(
            db_manager=small_db_manager,
            n_folds=3,
            random_state=42
        )
        
        data_loaders = cv_manager.get_fold_data_loaders(
            fold_idx=0,
            batch_size=2,
            num_workers=0,
            load_volumes=False
        )
        
        assert 'train' in data_loaders
        assert 'val' in data_loaders
        
        # Test that loaders work with custom collate function
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        # Manually set collate function to handle None values
        train_loader.collate_fn = custom_collate_fn
        val_loader.collate_fn = custom_collate_fn
        
        # Test getting a batch from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert 'clinical_features' in train_batch
        assert 'alzheimer_score' in train_batch
        assert 'clinical_features' in val_batch
        assert 'alzheimer_score' in val_batch
    
    def test_get_fold_statistics(self, small_db_manager):
        """Test getting fold statistics."""
        cv_manager = CrossValidationDataManager(
            db_manager=small_db_manager,
            n_folds=3,
            random_state=42
        )
        
        fold_stats = cv_manager.get_fold_statistics(0)
        
        assert 'fold_idx' in fold_stats
        assert 'train_size' in fold_stats
        assert 'val_size' in fold_stats
        assert 'train_score_stats' in fold_stats
        assert 'val_score_stats' in fold_stats
        
        assert fold_stats['fold_idx'] == 0
        assert fold_stats['train_size'] > 0
        assert fold_stats['val_size'] > 0
    
    def test_get_all_fold_statistics(self, small_db_manager):
        """Test getting all fold statistics."""
        cv_manager = CrossValidationDataManager(
            db_manager=small_db_manager,
            n_folds=3,
            random_state=42
        )
        
        all_stats = cv_manager.get_all_fold_statistics()
        
        assert len(all_stats) == 3
        for i, stats in enumerate(all_stats):
            assert stats['fold_idx'] == i
    
    def test_invalid_fold_index(self, small_db_manager):
        """Test error handling for invalid fold index."""
        cv_manager = CrossValidationDataManager(
            db_manager=small_db_manager,
            n_folds=3,
            random_state=42
        )
        
        with pytest.raises(ValueError, match="Fold index .* out of range"):
            cv_manager.get_fold_data_loaders(fold_idx=5)
        
        with pytest.raises(ValueError, match="Fold index .* out of range"):
            cv_manager.get_fold_statistics(fold_idx=5)


class TestDataLoaders:
    """Test data loader creation functions."""
    
    def test_create_data_loaders_basic(self, small_db_manager):
        """Test basic data loader creation."""
        splitter = DataSplitter(small_db_manager, random_state=42)
        train_ids, val_ids, test_ids = splitter.train_val_test_split(
            train_size=0.5, val_size=0.25, test_size=0.25, stratify=False
        )
        
        data_loaders = create_data_loaders(
            db_manager=small_db_manager,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            batch_size=2,
            num_workers=0,
            load_volumes=False
        )
        
        assert 'train' in data_loaders
        assert 'val' in data_loaders
        assert 'test' in data_loaders
        
        # Test each loader with custom collate function
        for split_name, loader in data_loaders.items():
            loader.collate_fn = custom_collate_fn
            batch = next(iter(loader))
            assert 'clinical_features' in batch
            assert 'alzheimer_score' in batch
            assert batch['clinical_features'].shape[1] == 116
    
    def test_create_data_loaders_without_test(self, small_db_manager):
        """Test data loader creation without test set."""
        splitter = DataSplitter(small_db_manager, random_state=42)
        train_ids, val_ids, _ = splitter.train_val_test_split(
            train_size=0.7, val_size=0.3, test_size=None, stratify=False
        )
        
        data_loaders = create_data_loaders(
            db_manager=small_db_manager,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=None,
            batch_size=2,
            num_workers=0,
            load_volumes=False
        )
        
        assert 'train' in data_loaders
        assert 'val' in data_loaders
        assert 'test' not in data_loaders
    
    def test_create_data_loaders_with_difference_channel(self, small_db_manager):
        """Test data loader creation with difference channel."""
        splitter = DataSplitter(small_db_manager, random_state=42)
        train_ids, val_ids, _ = splitter.train_val_test_split(
            train_size=0.7, val_size=0.3, test_size=None, stratify=False
        )
        
        data_loaders = create_data_loaders(
            db_manager=small_db_manager,
            train_ids=train_ids,
            val_ids=val_ids,
            batch_size=2,
            num_workers=0,
            load_volumes=False,
            include_difference_channel=True
        )
        
        # Should work without errors
        assert 'train' in data_loaders
        assert 'val' in data_loaders


class TestDatasetStatistics:
    """Test dataset statistics functions."""
    
    def test_get_dataset_statistics(self, db_manager):
        """Test getting dataset statistics."""
        stats = get_dataset_statistics(db_manager)
        
        assert 'total_subjects' in stats
        assert 'score_stats' in stats
        assert 'age_stats' in stats
        assert 'sex_distribution' in stats
        assert 'dataset_sources' in stats
        
        assert stats['total_subjects'] == 20
        
        # Check score stats
        score_stats = stats['score_stats']
        assert 'mean' in score_stats
        assert 'std' in score_stats
        assert 'min' in score_stats
        assert 'max' in score_stats
        assert 'median' in score_stats
        assert 'quartiles' in score_stats
        
        # Check distributions
        assert 'M' in stats['sex_distribution']
        assert 'F' in stats['sex_distribution']
        assert stats['sex_distribution']['M'] == 10
        assert stats['sex_distribution']['F'] == 10
        
        assert 'TEST_ADNI' in stats['dataset_sources']
        assert 'TEST_OASIS' in stats['dataset_sources']
    
    def test_get_dataset_statistics_empty(self, temp_db):
        """Test dataset statistics with empty database."""
        from gazimed.data.database import DatabaseManager
        db_manager = DatabaseManager(temp_db)
        db_manager.create_tables()
        
        stats = get_dataset_statistics(db_manager)
        
        assert 'error' in stats
        assert stats['error'] == "No valid subjects found"


class TestVolumeLoading:
    """Test volume loading functionality (mocked)."""
    
    @patch('gazimed.data.dataset.NIBABEL_AVAILABLE', False)
    def test_volume_loading_without_nibabel(self, db_manager):
        """Test error handling when nibabel is not available."""
        # Create dataset without loading volumes first
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=False,
            validate_files=False
        )
        
        # Test that the nibabel error is raised when trying to load volumes
        with pytest.raises(RuntimeError, match="nibabel is required"):
            dataset._load_nifti_volume("/fake/path.nii.gz")
    
    def test_volume_loading_difference_channel_logic(self, db_manager):
        """Test the difference channel logic without actual file loading."""
        dataset = AlzheimersDataset(
            db_manager=db_manager,
            load_volumes=False,
            include_difference_channel=True,
            validate_files=False
        )
        
        # Test that dataset can be created with difference channel option
        assert dataset.include_difference_channel == True
        assert len(dataset) == 20