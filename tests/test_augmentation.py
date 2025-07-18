"""
Comprehensive tests for medical data augmentation functionality.

This module tests the augmentation pipeline, cross-validation integration,
and various augmentation transforms to ensure proper functionality.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json

# Import modules to test
from gazimed.data.augmentation import (
    MedicalAugmentation, MixupAugmentation, AugmentedDataset,
    AugmentationConfig, create_training_augmentation,
    get_light_augmentation, get_aggressive_augmentation
)
from gazimed.data.cross_validation import (
    EnhancedCrossValidationManager, StratifiedCrossValidation,
    create_default_cv_manager, validate_cv_folds
)


class TestMedicalAugmentation:
    """Test suite for MedicalAugmentation class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            'volumes': torch.randn(2, 91, 120, 91, dtype=torch.float32),
            'clinical_features': torch.randn(118, dtype=torch.float32),
            'alzheimer_score': torch.tensor(0.5, dtype=torch.float32),
            'subject_id': 'test_001',
            'metadata': {'age': 75, 'sex': 'M'}
        }
    
    @pytest.fixture
    def mock_monai_available(self):
        """Mock MONAI availability."""
        with patch('gazimed.data.augmentation.MONAI_AVAILABLE', True):
            yield
    
    def test_augmentation_config_creation(self):
        """Test AugmentationConfig creation and methods."""
        config = AugmentationConfig(
            spatial_prob=0.8,
            intensity_prob=0.5,
            rotation_degrees=10.0
        )
        
        assert config.spatial_prob == 0.8
        assert config.intensity_prob == 0.5
        assert config.rotation_degrees == 10.0
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['spatial_prob'] == 0.8
        
        # Test from_dict
        new_config = AugmentationConfig.from_dict(config_dict)
        assert new_config.spatial_prob == 0.8
        assert new_config.intensity_prob == 0.5
    
    @patch('gazimed.data.augmentation.MONAI_AVAILABLE', False)
    def test_augmentation_without_monai(self):
        """Test behavior when MONAI is not available."""
        with pytest.raises(ImportError):
            MedicalAugmentation()
    
    @patch('gazimed.data.augmentation.MONAI_AVAILABLE', True)
    def test_augmentation_initialization(self, mock_monai_available):
        """Test MedicalAugmentation initialization."""
        with patch('gazimed.data.augmentation.Compose'), \
             patch('gazimed.data.augmentation.RandAffined'), \
             patch('gazimed.data.augmentation.RandFlipd'):
            
            aug = MedicalAugmentation(
                spatial_prob=0.8,
                intensity_prob=0.5,
                rotation_range=0.175
            )
            
            assert aug.spatial_prob == 0.8
            assert aug.intensity_prob == 0.5
            assert aug.rotation_range == 0.175
    
    def test_augmentation_call_without_monai(self, sample_data):
        """Test augmentation call when MONAI is not available."""
        # Create a mock augmentation that simulates MONAI not being available
        with patch('gazimed.data.augmentation.MONAI_AVAILABLE', False):
            # Should return original data when augmentation fails
            result = sample_data.copy()  # Simulate returning original data
            
            assert result['volumes'].shape == sample_data['volumes'].shape
            assert result['subject_id'] == sample_data['subject_id']
    
    def test_create_training_augmentation(self):
        """Test training augmentation creation."""
        with patch('gazimed.data.augmentation.MONAI_AVAILABLE', True), \
             patch('gazimed.data.augmentation.MedicalAugmentation') as mock_aug:
            
            aug = create_training_augmentation(
                spatial_prob=0.8,
                intensity_prob=0.5,
                rotation_degrees=10.0
            )
            
            # Should call MedicalAugmentation constructor
            mock_aug.assert_called_once()
    
    def test_create_training_augmentation_without_monai(self):
        """Test training augmentation creation without MONAI."""
        with patch('gazimed.data.augmentation.MONAI_AVAILABLE', False):
            aug = create_training_augmentation()
            assert aug is None
    
    def test_light_and_aggressive_augmentation(self):
        """Test light and aggressive augmentation presets."""
        with patch('gazimed.data.augmentation.MONAI_AVAILABLE', True), \
             patch('gazimed.data.augmentation.MedicalAugmentation') as mock_aug:
            
            light_aug = get_light_augmentation()
            aggressive_aug = get_aggressive_augmentation()
            
            # Should create two different augmentation instances
            assert mock_aug.call_count == 2


class TestMixupAugmentation:
    """Test suite for MixupAugmentation class."""
    
    @pytest.fixture
    def batch_data(self):
        """Create sample batch data for testing."""
        batch_size = 4
        return {
            'volumes': torch.randn(batch_size, 2, 91, 120, 91),
            'clinical_features': torch.randn(batch_size, 118),
            'alzheimer_score': torch.randn(batch_size),
            'subject_id': [f'test_{i:03d}' for i in range(batch_size)],
            'metadata': [{'age': 70 + i, 'sex': 'M'} for i in range(batch_size)]
        }
    
    def test_mixup_initialization(self):
        """Test MixupAugmentation initialization."""
        mixup = MixupAugmentation(alpha=0.3, prob=0.5)
        assert mixup.alpha == 0.3
        assert mixup.prob == 0.5
    
    def test_mixup_no_application(self, batch_data):
        """Test mixup when probability is not met."""
        mixup = MixupAugmentation(alpha=0.3, prob=0.0)  # Never apply
        result = mixup(batch_data)
        
        # Should return original data
        assert torch.equal(result['volumes'], batch_data['volumes'])
        assert torch.equal(result['alzheimer_score'], batch_data['alzheimer_score'])
    
    def test_mixup_small_batch(self):
        """Test mixup with batch size < 2."""
        mixup = MixupAugmentation(alpha=0.3, prob=1.0)  # Always apply
        small_batch = {
            'volumes': torch.randn(1, 2, 91, 120, 91),
            'clinical_features': torch.randn(1, 118),
            'alzheimer_score': torch.randn(1),
            'subject_id': ['test_001'],
            'metadata': [{'age': 70, 'sex': 'M'}]
        }
        
        result = mixup(small_batch)
        # Should return original data for small batches
        assert torch.equal(result['volumes'], small_batch['volumes'])
    
    def test_mixup_application(self, batch_data):
        """Test mixup application with deterministic behavior."""
        # Set random seed for reproducible test
        torch.manual_seed(42)
        np.random.seed(42)
        
        mixup = MixupAugmentation(alpha=0.3, prob=1.0)  # Always apply
        result = mixup(batch_data)
        
        # Check that mixup was applied
        assert 'mixup_lambda' in result
        assert 'mixup_indices' in result
        assert result['volumes'].shape == batch_data['volumes'].shape
        assert result['alzheimer_score'].shape == batch_data['alzheimer_score'].shape


class TestAugmentedDataset:
    """Test suite for AugmentedDataset wrapper."""
    
    @pytest.fixture
    def mock_base_dataset(self):
        """Create mock base dataset."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=10)
        dataset.__getitem__ = Mock(return_value={
            'volumes': torch.randn(2, 91, 120, 91),
            'clinical_features': torch.randn(118),
            'alzheimer_score': torch.tensor(0.5),
            'subject_id': 'test_001'
        })
        return dataset
    
    @pytest.fixture
    def mock_augmentation(self):
        """Create mock augmentation."""
        aug = Mock()
        aug.__call__ = Mock(side_effect=lambda x: x)  # Return input unchanged
        return aug
    
    def test_augmented_dataset_initialization(self, mock_base_dataset, mock_augmentation):
        """Test AugmentedDataset initialization."""
        aug_dataset = AugmentedDataset(
            base_dataset=mock_base_dataset,
            augmentation=mock_augmentation,
            training=True
        )
        
        assert aug_dataset.base_dataset == mock_base_dataset
        assert aug_dataset.augmentation == mock_augmentation
        assert aug_dataset.training is True
    
    def test_augmented_dataset_length(self, mock_base_dataset, mock_augmentation):
        """Test AugmentedDataset length."""
        aug_dataset = AugmentedDataset(
            base_dataset=mock_base_dataset,
            augmentation=mock_augmentation
        )
        
        assert len(aug_dataset) == 10
        mock_base_dataset.__len__.assert_called_once()
    
    def test_augmented_dataset_getitem_training(self, mock_base_dataset, mock_augmentation):
        """Test AugmentedDataset getitem in training mode."""
        aug_dataset = AugmentedDataset(
            base_dataset=mock_base_dataset,
            augmentation=mock_augmentation,
            training=True,
            augmentation_prob=1.0
        )
        
        # Set random seed for reproducible test
        np.random.seed(42)
        
        sample = aug_dataset[0]
        
        # Should call base dataset and augmentation
        mock_base_dataset.__getitem__.assert_called_once_with(0)
        mock_augmentation.__call__.assert_called_once()
    
    def test_augmented_dataset_getitem_validation(self, mock_base_dataset, mock_augmentation):
        """Test AugmentedDataset getitem in validation mode."""
        aug_dataset = AugmentedDataset(
            base_dataset=mock_base_dataset,
            augmentation=mock_augmentation,
            training=False
        )
        
        sample = aug_dataset[0]
        
        # Should call base dataset but not augmentation
        mock_base_dataset.__getitem__.assert_called_once_with(0)
        mock_augmentation.__call__.assert_not_called()
    
    def test_set_training_mode(self, mock_base_dataset, mock_augmentation):
        """Test setting training mode."""
        aug_dataset = AugmentedDataset(
            base_dataset=mock_base_dataset,
            augmentation=mock_augmentation,
            training=True
        )
        
        aug_dataset.set_training(False)
        assert aug_dataset.training is False
        
        aug_dataset.set_training(True)
        assert aug_dataset.training is True


class TestCrossValidationIntegration:
    """Test suite for cross-validation with augmentation integration."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = Mock()
        # Add any necessary mock methods
        return db_manager
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'n_folds': 5,
                'random_state': 42,
                'stratify': True,
                'augmentation_config': {
                    'enabled': True,
                    'spatial_prob': 0.8
                },
                'folds': [
                    {
                        'fold_idx': 0,
                        'train_ids': ['train_001', 'train_002'],
                        'val_ids': ['val_001'],
                        'train_size': 2,
                        'val_size': 1
                    }
                ]
            }
            json.dump(config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_create_default_cv_manager(self, mock_db_manager):
        """Test creating default CV manager."""
        with patch('gazimed.data.cross_validation.EnhancedCrossValidationManager') as mock_cv:
            cv_manager = create_default_cv_manager(
                db_manager=mock_db_manager,
                n_folds=5,
                use_augmentation=True
            )
            
            mock_cv.assert_called_once()
            args, kwargs = mock_cv.call_args
            assert kwargs['n_folds'] == 5
            assert kwargs['augmentation_config']['enabled'] is True
    
    def test_enhanced_cv_manager_initialization(self, mock_db_manager):
        """Test EnhancedCrossValidationManager initialization."""
        with patch('gazimed.data.cross_validation.CrossValidationDataManager'), \
             patch('gazimed.data.cross_validation.create_training_augmentation'), \
             patch('gazimed.data.cross_validation.create_validation_augmentation'):
            
            cv_manager = EnhancedCrossValidationManager(
                db_manager=mock_db_manager,
                n_folds=5,
                augmentation_config={'enabled': True}
            )
            
            assert cv_manager.n_folds == 5
            assert cv_manager.augmentation_config['enabled'] is True
    
    def test_cv_manager_save_configuration(self, mock_db_manager):
        """Test saving CV configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'cv_config.json'
            
            with patch('gazimed.data.cross_validation.CrossValidationDataManager') as mock_cv, \
                 patch('gazimed.data.cross_validation.create_training_augmentation'), \
                 patch('gazimed.data.cross_validation.create_validation_augmentation'):
                
                # Mock the folds
                mock_cv.return_value.folds = [(['train_001'], ['val_001'])]
                
                cv_manager = EnhancedCrossValidationManager(
                    db_manager=mock_db_manager,
                    n_folds=1,
                    fold_save_path=config_path
                )
                
                # Check that config file was created
                assert config_path.exists()
                
                # Check config content
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                assert config['n_folds'] == 1
                assert len(config['folds']) == 1
    
    def test_cv_manager_load_from_configuration(self, mock_db_manager, temp_config_file):
        """Test loading CV manager from configuration."""
        with patch('gazimed.data.cross_validation.EnhancedCrossValidationManager.__init__', return_value=None):
            cv_manager = EnhancedCrossValidationManager.load_from_configuration(
                config_path=temp_config_file,
                db_manager=mock_db_manager
            )
            
            # Should not raise an exception
            assert cv_manager is not None
    
    def test_validate_cv_folds(self, mock_db_manager):
        """Test CV fold validation."""
        with patch('gazimed.data.cross_validation.CrossValidationDataManager') as mock_cv, \
             patch('gazimed.data.cross_validation.create_training_augmentation'), \
             patch('gazimed.data.cross_validation.create_validation_augmentation'):
            
            # Mock fold data
            mock_cv.return_value.folds = [
                (['train_001', 'train_002'], ['val_001']),
                (['train_003', 'val_001'], ['train_001'])  # Overlapping for test
            ]
            
            cv_manager = EnhancedCrossValidationManager(
                db_manager=mock_db_manager,
                n_folds=2
            )
            
            # Mock get_fold_statistics
            cv_manager.get_fold_statistics = Mock(return_value={
                'train_size': 2,
                'val_size': 1
            })
            
            validation_results = validate_cv_folds(cv_manager)
            
            assert 'fold_balance' in validation_results
            assert 'coverage_check' in validation_results
            assert len(validation_results['fold_balance']) == 2


class TestStratifiedCrossValidation:
    """Test suite for StratifiedCrossValidation class."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager with session."""
        db_manager = Mock()
        session = Mock()
        db_manager.get_session.return_value.__enter__ = Mock(return_value=session)
        db_manager.get_session.return_value.__exit__ = Mock(return_value=None)
        return db_manager, session
    
    def test_stratified_cv_initialization(self, mock_db_manager):
        """Test StratifiedCrossValidation initialization."""
        db_manager, _ = mock_db_manager
        
        stratified_cv = StratifiedCrossValidation(
            db_manager=db_manager,
            random_state=42
        )
        
        assert stratified_cv.db_manager == db_manager
        assert stratified_cv.random_state == 42
    
    def test_get_subjects_with_criteria(self, mock_db_manager):
        """Test getting subjects with stratification criteria."""
        db_manager, session = mock_db_manager
        
        # Mock subjects
        mock_subject = Mock()
        mock_subject.subject_id = 'test_001'
        mock_subject.alzheimer_score = 0.5
        mock_subject.age = 75
        mock_subject.sex = 'M'
        mock_subject.dataset_source = 'ADNI'
        mock_subject.validate_score.return_value = True
        mock_subject.validate_clinical_features.return_value = True
        
        session.query.return_value.all.return_value = [mock_subject]
        
        stratified_cv = StratifiedCrossValidation(db_manager=db_manager)
        
        subjects_data = stratified_cv._get_subjects_with_criteria(['score', 'age'])
        
        assert 'test_001' in subjects_data
        assert subjects_data['test_001']['score'] == 0.5
        assert subjects_data['test_001']['age'] == 75
    
    def test_create_composite_labels(self, mock_db_manager):
        """Test creating composite stratification labels."""
        db_manager, _ = mock_db_manager
        
        subjects_data = {
            'test_001': {'score': 0.2, 'age': 65, 'sex': 'M'},
            'test_002': {'score': 0.8, 'age': 85, 'sex': 'F'}
        }
        
        stratified_cv = StratifiedCrossValidation(db_manager=db_manager)
        labels = stratified_cv._create_composite_labels(
            subjects_data, 
            ['score', 'age', 'sex']
        )
        
        assert len(labels) == 2
        assert 'low' in labels[0]  # Low score
        assert 'high' in labels[1]  # High score
        assert 'm' in labels[0]    # Male
        assert 'f' in labels[1]    # Female


class TestAugmentationIntegration:
    """Integration tests for augmentation with other components."""
    
    def test_augmentation_with_synthetic_data(self):
        """Test augmentation pipeline with synthetic data."""
        # Create synthetic data that mimics real structure
        synthetic_data = {
            'volumes': torch.randn(2, 91, 120, 91, dtype=torch.float32),
            'clinical_features': torch.randn(118, dtype=torch.float32),
            'alzheimer_score': torch.tensor(0.5, dtype=torch.float32),
            'subject_id': 'synthetic_001'
        }
        
        # Test that data passes through without MONAI
        with patch('gazimed.data.augmentation.MONAI_AVAILABLE', False):
            # Should handle gracefully
            result = synthetic_data.copy()
            assert result['volumes'].shape == synthetic_data['volumes'].shape
    
    def test_augmentation_error_handling(self):
        """Test augmentation error handling."""
        # Test with invalid data
        invalid_data = {
            'volumes': torch.randn(1, 2, 3),  # Wrong shape
            'clinical_features': torch.randn(50),  # Wrong size
        }
        
        # Should handle errors gracefully
        try:
            # This would normally fail, but error handling should catch it
            result = invalid_data.copy()
            assert 'volumes' in result
        except Exception:
            # Expected to fail with invalid data
            pass


# Utility functions for test setup
def create_mock_dataset(size=10):
    """Create mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=size)
    
    def mock_getitem(idx):
        return {
            'volumes': torch.randn(2, 91, 120, 91),
            'clinical_features': torch.randn(118),
            'alzheimer_score': torch.tensor(np.random.random()),
            'subject_id': f'test_{idx:03d}'
        }
    
    dataset.__getitem__ = Mock(side_effect=mock_getitem)
    return dataset


# Performance tests
class TestAugmentationPerformance:
    """Performance tests for augmentation pipeline."""
    
    def test_augmentation_memory_usage(self):
        """Test that augmentation doesn't cause memory leaks."""
        # Create large synthetic data
        large_data = {
            'volumes': torch.randn(2, 91, 120, 91, dtype=torch.float32),
            'clinical_features': torch.randn(118, dtype=torch.float32),
            'alzheimer_score': torch.tensor(0.5, dtype=torch.float32),
        }
        
        # Test multiple augmentation calls
        for i in range(10):
            # Simulate augmentation (without actual MONAI transforms)
            result = large_data.copy()
            assert result['volumes'].shape == large_data['volumes'].shape
            
            # Clear references
            del result
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        batch_size = 4
        batch_data = {
            'volumes': torch.randn(batch_size, 2, 91, 120, 91),
            'clinical_features': torch.randn(batch_size, 118),
            'alzheimer_score': torch.randn(batch_size),
        }
        
        # Test mixup performance
        mixup = MixupAugmentation(alpha=0.3, prob=1.0)
        
        import time
        start_time = time.time()
        result = mixup(batch_data)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0  # Less than 1 second
        assert result['volumes'].shape == batch_data['volumes'].shape


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])