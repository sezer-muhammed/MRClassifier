"""
Enhanced cross-validation utilities with augmentation integration.

This module extends the existing cross-validation functionality with
proper augmentation integration, fold persistence, and advanced
stratification strategies for medical data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from gazimed.data.dataset import (
    AlzheimersDataset, DataSplitter, CrossValidationDataManager
)
from gazimed.data.augmentation import (
    MedicalAugmentation, AugmentedDataset, MixupAugmentation,
    create_training_augmentation, create_validation_augmentation
)

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedCrossValidationManager:
    """
    Enhanced cross-validation manager with augmentation integration.
    
    Extends the base CrossValidationDataManager with augmentation support,
    fold persistence, and advanced configuration options.
    """
    
    def __init__(self,
                 db_manager,
                 n_folds: int = 5,
                 random_state: int = 42,
                 stratify: bool = True,
                 augmentation_config: Optional[Dict[str, Any]] = None,
                 fold_save_path: Optional[Union[str, Path]] = None):
        """
        Initialize enhanced cross-validation manager.
        
        Args:
            db_manager: Database manager instance
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducible splits
            stratify: Whether to stratify folds based on scores
            augmentation_config: Configuration for augmentation pipeline
            fold_save_path: Path to save/load fold configurations
        """
        self.db_manager = db_manager
        self.n_folds = n_folds
        self.random_state = random_state
        self.stratify = stratify
        self.augmentation_config = augmentation_config or {}
        self.fold_save_path = Path(fold_save_path) if fold_save_path else None
        
        # Initialize base cross-validation manager
        self.base_cv_manager = CrossValidationDataManager(
            db_manager=db_manager,
            n_folds=n_folds,
            random_state=random_state,
            stratify=stratify
        )
        
        # Create augmentation pipelines
        self.train_augmentation = self._create_train_augmentation()
        self.val_augmentation = self._create_val_augmentation()
        self.mixup_augmentation = self._create_mixup_augmentation()
        
        # Save fold configuration if path provided
        if self.fold_save_path:
            self._save_fold_configuration()
        
        logger.info(f"Enhanced CV manager initialized with {n_folds} folds, "
                   f"stratify={stratify}, augmentation={'enabled' if self.train_augmentation else 'disabled'}")
    
    def _create_train_augmentation(self) -> Optional[MedicalAugmentation]:
        """Create training augmentation pipeline."""
        if not self.augmentation_config.get('enabled', True):
            return None
        
        return create_training_augmentation(
            spatial_prob=self.augmentation_config.get('spatial_prob', 0.8),
            intensity_prob=self.augmentation_config.get('intensity_prob', 0.5),
            mixup_prob=self.augmentation_config.get('mixup_prob', 0.3),
            rotation_degrees=self.augmentation_config.get('rotation_degrees', 10.0)
        )
    
    def _create_val_augmentation(self) -> Optional[MedicalAugmentation]:
        """Create validation augmentation pipeline (typically none)."""
        return create_validation_augmentation()
    
    def _create_mixup_augmentation(self) -> Optional[MixupAugmentation]:
        """Create mixup augmentation for batch-level mixing."""
        if not self.augmentation_config.get('mixup_enabled', True):
            return None
        
        return MixupAugmentation(
            alpha=self.augmentation_config.get('mixup_alpha', 0.3),
            prob=self.augmentation_config.get('mixup_prob', 0.3)
        )
    
    def get_fold_data_loaders(self,
                             fold_idx: int,
                             batch_size: int = 4,
                             num_workers: int = 4,
                             load_volumes: bool = True,
                             include_difference_channel: bool = False,
                             use_augmentation: bool = True) -> Dict[str, DataLoader]:
        """
        Get augmented data loaders for a specific fold.
        
        Args:
            fold_idx: Index of the fold (0 to n_folds-1)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            load_volumes: Whether to load NIfTI volumes
            include_difference_channel: Whether to include PET-MRI difference channel
            use_augmentation: Whether to apply augmentation to training data
            
        Returns:
            Dictionary with 'train' and 'val' DataLoaders
        """
        if fold_idx >= self.n_folds:
            raise ValueError(f"Fold index {fold_idx} out of range (0-{self.n_folds-1})")
        
        train_ids, val_ids = self.base_cv_manager.folds[fold_idx]
        
        # Create base datasets
        train_dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            subject_ids=train_ids,
            load_volumes=load_volumes,
            cache_volumes=False,
            include_difference_channel=include_difference_channel
        )
        
        val_dataset = AlzheimersDataset(
            db_manager=self.db_manager,
            subject_ids=val_ids,
            load_volumes=load_volumes,
            cache_volumes=True,
            include_difference_channel=include_difference_channel
        )
        
        # Wrap with augmentation if enabled
        if use_augmentation and self.train_augmentation:
            train_dataset = AugmentedDataset(
                base_dataset=train_dataset,
                augmentation=self.train_augmentation,
                training=True
            )
        
        if self.val_augmentation:
            val_dataset = AugmentedDataset(
                base_dataset=val_dataset,
                augmentation=self.val_augmentation,
                training=False
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._create_collate_fn()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader
        }
    
    def _create_collate_fn(self):
        """Create collate function with optional mixup."""
        def collate_with_mixup(batch):
            # Standard collation
            collated = torch.utils.data.default_collate(batch)
            
            # Apply mixup if enabled
            if self.mixup_augmentation and collated['volumes'].size(0) > 1:
                collated = self.mixup_augmentation(collated)
            
            return collated
        
        return collate_with_mixup if self.mixup_augmentation else None
    
    def get_fold_statistics(self, fold_idx: int) -> Dict[str, Any]:
        """Get enhanced statistics for a specific fold."""
        base_stats = self.base_cv_manager.get_fold_statistics(fold_idx)
        
        # Add augmentation information
        aug_info = {
            'augmentation_enabled': self.train_augmentation is not None,
            'mixup_enabled': self.mixup_augmentation is not None,
            'augmentation_config': self.augmentation_config
        }
        
        base_stats.update(aug_info)
        return base_stats
    
    def get_all_fold_statistics(self) -> List[Dict[str, Any]]:
        """Get enhanced statistics for all folds."""
        return [self.get_fold_statistics(i) for i in range(self.n_folds)]
    
    def _save_fold_configuration(self):
        """Save fold configuration to disk for reproducibility."""
        if not self.fold_save_path:
            return
        
        config = {
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'stratify': self.stratify,
            'augmentation_config': self.augmentation_config,
            'folds': []
        }
        
        # Save fold subject IDs
        for i, (train_ids, val_ids) in enumerate(self.base_cv_manager.folds):
            config['folds'].append({
                'fold_idx': i,
                'train_ids': train_ids,
                'val_ids': val_ids,
                'train_size': len(train_ids),
                'val_size': len(val_ids)
            })
        
        # Ensure directory exists
        self.fold_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.fold_save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved fold configuration to {self.fold_save_path}")
    
    @classmethod
    def load_from_configuration(cls, config_path: Union[str, Path], db_manager):
        """Load cross-validation manager from saved configuration."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create manager with saved configuration
        manager = cls(
            db_manager=db_manager,
            n_folds=config['n_folds'],
            random_state=config['random_state'],
            stratify=config['stratify'],
            augmentation_config=config['augmentation_config'],
            fold_save_path=config_path
        )
        
        logger.info(f"Loaded fold configuration from {config_path}")
        return manager
    
    def run_cross_validation_experiment(self,
                                      model_factory,
                                      training_config: Dict[str, Any],
                                      save_results_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Run complete cross-validation experiment.
        
        Args:
            model_factory: Function that creates model instances
            training_config: Configuration for training (epochs, lr, etc.)
            save_results_path: Path to save experiment results
            
        Returns:
            Dictionary with cross-validation results
        """
        results = {
            'fold_results': [],
            'summary_stats': {},
            'config': {
                'n_folds': self.n_folds,
                'training_config': training_config,
                'augmentation_config': self.augmentation_config
            }
        }
        
        fold_metrics = []
        
        for fold_idx in range(self.n_folds):
            logger.info(f"Starting fold {fold_idx + 1}/{self.n_folds}")
            
            # Get data loaders for this fold
            data_loaders = self.get_fold_data_loaders(
                fold_idx=fold_idx,
                batch_size=training_config.get('batch_size', 4),
                use_augmentation=training_config.get('use_augmentation', True)
            )
            
            # Create model for this fold
            model = model_factory()
            
            # Train model (this would be implemented in training module)
            fold_result = self._train_fold(
                model=model,
                data_loaders=data_loaders,
                training_config=training_config,
                fold_idx=fold_idx
            )
            
            results['fold_results'].append(fold_result)
            fold_metrics.append(fold_result['val_metrics'])
            
            logger.info(f"Fold {fold_idx + 1} completed: "
                       f"Val Loss: {fold_result['val_metrics']['loss']:.4f}")
        
        # Calculate summary statistics
        results['summary_stats'] = self._calculate_cv_summary(fold_metrics)
        
        # Save results if path provided
        if save_results_path:
            self._save_cv_results(results, save_results_path)
        
        return results
    
    def _train_fold(self, model, data_loaders, training_config, fold_idx):
        """
        Train model for a single fold.
        
        This is a placeholder - actual training would be implemented
        in the training module.
        """
        # Placeholder implementation
        return {
            'fold_idx': fold_idx,
            'train_metrics': {'loss': 0.5, 'mae': 0.3},
            'val_metrics': {'loss': 0.6, 'mae': 0.35},
            'best_epoch': 50,
            'training_time': 3600
        }
    
    def _calculate_cv_summary(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate summary statistics across folds."""
        metrics_names = fold_metrics[0].keys()
        summary = {}
        
        for metric in metrics_names:
            values = [fold[metric] for fold in fold_metrics]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return summary
    
    def _save_cv_results(self, results: Dict[str, Any], save_path: Union[str, Path]):
        """Save cross-validation results to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved CV results to {save_path}")


class StratifiedCrossValidation:
    """
    Advanced stratified cross-validation for medical data.
    
    Provides more sophisticated stratification strategies beyond simple
    score-based stratification, including multi-criteria stratification.
    """
    
    def __init__(self, db_manager, random_state: int = 42):
        """Initialize stratified cross-validation."""
        self.db_manager = db_manager
        self.random_state = random_state
    
    def create_multi_criteria_folds(self,
                                   n_folds: int = 5,
                                   stratify_by: List[str] = None) -> List[Tuple[List[str], List[str]]]:
        """
        Create folds with multi-criteria stratification.
        
        Args:
            n_folds: Number of folds
            stratify_by: List of criteria to stratify by
                       Options: ['score', 'age', 'sex', 'dataset_source']
            
        Returns:
            List of (train_ids, val_ids) tuples for each fold
        """
        if stratify_by is None:
            stratify_by = ['score']
        
        # Get all subjects with their stratification criteria
        subjects_data = self._get_subjects_with_criteria(stratify_by)
        
        if not subjects_data:
            raise ValueError("No valid subjects found for stratification")
        
        # Create composite stratification labels
        stratification_labels = self._create_composite_labels(subjects_data, stratify_by)
        
        # Perform stratified split
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        subject_ids = list(subjects_data.keys())
        splits = skf.split(subject_ids, stratification_labels)
        
        folds = []
        for train_idx, val_idx in splits:
            train_ids = [subject_ids[i] for i in train_idx]
            val_ids = [subject_ids[i] for i in val_idx]
            folds.append((train_ids, val_ids))
        
        return folds
    
    def _get_subjects_with_criteria(self, criteria: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get subjects with their stratification criteria values."""
        with self.db_manager.get_session() as session:
            from gazimed.data.database import Subject
            subjects = session.query(Subject).all()
            
            subjects_data = {}
            for subject in subjects:
                if not (subject.validate_score() and subject.validate_clinical_features()):
                    continue
                
                subject_info = {'subject_id': subject.subject_id}
                
                if 'score' in criteria:
                    subject_info['score'] = subject.alzheimer_score
                if 'age' in criteria:
                    subject_info['age'] = subject.age
                if 'sex' in criteria:
                    subject_info['sex'] = subject.sex
                if 'dataset_source' in criteria:
                    subject_info['dataset_source'] = subject.dataset_source
                
                subjects_data[subject.subject_id] = subject_info
        
        return subjects_data
    
    def _create_composite_labels(self, subjects_data: Dict[str, Dict[str, Any]], 
                               criteria: List[str]) -> List[str]:
        """Create composite stratification labels."""
        labels = []
        
        for subject_id, data in subjects_data.items():
            label_parts = []
            
            if 'score' in criteria:
                # Discretize score into bins
                score = data['score']
                if score <= 0.33:
                    label_parts.append('low')
                elif score <= 0.66:
                    label_parts.append('med')
                else:
                    label_parts.append('high')
            
            if 'age' in criteria and data.get('age'):
                # Discretize age into bins
                age = data['age']
                if age < 65:
                    label_parts.append('young')
                elif age < 80:
                    label_parts.append('middle')
                else:
                    label_parts.append('old')
            
            if 'sex' in criteria and data.get('sex'):
                label_parts.append(data['sex'].lower())
            
            if 'dataset_source' in criteria and data.get('dataset_source'):
                label_parts.append(data['dataset_source'].lower())
            
            # Combine all criteria into single label
            composite_label = '_'.join(label_parts)
            labels.append(composite_label)
        
        return labels


# Utility functions
def create_default_cv_manager(db_manager, 
                            n_folds: int = 5,
                            use_augmentation: bool = True) -> EnhancedCrossValidationManager:
    """Create cross-validation manager with default settings."""
    augmentation_config = {
        'enabled': use_augmentation,
        'spatial_prob': 0.8,
        'intensity_prob': 0.5,
        'mixup_prob': 0.3,
        'rotation_degrees': 10.0
    } if use_augmentation else {'enabled': False}
    
    return EnhancedCrossValidationManager(
        db_manager=db_manager,
        n_folds=n_folds,
        augmentation_config=augmentation_config
    )


def validate_cv_folds(cv_manager: EnhancedCrossValidationManager) -> Dict[str, Any]:
    """Validate cross-validation folds for balance and coverage."""
    validation_results = {
        'fold_balance': [],
        'coverage_check': True,
        'stratification_quality': {}
    }
    
    all_train_ids = set()
    all_val_ids = set()
    
    for i in range(cv_manager.n_folds):
        fold_stats = cv_manager.get_fold_statistics(i)
        
        # Check fold balance
        train_size = fold_stats['train_size']
        val_size = fold_stats['val_size']
        total_size = train_size + val_size
        
        validation_results['fold_balance'].append({
            'fold_idx': i,
            'train_ratio': train_size / total_size,
            'val_ratio': val_size / total_size,
            'size_difference': abs(train_size - val_size)
        })
        
        # Collect IDs for coverage check
        train_ids, val_ids = cv_manager.base_cv_manager.folds[i]
        all_train_ids.update(train_ids)
        all_val_ids.update(val_ids)
    
    # Check coverage (all subjects should appear in validation exactly once)
    if len(all_train_ids & all_val_ids) > 0:
        validation_results['coverage_check'] = False
        validation_results['overlap_subjects'] = list(all_train_ids & all_val_ids)
    
    return validation_results


if __name__ == "__main__":
    # Example usage would go here
    print("Enhanced cross-validation module loaded successfully")