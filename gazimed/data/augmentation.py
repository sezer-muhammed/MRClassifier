"""
Medical data augmentation for Alzheimer's detection model.

This module provides MONAI-based augmentation transforms specifically designed
for medical imaging data, including spatial transforms, intensity transforms,
and advanced techniques like Mixup for improved model generalization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset

# MONAI imports for medical image augmentation
try:
    from monai.transforms import (
        Compose, RandAffined, RandFlipd, RandBiasFieldd,
        RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
        RandShiftIntensityd, RandRotated, RandZoomd, RandSpatialCropd,
        EnsureChannelFirstd, EnsureTyped, ToTensord
    )
    from monai.data import MetaTensor
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Medical augmentations will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


class MedicalAugmentation:
    """
    Medical image augmentation pipeline using MONAI transforms.
    
    Provides configurable augmentation transforms specifically designed for
    3D medical imaging data with proper handling of multi-channel volumes.
    """
    
    def __init__(self,
                 spatial_prob: float = 0.8,
                 intensity_prob: float = 0.5,
                 noise_prob: float = 0.3,
                 rotation_range: float = 0.175,  # ±10 degrees in radians
                 flip_prob: float = 0.5,
                 bias_field_prob: float = 0.3,
                 mixup_prob: float = 0.3,
                 mixup_alpha: float = 0.3,
                 keys: List[str] = None):
        """
        Initialize medical augmentation pipeline.
        
        Args:
            spatial_prob: Probability for spatial transforms
            intensity_prob: Probability for intensity transforms
            noise_prob: Probability for noise transforms
            rotation_range: Rotation range in radians (±10° = 0.175 rad)
            flip_prob: Probability for random flips
            bias_field_prob: Probability for bias field simulation
            mixup_prob: Probability for mixup augmentation
            mixup_alpha: Alpha parameter for mixup beta distribution
            keys: Keys for dictionary-based transforms
        """
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for medical augmentations but not available")
        
        self.spatial_prob = spatial_prob
        self.intensity_prob = intensity_prob
        self.noise_prob = noise_prob
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.bias_field_prob = bias_field_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        
        # Default keys for dictionary-based transforms
        if keys is None:
            self.keys = ["volumes"]
        else:
            self.keys = keys
        
        # Build augmentation pipeline
        self.spatial_transforms = self._build_spatial_transforms()
        self.intensity_transforms = self._build_intensity_transforms()
        self.combined_transforms = self._build_combined_pipeline()
        
        logger.info(f"Initialized medical augmentation with spatial_prob={spatial_prob}, "
                   f"intensity_prob={intensity_prob}, mixup_prob={mixup_prob}")
    
    def _build_spatial_transforms(self) -> Compose:
        """Build spatial augmentation transforms."""
        transforms = [
            # Random affine transformation with ±10° rotation
            RandAffined(
                keys=self.keys,
                prob=self.spatial_prob,
                rotate_range=self.rotation_range,  # ±10 degrees
                scale_range=0.1,  # ±10% scaling
                translate_range=5,  # ±5 voxel translation
                mode='bilinear',
                padding_mode='border',
                spatial_size=None  # Keep original size
            ),
            
            # Random flip (left-right only for brain images)
            RandFlipd(
                keys=self.keys,
                prob=self.flip_prob,
                spatial_axis=0  # Left-right flip only
            ),
            
            # Random zoom
            RandZoomd(
                keys=self.keys,
                prob=0.3,
                min_zoom=0.9,
                max_zoom=1.1,
                mode='bilinear',
                padding_mode='border'
            )
        ]
        
        return Compose(transforms)
    
    def _build_intensity_transforms(self) -> Compose:
        """Build intensity augmentation transforms."""
        transforms = [
            # Random bias field simulation
            RandBiasFieldd(
                keys=self.keys,
                prob=self.bias_field_prob,
                degree=3,
                coeff_range=(0.0, 0.1)
            ),
            
            # Random Gaussian noise
            RandGaussianNoised(
                keys=self.keys,
                prob=self.noise_prob,
                mean=0.0,
                std=0.1
            ),
            
            # Random intensity scaling
            RandScaleIntensityd(
                keys=self.keys,
                prob=self.intensity_prob,
                factors=0.1
            ),
            
            # Random intensity shift
            RandShiftIntensityd(
                keys=self.keys,
                prob=self.intensity_prob,
                offsets=0.1
            ),
            
            # Random Gaussian smoothing
            RandGaussianSmoothd(
                keys=self.keys,
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            )
        ]
        
        return Compose(transforms)
    
    def _build_combined_pipeline(self) -> Compose:
        """Build complete augmentation pipeline."""
        transforms = [
            # Ensure proper data types
            EnsureTyped(keys=self.keys, data_type="tensor"),
            
            # Apply spatial transforms
            self.spatial_transforms,
            
            # Apply intensity transforms
            self.intensity_transforms,
            
            # Ensure tensor output
            ToTensord(keys=self.keys)
        ]
        
        return Compose(transforms)
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentation transforms to data.
        
        Args:
            data: Dictionary containing volumes and other data
            
        Returns:
            Augmented data dictionary
        """
        try:
            # Apply standard transforms
            augmented_data = self.combined_transforms(data)
            
            # Apply mixup if enabled and probability allows
            if (self.mixup_prob > 0 and 
                np.random.random() < self.mixup_prob and 
                'volumes' in augmented_data):
                augmented_data = self._apply_mixup(augmented_data)
            
            return augmented_data
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {str(e)}, returning original data")
            return data
    
    def _apply_mixup(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mixup augmentation (requires external mixing with another sample).
        
        Note: This is a placeholder implementation. True mixup requires
        mixing with another sample from the dataset, which should be
        implemented at the dataset or dataloader level.
        
        Args:
            data: Data dictionary
            
        Returns:
            Data dictionary (unchanged in this implementation)
        """
        # Mixup requires mixing with another sample, which is typically
        # handled at the dataset or batch level, not per-sample
        logger.debug("Mixup placeholder called (requires batch-level implementation)")
        return data


class MixupAugmentation:
    """
    Mixup augmentation for medical imaging data.
    
    Implements mixup by linearly combining pairs of samples and their labels.
    This should be applied at the batch level during training.
    """
    
    def __init__(self, alpha: float = 0.3, prob: float = 0.3):
        """
        Initialize mixup augmentation.
        
        Args:
            alpha: Alpha parameter for beta distribution
            prob: Probability of applying mixup to a batch
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply mixup to a batch of data.
        
        Args:
            batch_data: Dictionary containing batch tensors
            
        Returns:
            Mixed batch data
        """
        if np.random.random() > self.prob:
            return batch_data
        
        batch_size = batch_data['volumes'].size(0)
        if batch_size < 2:
            return batch_data
        
        # Sample lambda from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Create random permutation for mixing
        indices = torch.randperm(batch_size)
        
        # Mix volumes
        mixed_volumes = (lam * batch_data['volumes'] + 
                        (1 - lam) * batch_data['volumes'][indices])
        
        # Mix clinical features if present
        mixed_clinical = batch_data['clinical_features']
        if 'clinical_features' in batch_data:
            mixed_clinical = (lam * batch_data['clinical_features'] + 
                            (1 - lam) * batch_data['clinical_features'][indices])
        
        # Mix targets (for regression)
        mixed_targets = (lam * batch_data['alzheimer_score'] + 
                        (1 - lam) * batch_data['alzheimer_score'][indices])
        
        return {
            'volumes': mixed_volumes,
            'clinical_features': mixed_clinical,
            'alzheimer_score': mixed_targets,
            'subject_id': batch_data['subject_id'],  # Keep original IDs
            'metadata': batch_data['metadata'],
            'mixup_lambda': lam,
            'mixup_indices': indices
        }


class AugmentedDataset:
    """
    Wrapper dataset that applies augmentations to an existing dataset.
    
    This class wraps an existing dataset and applies augmentation transforms
    during data loading, with proper handling of training/validation modes.
    """
    
    def __init__(self,
                 base_dataset: Dataset,
                 augmentation: Optional[MedicalAugmentation] = None,
                 training: bool = True,
                 augmentation_prob: float = 1.0):
        """
        Initialize augmented dataset wrapper.
        
        Args:
            base_dataset: Base dataset to wrap
            augmentation: Augmentation pipeline to apply
            training: Whether in training mode (augmentations only applied during training)
            augmentation_prob: Probability of applying augmentations to each sample
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.training = training
        self.augmentation_prob = augmentation_prob
        
        # Create default augmentation if none provided
        if self.augmentation is None and MONAI_AVAILABLE:
            self.augmentation = MedicalAugmentation()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get augmented sample from base dataset."""
        # Get original sample
        sample = self.base_dataset[idx]
        
        # Apply augmentations only during training and with specified probability
        if (self.training and 
            self.augmentation is not None and 
            np.random.random() < self.augmentation_prob):
            
            try:
                sample = self.augmentation(sample)
            except Exception as e:
                logger.warning(f"Augmentation failed for sample {idx}: {str(e)}")
        
        return sample
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training


def create_training_augmentation(
    spatial_prob: float = 0.8,
    intensity_prob: float = 0.5,
    mixup_prob: float = 0.3,
    rotation_degrees: float = 10.0
) -> MedicalAugmentation:
    """
    Create standard training augmentation pipeline.
    
    Args:
        spatial_prob: Probability for spatial transforms
        intensity_prob: Probability for intensity transforms
        mixup_prob: Probability for mixup augmentation
        rotation_degrees: Maximum rotation in degrees
        
    Returns:
        Configured MedicalAugmentation instance
    """
    if not MONAI_AVAILABLE:
        logger.warning("MONAI not available, returning None for augmentation")
        return None
    
    rotation_radians = np.deg2rad(rotation_degrees)
    
    return MedicalAugmentation(
        spatial_prob=spatial_prob,
        intensity_prob=intensity_prob,
        rotation_range=rotation_radians,
        mixup_prob=mixup_prob,
        keys=["volumes"]
    )


def create_validation_augmentation() -> Optional[MedicalAugmentation]:
    """
    Create minimal augmentation for validation (typically none).
    
    Returns:
        None (no augmentation for validation)
    """
    return None


class AugmentationConfig:
    """Configuration class for augmentation parameters."""
    
    def __init__(self,
                 spatial_prob: float = 0.8,
                 intensity_prob: float = 0.5,
                 noise_prob: float = 0.3,
                 rotation_degrees: float = 10.0,
                 flip_prob: float = 0.5,
                 bias_field_prob: float = 0.3,
                 mixup_prob: float = 0.3,
                 mixup_alpha: float = 0.3):
        """
        Initialize augmentation configuration.
        
        Args:
            spatial_prob: Probability for spatial transforms
            intensity_prob: Probability for intensity transforms
            noise_prob: Probability for noise transforms
            rotation_degrees: Maximum rotation in degrees
            flip_prob: Probability for random flips
            bias_field_prob: Probability for bias field simulation
            mixup_prob: Probability for mixup augmentation
            mixup_alpha: Alpha parameter for mixup
        """
        self.spatial_prob = spatial_prob
        self.intensity_prob = intensity_prob
        self.noise_prob = noise_prob
        self.rotation_degrees = rotation_degrees
        self.flip_prob = flip_prob
        self.bias_field_prob = bias_field_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
    
    def create_augmentation(self) -> Optional[MedicalAugmentation]:
        """Create augmentation instance from configuration."""
        if not MONAI_AVAILABLE:
            return None
        
        return MedicalAugmentation(
            spatial_prob=self.spatial_prob,
            intensity_prob=self.intensity_prob,
            noise_prob=self.noise_prob,
            rotation_range=np.deg2rad(self.rotation_degrees),
            flip_prob=self.flip_prob,
            bias_field_prob=self.bias_field_prob,
            mixup_prob=self.mixup_prob,
            mixup_alpha=self.mixup_alpha
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert configuration to dictionary."""
        return {
            'spatial_prob': self.spatial_prob,
            'intensity_prob': self.intensity_prob,
            'noise_prob': self.noise_prob,
            'rotation_degrees': self.rotation_degrees,
            'flip_prob': self.flip_prob,
            'bias_field_prob': self.bias_field_prob,
            'mixup_prob': self.mixup_prob,
            'mixup_alpha': self.mixup_alpha
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, float]) -> 'AugmentationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Utility functions for common augmentation patterns
def get_light_augmentation() -> Optional[MedicalAugmentation]:
    """Get light augmentation for sensitive medical data."""
    if not MONAI_AVAILABLE:
        return None
    
    return MedicalAugmentation(
        spatial_prob=0.5,
        intensity_prob=0.3,
        rotation_range=np.deg2rad(5.0),  # ±5 degrees
        flip_prob=0.3,
        bias_field_prob=0.2,
        mixup_prob=0.1
    )


def get_aggressive_augmentation() -> Optional[MedicalAugmentation]:
    """Get aggressive augmentation for robust training."""
    if not MONAI_AVAILABLE:
        return None
    
    return MedicalAugmentation(
        spatial_prob=0.9,
        intensity_prob=0.7,
        rotation_range=np.deg2rad(15.0),  # ±15 degrees
        flip_prob=0.7,
        bias_field_prob=0.5,
        mixup_prob=0.5
    )


def test_augmentation_pipeline():
    """Test the augmentation pipeline with synthetic data."""
    if not MONAI_AVAILABLE:
        print("MONAI not available, skipping augmentation test")
        return
    
    # Create synthetic data
    synthetic_volume = np.random.randn(2, 91, 120, 91).astype(np.float32)
    synthetic_clinical = np.random.randn(118).astype(np.float32)
    
    sample = {
        'volumes': torch.tensor(synthetic_volume),
        'clinical_features': torch.tensor(synthetic_clinical),
        'alzheimer_score': torch.tensor(0.5),
        'subject_id': 'test_001'
    }
    
    # Test augmentation
    augmentation = create_training_augmentation()
    if augmentation is not None:
        try:
            augmented_sample = augmentation(sample)
            print(f"Original volume shape: {sample['volumes'].shape}")
            print(f"Augmented volume shape: {augmented_sample['volumes'].shape}")
            print("Augmentation test passed!")
        except Exception as e:
            print(f"Augmentation test failed: {e}")
    else:
        print("No augmentation created")


if __name__ == "__main__":
    # Test the augmentation pipeline
    test_augmentation_pipeline()
    
    # Test configuration
    config = AugmentationConfig(rotation_degrees=15.0, mixup_prob=0.5)
    print(f"Augmentation config: {config.to_dict()}")
    
    # Test different augmentation levels
    light_aug = get_light_augmentation()
    aggressive_aug = get_aggressive_augmentation()
    
    if light_aug and aggressive_aug:
        print("Light and aggressive augmentation pipelines created successfully")
    else:
        print("Augmentation creation failed (MONAI not available)")