"""
NIfTI preprocessing pipeline for Alzheimer's detection model.

This module provides comprehensive preprocessing functionality for paired T1-weighted MRI 
and ^18F-FDG PET brain imaging volumes, including N4 bias correction, MNI registration,
resampling, and normalization.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# Configure logging
logger = logging.getLogger(__name__)

# Suppress SimpleITK warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="SimpleITK")


class NIfTILoader:
    """
    NIfTI file loader with N4 bias field correction.
    
    This class handles loading of NIfTI files and applies N4 bias field correction
    to improve image quality and standardize intensity distributions.
    """
    
    def __init__(self, 
                 apply_n4_correction: bool = True,
                 n4_iterations: int = 50,
                 n4_convergence_threshold: float = 0.001):
        """
        Initialize NIfTI loader.
        
        Args:
            apply_n4_correction: Whether to apply N4 bias field correction
            n4_iterations: Number of iterations for N4 correction
            n4_convergence_threshold: Convergence threshold for N4 correction
        """
        self.apply_n4_correction = apply_n4_correction
        self.n4_iterations = n4_iterations
        self.n4_convergence_threshold = n4_convergence_threshold
        
        # Initialize N4 bias corrector if needed
        if self.apply_n4_correction:
            self.n4_corrector = sitk.N4BiasFieldCorrectionImageFilter()
            self.n4_corrector.SetMaximumNumberOfIterations([n4_iterations])
            self.n4_corrector.SetConvergenceThreshold(n4_convergence_threshold)
    
    def load_nifti(self, nifti_path: Union[str, Path]) -> np.ndarray:
        """
        Load NIfTI file and return as numpy array.
        
        Args:
            nifti_path: Path to NIfTI file
            
        Returns:
            3D numpy array with shape (H, W, D)
            
        Raises:
            FileNotFoundError: If NIfTI file doesn't exist
            ValueError: If file cannot be loaded or has invalid dimensions
        """
        nifti_path = Path(nifti_path)
        
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
        
        try:
            # Load NIfTI file using nibabel
            nifti_img = nib.load(str(nifti_path))
            volume_data = nifti_img.get_fdata()
            
            # Ensure 3D volume
            if volume_data.ndim != 3:
                if volume_data.ndim == 4 and volume_data.shape[3] == 1:
                    # Remove singleton dimension
                    volume_data = volume_data.squeeze(axis=3)
                else:
                    raise ValueError(f"Expected 3D volume, got shape {volume_data.shape}")
            
            # Convert to float32 for processing
            volume_data = volume_data.astype(np.float32)
            
            logger.debug(f"Loaded NIfTI file {nifti_path} with shape {volume_data.shape}")
            return volume_data
            
        except Exception as e:
            raise ValueError(f"Failed to load NIfTI file {nifti_path}: {str(e)}")
    
    def apply_n4_bias_correction(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply N4 bias field correction to volume.
        
        Args:
            volume: 3D numpy array
            
        Returns:
            Bias-corrected 3D numpy array
        """
        if not self.apply_n4_correction:
            return volume
        
        try:
            # Convert numpy array to SimpleITK image
            sitk_image = sitk.GetImageFromArray(volume)
            
            # Apply N4 bias field correction
            corrected_image = self.n4_corrector.Execute(sitk_image)
            
            # Convert back to numpy array
            corrected_volume = sitk.GetArrayFromImage(corrected_image)
            
            logger.debug("Applied N4 bias field correction")
            return corrected_volume.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"N4 bias correction failed: {str(e)}, returning original volume")
            return volume
    
    def load_and_correct(self, nifti_path: Union[str, Path]) -> np.ndarray:
        """
        Load NIfTI file and apply N4 bias correction.
        
        Args:
            nifti_path: Path to NIfTI file
            
        Returns:
            Preprocessed 3D numpy array
        """
        # Load NIfTI file
        volume = self.load_nifti(nifti_path)
        
        # Apply N4 bias correction
        if self.apply_n4_correction:
            volume = self.apply_n4_bias_correction(volume)
        
        return volume
    
    def validate_volume_shape(self, volume: np.ndarray, 
                            expected_shape: Optional[Tuple[int, int, int]] = None) -> bool:
        """
        Validate volume shape and properties.
        
        Args:
            volume: 3D numpy array to validate
            expected_shape: Expected shape tuple (optional)
            
        Returns:
            True if volume is valid, False otherwise
        """
        if volume.ndim != 3:
            logger.error(f"Volume must be 3D, got {volume.ndim}D")
            return False
        
        if expected_shape and volume.shape != expected_shape:
            logger.error(f"Volume shape {volume.shape} doesn't match expected {expected_shape}")
            return False
        
        if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
            logger.error("Volume contains NaN or infinite values")
            return False
        
        if volume.size == 0:
            logger.error("Volume is empty")
            return False
        
        return True


def load_nifti_with_correction(nifti_path: Union[str, Path], 
                              apply_n4: bool = True) -> np.ndarray:
    """
    Convenience function to load NIfTI file with optional N4 bias correction.
    
    Args:
        nifti_path: Path to NIfTI file
        apply_n4: Whether to apply N4 bias correction
        
    Returns:
        Preprocessed 3D numpy array
    """
    loader = NIfTILoader(apply_n4_correction=apply_n4)
    return loader.load_and_correct(nifti_path)


def validate_nifti_pair(mri_path: Union[str, Path], 
                       pet_path: Union[str, Path]) -> bool:
    """
    Validate that MRI and PET NIfTI files exist and can be loaded.
    
    Args:
        mri_path: Path to MRI NIfTI file
        pet_path: Path to PET NIfTI file
        
    Returns:
        True if both files are valid, False otherwise
    """
    try:
        loader = NIfTILoader(apply_n4_correction=False)  # Skip correction for validation
        
        # Check if files exist
        mri_path = Path(mri_path)
        pet_path = Path(pet_path)
        
        if not mri_path.exists():
            logger.error(f"MRI file not found: {mri_path}")
            return False
        
        if not pet_path.exists():
            logger.error(f"PET file not found: {pet_path}")
            return False
        
        # Try loading both files
        mri_volume = loader.load_nifti(mri_path)
        pet_volume = loader.load_nifti(pet_path)
        
        # Validate volumes
        if not loader.validate_volume_shape(mri_volume):
            logger.error(f"Invalid MRI volume: {mri_path}")
            return False
        
        if not loader.validate_volume_shape(pet_volume):
            logger.error(f"Invalid PET volume: {pet_path}")
            return False
        
        # Check if volumes have compatible shapes (they should match after preprocessing)
        if mri_volume.shape != pet_volume.shape:
            logger.warning(f"MRI shape {mri_volume.shape} != PET shape {pet_volume.shape}")
            # This is acceptable as they will be registered and resampled later
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False


class MNIRegistrationResampler:
    """
    MNI152 registration and resampling for brain volumes.
    
    This class handles affine registration to MNI152 template space and 
    resampling to standardized 1mm³ isotropic resolution with target dimensions (91, 120, 91).
    """
    
    def __init__(self, 
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (91, 120, 91),
                 interpolation_method: str = 'linear'):
        """
        Initialize MNI registration and resampling.
        
        Args:
            target_spacing: Target voxel spacing in mm (x, y, z)
            target_size: Target volume dimensions (x, y, z)
            interpolation_method: Interpolation method ('linear', 'nearest', 'cubic')
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.interpolation_method = interpolation_method
        
        # Map interpolation methods to SimpleITK constants
        self.interpolation_map = {
            'linear': sitk.sitkLinear,
            'nearest': sitk.sitkNearestNeighbor,
            'cubic': sitk.sitkBSpline
        }
        
        if interpolation_method not in self.interpolation_map:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
        
        self.interpolator = self.interpolation_map[interpolation_method]
        
        # Initialize registration components
        self.registration_method = sitk.ImageRegistrationMethod()
        self._setup_registration()
    
    def _setup_registration(self):
        """Set up the registration method with appropriate parameters."""
        # Similarity metric
        self.registration_method.SetMetricAsMeanSquares()
        
        # Optimizer
        self.registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=0.001,
            numberOfIterations=200
        )
        self.registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Setup for the multi-resolution framework
        self.registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        self.registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        self.registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Don't optimize in-place, we would possibly like to run this multiple times
        self.registration_method.SetInitialTransform(sitk.AffineTransform(3), inPlace=False)
        
        # Set interpolator
        self.registration_method.SetInterpolator(self.interpolator)
    
    def create_mni_template(self) -> sitk.Image:
        """
        Create a synthetic MNI152 template for registration.
        
        Since we don't have access to the actual MNI152 template, we create
        a synthetic template with the target dimensions and spacing.
        
        Returns:
            SimpleITK image representing MNI template space
        """
        # Create template image with target dimensions and spacing
        template_image = sitk.Image(self.target_size, sitk.sitkFloat32)
        template_image.SetSpacing(self.target_spacing)
        
        # Set origin to center the image
        origin = [-(size * spacing) / 2.0 for size, spacing in zip(self.target_size, self.target_spacing)]
        template_image.SetOrigin(origin)
        
        # Create a brain-like template (simple ellipsoid)
        template_array = sitk.GetArrayFromImage(template_image)
        center = [s // 2 for s in template_array.shape]
        
        for i in range(template_array.shape[0]):
            for j in range(template_array.shape[1]):
                for k in range(template_array.shape[2]):
                    # Create ellipsoid shape
                    dist_sq = ((i - center[0]) / (center[0] * 0.8))**2 + \
                             ((j - center[1]) / (center[1] * 0.8))**2 + \
                             ((k - center[2]) / (center[2] * 0.8))**2
                    
                    if dist_sq <= 1.0:
                        template_array[i, j, k] = 1000.0 * (1.0 - dist_sq)
        
        template_image = sitk.GetImageFromArray(template_array)
        template_image.SetSpacing(self.target_spacing)
        template_image.SetOrigin(origin)
        
        return template_image
    
    def register_to_mni(self, volume: np.ndarray, 
                       original_spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Register volume to MNI152 template space using affine transformation.
        
        Args:
            volume: 3D numpy array to register
            original_spacing: Original voxel spacing (if None, assumes 1mm isotropic)
            
        Returns:
            Registered 3D numpy array
        """
        try:
            # Convert numpy array to SimpleITK image
            moving_image = sitk.GetImageFromArray(volume)
            
            # Set spacing if provided
            if original_spacing is not None:
                moving_image.SetSpacing(original_spacing)
            else:
                # Assume 1mm isotropic if not specified
                moving_image.SetSpacing((1.0, 1.0, 1.0))
            
            # Create MNI template
            fixed_image = self.create_mni_template()
            
            # Perform registration
            try:
                final_transform = self.registration_method.Execute(fixed_image, moving_image)
                
                # Apply transformation and resample
                registered_image = sitk.Resample(
                    moving_image,
                    fixed_image,
                    final_transform,
                    self.interpolator,
                    0.0,  # Default pixel value
                    moving_image.GetPixelID()
                )
                
                # Convert back to numpy array
                registered_volume = sitk.GetArrayFromImage(registered_image)
                
                logger.debug(f"Successfully registered volume to MNI space with shape {registered_volume.shape}")
                return registered_volume.astype(np.float32)
                
            except Exception as reg_error:
                logger.warning(f"Registration failed: {str(reg_error)}, falling back to resampling only")
                # Fall back to simple resampling without registration
                return self.resample_to_target(volume, original_spacing)
                
        except Exception as e:
            logger.error(f"MNI registration failed: {str(e)}")
            raise ValueError(f"Failed to register volume to MNI space: {str(e)}")
    
    def resample_to_target(self, volume: np.ndarray, 
                          original_spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Resample volume to target spacing and dimensions without registration.
        
        Args:
            volume: 3D numpy array to resample
            original_spacing: Original voxel spacing (if None, assumes 1mm isotropic)
            
        Returns:
            Resampled 3D numpy array with target dimensions
        """
        try:
            # Convert numpy array to SimpleITK image
            image = sitk.GetImageFromArray(volume)
            
            # Set original spacing
            if original_spacing is not None:
                image.SetSpacing(original_spacing)
            else:
                image.SetSpacing((1.0, 1.0, 1.0))
            
            # Create reference image with target properties
            reference_image = sitk.Image(self.target_size, sitk.sitkFloat32)
            reference_image.SetSpacing(self.target_spacing)
            
            # Set origin to center the image
            origin = [-(size * spacing) / 2.0 for size, spacing in zip(self.target_size, self.target_spacing)]
            reference_image.SetOrigin(origin)
            
            # Resample to target dimensions and spacing
            resampled_image = sitk.Resample(
                image,
                reference_image,
                sitk.Transform(),  # Identity transform
                self.interpolator,
                0.0,  # Default pixel value
                image.GetPixelID()
            )
            
            # Convert back to numpy array
            resampled_volume = sitk.GetArrayFromImage(resampled_image)
            
            logger.debug(f"Resampled volume to target shape {resampled_volume.shape}")
            return resampled_volume.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Resampling failed: {str(e)}")
            raise ValueError(f"Failed to resample volume: {str(e)}")
    
    def validate_output_dimensions(self, volume: np.ndarray) -> bool:
        """
        Validate that output volume has the expected target dimensions.
        
        Args:
            volume: 3D numpy array to validate
            
        Returns:
            True if dimensions match target, False otherwise
        """
        expected_shape = self.target_size
        if volume.shape != expected_shape:
            logger.error(f"Output shape {volume.shape} doesn't match target {expected_shape}")
            return False
        
        return True


def register_and_resample_volume(volume: np.ndarray,
                               original_spacing: Optional[Tuple[float, float, float]] = None,
                               target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                               target_size: Tuple[int, int, int] = (91, 120, 91),
                               interpolation: str = 'linear') -> np.ndarray:
    """
    Convenience function to register volume to MNI space and resample to target dimensions.
    
    Args:
        volume: 3D numpy array to process
        original_spacing: Original voxel spacing in mm
        target_spacing: Target voxel spacing in mm
        target_size: Target volume dimensions
        interpolation: Interpolation method
        
    Returns:
        Registered and resampled 3D numpy array
    """
    registrar = MNIRegistrationResampler(
        target_spacing=target_spacing,
        target_size=target_size,
        interpolation_method=interpolation
    )
    
    registered_volume = registrar.register_to_mni(volume, original_spacing)
    
    # Validate output dimensions
    if not registrar.validate_output_dimensions(registered_volume):
        raise ValueError(f"Output dimensions validation failed")
    
    return registered_volume


class VolumeNormalizer:
    """
    Volume normalization and combination utilities.
    
    This class handles Z-score normalization of individual volumes and 
    combination of MRI and PET volumes into multi-channel tensors.
    """
    
    def __init__(self, 
                 normalization_method: str = 'zscore',
                 include_difference_channel: bool = False,
                 mask_background: bool = True,
                 background_threshold: float = 0.01):
        """
        Initialize volume normalizer.
        
        Args:
            normalization_method: Normalization method ('zscore', 'minmax', 'none')
            include_difference_channel: Whether to include PET-MRI difference channel
            mask_background: Whether to mask background voxels during normalization
            background_threshold: Threshold for background masking (fraction of max intensity)
        """
        self.normalization_method = normalization_method
        self.include_difference_channel = include_difference_channel
        self.mask_background = mask_background
        self.background_threshold = background_threshold
        
        if normalization_method not in ['zscore', 'minmax', 'none']:
            raise ValueError(f"Unsupported normalization method: {normalization_method}")
    
    def create_brain_mask(self, volume: np.ndarray) -> np.ndarray:
        """
        Create a brain mask to exclude background voxels from normalization.
        
        Args:
            volume: 3D numpy array
            
        Returns:
            Binary mask array (True for brain voxels, False for background)
        """
        # Simple thresholding approach
        max_intensity = np.max(volume)
        threshold = max_intensity * self.background_threshold
        
        # Create initial mask
        mask = volume > threshold
        
        # Apply morphological operations to clean up the mask
        from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
        
        # Fill holes and smooth the mask
        mask = binary_fill_holes(mask)
        mask = binary_erosion(mask, iterations=1)
        mask = binary_dilation(mask, iterations=2)
        
        return mask
    
    def zscore_normalize(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Z-score normalization to volume.
        
        Args:
            volume: 3D numpy array to normalize
            mask: Optional binary mask for brain region
            
        Returns:
            Z-score normalized volume
        """
        if mask is not None:
            # Calculate statistics only within the mask
            masked_values = volume[mask]
            if len(masked_values) == 0:
                logger.warning("Empty mask provided, using whole volume for normalization")
                mean_val = np.mean(volume)
                std_val = np.std(volume)
            else:
                mean_val = np.mean(masked_values)
                std_val = np.std(masked_values)
        else:
            mean_val = np.mean(volume)
            std_val = np.std(volume)
        
        # Avoid division by zero
        if std_val == 0:
            logger.warning("Standard deviation is zero, returning zero-centered volume")
            return volume - mean_val
        
        # Apply Z-score normalization
        normalized_volume = (volume - mean_val) / std_val
        
        return normalized_volume.astype(np.float32)
    
    def minmax_normalize(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply min-max normalization to volume (scale to 0-1 range).
        
        Args:
            volume: 3D numpy array to normalize
            mask: Optional binary mask for brain region
            
        Returns:
            Min-max normalized volume
        """
        if mask is not None:
            masked_values = volume[mask]
            if len(masked_values) == 0:
                logger.warning("Empty mask provided, using whole volume for normalization")
                min_val = np.min(volume)
                max_val = np.max(volume)
            else:
                min_val = np.min(masked_values)
                max_val = np.max(masked_values)
        else:
            min_val = np.min(volume)
            max_val = np.max(volume)
        
        # Avoid division by zero
        if max_val == min_val:
            logger.warning("Min and max values are equal, returning zero array")
            return np.zeros_like(volume)
        
        # Apply min-max normalization
        normalized_volume = (volume - min_val) / (max_val - min_val)
        
        return normalized_volume.astype(np.float32)
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize volume using the specified method.
        
        Args:
            volume: 3D numpy array to normalize
            
        Returns:
            Normalized volume
        """
        if self.normalization_method == 'none':
            return volume.astype(np.float32)
        
        # Create brain mask if needed
        mask = None
        if self.mask_background:
            mask = self.create_brain_mask(volume)
        
        # Apply normalization
        if self.normalization_method == 'zscore':
            return self.zscore_normalize(volume, mask)
        elif self.normalization_method == 'minmax':
            return self.minmax_normalize(volume, mask)
        else:
            return volume.astype(np.float32)
    
    def combine_volumes(self, mri_volume: np.ndarray, pet_volume: np.ndarray) -> np.ndarray:
        """
        Combine MRI and PET volumes into multi-channel tensor.
        
        Args:
            mri_volume: 3D MRI volume
            pet_volume: 3D PET volume
            
        Returns:
            Combined volume with shape (2, H, W, D) or (3, H, W, D) if difference channel included
        """
        # Validate input shapes
        if mri_volume.shape != pet_volume.shape:
            raise ValueError(f"MRI shape {mri_volume.shape} != PET shape {pet_volume.shape}")
        
        # Normalize volumes
        mri_normalized = self.normalize_volume(mri_volume)
        pet_normalized = self.normalize_volume(pet_volume)
        
        # Combine into multi-channel tensor
        combined = np.stack([mri_normalized, pet_normalized], axis=0)
        
        # Add difference channel if requested
        if self.include_difference_channel:
            difference_channel = pet_normalized - mri_normalized
            combined = np.concatenate([combined, difference_channel[np.newaxis, ...]], axis=0)
        
        logger.debug(f"Combined volumes into tensor with shape {combined.shape}")
        return combined.astype(np.float32)
    
    def validate_combined_volume(self, combined_volume: np.ndarray) -> bool:
        """
        Validate combined volume properties.
        
        Args:
            combined_volume: Multi-channel volume to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check dimensions
        if combined_volume.ndim != 4:
            logger.error(f"Combined volume must be 4D, got {combined_volume.ndim}D")
            return False
        
        # Check number of channels
        expected_channels = 3 if self.include_difference_channel else 2
        if combined_volume.shape[0] != expected_channels:
            logger.error(f"Expected {expected_channels} channels, got {combined_volume.shape[0]}")
            return False
        
        # Check for invalid values
        if np.any(np.isnan(combined_volume)) or np.any(np.isinf(combined_volume)):
            logger.error("Combined volume contains NaN or infinite values")
            return False
        
        return True


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for paired MRI/PET volumes.
    
    This class combines all preprocessing steps: loading, N4 correction,
    MNI registration, resampling, normalization, and volume combination.
    """
    
    def __init__(self,
                 apply_n4_correction: bool = True,
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (91, 120, 91),
                 normalization_method: str = 'zscore',
                 include_difference_channel: bool = False,
                 interpolation_method: str = 'linear'):
        """
        Initialize preprocessing pipeline.
        
        Args:
            apply_n4_correction: Whether to apply N4 bias correction
            target_spacing: Target voxel spacing in mm
            target_size: Target volume dimensions
            normalization_method: Volume normalization method
            include_difference_channel: Whether to include difference channel
            interpolation_method: Interpolation method for resampling
        """
        self.loader = NIfTILoader(apply_n4_correction=apply_n4_correction)
        self.registrar = MNIRegistrationResampler(
            target_spacing=target_spacing,
            target_size=target_size,
            interpolation_method=interpolation_method
        )
        self.normalizer = VolumeNormalizer(
            normalization_method=normalization_method,
            include_difference_channel=include_difference_channel
        )
        
        self.target_size = target_size
        self.include_difference_channel = include_difference_channel
    
    def process_volume_pair(self, mri_path: Union[str, Path], 
                           pet_path: Union[str, Path]) -> np.ndarray:
        """
        Process a pair of MRI and PET volumes through the complete pipeline.
        
        Args:
            mri_path: Path to MRI NIfTI file
            pet_path: Path to PET NIfTI file
            
        Returns:
            Preprocessed multi-channel volume with shape (2-3, 91, 120, 91)
        """
        try:
            # Step 1: Load and apply N4 bias correction
            logger.info(f"Loading MRI: {mri_path}")
            mri_volume = self.loader.load_and_correct(mri_path)
            
            logger.info(f"Loading PET: {pet_path}")
            pet_volume = self.loader.load_and_correct(pet_path)
            
            # Step 2: Register to MNI space and resample
            logger.info("Registering MRI to MNI space")
            mri_registered = self.registrar.register_to_mni(mri_volume)
            
            logger.info("Registering PET to MNI space")
            pet_registered = self.registrar.register_to_mni(pet_volume)
            
            # Step 3: Validate dimensions
            if not self.registrar.validate_output_dimensions(mri_registered):
                raise ValueError("MRI registration failed dimension validation")
            
            if not self.registrar.validate_output_dimensions(pet_registered):
                raise ValueError("PET registration failed dimension validation")
            
            # Step 4: Normalize and combine volumes
            logger.info("Normalizing and combining volumes")
            combined_volume = self.normalizer.combine_volumes(mri_registered, pet_registered)
            
            # Step 5: Final validation
            if not self.normalizer.validate_combined_volume(combined_volume):
                raise ValueError("Combined volume failed validation")
            
            logger.info(f"Successfully processed volume pair with final shape {combined_volume.shape}")
            return combined_volume
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {mri_path}, {pet_path}: {str(e)}")
            raise ValueError(f"Preprocessing pipeline failed: {str(e)}")
    
    def get_output_shape(self) -> Tuple[int, int, int, int]:
        """
        Get the expected output shape of the preprocessing pipeline.
        
        Returns:
            Expected output shape (channels, height, width, depth)
        """
        channels = 3 if self.include_difference_channel else 2
        return (channels, *self.target_size)


def preprocess_volume_pair(mri_path: Union[str, Path], 
                          pet_path: Union[str, Path],
                          apply_n4: bool = True,
                          include_difference: bool = False,
                          normalization: str = 'zscore') -> np.ndarray:
    """
    Convenience function to preprocess a pair of MRI/PET volumes.
    
    Args:
        mri_path: Path to MRI NIfTI file
        pet_path: Path to PET NIfTI file
        apply_n4: Whether to apply N4 bias correction
        include_difference: Whether to include difference channel
        normalization: Normalization method ('zscore', 'minmax', 'none')
        
    Returns:
        Preprocessed multi-channel volume
    """
    pipeline = PreprocessingPipeline(
        apply_n4_correction=apply_n4,
        normalization_method=normalization,
        include_difference_channel=include_difference
    )
    
    return pipeline.process_volume_pair(mri_path, pet_path)