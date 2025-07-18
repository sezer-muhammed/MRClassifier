"""
NIfTI preprocessing pipeline for Alzheimer's detection model.

This module provides comprehensive preprocessing functionality for paired T1-weighted MRI 
and ^18F-FDG PET brain imaging volumes, including N4 bias correction, MNI registration,
resampling, and normalization.
"""

import logging
import warnings
import os
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# Configure logging
logger = logging.getLogger(__name__)

# Comprehensive SimpleITK warning suppression
# Suppress Python-level warnings
warnings.filterwarnings("ignore", category=UserWarning, module="SimpleITK")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="SimpleITK")

# Suppress SimpleITK C++ level warnings by redirecting ITK output
try:
    # Set ITK global warning display to off
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
except AttributeError:
    # Fallback for older SimpleITK versions
    pass

# Additional environment variable to suppress ITK warnings
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['ITK_USE_THREADPOOL'] = '0'


# Timing utilities for performance profiling
class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        logger.log(self.log_level, f"{self.name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


def time_function(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.debug(f"{func.__name__} took {duration:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.error(f"{func.__name__} failed after {duration:.4f} seconds: {str(e)}")
            raise
    return wrapper


class PerformanceProfiler:
    """Class to track and report performance metrics."""
    
    def __init__(self):
        self.timings: Dict[str, list] = {}
        self.counters: Dict[str, int] = {}
    
    def record_time(self, operation: str, duration: float):
        """Record timing for an operation."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
    
    def increment_counter(self, counter: str):
        """Increment a counter."""
        self.counters[counter] = self.counters.get(counter, 0) + 1
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return {}
        
        times = self.timings[operation]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'min': min(times),
            'max': max(times),
            'std': np.std(times)
        }
    
    def report(self) -> str:
        """Generate a performance report."""
        report = ["\n=== Performance Report ==="]
        
        for operation, times in self.timings.items():
            stats = self.get_stats(operation)
            if stats:
                report.append(f"\n{operation}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Total: {stats['total']:.4f}s")
                report.append(f"  Mean: {stats['mean']:.4f}s")
                report.append(f"  Median: {stats['median']:.4f}s")
                report.append(f"  Min: {stats['min']:.4f}s")
                report.append(f"  Max: {stats['max']:.4f}s")
                report.append(f"  Std: {stats['std']:.4f}s")
        
        if self.counters:
            report.append("\nCounters:")
            for counter, value in self.counters.items():
                report.append(f"  {counter}: {value}")
        
        return "\n".join(report)


# Global profiler instance
profiler = PerformanceProfiler()


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
    
    @time_function
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
    
    @time_function
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
    For pre-aligned data, registration can be skipped to avoid unnecessary processing.
    """
    
    def __init__(self, 
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 target_size: Tuple[int, int, int] = (91, 120, 91),
                 interpolation_method: str = 'linear',
                 skip_registration: bool = True):
        """
        Initialize MNI registration and resampling.
        
        Args:
            target_spacing: Target voxel spacing in mm (x, y, z)
            target_size: Target volume dimensions (x, y, z)
            interpolation_method: Interpolation method ('linear', 'nearest', 'cubic')
            skip_registration: If True, skip registration and only resample (for pre-aligned data)
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.interpolation_method = interpolation_method
        self.skip_registration = skip_registration
        
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
    
    @time_function
    def register_to_mni(self, volume: np.ndarray, 
                       original_spacing: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Register volume to MNI152 template space using affine transformation.
        If skip_registration is True, only resampling is performed.
        
        Args:
            volume: 3D numpy array to register
            original_spacing: Original voxel spacing (if None, assumes 1mm isotropic)
            
        Returns:
            Registered/resampled 3D numpy array
        """
        # Skip registration for pre-aligned data
        if self.skip_registration:
            logger.debug("Skipping registration for pre-aligned data, performing resampling only")
            return self.resample_to_target(volume, original_spacing)
        
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
    
    @time_function
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
    
    @time_function
    def zscore_normalize(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Z-score normalization to volume.
        
        Args:
            volume: 3D numpy array to normalize
            mask: Optional binary mask for brain region
            
        Returns:
            Z-score normalized volume
        """
        # Check for and handle NaN/infinite values in input
        if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
            logger.warning("Input volume contains NaN or infinite values, cleaning them")
            # Replace NaN with 0 and infinite values with volume max/min
            volume = np.nan_to_num(volume, nan=0.0, posinf=np.nanmax(volume[np.isfinite(volume)]), 
                                 neginf=np.nanmin(volume[np.isfinite(volume)]))
        
        if mask is not None:
            # Calculate statistics only within the mask
            masked_values = volume[mask]
            if len(masked_values) == 0:
                logger.warning("Empty mask provided, using whole volume for normalization")
                # Use finite values only for statistics
                finite_values = volume[np.isfinite(volume)]
                if len(finite_values) == 0:
                    logger.error("No finite values found in volume")
                    return np.zeros_like(volume, dtype=np.float32)
                mean_val = np.mean(finite_values)
                std_val = np.std(finite_values)
            else:
                # Use only finite values from mask
                finite_masked = masked_values[np.isfinite(masked_values)]
                if len(finite_masked) == 0:
                    logger.error("No finite values found in masked region")
                    return np.zeros_like(volume, dtype=np.float32)
                mean_val = np.mean(finite_masked)
                std_val = np.std(finite_masked)
        else:
            # Use finite values only for statistics
            finite_values = volume[np.isfinite(volume)]
            if len(finite_values) == 0:
                logger.error("No finite values found in volume")
                return np.zeros_like(volume, dtype=np.float32)
            mean_val = np.mean(finite_values)
            std_val = np.std(finite_values)
        
        # Avoid division by zero or very small std
        if std_val == 0 or std_val < 1e-8:
            logger.warning(f"Standard deviation is {std_val}, returning zero-centered volume")
            normalized_volume = volume - mean_val
        else:
            # Apply Z-score normalization
            normalized_volume = (volume - mean_val) / std_val
        
        # Final check for NaN/infinite values and clean them
        normalized_volume = np.nan_to_num(normalized_volume, nan=0.0, posinf=10.0, neginf=-10.0)
        
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
        # Check for and handle NaN/infinite values in input
        if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
            logger.warning("Input volume contains NaN or infinite values, cleaning them")
            # Replace NaN with 0 and infinite values with volume max/min
            volume = np.nan_to_num(volume, nan=0.0, posinf=np.nanmax(volume[np.isfinite(volume)]), 
                                 neginf=np.nanmin(volume[np.isfinite(volume)]))
        
        if mask is not None:
            masked_values = volume[mask]
            if len(masked_values) == 0:
                logger.warning("Empty mask provided, using whole volume for normalization")
                # Use finite values only for statistics
                finite_values = volume[np.isfinite(volume)]
                if len(finite_values) == 0:
                    logger.error("No finite values found in volume")
                    return np.zeros_like(volume, dtype=np.float32)
                min_val = np.min(finite_values)
                max_val = np.max(finite_values)
            else:
                # Use only finite values from mask
                finite_masked = masked_values[np.isfinite(masked_values)]
                if len(finite_masked) == 0:
                    logger.error("No finite values found in masked region")
                    return np.zeros_like(volume, dtype=np.float32)
                min_val = np.min(finite_masked)
                max_val = np.max(finite_masked)
        else:
            # Use finite values only for statistics
            finite_values = volume[np.isfinite(volume)]
            if len(finite_values) == 0:
                logger.error("No finite values found in volume")
                return np.zeros_like(volume, dtype=np.float32)
            min_val = np.min(finite_values)
            max_val = np.max(finite_values)
        
        # Avoid division by zero or very small range
        if max_val == min_val or abs(max_val - min_val) < 1e-8:
            logger.warning(f"Min ({min_val}) and max ({max_val}) values are too close, returning zero array")
            return np.zeros_like(volume, dtype=np.float32)
        
        # Apply min-max normalization
        normalized_volume = (volume - min_val) / (max_val - min_val)
        
        # Final check for NaN/infinite values and clean them
        normalized_volume = np.nan_to_num(normalized_volume, nan=0.0, posinf=1.0, neginf=0.0)
        
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
    
    @time_function
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
        
        # Check for NaN/infinite values in input volumes before normalization
        if np.any(np.isnan(mri_volume)) or np.any(np.isinf(mri_volume)):
            logger.warning("MRI volume contains NaN or infinite values before normalization")
        if np.any(np.isnan(pet_volume)) or np.any(np.isinf(pet_volume)):
            logger.warning("PET volume contains NaN or infinite values before normalization")
        
        # Normalize volumes
        mri_normalized = self.normalize_volume(mri_volume)
        pet_normalized = self.normalize_volume(pet_volume)
        
        # Validate normalized volumes
        if np.any(np.isnan(mri_normalized)) or np.any(np.isinf(mri_normalized)):
            logger.error("MRI volume contains NaN or infinite values after normalization")
            mri_normalized = np.nan_to_num(mri_normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if np.any(np.isnan(pet_normalized)) or np.any(np.isinf(pet_normalized)):
            logger.error("PET volume contains NaN or infinite values after normalization")
            pet_normalized = np.nan_to_num(pet_normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Combine into multi-channel tensor
        combined = np.stack([mri_normalized, pet_normalized], axis=0)
        
        # Add difference channel if requested
        if self.include_difference_channel:
            difference_channel = pet_normalized - mri_normalized
            # Clean difference channel
            difference_channel = np.nan_to_num(difference_channel, nan=0.0, posinf=10.0, neginf=-10.0)
            combined = np.concatenate([combined, difference_channel[np.newaxis, ...]], axis=0)
        
        # Final validation of combined volume
        if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
            logger.error("Combined volume still contains NaN or infinite values, performing final cleanup")
            combined = np.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0)
        
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
                 interpolation_method: str = 'linear',
                 skip_registration: bool = True):
        """
        Initialize preprocessing pipeline.
        
        Args:
            apply_n4_correction: Whether to apply N4 bias correction
            target_spacing: Target voxel spacing in mm
            target_size: Target volume dimensions
            normalization_method: Volume normalization method
            include_difference_channel: Whether to include difference channel
            interpolation_method: Interpolation method for resampling
            skip_registration: Skip registration for pre-aligned data (default: True)
        """
        self.loader = NIfTILoader(apply_n4_correction=apply_n4_correction)
        self.registrar = MNIRegistrationResampler(
            target_spacing=target_spacing,
            target_size=target_size,
            interpolation_method=interpolation_method,
            skip_registration=skip_registration
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
        pipeline_start = time.perf_counter()
        
        try:
            # Step 1: Load and apply N4 bias correction
            with Timer(f"Loading MRI {Path(mri_path).name}", logging.INFO):
                logger.info(f"Loading MRI: {mri_path}")
                mri_volume = self.loader.load_and_correct(mri_path)
                profiler.increment_counter("mri_volumes_loaded")
            
            with Timer(f"Loading PET {Path(pet_path).name}", logging.INFO):
                logger.info(f"Loading PET: {pet_path}")
                pet_volume = self.loader.load_and_correct(pet_path)
                profiler.increment_counter("pet_volumes_loaded")
            
            # Step 2: Register to MNI space and resample
            with Timer("MRI registration/resampling", logging.INFO):
                logger.info("Processing MRI (registration/resampling)")
                mri_registered = self.registrar.register_to_mni(mri_volume)
            
            with Timer("PET registration/resampling", logging.INFO):
                logger.info("Processing PET (registration/resampling)")
                pet_registered = self.registrar.register_to_mni(pet_volume)
            
            # Step 3: Validate dimensions
            with Timer("Dimension validation", logging.DEBUG):
                if not self.registrar.validate_output_dimensions(mri_registered):
                    raise ValueError("MRI registration failed dimension validation")
                
                if not self.registrar.validate_output_dimensions(pet_registered):
                    raise ValueError("PET registration failed dimension validation")
            
            # Step 4: Normalize and combine volumes
            with Timer("Volume normalization and combination", logging.INFO):
                logger.info("Normalizing and combining volumes")
                combined_volume = self.normalizer.combine_volumes(mri_registered, pet_registered)
            
            # Step 5: Final validation
            with Timer("Final validation", logging.DEBUG):
                if not self.normalizer.validate_combined_volume(combined_volume):
                    raise ValueError("Combined volume failed validation")
            
            # Record total pipeline time
            pipeline_end = time.perf_counter()
            pipeline_duration = pipeline_end - pipeline_start
            profiler.record_time("complete_pipeline", pipeline_duration)
            profiler.increment_counter("successful_pairs_processed")
            
            logger.info(f"Successfully processed volume pair with final shape {combined_volume.shape} in {pipeline_duration:.4f}s")
            return combined_volume
            
        except Exception as e:
            pipeline_end = time.perf_counter()
            pipeline_duration = pipeline_end - pipeline_start
            profiler.record_time("failed_pipeline", pipeline_duration)
            profiler.increment_counter("failed_pairs_processed")
            
            logger.error(f"Preprocessing failed for {mri_path}, {pet_path} after {pipeline_duration:.4f}s: {str(e)}")
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


def get_performance_report() -> str:
    """
    Get a comprehensive performance report from the global profiler.
    
    Returns:
        Formatted performance report string
    """
    return profiler.report()


def reset_performance_profiler():
    """
    Reset the global performance profiler to start fresh measurements.
    """
    global profiler
    profiler = PerformanceProfiler()
    logger.info("Performance profiler reset")


def log_performance_summary():
    """
    Log a summary of current performance metrics.
    """
    report = profiler.report()
    if report.strip():
        logger.info(report)
    else:
        logger.info("No performance data available yet")