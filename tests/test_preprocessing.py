"""
Unit tests for NIfTI preprocessing pipeline.
"""

import numpy as np
import pytest
import tempfile
import nibabel as nib
from pathlib import Path
from unittest.mock import patch, MagicMock

from gazimed.data.preprocessing import (
    NIfTILoader,
    load_nifti_with_correction,
    validate_nifti_pair,
    MNIRegistrationResampler,
    register_and_resample_volume,
    VolumeNormalizer,
    PreprocessingPipeline,
    preprocess_volume_pair
)


class TestNIfTILoader:
    """Test cases for NIfTILoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = NIfTILoader(apply_n4_correction=True)
        self.loader_no_n4 = NIfTILoader(apply_n4_correction=False)
    
    def create_test_nifti(self, shape=(91, 120, 91), data=None):
        """Create a test NIfTI file."""
        if data is None:
            # Create synthetic brain-like data
            data = np.random.rand(*shape).astype(np.float32) * 1000
            # Add some structure to mimic brain tissue
            center = tuple(s // 2 for s in shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                        if dist < min(shape) // 4:
                            data[i, j, k] *= 2  # Enhance center region
        
        # Create NIfTI image
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        nib.save(nifti_img, temp_file.name)
        temp_file.close()
        
        return temp_file.name, data
    
    def test_init_with_n4_correction(self):
        """Test initialization with N4 correction enabled."""
        loader = NIfTILoader(apply_n4_correction=True)
        assert loader.apply_n4_correction is True
        assert hasattr(loader, 'n4_corrector')
        assert loader.n4_iterations == 50
        assert loader.n4_convergence_threshold == 0.001
    
    def test_init_without_n4_correction(self):
        """Test initialization with N4 correction disabled."""
        loader = NIfTILoader(apply_n4_correction=False)
        assert loader.apply_n4_correction is False
        assert not hasattr(loader, 'n4_corrector')
    
    def test_load_nifti_success(self):
        """Test successful NIfTI loading."""
        nifti_path, original_data = self.create_test_nifti()
        
        try:
            loaded_data = self.loader_no_n4.load_nifti(nifti_path)
            
            assert loaded_data.shape == original_data.shape
            assert loaded_data.dtype == np.float32
            assert np.allclose(loaded_data, original_data, rtol=1e-5)
            
        finally:
            Path(nifti_path).unlink()  # Clean up
    
    def test_load_nifti_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_nifti("nonexistent_file.nii.gz")
    
    def test_load_nifti_4d_volume(self):
        """Test loading 4D volume with singleton dimension."""
        # Create 4D data with singleton time dimension
        data_4d = np.random.rand(91, 120, 91, 1).astype(np.float32) * 1000
        nifti_path, _ = self.create_test_nifti(data=data_4d)
        
        try:
            loaded_data = self.loader_no_n4.load_nifti(nifti_path)
            
            assert loaded_data.shape == (91, 120, 91)
            assert loaded_data.dtype == np.float32
            
        finally:
            Path(nifti_path).unlink()
    
    def test_load_nifti_invalid_dimensions(self):
        """Test loading volume with invalid dimensions."""
        # Create 2D data (invalid)
        data_2d = np.random.rand(91, 120).astype(np.float32)
        nifti_path, _ = self.create_test_nifti(data=data_2d)
        
        try:
            with pytest.raises(ValueError, match="Expected 3D volume"):
                self.loader.load_nifti(nifti_path)
        finally:
            Path(nifti_path).unlink()
    
    def test_apply_n4_bias_correction_success(self):
        """Test successful N4 bias correction."""
        # Create test data
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        corrected_volume = test_volume * 1.1  # Simulate correction
        
        # Mock all SimpleITK operations to avoid actual execution
        with patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image, \
             patch('gazimed.data.preprocessing.sitk.GetArrayFromImage') as mock_get_array, \
             patch.object(self.loader, 'n4_corrector') as mock_corrector:
            
            mock_sitk_image = MagicMock()
            mock_corrected_image = MagicMock()
            
            mock_get_image.return_value = mock_sitk_image
            mock_corrector.Execute.return_value = mock_corrected_image
            mock_get_array.return_value = corrected_volume
            
            result = self.loader.apply_n4_bias_correction(test_volume)
            
            assert result.shape == test_volume.shape
            assert result.dtype == np.float32
            mock_get_image.assert_called_once()
            mock_corrector.Execute.assert_called_once_with(mock_sitk_image)
            mock_get_array.assert_called_once_with(mock_corrected_image)
    
    def test_apply_n4_bias_correction_disabled(self):
        """Test N4 correction when disabled."""
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        result = self.loader_no_n4.apply_n4_bias_correction(test_volume)
        
        assert np.array_equal(result, test_volume)
    
    def test_apply_n4_bias_correction_failure(self):
        """Test N4 correction failure handling."""
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        
        # Mock SimpleITK operations to simulate failure
        with patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image, \
             patch.object(self.loader, 'n4_corrector') as mock_corrector:
            
            mock_get_image.side_effect = Exception("N4 failed")
            
            result = self.loader.apply_n4_bias_correction(test_volume)
            
            # Should return original volume on failure
            assert np.array_equal(result, test_volume)
    
    def test_load_and_correct_with_n4(self):
        """Test complete loading and correction pipeline."""
        nifti_path, original_data = self.create_test_nifti()
        
        try:
            with patch.object(self.loader, 'apply_n4_bias_correction') as mock_n4:
                mock_n4.return_value = original_data * 1.1  # Simulate correction
                
                result = self.loader.load_and_correct(nifti_path)
                
                assert result.shape == original_data.shape
                mock_n4.assert_called_once()
                
        finally:
            Path(nifti_path).unlink()
    
    def test_load_and_correct_without_n4(self):
        """Test loading without N4 correction."""
        nifti_path, original_data = self.create_test_nifti()
        
        try:
            result = self.loader_no_n4.load_and_correct(nifti_path)
            
            assert result.shape == original_data.shape
            assert np.allclose(result, original_data, rtol=1e-5)
            
        finally:
            Path(nifti_path).unlink()
    
    def test_validate_volume_shape_valid(self):
        """Test volume validation with valid data."""
        volume = np.random.rand(91, 120, 91).astype(np.float32)
        
        assert self.loader.validate_volume_shape(volume) is True
        assert self.loader.validate_volume_shape(volume, (91, 120, 91)) is True
    
    def test_validate_volume_shape_invalid_dimensions(self):
        """Test volume validation with invalid dimensions."""
        volume_2d = np.random.rand(91, 120).astype(np.float32)
        
        assert self.loader.validate_volume_shape(volume_2d) is False
    
    def test_validate_volume_shape_wrong_shape(self):
        """Test volume validation with wrong expected shape."""
        volume = np.random.rand(91, 120, 91).astype(np.float32)
        
        assert self.loader.validate_volume_shape(volume, (100, 100, 100)) is False
    
    def test_validate_volume_shape_nan_values(self):
        """Test volume validation with NaN values."""
        volume = np.random.rand(50, 50, 50).astype(np.float32)
        volume[0, 0, 0] = np.nan
        
        assert self.loader.validate_volume_shape(volume) is False
    
    def test_validate_volume_shape_infinite_values(self):
        """Test volume validation with infinite values."""
        volume = np.random.rand(50, 50, 50).astype(np.float32)
        volume[0, 0, 0] = np.inf
        
        assert self.loader.validate_volume_shape(volume) is False
    
    def test_validate_volume_shape_empty(self):
        """Test volume validation with empty array."""
        volume = np.array([]).astype(np.float32)
        
        assert self.loader.validate_volume_shape(volume) is False


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def create_test_nifti(self, shape=(91, 120, 91)):
        """Create a test NIfTI file."""
        data = np.random.rand(*shape).astype(np.float32) * 1000
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        nib.save(nifti_img, temp_file.name)
        temp_file.close()
        
        return temp_file.name, data
    
    def test_load_nifti_with_correction_enabled(self):
        """Test convenience function with N4 correction enabled."""
        nifti_path, original_data = self.create_test_nifti()
        
        try:
            with patch('gazimed.data.preprocessing.NIfTILoader') as mock_loader_class:
                mock_loader = MagicMock()
                mock_loader_class.return_value = mock_loader
                mock_loader.load_and_correct.return_value = original_data
                
                result = load_nifti_with_correction(nifti_path, apply_n4=True)
                
                mock_loader_class.assert_called_once_with(apply_n4_correction=True)
                mock_loader.load_and_correct.assert_called_once_with(nifti_path)
                assert np.array_equal(result, original_data)
                
        finally:
            Path(nifti_path).unlink()
    
    def test_load_nifti_with_correction_disabled(self):
        """Test convenience function with N4 correction disabled."""
        nifti_path, original_data = self.create_test_nifti()
        
        try:
            with patch('gazimed.data.preprocessing.NIfTILoader') as mock_loader_class:
                mock_loader = MagicMock()
                mock_loader_class.return_value = mock_loader
                mock_loader.load_and_correct.return_value = original_data
                
                result = load_nifti_with_correction(nifti_path, apply_n4=False)
                
                mock_loader_class.assert_called_once_with(apply_n4_correction=False)
                mock_loader.load_and_correct.assert_called_once_with(nifti_path)
                assert np.array_equal(result, original_data)
                
        finally:
            Path(nifti_path).unlink()
    
    def test_validate_nifti_pair_success(self):
        """Test successful validation of NIfTI pair."""
        mri_path, _ = self.create_test_nifti()
        pet_path, _ = self.create_test_nifti()
        
        try:
            result = validate_nifti_pair(mri_path, pet_path)
            assert result is True
            
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()
    
    def test_validate_nifti_pair_missing_mri(self):
        """Test validation with missing MRI file."""
        pet_path, _ = self.create_test_nifti()
        
        try:
            result = validate_nifti_pair("nonexistent_mri.nii.gz", pet_path)
            assert result is False
            
        finally:
            Path(pet_path).unlink()
    
    def test_validate_nifti_pair_missing_pet(self):
        """Test validation with missing PET file."""
        mri_path, _ = self.create_test_nifti()
        
        try:
            result = validate_nifti_pair(mri_path, "nonexistent_pet.nii.gz")
            assert result is False
            
        finally:
            Path(mri_path).unlink()
    
    def test_validate_nifti_pair_different_shapes(self):
        """Test validation with different shaped volumes."""
        mri_path, _ = self.create_test_nifti(shape=(91, 120, 91))
        pet_path, _ = self.create_test_nifti(shape=(100, 100, 100))
        
        try:
            # Should still return True as shapes will be standardized later
            result = validate_nifti_pair(mri_path, pet_path)
            assert result is True
            
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()
    
    def test_validate_nifti_pair_loading_error(self):
        """Test validation with file loading error."""
        # Create invalid NIfTI files
        temp_mri = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        temp_pet = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        
        # Write invalid data
        temp_mri.write(b"invalid nifti data")
        temp_pet.write(b"invalid nifti data")
        temp_mri.close()
        temp_pet.close()
        
        try:
            result = validate_nifti_pair(temp_mri.name, temp_pet.name)
            assert result is False
            
        finally:
            Path(temp_mri.name).unlink()
            Path(temp_pet.name).unlink()


class TestMNIRegistrationResampler:
    """Test cases for MNI registration and resampling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resampler = MNIRegistrationResampler()
        self.custom_resampler = MNIRegistrationResampler(
            target_spacing=(2.0, 2.0, 2.0),
            target_size=(45, 60, 45),
            interpolation_method='nearest'
        )
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        resampler = MNIRegistrationResampler()
        
        assert resampler.target_spacing == (1.0, 1.0, 1.0)
        assert resampler.target_size == (91, 120, 91)
        assert resampler.interpolation_method == 'linear'
        assert hasattr(resampler, 'registration_method')
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        resampler = MNIRegistrationResampler(
            target_spacing=(2.0, 2.0, 2.0),
            target_size=(50, 60, 70),
            interpolation_method='cubic'
        )
        
        assert resampler.target_spacing == (2.0, 2.0, 2.0)
        assert resampler.target_size == (50, 60, 70)
        assert resampler.interpolation_method == 'cubic'
    
    def test_init_invalid_interpolation(self):
        """Test initialization with invalid interpolation method."""
        with pytest.raises(ValueError, match="Unsupported interpolation method"):
            MNIRegistrationResampler(interpolation_method='invalid')
    
    def test_create_mni_template(self):
        """Test MNI template creation."""
        template = self.resampler.create_mni_template()
        
        # Check that template is a SimpleITK image
        assert hasattr(template, 'GetSize')
        assert hasattr(template, 'GetSpacing')
        assert hasattr(template, 'GetOrigin')
        
        # Check dimensions
        size = template.GetSize()
        assert size == self.resampler.target_size
        
        # Check spacing
        spacing = template.GetSpacing()
        assert spacing == self.resampler.target_spacing
    
    def test_resample_to_target_success(self):
        """Test successful resampling to target dimensions."""
        # Create test volume with different dimensions
        test_volume = np.random.rand(100, 100, 100).astype(np.float32) * 1000
        
        with patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image, \
             patch('gazimed.data.preprocessing.sitk.GetArrayFromImage') as mock_get_array, \
             patch('gazimed.data.preprocessing.sitk.Resample') as mock_resample:
            
            # Mock the resampled result
            resampled_data = np.random.rand(91, 120, 91).astype(np.float32) * 1000
            
            mock_sitk_image = MagicMock()
            mock_resampled_image = MagicMock()
            
            mock_get_image.return_value = mock_sitk_image
            mock_resample.return_value = mock_resampled_image
            mock_get_array.return_value = resampled_data
            
            result = self.resampler.resample_to_target(test_volume)
            
            assert result.shape == (91, 120, 91)
            assert result.dtype == np.float32
            mock_get_image.assert_called_once()
            mock_resample.assert_called_once()
            mock_get_array.assert_called_once()
    
    def test_resample_to_target_with_spacing(self):
        """Test resampling with custom original spacing."""
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        original_spacing = (2.0, 2.0, 2.0)
        
        with patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image, \
             patch('gazimed.data.preprocessing.sitk.GetArrayFromImage') as mock_get_array, \
             patch('gazimed.data.preprocessing.sitk.Resample') as mock_resample:
            
            resampled_data = np.random.rand(91, 120, 91).astype(np.float32)
            
            mock_sitk_image = MagicMock()
            mock_resampled_image = MagicMock()
            
            mock_get_image.return_value = mock_sitk_image
            mock_resample.return_value = mock_resampled_image
            mock_get_array.return_value = resampled_data
            
            result = self.resampler.resample_to_target(test_volume, original_spacing)
            
            # Verify that SetSpacing was called with the original spacing
            mock_sitk_image.SetSpacing.assert_called_with(original_spacing)
            assert result.shape == (91, 120, 91)
    
    def test_resample_to_target_failure(self):
        """Test resampling failure handling."""
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        
        with patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image:
            mock_get_image.side_effect = Exception("Resampling failed")
            
            with pytest.raises(ValueError, match="Failed to resample volume"):
                self.resampler.resample_to_target(test_volume)
    
    def test_register_to_mni_success(self):
        """Test successful MNI registration."""
        test_volume = np.random.rand(100, 100, 100).astype(np.float32) * 1000
        
        with patch.object(self.resampler, 'create_mni_template') as mock_template, \
             patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image, \
             patch('gazimed.data.preprocessing.sitk.GetArrayFromImage') as mock_get_array, \
             patch('gazimed.data.preprocessing.sitk.Resample') as mock_resample, \
             patch.object(self.resampler.registration_method, 'Execute') as mock_execute:
            
            # Mock template and images
            mock_fixed_image = MagicMock()
            mock_moving_image = MagicMock()
            mock_registered_image = MagicMock()
            mock_transform = MagicMock()
            
            mock_template.return_value = mock_fixed_image
            mock_get_image.return_value = mock_moving_image
            mock_execute.return_value = mock_transform
            mock_resample.return_value = mock_registered_image
            
            # Mock the registered result
            registered_data = np.random.rand(91, 120, 91).astype(np.float32) * 1000
            mock_get_array.return_value = registered_data
            
            result = self.resampler.register_to_mni(test_volume)
            
            assert result.shape == (91, 120, 91)
            assert result.dtype == np.float32
            mock_execute.assert_called_once()
            mock_resample.assert_called_once()
    
    def test_register_to_mni_registration_failure_fallback(self):
        """Test MNI registration with fallback to resampling only."""
        test_volume = np.random.rand(100, 100, 100).astype(np.float32) * 1000
        
        with patch.object(self.resampler, 'create_mni_template') as mock_template, \
             patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image, \
             patch.object(self.resampler.registration_method, 'Execute') as mock_execute, \
             patch.object(self.resampler, 'resample_to_target') as mock_resample_fallback:
            
            # Mock registration failure
            mock_template.return_value = MagicMock()
            mock_get_image.return_value = MagicMock()
            mock_execute.side_effect = Exception("Registration failed")
            
            # Mock fallback result
            fallback_data = np.random.rand(91, 120, 91).astype(np.float32) * 1000
            mock_resample_fallback.return_value = fallback_data
            
            result = self.resampler.register_to_mni(test_volume)
            
            # Should fall back to resampling only
            mock_resample_fallback.assert_called_once_with(test_volume, None)
            assert result.shape == (91, 120, 91)
    
    def test_register_to_mni_complete_failure(self):
        """Test complete failure in MNI registration."""
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        
        with patch('gazimed.data.preprocessing.sitk.GetImageFromArray') as mock_get_image:
            mock_get_image.side_effect = Exception("Complete failure")
            
            with pytest.raises(ValueError, match="Failed to register volume to MNI space"):
                self.resampler.register_to_mni(test_volume)
    
    def test_validate_output_dimensions_success(self):
        """Test successful output dimension validation."""
        volume = np.random.rand(91, 120, 91).astype(np.float32)
        
        assert self.resampler.validate_output_dimensions(volume) is True
    
    def test_validate_output_dimensions_failure(self):
        """Test failed output dimension validation."""
        volume = np.random.rand(100, 100, 100).astype(np.float32)
        
        assert self.resampler.validate_output_dimensions(volume) is False
    
    def test_custom_target_dimensions(self):
        """Test resampler with custom target dimensions."""
        volume = np.random.rand(45, 60, 45).astype(np.float32)
        
        assert self.custom_resampler.validate_output_dimensions(volume) is True
        assert self.custom_resampler.target_size == (45, 60, 45)
        assert self.custom_resampler.target_spacing == (2.0, 2.0, 2.0)


class TestRegistrationConvenienceFunctions:
    """Test convenience functions for registration and resampling."""
    
    def test_register_and_resample_volume_success(self):
        """Test successful volume registration and resampling."""
        test_volume = np.random.rand(100, 100, 100).astype(np.float32) * 1000
        
        with patch('gazimed.data.preprocessing.MNIRegistrationResampler') as mock_resampler_class:
            mock_resampler = MagicMock()
            mock_resampler_class.return_value = mock_resampler
            
            # Mock successful registration and validation
            registered_data = np.random.rand(91, 120, 91).astype(np.float32) * 1000
            mock_resampler.register_to_mni.return_value = registered_data
            mock_resampler.validate_output_dimensions.return_value = True
            
            result = register_and_resample_volume(test_volume)
            
            # Verify resampler was created with default parameters
            mock_resampler_class.assert_called_once_with(
                target_spacing=(1.0, 1.0, 1.0),
                target_size=(91, 120, 91),
                interpolation_method='linear'
            )
            
            # Verify registration was called
            mock_resampler.register_to_mni.assert_called_once_with(test_volume, None)
            mock_resampler.validate_output_dimensions.assert_called_once_with(registered_data)
            
            assert np.array_equal(result, registered_data)
    
    def test_register_and_resample_volume_custom_params(self):
        """Test volume registration with custom parameters."""
        test_volume = np.random.rand(50, 50, 50).astype(np.float32)
        original_spacing = (2.0, 2.0, 2.0)
        target_spacing = (1.5, 1.5, 1.5)
        target_size = (60, 80, 60)
        
        with patch('gazimed.data.preprocessing.MNIRegistrationResampler') as mock_resampler_class:
            mock_resampler = MagicMock()
            mock_resampler_class.return_value = mock_resampler
            
            registered_data = np.random.rand(60, 80, 60).astype(np.float32)
            mock_resampler.register_to_mni.return_value = registered_data
            mock_resampler.validate_output_dimensions.return_value = True
            
            result = register_and_resample_volume(
                test_volume,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                target_size=target_size,
                interpolation='cubic'
            )
            
            # Verify resampler was created with custom parameters
            mock_resampler_class.assert_called_once_with(
                target_spacing=target_spacing,
                target_size=target_size,
                interpolation_method='cubic'
            )
            
            mock_resampler.register_to_mni.assert_called_once_with(test_volume, original_spacing)
            assert np.array_equal(result, registered_data)
    
    def test_register_and_resample_volume_validation_failure(self):
        """Test volume registration with validation failure."""
        test_volume = np.random.rand(100, 100, 100).astype(np.float32)
        
        with patch('gazimed.data.preprocessing.MNIRegistrationResampler') as mock_resampler_class:
            mock_resampler = MagicMock()
            mock_resampler_class.return_value = mock_resampler
            
            # Mock registration success but validation failure
            registered_data = np.random.rand(100, 100, 100).astype(np.float32)  # Wrong dimensions
            mock_resampler.register_to_mni.return_value = registered_data
            mock_resampler.validate_output_dimensions.return_value = False
            
            with pytest.raises(ValueError, match="Output dimensions validation failed"):
                register_and_resample_volume(test_volume)


class TestVolumeNormalizer:
    """Test cases for VolumeNormalizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = VolumeNormalizer()
        self.normalizer_with_diff = VolumeNormalizer(include_difference_channel=True)
        self.normalizer_minmax = VolumeNormalizer(normalization_method='minmax')
        self.normalizer_none = VolumeNormalizer(normalization_method='none')
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        normalizer = VolumeNormalizer()
        
        assert normalizer.normalization_method == 'zscore'
        assert normalizer.include_difference_channel is False
        assert normalizer.mask_background is True
        assert normalizer.background_threshold == 0.01
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        normalizer = VolumeNormalizer(
            normalization_method='minmax',
            include_difference_channel=True,
            mask_background=False,
            background_threshold=0.05
        )
        
        assert normalizer.normalization_method == 'minmax'
        assert normalizer.include_difference_channel is True
        assert normalizer.mask_background is False
        assert normalizer.background_threshold == 0.05
    
    def test_init_invalid_normalization_method(self):
        """Test initialization with invalid normalization method."""
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            VolumeNormalizer(normalization_method='invalid')
    
    def test_create_brain_mask(self):
        """Test brain mask creation."""
        # Create test volume with clear background/foreground
        volume = np.zeros((50, 50, 50), dtype=np.float32)
        # Add brain region in center
        volume[15:35, 15:35, 15:35] = 1000.0
        
        with patch('scipy.ndimage.binary_fill_holes') as mock_fill, \
             patch('scipy.ndimage.binary_erosion') as mock_erosion, \
             patch('scipy.ndimage.binary_dilation') as mock_dilation:
            
            # Mock morphological operations
            mock_fill.return_value = volume > 10.0  # Simple threshold
            mock_erosion.return_value = volume > 10.0
            mock_dilation.return_value = volume > 10.0
            
            mask = self.normalizer.create_brain_mask(volume)
            
            assert mask.shape == volume.shape
            assert mask.dtype == bool
            mock_fill.assert_called_once()
            mock_erosion.assert_called_once()
            mock_dilation.assert_called_once()
    
    def test_zscore_normalize_without_mask(self):
        """Test Z-score normalization without mask."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000 + 500
        
        result = self.normalizer.zscore_normalize(volume)
        
        # Check that result is approximately zero-mean and unit variance
        assert result.dtype == np.float32
        assert abs(np.mean(result)) < 0.1  # Should be close to zero
        assert abs(np.std(result) - 1.0) < 0.1  # Should be close to 1
    
    def test_zscore_normalize_with_mask(self):
        """Test Z-score normalization with mask."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000 + 500
        mask = np.ones((50, 50, 50), dtype=bool)
        mask[:10, :10, :10] = False  # Exclude corner region
        
        result = self.normalizer.zscore_normalize(volume, mask)
        
        assert result.dtype == np.float32
        assert result.shape == volume.shape
        # Mean should be close to zero for masked region
        masked_mean = np.mean(result[mask])
        assert abs(masked_mean) < 0.1
    
    def test_zscore_normalize_zero_std(self):
        """Test Z-score normalization with zero standard deviation."""
        volume = np.ones((50, 50, 50), dtype=np.float32) * 500  # Constant volume
        
        result = self.normalizer.zscore_normalize(volume)
        
        # Should return zero-centered volume
        assert np.allclose(result, np.zeros_like(volume))
    
    def test_zscore_normalize_empty_mask(self):
        """Test Z-score normalization with empty mask."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000
        mask = np.zeros((50, 50, 50), dtype=bool)  # Empty mask
        
        result = self.normalizer.zscore_normalize(volume, mask)
        
        # Should fall back to whole volume normalization
        assert result.dtype == np.float32
        assert abs(np.mean(result)) < 0.1
    
    def test_minmax_normalize_without_mask(self):
        """Test min-max normalization without mask."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000 + 500
        
        result = self.normalizer_minmax.minmax_normalize(volume)
        
        assert result.dtype == np.float32
        assert np.min(result) >= 0.0
        assert np.max(result) <= 1.0
        assert abs(np.min(result) - 0.0) < 0.01
        assert abs(np.max(result) - 1.0) < 0.01
    
    def test_minmax_normalize_with_mask(self):
        """Test min-max normalization with mask."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000 + 500
        mask = np.ones((50, 50, 50), dtype=bool)
        mask[:10, :10, :10] = False
        
        result = self.normalizer_minmax.minmax_normalize(volume, mask)
        
        assert result.dtype == np.float32
        assert result.shape == volume.shape
    
    def test_minmax_normalize_constant_volume(self):
        """Test min-max normalization with constant volume."""
        volume = np.ones((50, 50, 50), dtype=np.float32) * 500
        
        result = self.normalizer_minmax.minmax_normalize(volume)
        
        # Should return zero array
        assert np.allclose(result, np.zeros_like(volume))
    
    def test_normalize_volume_zscore(self):
        """Test volume normalization with Z-score method."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000
        
        with patch.object(self.normalizer, 'create_brain_mask') as mock_mask, \
             patch.object(self.normalizer, 'zscore_normalize') as mock_zscore:
            
            mock_mask.return_value = np.ones((50, 50, 50), dtype=bool)
            mock_zscore.return_value = volume * 0.5  # Mock normalized result
            
            result = self.normalizer.normalize_volume(volume)
            
            mock_mask.assert_called_once_with(volume)
            mock_zscore.assert_called_once()
            assert np.array_equal(result, volume * 0.5)
    
    def test_normalize_volume_minmax(self):
        """Test volume normalization with min-max method."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000
        
        with patch.object(self.normalizer_minmax, 'create_brain_mask') as mock_mask, \
             patch.object(self.normalizer_minmax, 'minmax_normalize') as mock_minmax:
            
            mock_mask.return_value = np.ones((50, 50, 50), dtype=bool)
            mock_minmax.return_value = volume * 0.001  # Mock normalized result
            
            result = self.normalizer_minmax.normalize_volume(volume)
            
            mock_mask.assert_called_once_with(volume)
            mock_minmax.assert_called_once()
            assert np.array_equal(result, volume * 0.001)
    
    def test_normalize_volume_none(self):
        """Test volume normalization with no normalization."""
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000
        
        result = self.normalizer_none.normalize_volume(volume)
        
        assert np.array_equal(result, volume.astype(np.float32))
    
    def test_combine_volumes_without_difference(self):
        """Test volume combination without difference channel."""
        mri_volume = np.random.rand(91, 120, 91).astype(np.float32) * 1000
        pet_volume = np.random.rand(91, 120, 91).astype(np.float32) * 800
        
        with patch.object(self.normalizer, 'normalize_volume') as mock_normalize:
            # Mock normalization to return scaled versions
            mock_normalize.side_effect = lambda x: x * 0.001
            
            result = self.normalizer.combine_volumes(mri_volume, pet_volume)
            
            assert result.shape == (2, 91, 120, 91)
            assert result.dtype == np.float32
            assert mock_normalize.call_count == 2
    
    def test_combine_volumes_with_difference(self):
        """Test volume combination with difference channel."""
        mri_volume = np.random.rand(91, 120, 91).astype(np.float32) * 1000
        pet_volume = np.random.rand(91, 120, 91).astype(np.float32) * 800
        
        with patch.object(self.normalizer_with_diff, 'normalize_volume') as mock_normalize:
            # Mock normalization to return the input (for simplicity)
            mock_normalize.side_effect = lambda x: x
            
            result = self.normalizer_with_diff.combine_volumes(mri_volume, pet_volume)
            
            assert result.shape == (3, 91, 120, 91)
            assert result.dtype == np.float32
            # Check that difference channel is PET - MRI
            np.testing.assert_array_equal(result[2], pet_volume - mri_volume)
    
    def test_combine_volumes_shape_mismatch(self):
        """Test volume combination with mismatched shapes."""
        mri_volume = np.random.rand(91, 120, 91).astype(np.float32)
        pet_volume = np.random.rand(100, 100, 100).astype(np.float32)
        
        with pytest.raises(ValueError, match="MRI shape .* != PET shape"):
            self.normalizer.combine_volumes(mri_volume, pet_volume)
    
    def test_validate_combined_volume_success(self):
        """Test successful combined volume validation."""
        combined_volume = np.random.rand(2, 91, 120, 91).astype(np.float32)
        
        assert self.normalizer.validate_combined_volume(combined_volume) is True
    
    def test_validate_combined_volume_with_difference_success(self):
        """Test successful validation with difference channel."""
        combined_volume = np.random.rand(3, 91, 120, 91).astype(np.float32)
        
        assert self.normalizer_with_diff.validate_combined_volume(combined_volume) is True
    
    def test_validate_combined_volume_wrong_dimensions(self):
        """Test validation with wrong number of dimensions."""
        combined_volume = np.random.rand(91, 120, 91).astype(np.float32)  # 3D instead of 4D
        
        assert self.normalizer.validate_combined_volume(combined_volume) is False
    
    def test_validate_combined_volume_wrong_channels(self):
        """Test validation with wrong number of channels."""
        combined_volume = np.random.rand(4, 91, 120, 91).astype(np.float32)  # 4 channels instead of 2
        
        assert self.normalizer.validate_combined_volume(combined_volume) is False
    
    def test_validate_combined_volume_nan_values(self):
        """Test validation with NaN values."""
        combined_volume = np.random.rand(2, 91, 120, 91).astype(np.float32)
        combined_volume[0, 0, 0, 0] = np.nan
        
        assert self.normalizer.validate_combined_volume(combined_volume) is False
    
    def test_validate_combined_volume_infinite_values(self):
        """Test validation with infinite values."""
        combined_volume = np.random.rand(2, 91, 120, 91).astype(np.float32)
        combined_volume[0, 0, 0, 0] = np.inf
        
        assert self.normalizer.validate_combined_volume(combined_volume) is False


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = PreprocessingPipeline()
        self.pipeline_with_diff = PreprocessingPipeline(include_difference_channel=True)
    
    def create_test_nifti(self, shape=(91, 120, 91)):
        """Create a test NIfTI file."""
        data = np.random.rand(*shape).astype(np.float32) * 1000
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        nib.save(nifti_img, temp_file.name)
        temp_file.close()
        
        return temp_file.name, data
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        pipeline = PreprocessingPipeline()
        
        assert hasattr(pipeline, 'loader')
        assert hasattr(pipeline, 'registrar')
        assert hasattr(pipeline, 'normalizer')
        assert pipeline.target_size == (91, 120, 91)
        assert pipeline.include_difference_channel is False
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        pipeline = PreprocessingPipeline(
            apply_n4_correction=False,
            target_size=(100, 100, 100),
            normalization_method='minmax',
            include_difference_channel=True
        )
        
        assert pipeline.target_size == (100, 100, 100)
        assert pipeline.include_difference_channel is True
    
    def test_process_volume_pair_success(self):
        """Test successful volume pair processing."""
        mri_path, mri_data = self.create_test_nifti()
        pet_path, pet_data = self.create_test_nifti()
        
        try:
            with patch.object(self.pipeline.loader, 'load_and_correct') as mock_load, \
                 patch.object(self.pipeline.registrar, 'register_to_mni') as mock_register, \
                 patch.object(self.pipeline.registrar, 'validate_output_dimensions') as mock_validate_dims, \
                 patch.object(self.pipeline.normalizer, 'combine_volumes') as mock_combine, \
                 patch.object(self.pipeline.normalizer, 'validate_combined_volume') as mock_validate_combined:
                
                # Mock successful processing
                registered_data = np.random.rand(91, 120, 91).astype(np.float32) * 1000
                combined_data = np.random.rand(2, 91, 120, 91).astype(np.float32)
                
                mock_load.side_effect = [mri_data, pet_data]
                mock_register.return_value = registered_data
                mock_validate_dims.return_value = True
                mock_combine.return_value = combined_data
                mock_validate_combined.return_value = True
                
                result = self.pipeline.process_volume_pair(mri_path, pet_path)
                
                # Verify all steps were called
                assert mock_load.call_count == 2
                assert mock_register.call_count == 2
                assert mock_validate_dims.call_count == 2
                mock_combine.assert_called_once()
                mock_validate_combined.assert_called_once()
                
                assert np.array_equal(result, combined_data)
                
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()
    
    def test_process_volume_pair_mri_validation_failure(self):
        """Test processing with MRI validation failure."""
        mri_path, mri_data = self.create_test_nifti()
        pet_path, pet_data = self.create_test_nifti()
        
        try:
            with patch.object(self.pipeline.loader, 'load_and_correct') as mock_load, \
                 patch.object(self.pipeline.registrar, 'register_to_mni') as mock_register, \
                 patch.object(self.pipeline.registrar, 'validate_output_dimensions') as mock_validate_dims:
                
                mock_load.side_effect = [mri_data, pet_data]
                mock_register.return_value = np.random.rand(100, 100, 100).astype(np.float32)  # Wrong size
                mock_validate_dims.side_effect = [False, True]  # MRI fails, PET passes
                
                with pytest.raises(ValueError, match="MRI registration failed dimension validation"):
                    self.pipeline.process_volume_pair(mri_path, pet_path)
                    
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()
    
    def test_process_volume_pair_combined_validation_failure(self):
        """Test processing with combined volume validation failure."""
        mri_path, mri_data = self.create_test_nifti()
        pet_path, pet_data = self.create_test_nifti()
        
        try:
            with patch.object(self.pipeline.loader, 'load_and_correct') as mock_load, \
                 patch.object(self.pipeline.registrar, 'register_to_mni') as mock_register, \
                 patch.object(self.pipeline.registrar, 'validate_output_dimensions') as mock_validate_dims, \
                 patch.object(self.pipeline.normalizer, 'combine_volumes') as mock_combine, \
                 patch.object(self.pipeline.normalizer, 'validate_combined_volume') as mock_validate_combined:
                
                mock_load.side_effect = [mri_data, pet_data]
                mock_register.return_value = np.random.rand(91, 120, 91).astype(np.float32)
                mock_validate_dims.return_value = True
                mock_combine.return_value = np.random.rand(2, 91, 120, 91).astype(np.float32)
                mock_validate_combined.return_value = False  # Combined validation fails
                
                with pytest.raises(ValueError, match="Combined volume failed validation"):
                    self.pipeline.process_volume_pair(mri_path, pet_path)
                    
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()
    
    def test_get_output_shape_without_difference(self):
        """Test output shape without difference channel."""
        shape = self.pipeline.get_output_shape()
        
        assert shape == (2, 91, 120, 91)
    
    def test_get_output_shape_with_difference(self):
        """Test output shape with difference channel."""
        shape = self.pipeline_with_diff.get_output_shape()
        
        assert shape == (3, 91, 120, 91)


class TestPreprocessingConvenienceFunctions:
    """Test convenience functions for preprocessing."""
    
    def create_test_nifti(self, shape=(91, 120, 91)):
        """Create a test NIfTI file."""
        data = np.random.rand(*shape).astype(np.float32) * 1000
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        nib.save(nifti_img, temp_file.name)
        temp_file.close()
        
        return temp_file.name, data
    
    def test_preprocess_volume_pair_default_params(self):
        """Test convenience function with default parameters."""
        mri_path, _ = self.create_test_nifti()
        pet_path, _ = self.create_test_nifti()
        
        try:
            with patch('gazimed.data.preprocessing.PreprocessingPipeline') as mock_pipeline_class:
                mock_pipeline = MagicMock()
                mock_pipeline_class.return_value = mock_pipeline
                
                combined_data = np.random.rand(2, 91, 120, 91).astype(np.float32)
                mock_pipeline.process_volume_pair.return_value = combined_data
                
                result = preprocess_volume_pair(mri_path, pet_path)
                
                # Verify pipeline was created with default parameters
                mock_pipeline_class.assert_called_once_with(
                    apply_n4_correction=True,
                    normalization_method='zscore',
                    include_difference_channel=False
                )
                
                mock_pipeline.process_volume_pair.assert_called_once_with(mri_path, pet_path)
                assert np.array_equal(result, combined_data)
                
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()
    
    def test_preprocess_volume_pair_custom_params(self):
        """Test convenience function with custom parameters."""
        mri_path, _ = self.create_test_nifti()
        pet_path, _ = self.create_test_nifti()
        
        try:
            with patch('gazimed.data.preprocessing.PreprocessingPipeline') as mock_pipeline_class:
                mock_pipeline = MagicMock()
                mock_pipeline_class.return_value = mock_pipeline
                
                combined_data = np.random.rand(3, 91, 120, 91).astype(np.float32)
                mock_pipeline.process_volume_pair.return_value = combined_data
                
                result = preprocess_volume_pair(
                    mri_path, pet_path,
                    apply_n4=False,
                    include_difference=True,
                    normalization='minmax'
                )
                
                # Verify pipeline was created with custom parameters
                mock_pipeline_class.assert_called_once_with(
                    apply_n4_correction=False,
                    normalization_method='minmax',
                    include_difference_channel=True
                )
                
                assert np.array_equal(result, combined_data)
                
        finally:
            Path(mri_path).unlink()
            Path(pet_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])