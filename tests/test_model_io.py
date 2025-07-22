"""Unit tests for Pydantic input/output validation."""

import pytest
import torch
from pydantic import ValidationError

from gazimed.core.model_io import ModelInput, ModelOutput


class TestModelInput:
    """Test cases for ModelInput validation."""
    
    def test_valid_imaging_input(self):
        """Test creating valid imaging input."""
        image = torch.randn(2, 3, 96, 96, 96)
        
        input_obj = ModelInput(
            image=image,
            batch_size=2,
            input_shape=(96, 96, 96),
            in_channels=3
        )
        
        assert input_obj.image.shape == (2, 3, 96, 96, 96)
        assert input_obj.batch_size == 2
        assert input_obj.input_shape == (96, 96, 96)
        assert input_obj.in_channels == 3
        assert input_obj.clinical_features_size is None
    
    def test_valid_multimodal_input(self):
        """Test creating valid multimodal input."""
        image = torch.randn(2, 2, 64, 64, 64)
        features = torch.randn(2, 116)
        
        input_obj = ModelInput(
            image=image,
            features=features,
            batch_size=2,
            input_shape=(64, 64, 64),
            in_channels=2,
            clinical_features_size=116
        )
        
        assert input_obj.image.shape == (2, 2, 64, 64, 64)
        assert input_obj.features.shape == (2, 116)
        assert input_obj.clinical_features_size == 116
    
    def test_invalid_batch_size(self):
        """Test validation fails for invalid batch size."""
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            ModelInput(
                batch_size=0,
                input_shape=(96, 96, 96),
                in_channels=3
            )
        
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            ModelInput(
                batch_size=-1,
                input_shape=(96, 96, 96),
                in_channels=3
            )
    
    def test_invalid_input_shape(self):
        """Test validation fails for invalid input shape."""
        with pytest.raises(ValidationError, match="input_shape must have exactly 3 dimensions"):
            ModelInput(
                batch_size=2,
                input_shape=(96, 96),  # Only 2 dimensions
                in_channels=3
            )
        
        with pytest.raises(ValidationError, match="All dimensions in input_shape must be positive"):
            ModelInput(
                batch_size=2,
                input_shape=(96, 0, 96),  # Zero dimension
                in_channels=3
            )
    
    def test_invalid_channels(self):
        """Test validation fails for invalid channel count."""
        with pytest.raises(ValidationError, match="in_channels must be positive"):
            ModelInput(
                batch_size=2,
                input_shape=(96, 96, 96),
                in_channels=0
            )
    
    def test_invalid_clinical_features_size(self):
        """Test validation fails for invalid clinical features size."""
        with pytest.raises(ValidationError, match="clinical_features_size must be positive"):
            ModelInput(
                batch_size=2,
                input_shape=(96, 96, 96),
                in_channels=3,
                clinical_features_size=-1
            )
    
    def test_image_shape_mismatch(self):
        """Test validation fails when image shape doesn't match expected."""
        image = torch.randn(2, 3, 64, 64, 64)  # Wrong shape
        
        with pytest.raises(ValidationError, match="Image tensor shape .* doesn't match expected"):
            ModelInput(
                image=image,
                batch_size=2,
                input_shape=(96, 96, 96),  # Expected different shape
                in_channels=3
            )
    
    def test_features_shape_mismatch(self):
        """Test validation fails when features shape doesn't match expected."""
        features = torch.randn(2, 100)  # Wrong feature size
        
        with pytest.raises(ValidationError, match="Features tensor shape .* doesn't match expected"):
            ModelInput(
                features=features,
                batch_size=2,
                input_shape=(96, 96, 96),
                in_channels=3,
                clinical_features_size=116  # Expected different size
            )
    
    def test_batch_size_mismatch_image(self):
        """Test validation fails when image batch size doesn't match."""
        image = torch.randn(3, 3, 96, 96, 96)  # Batch size 3
        
        with pytest.raises(ValidationError, match="Image tensor shape .* doesn't match expected"):
            ModelInput(
                image=image,
                batch_size=2,  # Expected batch size 2
                input_shape=(96, 96, 96),
                in_channels=3
            )
    
    def test_batch_size_mismatch_features(self):
        """Test validation fails when features batch size doesn't match."""
        features = torch.randn(3, 116)  # Batch size 3
        
        with pytest.raises(ValidationError, match="Features tensor shape .* doesn't match expected"):
            ModelInput(
                features=features,
                batch_size=2,  # Expected batch size 2
                input_shape=(96, 96, 96),
                in_channels=3,
                clinical_features_size=116
            )
    
    def test_features_without_clinical_features_size(self):
        """Test features validation when clinical_features_size is not specified."""
        features = torch.randn(2, 50)
        
        input_obj = ModelInput(
            features=features,
            batch_size=2,
            input_shape=(96, 96, 96),
            in_channels=3
            # clinical_features_size not specified
        )
        
        assert input_obj.features.shape == (2, 50)
        assert input_obj.clinical_features_size is None


class TestModelOutput:
    """Test cases for ModelOutput validation."""
    
    def test_valid_output(self):
        """Test creating valid model output."""
        predictions = torch.randn(2, 1)
        probabilities = torch.sigmoid(predictions)
        logits = torch.randn(2, 1)
        
        output = ModelOutput(
            predictions=predictions,
            probabilities=probabilities,
            logits=logits,
            input_shape=(96, 96, 96),
            in_channels=3
        )
        
        assert output.predictions.shape == (2, 1)
        assert output.probabilities.shape == (2, 1)
        assert output.logits.shape == (2, 1)
        assert output.input_shape == (96, 96, 96)
        assert output.in_channels == 3
    
    def test_empty_predictions_validation(self):
        """Test validation fails for empty predictions tensor."""
        empty_tensor = torch.empty(0)
        
        with pytest.raises(ValidationError, match="predictions tensor cannot be empty"):
            ModelOutput(
                predictions=empty_tensor,
                input_shape=(96, 96, 96),
                in_channels=3
            )
    
    def test_invalid_probabilities_range(self):
        """Test validation fails for probabilities outside [0, 1] range."""
        predictions = torch.randn(2, 1)
        invalid_probs = torch.tensor([[1.5], [-0.1]])  # Outside [0, 1]
        
        with pytest.raises(ValidationError, match="probabilities must be in range"):
            ModelOutput(
                predictions=predictions,
                probabilities=invalid_probs,
                input_shape=(96, 96, 96),
                in_channels=3
            )
    
    def test_optional_fields(self):
        """Test all optional fields can be set."""
        predictions = torch.randn(2, 1)
        attention_maps = torch.randn(2, 8, 96, 96, 96)
        feature_importance = torch.randn(2, 116)
        
        output = ModelOutput(
            predictions=predictions,
            attention_maps=attention_maps,
            feature_importance=feature_importance,
            input_shape=(96, 96, 96),
            in_channels=3,
            clinical_features_size=116
        )
        
        assert output.attention_maps.shape == (2, 8, 96, 96, 96)
        assert output.feature_importance.shape == (2, 116)
        assert output.clinical_features_size == 116


class TestValidationAssignment:
    """Test validation on assignment (Config.validate_assignment = True)."""
    
    def test_assignment_validation(self):
        """Test that validation occurs on field assignment."""
        input_obj = ModelInput(
            batch_size=2,
            input_shape=(96, 96, 96),
            in_channels=3
        )
        
        # This should trigger validation and fail
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            input_obj.batch_size = -1
    
    def test_tensor_assignment_validation(self):
        """Test tensor validation on assignment."""
        input_obj = ModelInput(
            batch_size=2,
            input_shape=(96, 96, 96),
            in_channels=3
        )
        
        # Valid assignment
        valid_image = torch.randn(2, 3, 96, 96, 96)
        input_obj.image = valid_image
        assert input_obj.image.shape == (2, 3, 96, 96, 96)
        
        # Invalid assignment should fail
        invalid_image = torch.randn(2, 3, 64, 64, 64)  # Wrong shape
        with pytest.raises(ValidationError, match="Image tensor shape .* doesn't match expected"):
            input_obj.image = invalid_image


if __name__ == "__main__":
    pytest.main([__file__])