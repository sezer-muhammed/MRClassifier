"""
Tests for Clinical Features Encoder Module
"""

import pytest
import torch
import torch.nn as nn
from gazimed.models.clinical_encoder import (
    ClinicalFeaturesEncoder,
    CategoricalEncoder,
    MultiModalClinicalEncoder,
    ClinicalAttentionEncoder,
    ClinicalFeatureNormalizer,
    create_clinical_feature_mask,
    impute_missing_features,
    standardize_clinical_features
)


class TestClinicalFeaturesEncoder:
    """Test cases for ClinicalFeaturesEncoder."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        encoder = ClinicalFeaturesEncoder()
        
        assert encoder.input_dim == 118
        assert encoder.output_dim == 128
        assert encoder.use_residual == True
        
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        encoder = ClinicalFeaturesEncoder(
            input_dim=100,
            hidden_dims=[128, 256, 128],
            output_dim=64,
            dropout=0.2,
            activation='gelu',
            use_residual=False
        )
        
        assert encoder.input_dim == 100
        assert encoder.output_dim == 64
        assert encoder.use_residual == False
        
    def test_forward_pass(self):
        """Test forward pass."""
        encoder = ClinicalFeaturesEncoder(
            input_dim=118,
            hidden_dims=[256, 128],
            output_dim=64
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 118)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, 64)
        
    def test_forward_pass_with_residual(self):
        """Test forward pass with residual connection."""
        encoder = ClinicalFeaturesEncoder(
            input_dim=64,
            hidden_dims=[128, 64],
            output_dim=64,
            use_residual=True
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 64)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, 64)
        
    def test_forward_pass_invalid_input(self):
        """Test forward pass with invalid input dimension."""
        encoder = ClinicalFeaturesEncoder(input_dim=118)
        
        # Wrong input dimension
        x = torch.randn(2, 100)  # Should be 118
        
        with pytest.raises(AssertionError):
            encoder(x)
            
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'gelu', 'swish']
        
        for activation in activations:
            encoder = ClinicalFeaturesEncoder(
                input_dim=50,
                hidden_dims=[64],
                output_dim=32,
                activation=activation
            )
            
            x = torch.randn(2, 50)
            output = encoder(x)
            
            assert output.shape == (2, 32)
            
    def test_invalid_activation(self):
        """Test initialization with invalid activation."""
        with pytest.raises(ValueError):
            ClinicalFeaturesEncoder(activation='invalid')


class TestCategoricalEncoder:
    """Test cases for CategoricalEncoder."""
    
    def test_init(self):
        """Test initialization."""
        categorical_dims = {
            'gender': 3,  # Male, Female, Other
            'education': 5,  # 5 education levels
            'apoe': 4  # APOE genotype variants
        }
        
        encoder = CategoricalEncoder(categorical_dims)
        
        assert encoder.categorical_dims == categorical_dims
        assert len(encoder.embeddings) == 3
        assert encoder.output_dim == sum(encoder.embedding_dims.values())
        
    def test_init_custom_embeddings(self):
        """Test initialization with custom embedding dimensions."""
        categorical_dims = {'gender': 3, 'education': 5}
        embedding_dims = {'gender': 8, 'education': 16}
        
        encoder = CategoricalEncoder(categorical_dims, embedding_dims)
        
        assert encoder.embedding_dims == embedding_dims
        assert encoder.output_dim == 24  # 8 + 16
        
    def test_forward_pass(self):
        """Test forward pass."""
        categorical_dims = {'gender': 3, 'education': 5}
        encoder = CategoricalEncoder(categorical_dims)
        
        batch_size = 4
        categorical_features = {
            'gender': torch.randint(0, 3, (batch_size,)),
            'education': torch.randint(0, 5, (batch_size,))
        }
        
        output = encoder(categorical_features)
        
        assert output.shape == (batch_size, encoder.output_dim)
        
    def test_forward_pass_missing_features(self):
        """Test forward pass with missing categorical features."""
        categorical_dims = {'gender': 3, 'education': 5, 'apoe': 4}
        encoder = CategoricalEncoder(categorical_dims)
        
        batch_size = 2
        # Only provide some features
        categorical_features = {
            'gender': torch.randint(0, 3, (batch_size,))
            # Missing 'education' and 'apoe'
        }
        
        output = encoder(categorical_features)
        
        assert output.shape == (batch_size, encoder.output_dim)


class TestMultiModalClinicalEncoder:
    """Test cases for MultiModalClinicalEncoder."""
    
    def test_init_numerical_only(self):
        """Test initialization with numerical features only."""
        encoder = MultiModalClinicalEncoder(
            numerical_dim=118,
            categorical_dims=None
        )
        
        assert encoder.numerical_dim == 118
        assert encoder.categorical_encoder is None
        
    def test_init_with_categorical(self):
        """Test initialization with both numerical and categorical features."""
        categorical_dims = {'gender': 3, 'education': 5}
        
        encoder = MultiModalClinicalEncoder(
            numerical_dim=118,
            categorical_dims=categorical_dims
        )
        
        assert encoder.numerical_dim == 118
        assert encoder.categorical_dims == categorical_dims
        assert encoder.categorical_encoder is not None
        
    def test_forward_numerical_only(self):
        """Test forward pass with numerical features only."""
        encoder = MultiModalClinicalEncoder(
            numerical_dim=118,
            fusion_dim=256
        )
        
        batch_size = 3
        numerical_features = torch.randn(batch_size, 118)
        
        output = encoder(numerical_features)
        
        assert output.shape == (batch_size, 256)
        
    def test_forward_multimodal(self):
        """Test forward pass with both numerical and categorical features."""
        categorical_dims = {'gender': 3, 'education': 5}
        
        encoder = MultiModalClinicalEncoder(
            numerical_dim=118,
            categorical_dims=categorical_dims,
            fusion_dim=256
        )
        
        batch_size = 3
        numerical_features = torch.randn(batch_size, 118)
        categorical_features = {
            'gender': torch.randint(0, 3, (batch_size,)),
            'education': torch.randint(0, 5, (batch_size,))
        }
        
        output = encoder(numerical_features, categorical_features)
        
        assert output.shape == (batch_size, 256)


class TestClinicalAttentionEncoder:
    """Test cases for ClinicalAttentionEncoder."""
    
    def test_init(self):
        """Test initialization."""
        encoder = ClinicalAttentionEncoder(
            input_dim=118,
            embed_dim=256,
            num_heads=8,
            num_layers=2
        )
        
        assert encoder.input_dim == 118
        assert encoder.embed_dim == 256
        
    def test_forward_pass(self):
        """Test forward pass."""
        encoder = ClinicalAttentionEncoder(
            input_dim=118,
            embed_dim=128,
            num_heads=4,
            num_layers=2
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 118)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, 128)
        
    def test_forward_pass_different_sizes(self):
        """Test forward pass with different input sizes."""
        sizes = [50, 100, 200]
        
        for size in sizes:
            encoder = ClinicalAttentionEncoder(
                input_dim=size,
                embed_dim=64,
                num_heads=2,
                num_layers=1
            )
            
            x = torch.randn(2, size)
            output = encoder(x)
            
            assert output.shape == (2, 64)


class TestClinicalFeatureNormalizer:
    """Test cases for ClinicalFeatureNormalizer."""
    
    def test_init_batch_norm(self):
        """Test initialization with batch normalization."""
        normalizer = ClinicalFeatureNormalizer(
            num_features=118,
            normalization_type='batch'
        )
        
        assert normalizer.num_features == 118
        assert normalizer.normalization_type == 'batch'
        assert isinstance(normalizer.norm, nn.BatchNorm1d)
        
    def test_init_layer_norm(self):
        """Test initialization with layer normalization."""
        normalizer = ClinicalFeatureNormalizer(
            num_features=118,
            normalization_type='layer'
        )
        
        assert isinstance(normalizer.norm, nn.LayerNorm)
        
    def test_init_instance_norm(self):
        """Test initialization with instance normalization."""
        normalizer = ClinicalFeatureNormalizer(
            num_features=118,
            normalization_type='instance'
        )
        
        assert isinstance(normalizer.norm, nn.InstanceNorm1d)
        
    def test_invalid_normalization_type(self):
        """Test initialization with invalid normalization type."""
        with pytest.raises(ValueError):
            ClinicalFeatureNormalizer(
                num_features=118,
                normalization_type='invalid'
            )
            
    def test_forward_pass_without_mask(self):
        """Test forward pass without mask."""
        normalizer = ClinicalFeatureNormalizer(num_features=50)
        
        batch_size = 4
        x = torch.randn(batch_size, 50)
        
        output = normalizer(x)
        
        assert output.shape == (batch_size, 50)
        
    def test_forward_pass_with_mask(self):
        """Test forward pass with mask."""
        normalizer = ClinicalFeatureNormalizer(num_features=50)
        
        batch_size = 4
        x = torch.randn(batch_size, 50)
        mask = torch.randint(0, 2, (batch_size, 50)).float()
        
        output = normalizer(x, mask)
        
        assert output.shape == (batch_size, 50)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_clinical_feature_mask(self):
        """Test clinical feature mask creation."""
        features = torch.tensor([
            [1.0, 2.0, -999.0, 4.0],
            [5.0, -999.0, 7.0, -999.0]
        ])
        
        mask = create_clinical_feature_mask(features, missing_value=-999.0)
        
        expected_mask = torch.tensor([
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]
        ])
        
        assert torch.equal(mask, expected_mask)
        
    def test_impute_missing_features_zero(self):
        """Test missing feature imputation with zero strategy."""
        features = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        mask = torch.tensor([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        
        imputed = impute_missing_features(features, mask, strategy='zero')
        
        expected = torch.tensor([
            [1.0, 0.0, 3.0],
            [4.0, 5.0, 0.0]
        ])
        
        assert torch.equal(imputed, expected)
        
    def test_impute_missing_features_mean(self):
        """Test missing feature imputation with mean strategy."""
        features = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        mask = torch.tensor([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        
        imputed = impute_missing_features(features, mask, strategy='mean')
        
        # Expected means: [2.5, 5.0, 3.0] (mean of valid values for each feature)
        expected = torch.tensor([
            [1.0, 5.0, 3.0],  # Missing value at index 1 replaced with mean 5.0
            [4.0, 5.0, 3.0]   # Missing value at index 2 replaced with mean 3.0
        ])
        
        assert torch.allclose(imputed, expected)
        
    def test_standardize_clinical_features(self):
        """Test clinical feature standardization."""
        features = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        standardized = standardize_clinical_features(features)
        
        # Check that mean is approximately zero and std is approximately one
        assert torch.allclose(standardized.mean(dim=0), torch.zeros(3), atol=1e-6)
        assert torch.allclose(standardized.std(dim=0), torch.ones(3), atol=1e-6)
        
    def test_invalid_imputation_strategy(self):
        """Test invalid imputation strategy."""
        features = torch.randn(2, 3)
        mask = torch.ones(2, 3)
        
        with pytest.raises(ValueError):
            impute_missing_features(features, mask, strategy='invalid')


@pytest.fixture
def sample_clinical_features():
    """Create sample clinical features for testing."""
    batch_size = 4
    num_features = 118
    
    # Create features with some missing values
    features = torch.randn(batch_size, num_features)
    
    # Introduce some missing values
    missing_indices = torch.randint(0, num_features, (batch_size, 10))
    for i in range(batch_size):
        features[i, missing_indices[i]] = -999.0
        
    return features


def test_integration_clinical_encoder(sample_clinical_features):
    """Test integration of clinical encoder with sample data."""
    encoder = ClinicalFeaturesEncoder(
        input_dim=118,
        hidden_dims=[256, 128],
        output_dim=64
    )
    
    # Set to evaluation mode
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(sample_clinical_features)
        
    # Verify output shape
    assert output.shape == (4, 64)
    
    # Check that output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))


def test_integration_multimodal_encoder():
    """Test integration of multimodal clinical encoder."""
    categorical_dims = {
        'gender': 3,
        'education': 5,
        'apoe': 4
    }
    
    encoder = MultiModalClinicalEncoder(
        numerical_dim=118,
        categorical_dims=categorical_dims,
        fusion_dim=256
    )
    
    batch_size = 3
    numerical_features = torch.randn(batch_size, 118)
    categorical_features = {
        'gender': torch.randint(0, 3, (batch_size,)),
        'education': torch.randint(0, 5, (batch_size,)),
        'apoe': torch.randint(0, 4, (batch_size,))
    }
    
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(numerical_features, categorical_features)
        
    assert output.shape == (batch_size, 256)
    assert not torch.allclose(output, torch.zeros_like(output))


def test_gradient_flow_clinical_encoder():
    """Test that gradients flow properly through clinical encoder."""
    encoder = ClinicalFeaturesEncoder(
        input_dim=118,
        hidden_dims=[128, 64],
        output_dim=32
    )
    
    x = torch.randn(2, 118, requires_grad=True)
    
    output = encoder(x)
    
    # Create dummy loss
    loss = output.sum()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    # Check that model parameters have gradients
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_clinical_attention_encoder_attention_mechanism():
    """Test that attention encoder produces different outputs for different inputs."""
    encoder = ClinicalAttentionEncoder(
        input_dim=50,
        embed_dim=64,
        num_heads=2,
        num_layers=1
    )
    
    # Create two different inputs
    x1 = torch.randn(1, 50)
    x2 = torch.randn(1, 50)
    
    encoder.eval()
    
    with torch.no_grad():
        output1 = encoder(x1)
        output2 = encoder(x2)
        
    # Outputs should be different for different inputs
    assert not torch.allclose(output1, output2, atol=1e-4)


def test_feature_normalizer_with_missing_values():
    """Test feature normalizer handling of missing values."""
    normalizer = ClinicalFeatureNormalizer(num_features=10)
    
    # Create features with missing values
    features = torch.randn(3, 10)
    mask = torch.randint(0, 2, (3, 10)).float()
    
    normalizer.eval()
    
    with torch.no_grad():
        output = normalizer(features, mask)
        
    assert output.shape == (3, 10)
    
    # Check that missing value embeddings are used where mask is 0
    # This is a basic check - in practice, the exact behavior depends on normalization
    assert not torch.allclose(output, torch.zeros_like(output))