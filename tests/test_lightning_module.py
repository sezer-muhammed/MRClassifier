"""
Tests for the Alzheimer's Detection Lightning Module

This module contains unit tests for the PyTorch Lightning module
to ensure proper functionality and integration.
"""

import pytest
import torch
import pytorch_lightning as pl
from unittest.mock import Mock, patch
import numpy as np

from gazimed.models.lightning_module import AlzheimersLightningModule


class TestAlzheimersLightningModule:
    """Test suite for AlzheimersLightningModule."""
    
    @pytest.fixture
    def model_config(self):
        """Create a minimal model configuration for testing."""
        return {
            'img_size': (32, 32, 32),  # Smaller size for testing
            'patch_size': (4, 4, 4),
            'in_channels': 2,
            'embed_dim': 48,  # Smaller for testing
            'depths': [1, 1, 1, 1],  # Smaller depths
            'num_heads': [2, 4, 8, 16],
            'clinical_features_dim': 118,
            'clinical_hidden_dims': [64, 128, 64],
            'clinical_output_dim': 32,
            'fusion_dim': 64,
            'learning_rate': 1e-4,
            'weight_decay': 1e-2,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def lightning_module(self, model_config):
        """Create a Lightning module for testing."""
        return AlzheimersLightningModule(**model_config)
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 2
        return {
            'images': torch.randn(batch_size, 2, 32, 32, 32),  # (B, C, D, H, W)
            'clinical_features': torch.randn(batch_size, 118),  # (B, clinical_dim)
            'targets': torch.rand(batch_size),  # (B,) - continuous targets between 0-1
            'subject_ids': ['subject_001', 'subject_002']
        }
    
    def test_model_initialization(self, lightning_module):
        """Test that the model initializes correctly."""
        assert isinstance(lightning_module, pl.LightningModule)
        assert hasattr(lightning_module, 'image_encoder')
        assert hasattr(lightning_module, 'clinical_encoder')
        assert hasattr(lightning_module, 'cross_modal_fusion')
        assert hasattr(lightning_module, 'multimodal_fusion')
        assert hasattr(lightning_module, 'regression_head')
    
    def test_forward_pass(self, lightning_module, sample_batch):
        """Test forward pass of the model."""
        lightning_module.eval()
        
        with torch.no_grad():
            predictions = lightning_module(
                sample_batch['images'], 
                sample_batch['clinical_features']
            )
        
        # Check output shape and range
        assert predictions.shape == (2, 1)  # (batch_size, 1)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)  # Sigmoid output
    
    def test_training_step(self, lightning_module, sample_batch):
        """Test training step."""
        lightning_module.train()
        
        loss = lightning_module.training_step(sample_batch, batch_idx=0)
        
        # Check that loss is computed and is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_validation_step(self, lightning_module, sample_batch):
        """Test validation step."""
        lightning_module.eval()
        
        outputs = lightning_module.validation_step(sample_batch, batch_idx=0)
        
        # Check that outputs contain expected keys
        assert 'loss' in outputs
        assert 'predictions' in outputs
        assert 'targets' in outputs
        assert isinstance(outputs['loss'], torch.Tensor)
    
    def test_configure_optimizers(self, lightning_module):
        """Test optimizer configuration."""
        optimizer_config = lightning_module.configure_optimizers()
        
        # Check that optimizer and scheduler are configured
        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config
        
        optimizer = optimizer_config['optimizer']
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == lightning_module.learning_rate
        assert optimizer.param_groups[0]['weight_decay'] == lightning_module.weight_decay
    
    def test_metrics_creation(self, lightning_module):
        """Test that metrics are created correctly."""
        # Check that metrics are created for all stages
        assert hasattr(lightning_module, 'train_metrics')
        assert hasattr(lightning_module, 'val_metrics')
        assert hasattr(lightning_module, 'test_metrics')
        
        # Check that expected metrics are present
        expected_metrics = ['mse', 'mae', 'r2', 'pearson', 'auc', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
        
        for metric_name in expected_metrics:
            assert metric_name in lightning_module.train_metrics
            assert metric_name in lightning_module.val_metrics
            assert metric_name in lightning_module.test_metrics
    
    def test_predict_step(self, lightning_module, sample_batch):
        """Test prediction step."""
        lightning_module.eval()
        
        predictions = lightning_module.predict_step(sample_batch, batch_idx=0)
        
        # Check prediction outputs
        assert 'predictions' in predictions
        assert 'binary_predictions' in predictions
        assert 'subject_ids' in predictions
        
        # Check shapes and types
        assert predictions['predictions'].shape == (2,)  # (batch_size,)
        assert predictions['binary_predictions'].shape == (2,)
        assert predictions['binary_predictions'].dtype == torch.int32
    
    def test_model_summary(self, lightning_module):
        """Test model summary generation."""
        summary = lightning_module.get_model_summary()
        
        assert isinstance(summary, str)
        assert 'Alzheimer\'s Detection Model Summary' in summary
        assert 'Total Parameters' in summary
        assert 'Trainable Parameters' in summary
    
    @pytest.mark.parametrize("loss_type", ['mse', 'mae', 'huber'])
    def test_different_loss_types(self, model_config, loss_type):
        """Test different loss function types."""
        model_config['loss_type'] = loss_type
        lightning_module = AlzheimersLightningModule(**model_config)
        
        # Check that the correct loss function is set
        if loss_type == 'mse':
            assert isinstance(lightning_module.criterion, torch.nn.MSELoss)
        elif loss_type == 'mae':
            assert isinstance(lightning_module.criterion, torch.nn.L1Loss)
        elif loss_type == 'huber':
            assert isinstance(lightning_module.criterion, torch.nn.HuberLoss)
    
    def test_gradient_clipping_configuration(self, lightning_module):
        """Test gradient clipping configuration."""
        # Mock optimizer
        optimizer = torch.optim.AdamW(lightning_module.parameters(), lr=1e-4)
        
        # Test gradient clipping with default values
        with patch.object(lightning_module, 'clip_gradients') as mock_clip:
            lightning_module.configure_gradient_clipping(optimizer)
            mock_clip.assert_called_once()
    
    def test_mixed_precision_support(self, model_config):
        """Test mixed precision configuration."""
        model_config['use_mixed_precision'] = True
        lightning_module = AlzheimersLightningModule(**model_config)
        
        assert lightning_module.use_mixed_precision is True
        assert lightning_module.hparams.use_mixed_precision is True
    
    def test_hyperparameter_saving(self, lightning_module):
        """Test that hyperparameters are saved correctly."""
        # Check that hyperparameters are saved
        assert hasattr(lightning_module, 'hparams')
        
        # Check some key hyperparameters
        assert lightning_module.hparams.learning_rate == lightning_module.learning_rate
        assert lightning_module.hparams.weight_decay == lightning_module.weight_decay
        assert lightning_module.hparams.clinical_features_dim == 118
    
    def test_model_with_different_input_channels(self, model_config):
        """Test model with different input channel configurations."""
        # Test with 3 channels (MRI + PET + difference)
        model_config['in_channels'] = 3
        lightning_module = AlzheimersLightningModule(**model_config)
        
        # Create sample batch with 3 channels
        sample_batch = {
            'images': torch.randn(2, 3, 32, 32, 32),
            'clinical_features': torch.randn(2, 118),
            'targets': torch.rand(2)
        }
        
        lightning_module.eval()
        with torch.no_grad():
            predictions = lightning_module(
                sample_batch['images'], 
                sample_batch['clinical_features']
            )
        
        assert predictions.shape == (2, 1)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_lightning_trainer_integration(self, tmp_path):
        """Test integration with PyTorch Lightning Trainer."""
        # Create a minimal model
        model = AlzheimersLightningModule(
            img_size=(16, 16, 16),  # Very small for testing
            patch_size=(4, 4, 4),
            in_channels=2,
            embed_dim=32,
            depths=[1, 1],
            num_heads=[2, 4],
            clinical_features_dim=118,
            clinical_hidden_dims=[32, 64, 32],
            clinical_output_dim=16,
            fusion_dim=32,
            learning_rate=1e-3,
            dropout=0.1
        )
        
        # Create a simple trainer for testing
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            default_root_dir=str(tmp_path)
        )
        
        # Create dummy data
        train_data = [
            {
                'images': torch.randn(1, 2, 16, 16, 16),
                'clinical_features': torch.randn(1, 118),
                'targets': torch.rand(1)
            }
            for _ in range(4)
        ]
        
        # Mock dataloader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_data, batch_size=2)
        
        # Test that training runs without errors
        try:
            trainer.fit(model, train_loader)
            assert True  # Training completed successfully
        except Exception as e:
            pytest.fail(f"Training failed with error: {e}")


if __name__ == "__main__":
    pytest.main([__file__])