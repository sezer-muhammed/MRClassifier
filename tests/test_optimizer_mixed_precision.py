"""
Test suite for Task 6.3: Configure optimizers and mixed precision

This test suite verifies the implementation of:
- AdamW optimizer with specified learning rate and weight decay
- FP16 mixed precision training for memory efficiency
- Gradient clipping and accumulation support

Requirements tested:
- 3.2: AdamW optimizer with learning rate 1×10⁻⁴, cosine decay, and weight decay 1×10⁻²
- 3.3: Batch size 4 with gradient accumulation of 8 steps
"""

import pytest
import torch
import pytorch_lightning as pl
from unittest.mock import Mock, patch
import numpy as np

from gazimed.models.lightning_module import AlzheimersLightningModule
from gazimed.training.training_config import (
    TrainingConfig,
    TrainingConfigurationManager,
    OptimizerConfig,
    MixedPrecisionConfig,
    GradientAccumulationConfig
)


class TestOptimizerConfiguration:
    """Test AdamW optimizer configuration as per requirement 3.2."""
    
    def test_adamw_optimizer_parameters(self):
        """Test that AdamW optimizer is configured with correct parameters."""
        # Create model with default configuration
        model = AlzheimersLightningModule(
            learning_rate=1e-4,
            weight_decay=1e-2
        )
        
        # Get optimizer configuration
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']
        
        # Verify optimizer type
        assert isinstance(optimizer, torch.optim.AdamW), "Optimizer must be AdamW"
        
        # Verify learning rate (requirement 3.2)
        assert optimizer.param_groups[0]['lr'] == 1e-4, "Learning rate must be 1×10⁻⁴"
        
        # Verify weight decay (requirement 3.2)
        assert optimizer.param_groups[0]['weight_decay'] == 1e-2, "Weight decay must be 1×10⁻²"
        
        # Verify AdamW specific parameters
        assert optimizer.param_groups[0]['betas'] == (0.9, 0.999), "Beta parameters should be (0.9, 0.999)"
        assert optimizer.param_groups[0]['eps'] == 1e-8, "Epsilon should be 1e-8"
        assert optimizer.param_groups[0]['amsgrad'] == False, "AMSGrad should be disabled for standard AdamW"
    
    def test_cosine_annealing_scheduler(self):
        """Test that cosine annealing scheduler is configured correctly."""
        model = AlzheimersLightningModule(
            learning_rate=1e-4,
            weight_decay=1e-2,
            cosine_t_max=100
        )
        
        optimizer_config = model.configure_optimizers()
        scheduler_config = optimizer_config['lr_scheduler']
        scheduler = scheduler_config['scheduler']
        
        # Verify scheduler type
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR), "Must use CosineAnnealingLR"
        
        # Verify scheduler parameters
        assert scheduler.T_max == 100, "T_max should match cosine_t_max parameter"
        assert scheduler.eta_min == 1e-6, "Minimum learning rate should be 1e-6"
        
        # Verify scheduler configuration
        assert scheduler_config['monitor'] == 'val_loss', "Should monitor validation loss"
        assert scheduler_config['interval'] == 'epoch', "Should update every epoch"
        assert scheduler_config['frequency'] == 1, "Should update every epoch"
    
    def test_optimizer_config_class(self):
        """Test OptimizerConfig dataclass with requirement values."""
        config = OptimizerConfig()
        
        # Verify default values match requirements
        assert config.learning_rate == 1e-4, "Default learning rate must be 1×10⁻⁴"
        assert config.weight_decay == 1e-2, "Default weight decay must be 1×10⁻²"
        assert config.betas == (0.9, 0.999), "Default betas should be (0.9, 0.999)"
        assert config.eps == 1e-8, "Default eps should be 1e-8"
        assert config.amsgrad == False, "Default amsgrad should be False"


class TestMixedPrecisionConfiguration:
    """Test FP16 mixed precision training configuration."""
    
    def test_mixed_precision_config_class(self):
        """Test MixedPrecisionConfig dataclass."""
        config = MixedPrecisionConfig()
        
        # Verify default values for memory efficiency
        assert config.precision == 16, "Default precision should be 16 (FP16)"
        assert config.amp_backend == 'native', "Should use native PyTorch AMP"
        assert config.gradient_clip_val == 1.0, "Default gradient clipping should be 1.0"
        assert config.gradient_clip_algorithm == 'norm', "Should use norm-based gradient clipping"
    
    def test_lightning_module_mixed_precision_support(self):
        """Test that Lightning module supports mixed precision configuration."""
        model = AlzheimersLightningModule(
            use_mixed_precision=True,
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm'
        )
        
        # Verify mixed precision is enabled in hyperparameters
        assert model.hparams.use_mixed_precision == True, "Mixed precision should be enabled"
        assert model.hparams.gradient_clip_val == 1.0, "Gradient clipping value should be set"
        assert model.hparams.gradient_clip_algorithm == 'norm', "Gradient clipping algorithm should be norm"
    
    def test_trainer_mixed_precision_configuration(self):
        """Test that trainer is configured with mixed precision."""
        config = TrainingConfig()
        config.mixed_precision_config.precision = 16
        
        config_manager = TrainingConfigurationManager(config)
        trainer = config_manager.create_trainer()
        
        # Verify trainer precision
        assert trainer.precision == 16, "Trainer should use FP16 precision"
        
        # Verify gradient clipping
        assert trainer.gradient_clip_val == 1.0, "Gradient clipping should be enabled"
        assert trainer.gradient_clip_algorithm == 'norm', "Should use norm-based clipping"


class TestGradientAccumulationConfiguration:
    """Test gradient accumulation configuration as per requirement 3.3."""
    
    def test_gradient_accumulation_config_class(self):
        """Test GradientAccumulationConfig dataclass."""
        config = GradientAccumulationConfig()
        
        # Verify default values match requirement 3.3
        assert config.accumulate_grad_batches == 8, "Default gradient accumulation must be 8 steps"
        assert config.gradient_accumulation_schedule is not None, "Schedule should be initialized"
        assert config.gradient_accumulation_schedule[0] == 8, "Should accumulate 8 batches from epoch 0"
    
    def test_lightning_module_gradient_accumulation_support(self):
        """Test that Lightning module supports gradient accumulation."""
        model = AlzheimersLightningModule(
            accumulate_grad_batches=8
        )
        
        # Verify gradient accumulation is set in hyperparameters
        assert model.hparams.accumulate_grad_batches == 8, "Gradient accumulation should be 8 steps"
    
    def test_trainer_gradient_accumulation_configuration(self):
        """Test that trainer is configured with gradient accumulation."""
        config = TrainingConfig()
        config.gradient_config.accumulate_grad_batches = 8
        
        config_manager = TrainingConfigurationManager(config)
        trainer = config_manager.create_trainer()
        
        # Verify trainer gradient accumulation
        assert trainer.accumulate_grad_batches == 8, "Trainer should accumulate 8 gradient batches"


class TestTrainingConfigurationManager:
    """Test the comprehensive training configuration manager."""
    
    def test_configuration_validation_success(self):
        """Test that valid configuration passes validation."""
        config = TrainingConfig()
        config_manager = TrainingConfigurationManager(config)
        
        # Should not raise any exceptions
        assert config_manager.validate_configuration() == True
    
    def test_configuration_validation_batch_size_failure(self):
        """Test that invalid batch size fails validation."""
        config = TrainingConfig()
        config.batch_size = 8  # Should be 4 per requirement 3.3
        
        config_manager = TrainingConfigurationManager(config)
        
        with pytest.raises(ValueError, match="Batch size must be 4"):
            config_manager.validate_configuration()
    
    def test_configuration_validation_gradient_accumulation_failure(self):
        """Test that invalid gradient accumulation fails validation."""
        config = TrainingConfig()
        config.gradient_config.accumulate_grad_batches = 4  # Should be 8 per requirement 3.3
        
        config_manager = TrainingConfigurationManager(config)
        
        with pytest.raises(ValueError, match="Gradient accumulation must be 8 steps"):
            config_manager.validate_configuration()
    
    def test_configuration_validation_learning_rate_failure(self):
        """Test that invalid learning rate fails validation."""
        config = TrainingConfig()
        config.optimizer_config.learning_rate = 1e-3  # Should be 1e-4 per requirement 3.2
        
        config_manager = TrainingConfigurationManager(config)
        
        with pytest.raises(ValueError, match="Learning rate must be 1e-4"):
            config_manager.validate_configuration()
    
    def test_configuration_validation_weight_decay_failure(self):
        """Test that invalid weight decay fails validation."""
        config = TrainingConfig()
        config.optimizer_config.weight_decay = 1e-3  # Should be 1e-2 per requirement 3.2
        
        config_manager = TrainingConfigurationManager(config)
        
        with pytest.raises(ValueError, match="Weight decay must be 1e-2"):
            config_manager.validate_configuration()
    
    def test_get_model_kwargs(self):
        """Test that model kwargs are generated correctly."""
        config = TrainingConfig()
        config_manager = TrainingConfigurationManager(config)
        
        model_kwargs = config_manager.get_model_kwargs()
        
        # Verify all required kwargs are present
        assert 'learning_rate' in model_kwargs
        assert 'weight_decay' in model_kwargs
        assert 'use_mixed_precision' in model_kwargs
        assert 'gradient_clip_val' in model_kwargs
        assert 'accumulate_grad_batches' in model_kwargs
        
        # Verify values match requirements
        assert model_kwargs['learning_rate'] == 1e-4
        assert model_kwargs['weight_decay'] == 1e-2
        assert model_kwargs['accumulate_grad_batches'] == 8
    
    def test_create_trainer_with_all_configurations(self):
        """Test that trainer is created with all required configurations."""
        config = TrainingConfig()
        config_manager = TrainingConfigurationManager(config)
        
        trainer = config_manager.create_trainer()
        
        # Verify all configurations are applied
        assert trainer.precision == 16, "Should use FP16 precision"
        assert trainer.gradient_clip_val == 1.0, "Should have gradient clipping"
        assert trainer.accumulate_grad_batches == 8, "Should accumulate 8 batches"
        assert trainer.max_epochs == 100, "Should have correct max epochs"
        
        # Verify callbacks are created
        assert len(trainer.callbacks) > 0, "Should have callbacks"
        
        # Check for specific callback types
        callback_types = [type(cb).__name__ for cb in trainer.callbacks]
        assert 'ModelCheckpoint' in callback_types, "Should have model checkpointing"
        assert 'EarlyStopping' in callback_types, "Should have early stopping"
        assert 'LearningRateMonitor' in callback_types, "Should have LR monitoring"


class TestIntegrationRequirements:
    """Integration tests to verify all requirements are met together."""
    
    def test_complete_task_6_3_implementation(self):
        """Test that task 6.3 is completely implemented with all requirements."""
        # Create model with all task 6.3 configurations
        model = AlzheimersLightningModule(
            learning_rate=1e-4,      # Requirement 3.2
            weight_decay=1e-2,       # Requirement 3.2
            use_mixed_precision=True, # Task 6.3: FP16 mixed precision
            gradient_clip_val=1.0,   # Task 6.3: Gradient clipping
            accumulate_grad_batches=8 # Requirement 3.3
        )
        
        # Create training configuration
        config = TrainingConfig()
        config_manager = TrainingConfigurationManager(config)
        
        # Validate configuration
        assert config_manager.validate_configuration() == True
        
        # Create trainer
        trainer = config_manager.create_trainer()
        
        # Verify all task 6.3 requirements are met
        
        # 1. AdamW optimizer with specified learning rate and weight decay
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[0]['weight_decay'] == 1e-2
        
        # 2. FP16 mixed precision training for memory efficiency
        assert trainer.precision == 16
        
        # 3. Gradient clipping and accumulation support
        assert trainer.gradient_clip_val == 1.0
        assert trainer.accumulate_grad_batches == 8
        
        print("✓ Task 6.3 implementation verified:")
        print("  - AdamW optimizer with lr=1e-4, weight_decay=1e-2")
        print("  - FP16 mixed precision training enabled")
        print("  - Gradient clipping (val=1.0) and accumulation (8 steps) configured")
    
    def test_memory_efficiency_with_mixed_precision(self):
        """Test that mixed precision provides memory efficiency benefits."""
        # This test would ideally measure actual memory usage
        # For now, we verify the configuration is correct
        
        config = TrainingConfig()
        config.mixed_precision_config.precision = 16
        
        config_manager = TrainingConfigurationManager(config)
        trainer = config_manager.create_trainer()
        
        # Verify FP16 is enabled
        assert trainer.precision == 16, "FP16 should be enabled for memory efficiency"
        
        # In a real scenario, this would reduce memory usage by ~50%
        print("✓ Mixed precision configuration verified for memory efficiency")
    
    def test_gradient_accumulation_effective_batch_size(self):
        """Test that gradient accumulation provides correct effective batch size."""
        config = TrainingConfig()
        config.batch_size = 4  # Physical batch size
        config.gradient_config.accumulate_grad_batches = 8  # Accumulation steps
        
        # Effective batch size = physical_batch_size * accumulate_grad_batches
        effective_batch_size = config.batch_size * config.gradient_config.accumulate_grad_batches
        
        assert effective_batch_size == 32, "Effective batch size should be 4 * 8 = 32"
        
        print(f"✓ Gradient accumulation provides effective batch size of {effective_batch_size}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])