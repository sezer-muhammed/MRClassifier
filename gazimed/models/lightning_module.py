"""
PyTorch Lightning Module for Alzheimer's Detection Model

This module implements the complete PyTorch Lightning training module that combines
the Swin-UNETR backbone, cross-attention fusion, and clinical features encoder
for multimodal Alzheimer's disease detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from .swin_unetr import SwinUNETR
from .cross_attention import CrossModalAttention, HierarchicalFusion
from .clinical_encoder import ClinicalFeaturesEncoder


class AlzheimersLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Alzheimer's detection using multimodal data.
    
    This module combines:
    - Swin-UNETR backbone for 3D medical image processing
    - Cross-attention fusion for MRI/PET modality integration
    - Clinical features encoder for numerical features
    - Multimodal fusion for final prediction
    """
    
    def __init__(
        self,
        # Model architecture parameters
        img_size: Tuple[int, int, int] = (91, 109, 91),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 2,  # MRI + PET
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (7, 7, 7),
        
        # Clinical features parameters
        clinical_features_dim: int = 116,
        clinical_hidden_dims: List[int] = [256, 512, 256],
        clinical_output_dim: int = 128,
        
        # Cross-attention parameters
        cross_attn_heads: int = 2,
        cross_attn_dim: int = 48,
        
        # Fusion parameters
        fusion_dim: int = 256,
        
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        cosine_t_max: int = 100,
        
        # Regularization
        dropout: float = 0.3,
        
        # Loss parameters
        loss_type: str = 'mse',  # 'mse', 'mae', 'huber'
        
        # Mixed precision and gradient settings
        use_mixed_precision: bool = True,
        gradient_clip_val: Optional[float] = 1.0,
        gradient_clip_algorithm: str = 'norm',
        accumulate_grad_batches: int = 8,
        
        # Optimizer specific settings
        optimizer_betas: Tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_amsgrad: bool = False,
        
        **kwargs
    ):
        """
        Initialize Alzheimer's Lightning Module.
        
        Args:
            img_size: Input image size (D, H, W)
            patch_size: Patch size for patch embedding
            in_channels: Number of input channels (2 for MRI+PET, 3 for MRI+PET+diff)
            embed_dim: Embedding dimension
            depths: Depths of each Swin Transformer stage
            num_heads: Number of attention heads in each stage
            window_size: Window size for Swin attention
            clinical_features_dim: Number of clinical features
            clinical_hidden_dims: Hidden dimensions for clinical encoder
            clinical_output_dim: Output dimension for clinical encoder
            cross_attn_heads: Number of cross-attention heads
            cross_attn_dim: Dimension per cross-attention head
            fusion_dim: Final fusion dimension
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            cosine_t_max: T_max for cosine annealing scheduler
            dropout: Dropout rate
            loss_type: Type of loss function
            use_mixed_precision: Whether to use mixed precision training
            gradient_clip_val: Gradient clipping value (None to disable)
            gradient_clip_algorithm: Gradient clipping algorithm ('norm' or 'value')
            accumulate_grad_batches: Number of batches to accumulate gradients
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cosine_t_max = cosine_t_max
        self.loss_type = loss_type
        self.use_mixed_precision = use_mixed_precision
        
        # Image processing branch - Swin-UNETR encoders
        # Main encoder for combined input
        self.image_encoder = SwinUNETR(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.1
        )
        
        # Separate encoders for individual modalities (single channel each)
        self.mri_encoder = SwinUNETR(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,  # Single channel for MRI
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.1
        )
        
        self.pet_encoder = SwinUNETR(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,  # Single channel for PET
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.1
        )
        
        # Calculate feature dimensions from different stages
        feature_dims = [int(embed_dim * 2 ** i) for i in range(len(depths))]
        
        # Cross-modal attention fusion for MRI/PET
        self.cross_modal_fusion = HierarchicalFusion(
            feature_dims=feature_dims,
            target_dim=fusion_dim,
            num_heads=cross_attn_heads,
            num_layers=2,
            drop=dropout
        )
        
        # Clinical features encoder
        self.clinical_encoder = ClinicalFeaturesEncoder(
            input_dim=clinical_features_dim,
            hidden_dims=clinical_hidden_dims,
            output_dim=clinical_output_dim,
            dropout=dropout,
            activation='relu'
        )
        
        # Multimodal fusion layer
        multimodal_input_dim = fusion_dim + clinical_output_dim
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(multimodal_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Regression head for continuous score prediction (0-1)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
        # Loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Metrics for regression
        self.train_metrics = self._create_metrics('train')
        self.val_metrics = self._create_metrics('val')
        self.test_metrics = self._create_metrics('test')
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_metrics(self, stage: str) -> nn.ModuleDict:
        """Create metrics for a specific stage."""
        return nn.ModuleDict({
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'r2': torchmetrics.R2Score(),
            'pearson': torchmetrics.PearsonCorrCoef(),
            # For binary classification metrics (using 0.5 threshold)
            'auc': torchmetrics.AUROC(task='binary'),
            'accuracy': torchmetrics.Accuracy(task='binary'),
            'precision': torchmetrics.Precision(task='binary'),
            'recall': torchmetrics.Recall(task='binary'),
            'f1': torchmetrics.F1Score(task='binary'),
            'specificity': torchmetrics.Specificity(task='binary')
        })
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        images: torch.Tensor, 
        clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            images: Input images of shape (B, C, D, H, W)
            clinical_features: Clinical features of shape (B, clinical_dim)
            
        Returns:
            Alzheimer's score predictions of shape (B, 1)
        """
        # For cross-modal fusion, we need to separate MRI and PET features
        # Assuming input has 2 channels: [MRI, PET]
        if images.shape[1] == 2:
            mri_images = images[:, 0:1]  # First channel
            pet_images = images[:, 1:2]  # Second channel
            
            # Process each modality separately using dedicated encoders
            mri_features_list = self.mri_encoder(mri_images)
            pet_features_list = self.pet_encoder(pet_images)
            
            # Apply cross-modal fusion
            fused_image_features, _ = self.cross_modal_fusion(
                mri_features_list, pet_features_list
            )
        else:
            # If we have combined features, process with main encoder
            image_features_list = self.image_encoder(images)
            # Use hierarchical fusion on the same features
            fused_image_features, _ = self.cross_modal_fusion(
                image_features_list, image_features_list
            )
        
        # Process clinical features
        clinical_encoded = self.clinical_encoder(clinical_features)
        
        # Multimodal fusion
        combined_features = torch.cat([fused_image_features, clinical_encoded], dim=1)
        fused_features = self.multimodal_fusion(combined_features)
        
        # Final prediction
        predictions = self.regression_head(fused_features)
        
        return predictions
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with enhanced logging and NaN detection."""
        images = batch['volumes']  # Shape: [B, 2, H, W, D]
        clinical_features = batch['clinical_features']  # Shape: [B, 116]
        targets = batch['alzheimer_score']  # Shape: [B]
        
        # Quick validation check for first few batches
        if batch_idx < 3:
            print(f"Batch {batch_idx}: Images {images.shape}, NaN: {torch.isnan(images).any()}, Range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Check for NaN/Inf in inputs
        if torch.isnan(images).any() or torch.isinf(images).any():
            self.log('debug/nan_images', 1.0, on_step=True)
            print(f"ERROR: NaN/Inf detected in images at batch {batch_idx}")
            # Find which samples have NaN
            nan_mask = torch.isnan(images).any(dim=(1,2,3,4))
            print(f"Samples with NaN: {nan_mask.nonzero().flatten()}")
            
            # Replace NaN with zeros to prevent crash
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"After NaN replacement - min: {images.min()}, max: {images.max()}")
        
        if torch.isnan(clinical_features).any() or torch.isinf(clinical_features).any():
            self.log('debug/nan_clinical', 1.0, on_step=True)
            print(f"ERROR: NaN/Inf detected in clinical features at batch {batch_idx}")
            clinical_features = torch.nan_to_num(clinical_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            self.log('debug/nan_in_targets', 1.0, on_step=True)
            print(f"Warning: NaN/Inf detected in targets at batch {batch_idx}")
        
        # Forward pass
        predictions = self(images, clinical_features)
        predictions = predictions.squeeze(-1)  # Shape: [B]
        
        # Check for NaN/Inf in predictions
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            self.log('debug/nan_in_predictions', 1.0, on_step=True)
            print(f"Warning: NaN/Inf detected in predictions at batch {batch_idx}")
            print(f"Predictions stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
        
        # Compute loss with numerical stability
        loss = self.criterion(predictions, targets)
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.log('debug/nan_loss', 1.0, on_step=True)
            print(f"Warning: NaN/Inf loss detected at batch {batch_idx}")
            print(f"Loss: {loss}, Predictions: {predictions}, Targets: {targets}")
            # Return a small loss to prevent training from crashing
            return torch.tensor(0.01, requires_grad=True, device=loss.device)
        
        # Update metrics
        self.train_metrics['mse'](predictions, targets)
        self.train_metrics['mae'](predictions, targets)
        self.train_metrics['r2'](predictions, targets)
        self.train_metrics['pearson'](predictions, targets)
        
        # Binary classification metrics (threshold at 0.5)
        binary_preds = (predictions > 0.5).int()
        binary_targets = (targets > 0.5).int()
        
        self.train_metrics['accuracy'](binary_preds, binary_targets)
        self.train_metrics['precision'](binary_preds, binary_targets)
        self.train_metrics['recall'](binary_preds, binary_targets)
        self.train_metrics['f1'](binary_preds, binary_targets)
        self.train_metrics['auc'](predictions, binary_targets)
        
        # Enhanced logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        
        # Log prediction statistics
        self.log('train/pred_mean', predictions.mean(), on_step=True)
        self.log('train/pred_std', predictions.std(), on_step=True)
        self.log('train/pred_min', predictions.min(), on_step=True)
        self.log('train/pred_max', predictions.max(), on_step=True)
        
        # Log target statistics
        self.log('train/target_mean', targets.mean(), on_step=True)
        self.log('train/target_std', targets.std(), on_step=True)
        
        # Log gradient norms periodically
        if batch_idx % 10 == 0:
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.log('train/grad_norm', total_norm, on_step=True)
        
        # Log sample images and predictions periodically
        if batch_idx % 50 == 0 and batch_idx > 0:
            self._log_sample_images_and_predictions(images, predictions, targets, 'train', batch_idx)
        
        return loss
    
    def _log_sample_images_and_predictions(self, images, predictions, targets, stage, batch_idx):
        """Log sample images and predictions to TensorBoard."""
        try:
            # Only log a few samples to avoid memory issues
            num_samples = min(4, images.shape[0])
            
            # Get middle slices from 3D volumes for visualization
            # Images shape: [B, 2, H, W, D]
            mid_slice_idx = images.shape[-1] // 2
            
            # Extract middle slices: [B, 2, H, W]
            mri_slices = images[:num_samples, 0, :, :, mid_slice_idx]  # MRI channel
            pet_slices = images[:num_samples, 1, :, :, mid_slice_idx]  # PET channel
            
            # Normalize for visualization
            mri_slices = (mri_slices - mri_slices.min()) / (mri_slices.max() - mri_slices.min() + 1e-8)
            pet_slices = (pet_slices - pet_slices.min()) / (pet_slices.max() - pet_slices.min() + 1e-8)
            
            # Create a grid of images
            import torchvision.utils as vutils
            
            # Log MRI slices
            mri_grid = vutils.make_grid(mri_slices.unsqueeze(1), nrow=2, normalize=True)
            self.logger.experiment.add_image(f'{stage}/mri_samples', mri_grid, self.global_step)
            
            # Log PET slices
            pet_grid = vutils.make_grid(pet_slices.unsqueeze(1), nrow=2, normalize=True)
            self.logger.experiment.add_image(f'{stage}/pet_samples', pet_grid, self.global_step)
            
            # Log predictions vs targets as text
            pred_text = f"Predictions: {predictions[:num_samples].detach().cpu().numpy()}"
            target_text = f"Targets: {targets[:num_samples].detach().cpu().numpy()}"
            
            self.logger.experiment.add_text(f'{stage}/predictions', pred_text, self.global_step)
            self.logger.experiment.add_text(f'{stage}/targets', target_text, self.global_step)
            
        except Exception as e:
            print(f"Warning: Failed to log images for {stage}: {e}")
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with comprehensive metric logging.
        
        Implements validation step as specified in requirements 3.1, 3.2:
        - Computes validation loss and metrics
        - Logs metrics for monitoring training progress
        - Returns outputs for epoch-end aggregation
        """
        images = batch['volumes']  # Shape: [B, 2, H, W, D]
        clinical_features = batch['clinical_features']  # Shape: [B, 116]
        targets = batch['alzheimer_score']  # Shape: [B]
        
        # Forward pass
        predictions = self(images, clinical_features)
        predictions = predictions.squeeze(-1)  # Shape: [B]
        
        # Compute loss
        loss = self.criterion(predictions, targets)
        
        # Update metrics with error handling for small batch sizes
        try:
            self.val_metrics['mse'](predictions, targets)
            self.val_metrics['mae'](predictions, targets)
        except Exception as e:
            print(f"Warning: Failed to update basic metrics: {e}")
        
        # Metrics that require multiple samples - handle gracefully
        try:
            if len(predictions) > 1:
                self.val_metrics['r2'](predictions, targets)
                self.val_metrics['pearson'](predictions, targets)
        except Exception as e:
            print(f"Warning: Failed to update R2/Pearson metrics (batch size {len(predictions)}): {e}")
        
        # Binary classification metrics (threshold at 0.5)
        binary_preds = (predictions > 0.5).int()
        binary_targets = (targets > 0.5).int()
        
        try:
            self.val_metrics['accuracy'](binary_preds, binary_targets)
            if len(predictions) > 1:  # Precision/Recall/F1 may need multiple samples
                self.val_metrics['precision'](binary_preds, binary_targets)
                self.val_metrics['recall'](binary_preds, binary_targets)
                self.val_metrics['f1'](binary_preds, binary_targets)
            self.val_metrics['auc'](predictions, binary_targets)
        except Exception as e:
            print(f"Warning: Failed to update classification metrics: {e}")
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log prediction statistics
        self.log('val/pred_mean', predictions.mean(), on_step=False, on_epoch=True)
        self.log('val/pred_std', predictions.std(), on_step=False, on_epoch=True)
        
        # Log sample images periodically
        if batch_idx % 20 == 0 and batch_idx > 0:
            self._log_sample_images_and_predictions(images, predictions, targets, 'val', batch_idx)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'binary_predictions': binary_preds,
            'binary_targets': binary_targets
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        images = batch['volumes']  # Shape: [B, 2, H, W, D]
        clinical_features = batch['clinical_features']  # Shape: [B, 116]
        targets = batch['alzheimer_score']  # Shape: [B]
        
        # Forward pass
        predictions = self(images, clinical_features)
        predictions = predictions.squeeze(-1)  # Shape: [B]
        
        # Compute loss
        loss = self.criterion(predictions, targets)
        
        # Update metrics with error handling for small batch sizes
        try:
            self.test_metrics['mse'](predictions, targets)
            self.test_metrics['mae'](predictions, targets)
        except Exception as e:
            print(f"Warning: Failed to update basic test metrics: {e}")
        
        # Metrics that require multiple samples - handle gracefully
        try:
            if len(predictions) > 1:
                self.test_metrics['r2'](predictions, targets)
                self.test_metrics['pearson'](predictions, targets)
        except Exception as e:
            print(f"Warning: Failed to update R2/Pearson test metrics (batch size {len(predictions)}): {e}")
        
        # Binary classification metrics (threshold at 0.5)
        binary_preds = (predictions > 0.5).int()
        binary_targets = (targets > 0.5).int()
        
        try:
            self.test_metrics['accuracy'](binary_preds, binary_targets)
            if len(predictions) > 1:  # Precision/Recall/F1 may need multiple samples
                self.test_metrics['precision'](binary_preds, binary_targets)
                self.test_metrics['recall'](binary_preds, binary_targets)
                self.test_metrics['f1'](binary_preds, binary_targets)
            self.test_metrics['auc'](predictions, binary_targets)
        except Exception as e:
            print(f"Warning: Failed to update classification test metrics: {e}")
        
        # Log test loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'binary_predictions': binary_preds,
            'binary_targets': binary_targets
        }
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Compute and log training metrics
        for metric_name, metric in self.train_metrics.items():
            value = metric.compute()
            self.log(f'train_{metric_name}', value, on_epoch=True)
            metric.reset()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Compute and log validation metrics with error handling
        for metric_name, metric in self.val_metrics.items():
            try:
                value = metric.compute()
                self.log(f'val_{metric_name}', value, on_epoch=True, prog_bar=(metric_name in ['auc', 'r2']))
            except Exception as e:
                print(f"Warning: Failed to compute validation metric '{metric_name}': {e}")
                # Log a placeholder value for failed metrics
                if metric_name in ['r2', 'pearson']:
                    self.log(f'val_{metric_name}', 0.0, on_epoch=True, prog_bar=(metric_name in ['auc', 'r2']))
            finally:
                metric.reset()
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Compute and log test metrics with error handling
        for metric_name, metric in self.test_metrics.items():
            try:
                value = metric.compute()
                self.log(f'test_{metric_name}', value, on_epoch=True)
            except Exception as e:
                print(f"Warning: Failed to compute test metric '{metric_name}': {e}")
                # Log a placeholder value for failed metrics
                if metric_name in ['r2', 'pearson']:
                    self.log(f'test_{metric_name}', 0.0, on_epoch=True)
            finally:
                metric.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Implements optimizer configuration as specified in requirements 3.2, 3.3:
        - AdamW optimizer with learning rate 1×10⁻⁴ and weight decay 1×10⁻²
        - Cosine annealing learning rate scheduler with decay
        - Supports mixed precision training when enabled
        - Gradient accumulation of 8 steps (handled by trainer)
        """
        # AdamW optimizer as specified in requirements 3.2
        # Learning rate: 1×10⁻⁴, Weight decay: 1×10⁻²
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,  # 1e-4 as per requirements
            weight_decay=self.weight_decay,  # 1e-2 as per requirements
            betas=self.hparams.optimizer_betas,  # (0.9, 0.999)
            eps=self.hparams.optimizer_eps,  # 1e-8
            amsgrad=self.hparams.optimizer_amsgrad  # False for standard AdamW
        )
        
        # Cosine annealing scheduler with decay as specified in requirements 3.2
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cosine_t_max,
            eta_min=1e-6  # Minimum learning rate to prevent complete decay
        )
        
        # Return optimizer and scheduler configuration
        # Mixed precision and gradient accumulation are handled by the trainer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Monitor validation loss for scheduling
                'interval': 'epoch',    # Update scheduler every epoch
                'frequency': 1,         # Update every epoch
                'name': 'cosine_annealing',
                'strict': True          # Strict monitoring
            }
        }
    
    def configure_gradient_clipping(
        self, 
        optimizer: torch.optim.Optimizer, 
        gradient_clip_val: Optional[float] = None, 
        gradient_clip_algorithm: Optional[str] = None
    ) -> None:
        """
        Configure gradient clipping for training stability.
        
        Implements gradient clipping as specified in requirements 3.2, 3.3:
        - Supports gradient clipping by norm or value
        - Uses hyperparameter values if not explicitly provided
        - Helps prevent gradient explosion during training
        """
        # Use hyperparameter values if not provided
        clip_val = gradient_clip_val or self.hparams.gradient_clip_val
        clip_algorithm = gradient_clip_algorithm or self.hparams.gradient_clip_algorithm
        
        if clip_val is not None and clip_val > 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=clip_val, 
                gradient_clip_algorithm=clip_algorithm
            )
    
    def predict_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        images = batch['images']
        clinical_features = batch['clinical_features']
        
        # Forward pass
        predictions = self(images, clinical_features)
        predictions = predictions.squeeze(-1)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).int()
        
        return {
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'subject_ids': batch.get('subject_ids', None)
        }
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
        Alzheimer's Detection Model Summary:
        ===================================
        
        Architecture Components:
        - Image Encoder: Swin-UNETR
        - Cross-Modal Fusion: Hierarchical Attention
        - Clinical Encoder: Multi-layer MLP
        - Multimodal Fusion: Feed-forward layers
        - Regression Head: Sigmoid output (0-1)
        
        Model Parameters:
        - Total Parameters: {total_params:,}
        - Trainable Parameters: {trainable_params:,}
        - Non-trainable Parameters: {total_params - trainable_params:,}
        
        Input Specifications:
        - Image Size: {self.hparams.img_size}
        - Input Channels: {self.hparams.in_channels}
        - Clinical Features: {self.hparams.clinical_features_dim}
        
        Training Configuration:
        - Learning Rate: {self.learning_rate}
        - Weight Decay: {self.weight_decay}
        - Loss Function: {self.loss_type.upper()}
        - Mixed Precision: {self.use_mixed_precision}
        """
        
        return summary


class AlzheimersDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Alzheimer's detection data.
    
    This DataModule handles data loading, preprocessing, and splitting
    for the multimodal Alzheimer's detection task.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 0,  # Set to 0 to avoid multiprocessing issues with database connections
        pin_memory: bool = True,
        persistent_workers: bool = False,  # Disabled when num_workers=0
        **kwargs
    ):
        """
        Initialize Alzheimer's DataModule.
        
        Args:
            data_dir: Directory containing the data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            persistent_workers: Whether to use persistent workers
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and dataloaders for different stages using create_data_loaders utility."""
        from gazimed.data.database import DatabaseManager
        from gazimed.data.dataset import DataSplitter, create_data_loaders
        from gazimed.data.augmentation import create_training_augmentation, create_validation_augmentation
        import os

        # Initialize database manager
        self.db_manager = DatabaseManager(self.data_dir if os.path.isfile(self.data_dir) else os.path.join(self.data_dir, "gazimed_database.db"))
        self.splitter = DataSplitter(self.db_manager)

        # Split data
        self.train_ids, self.val_ids, self.test_ids = self.splitter.train_val_test_split()

        # Augmentations
        self.train_transform = create_training_augmentation()
        self.val_transform = create_validation_augmentation()

        # Use utility to create dataloaders with balanced sampling for training
        self._dataloaders = create_data_loaders(
            db_manager=self.db_manager,
            train_ids=self.train_ids,
            val_ids=self.val_ids,
            test_ids=self.test_ids,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_transform=self.train_transform,
            val_transform=self.val_transform,
            load_volumes=True,
            balanced_sampling=True  # Enable balanced sampling for 50/50 distribution
        )
    
    def train_dataloader(self):
        """Return training data loader."""
        return self._dataloaders['train']
    
    def val_dataloader(self):
        """Return validation data loader."""
        return self._dataloaders['val']
    
    def test_dataloader(self):
        """Return test data loader if available."""
        return self._dataloaders.get('test', None)
    
    def predict_dataloader(self):
        """Return prediction data loader (not implemented)."""
        return None