"""
Hybrid Alzheimer's Model

This module implements the main hybrid deep learning model that combines
3D brain imaging data (MRI + PET) with clinical features for Alzheimer's detection.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import numpy as np

from .swin_unet3d import SwinUNet3DBackbone
from .simple_backbone import Simple3DBackbone
from .clinical_processor import ClinicalFeatureProcessor
from .feature_fusion import FeatureFusion


class HybridAlzheimersModel(pl.LightningModule):
    """
    Hybrid Alzheimer's Model combining 3D imaging and clinical features
    
    This PyTorch Lightning module integrates:
    - SwinUNet3D backbone for 3D image processing (MRI + PET)
    - Clinical feature processor for 116 clinical features
    - Feature fusion module for combining modalities and final prediction
    """
    
    def __init__(
        self,
        # Model architecture parameters
    target_size: Tuple[int, int, int] = (96, 112, 96),
        swin_config: Optional[Dict] = None,
        clinical_dims: List[int] = [116, 32, 16, 8],  # Smaller network to save RAM
        fusion_dims: List[int] = [72, 32, 16, 1],  # Adjusted for 64+8=72 input
        
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        
        # Loss and metrics
        loss_type: str = "bce",
        
        # Regularization
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        
        # Other parameters
        fusion_strategy: str = "concatenate",
        use_simple_backbone: bool = False
    ):
        """
        Initialize Hybrid Alzheimer's Model
        
        Args:
            target_size: Target size for input images (H, W, D)
            swin_config: Configuration for SwinUNet3D backbone
            clinical_dims: Dimensions for clinical feature processor
            fusion_dims: Dimensions for feature fusion module
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
            scheduler_type: Type of learning rate scheduler ("cosine", "step", "plateau")
            loss_type: Type of loss function ("bce", "mse", "mae", "huber")
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            fusion_strategy: Strategy for feature fusion
            use_simple_backbone: Whether to use simple 3D CNN instead of SwinUNet3D
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Store configuration
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.loss_type = loss_type
        
        # Default SwinUNet3D configuration - optimized for lower RAM
        if swin_config is None:
            swin_config = {
                "in_channels": 2,
                "patch_size": 4,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "dropout_path_rate": 0.1,
                "feature_size": 96
            }
        
        # Initialize model components
        if use_simple_backbone:
            self.image_backbone = Simple3DBackbone(
                in_channels=swin_config["in_channels"],
                feature_size=swin_config["feature_size"],
                base_channels=32
            )
        else:
            self.image_backbone = SwinUNet3DBackbone(**swin_config)
        
        # Store the actual feature size from the backbone
        self.actual_image_features = swin_config["feature_size"]
        
        self.clinical_processor = ClinicalFeatureProcessor(
            input_dim=clinical_dims[0],
            hidden_dims=clinical_dims[1:],
            dropout_rate=dropout_rate,
            use_batch_norm=False  # Disable batch norm to allow batch size 1
        )
        
        self.feature_fusion = FeatureFusion(
            image_dim=self.actual_image_features,
            clinical_dim=clinical_dims[-1],
            fusion_dims=fusion_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=False,  # Disable batch norm to allow batch size 1
            fusion_strategy=fusion_strategy
        )
        
        # Initialize loss function
        self.loss_fn = self._get_loss_function(loss_type)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
    
    def _get_loss_function(self, loss_type: str):
        """Get loss function based on type"""
        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()  # Safe for autocast, includes sigmoid
        elif loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "mae":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, images: torch.Tensor, clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model
        
        Args:
            images: Input images tensor of shape (B, 2, H, W, D)
            clinical_features: Clinical features tensor of shape (B, 116)
            
        Returns:
            predictions: Alzheimer score predictions of shape (B, 1)
        """
        # Process images through SwinUNet3D backbone
        image_features = self.image_backbone(images)  # (B, 96)
        
        # Process clinical features
        processed_clinical = self.clinical_processor(clinical_features)  # (B, 16)
        
        # Fuse features and make prediction
        predictions = self.feature_fusion(image_features, processed_clinical)  # (B, 1)
        
        return predictions
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        # Extract data from batch
        images = batch["volumes"]  # (B, 2, H, W, D)
        clinical_features = batch["clinical_features"]  # (B, 116)
        targets = batch["alzheimer_score"].unsqueeze(-1)  # (B, 1)
        
        # Forward pass
        predictions = self(images, clinical_features)
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Calculate metrics
        predicted_probs = torch.sigmoid(predictions)  # Convert logits to probabilities
        mae = torch.mean(torch.abs(predicted_probs - targets))
        
        # Binary classification metrics
        predicted_binary = (predicted_probs > 0.5).float()
        accuracy = torch.mean((predicted_binary == targets).float())
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for epoch-level tracking
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        # Extract data from batch
        images = batch["volumes"]  # (B, 2, H, W, D)
        clinical_features = batch["clinical_features"]  # (B, 116)
        targets = batch["alzheimer_score"].unsqueeze(-1)  # (B, 1)
        
        # Forward pass
        predictions = self(images, clinical_features)
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Calculate metrics
        predicted_probs = torch.sigmoid(predictions)  # Convert logits to probabilities
        mae = torch.mean(torch.abs(predicted_probs - targets))
        
        # Binary classification metrics
        predicted_binary = (predicted_probs > 0.5).float()
        accuracy = torch.mean((predicted_binary == targets).float())
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for epoch-level tracking
        self.val_losses.append(loss.item())
        self.val_maes.append(mae.item())
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        # Extract data from batch
        images = batch["volumes"]  # (B, 2, H, W, D)
        clinical_features = batch["clinical_features"]  # (B, 116)
        targets = batch["alzheimer_score"].unsqueeze(-1)  # (B, 1)
        
        # Forward pass
        predictions = self(images, clinical_features)
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Calculate metrics
        predicted_probs = torch.sigmoid(predictions)  # Convert logits to probabilities
        mae = torch.mean(torch.abs(predicted_probs - targets))
        
        # Binary classification metrics
        predicted_binary = (predicted_probs > 0.5).float()
        accuracy = torch.mean((predicted_binary == targets).float())
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step"""
        # Extract data from batch
        images = batch["volumes"]  # (B, 2, H, W, D)
        clinical_features = batch["clinical_features"]  # (B, 116)
        
        # Forward pass (returns logits)
        logits = self(images, clinical_features)
        
        # Convert to probabilities for prediction
        predictions = torch.sigmoid(logits)
        
        return predictions
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Choose optimizer
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Choose scheduler
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if self.train_losses:
            avg_train_loss = float(np.mean(self.train_losses))
            print(f"\n[Epoch {self.current_epoch}] Train Loss: {avg_train_loss:.4f}")
            self.log("epoch_train_loss", avg_train_loss, prog_bar=True)
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if self.val_losses:
            avg_val_loss = float(np.mean(self.val_losses))
            avg_val_mae = float(np.mean(self.val_maes))
            print(f"[Epoch {self.current_epoch}] Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")
            self.log("epoch_val_loss", avg_val_loss, prog_bar=True)
            self.log("epoch_val_mae", avg_val_mae, prog_bar=True)
            self.val_losses.clear()
            self.val_maes.clear()
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """
        Save model with configuration and metadata
        
        Args:
            filepath: Path to save the model
            include_metadata: Whether to include training metadata
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save dictionary
        save_dict = {
            "model_state_dict": self.state_dict(),
            "hyperparameters": dict(self.hparams),
            "model_config": {
                "target_size": self.target_size,
                "architecture": "HybridAlzheimersModel",
                "version": "1.0"
            }
        }
        
        # Add metadata if requested
        if include_metadata:
            save_dict["metadata"] = {
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "model_components": {
                    "image_backbone": "SwinUNet3DBackbone",
                    "clinical_processor": "ClinicalFeatureProcessor", 
                    "feature_fusion": "FeatureFusion"
                }
            }
        
        # Save model
        torch.save(save_dict, filepath)
        
        # Save configuration as JSON for easy inspection
        config_path = filepath.with_suffix('.json')
        config_dict = {
            "hyperparameters": dict(self.hparams),
            "model_config": save_dict["model_config"]
        }
        if include_metadata:
            config_dict["metadata"] = save_dict["metadata"]
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Model saved to: {filepath}")
        print(f"Configuration saved to: {config_path}")
    
    @classmethod
    def load_model(cls, filepath: str, map_location: Optional[str] = None) -> 'HybridAlzheimersModel':
        """
        Load model from saved checkpoint
        
        Args:
            filepath: Path to the saved model
            map_location: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Extract hyperparameters
        hyperparameters = checkpoint.get("hyperparameters", {})
        
        # Create model instance
        model = cls(**hyperparameters)
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"Model loaded from: {filepath}")
        
        return model
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Component-wise parameter counts
        backbone_params = sum(p.numel() for p in self.image_backbone.parameters())
        clinical_params = sum(p.numel() for p in self.clinical_processor.parameters())
        fusion_params = sum(p.numel() for p in self.feature_fusion.parameters())
        
        return {
            "model_name": "HybridAlzheimersModel",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "component_parameters": {
                "image_backbone": backbone_params,
                "clinical_processor": clinical_params,
                "feature_fusion": fusion_params
            },
            "hyperparameters": dict(self.hparams),
            "target_size": self.target_size
        }