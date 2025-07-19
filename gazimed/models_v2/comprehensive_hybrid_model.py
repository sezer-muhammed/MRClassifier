"""
Comprehensive Hybrid Model for MR+PET+Clinical Alzheimer's Detection

- Dual-branch 3D ViT/UNETR (or 3D CNN fallback) for MR and PET
- Cross-attention fusion between MR and PET features
- Clinical feature MLP
- Final fusion and prediction head
- Compatible with existing dataloader and training script
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional, Any

# Try to import MONAI ViT/UNETR, fallback to 3D CNN if not available
try:
    from monai.networks.nets import UNETR
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


class ClinicalFeatureMLP(nn.Module):
    def __init__(self, input_dim=116, hidden_dims=[64, 32, 16], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    def forward(self, x):
        return self.mlp(x)

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(128, feature_dim)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, mr_feat, pet_feat):
        # mr_feat, pet_feat: (B, F)
        # Add sequence dim: (B, 1, F)
        mr_feat = mr_feat.unsqueeze(1)
        pet_feat = pet_feat.unsqueeze(1)
        # Cross attention: query=mr, key/value=pet
        attn_out, _ = self.attn(mr_feat, pet_feat, pet_feat)
        return attn_out.squeeze(1)

class ComprehensiveHybridModel(pl.LightningModule):
    def __init__(
        self,
        target_size: Tuple[int, int, int] = (96, 109, 96),
        backbone_type: str = "unetr",  # or "cnn"
        feature_dim: int = 96,
        clinical_hidden: List[int] = [64, 32, 16],
        fusion_hidden: List[int] = [128, 64, 32, 1],
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2
    ):
        super().__init__()
        self.save_hyperparameters()
        # MR and PET backbones
        if backbone_type == "unetr" and MONAI_AVAILABLE:
            self.mr_backbone = UNETR(
                in_channels=1, out_channels=feature_dim, img_size=target_size, feature_size=feature_dim
            )
            self.pet_backbone = UNETR(
                in_channels=1, out_channels=feature_dim, img_size=target_size, feature_size=feature_dim
            )
            self.pool = nn.AdaptiveAvgPool3d(1)
        else:
            self.mr_backbone = Simple3DCNN(in_channels=1, feature_dim=feature_dim)
            self.pet_backbone = Simple3DCNN(in_channels=1, feature_dim=feature_dim)
        # Clinical MLP
        self.clinical_mlp = ClinicalFeatureMLP(input_dim=116, hidden_dims=clinical_hidden, dropout=dropout)
        # Cross-attention fusion
        self.cross_attn = CrossAttentionFusion(embed_dim=feature_dim, num_heads=4, dropout=dropout)
        # Final fusion MLP
        fusion_in = feature_dim + self.clinical_mlp.output_dim
        layers = []
        prev_dim = fusion_in
        for h in fusion_hidden:
            layers.append(nn.Linear(prev_dim, h))
            if h != 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.fusion_mlp = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, images, clinical_features):
        # images: (B, 2, H, W, D)
        mr = images[:, 0:1]  # (B, 1, H, W, D)
        pet = images[:, 1:2]
        if hasattr(self.mr_backbone, 'forward'):  # UNETR or CNN
            mr_feat = self.mr_backbone(mr)
            pet_feat = self.pet_backbone(pet)
            if mr_feat.dim() == 5:
                mr_feat = self.pool(mr_feat).flatten(1)
                pet_feat = self.pool(pet_feat).flatten(1)
        else:
            mr_feat = self.mr_backbone(mr)
            pet_feat = self.pet_backbone(pet)
        # Cross-attention fusion
        fused_img_feat = self.cross_attn(mr_feat, pet_feat)
        # Clinical features
        clin_feat = self.clinical_mlp(clinical_features)
        # Final fusion
        final_feat = torch.cat([fused_img_feat, clin_feat], dim=1)
        out = self.fusion_mlp(final_feat)
        return out
    def training_step(self, batch, batch_idx):
        images = batch['volumes']
        clinical = batch['clinical_features']
        targets = batch['alzheimer_score'].unsqueeze(-1)
        logits = self(images, clinical)
        loss = self.loss_fn(logits, targets)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        images = batch['volumes']
        clinical = batch['clinical_features']
        targets = batch['alzheimer_score'].unsqueeze(-1)
        logits = self(images, clinical)
        loss = self.loss_fn(logits, targets)
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
