#!/usr/bin/env python3
"""
Cross-Dataset Evaluation for Deepfake Detection Models
Tests how well models trained on one dataset perform on others
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import json
import argparse
from datetime import datetime
import sys
import timm
import random
from PIL import Image


class MemoryOptimizedPatchEmbedding(nn.Module):
    """Memory-optimized patch embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        return x


class SimplifiedMAEDecoder(nn.Module):
    """Simplified MAE decoder to reduce memory usage"""
    def __init__(self, embed_dim=768, decoder_embed_dim=256, decoder_depth=2,
                 decoder_num_heads=8, patch_size=16, img_size=224):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        # Simplified decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Simplified positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, decoder_embed_dim))

        # Reduced transformer layers
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 2,  # Reduced from 4x
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ) for _ in range(decoder_depth)  # Reduced layers
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, mask, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        B, N_visible, D = x.shape

        # Add mask tokens efficiently
        N = self.n_patches
        mask_tokens = self.mask_token.expand(B, N - N_visible, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle more efficiently
        x_full = torch.gather(x_full, dim=1,
                             index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        # Add pos embed
        x_full = x_full + self.pos_embed

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)

        return pred


class MemoryEfficientMaskingStrategy:
    """Memory-efficient masking strategy"""
    def __init__(self, mask_ratio=0.75):
        self.mask_ratio = mask_ratio

    def __call__(self, x, patch_embed):
        B, C, H, W = x.shape

        # Get patches more efficiently
        with torch.no_grad():
            x_patches = patch_embed(x)  # [B, N, embed_dim]

        N = x_patches.shape[1]
        len_keep = int(N * (1 - self.mask_ratio))

        # More memory-efficient random selection
        noise = torch.rand(B, N, device=x.device, dtype=torch.float16)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep visible patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x_patches, dim=1,
                               index=ids_keep.unsqueeze(-1).expand(-1, -1, x_patches.shape[2]))

        # Generate binary mask
        mask = torch.ones([B, N], device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask.float(), ids_restore


# ============================================================================
# PART 2: Simplified Regularization (Memory-Friendly)
# ============================================================================

class LightweightMixUp:
    """Lightweight MixUp implementation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0 and random.random() < 0.3:  # 30% chance
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)

            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
        return x, y, None, 1.0

    def compute_loss(self, criterion, pred, y_a, y_b, lam):
        if y_b is not None:
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        return criterion(pred, y_a)


class CutMix:
    """CutMix augmentation for better regularization"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0 and random.random() < 0.3:  # 30% chance
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)

            # Get random box
            _, _, H, W = x.shape
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # Uniform sampling
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            # Apply CutMix
            x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

            # Adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            y_a, y_b = y, y[index]
            return x, y_a, y_b, lam
        return x, y, None, 1.0

    def compute_loss(self, criterion, pred, y_a, y_b, lam):
        if y_b is not None:
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        return criterion(pred, y_a)


class MemoryEfficientDropPath(nn.Module):
    """Memory-efficient DropPath"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FocalLoss(nn.Module):
    """Focal Loss to focus on hard examples and reduce false negatives"""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight  # Weight for positive class (fake)

    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Get probabilities
        probs = torch.sigmoid(inputs)

        # Focal term: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing with positive class weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply positive class weight to prioritize fake detection
        if self.pos_weight != 1.0:
            weight = targets * self.pos_weight + (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss * weight
        else:
            focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


# ============================================================================
# PART 3: Memory-Optimized Components
# ============================================================================

class OptimizedFrequencyExtractor(nn.Module):
    """Memory-optimized frequency extractor"""
    def __init__(self, out_channels=64):  # Reduced from 128
        super().__init__()

        self.freq_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def compute_frequency_features(self, x):
        # More memory-efficient frequency computation
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for FFT
            x_float = x.float()
            fft = torch.fft.fft2(x_float, dim=(-2, -1))
            fft_shift = torch.fft.fftshift(fft)

            magnitude = torch.abs(fft_shift)
            magnitude = torch.log(magnitude + 1e-8)
            phase = torch.angle(fft_shift)

            # Normalize to prevent memory explosion
            magnitude = F.layer_norm(magnitude, magnitude.shape[2:])
            phase = F.layer_norm(phase, phase.shape[2:])

        freq_features = torch.cat([magnitude, phase], dim=1)
        return freq_features

    def forward(self, x):
        freq = self.compute_frequency_features(x)
        freq_feat = self.freq_conv(freq)
        return freq_feat.flatten(1)


class OptimizedSelfSupervisedBackbone(nn.Module):
    """Memory-optimized SSL backbone"""
    def __init__(self, model_type='mae', frozen=False):
        super().__init__()
        self.model_type = model_type

        if model_type == 'mae':
            try:
                self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
                if hasattr(self.backbone, 'head'):
                    self.backbone.head = nn.Identity()
                self.feature_dim = 768
                self.is_vit = True
            except Exception as e:
                print(f"MAE not available ({e}), using EfficientNet")
                self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True)
                self.backbone.classifier = nn.Identity()
                self.feature_dim = 1280
                self.is_vit = False

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone.forward_features(x) if self.is_vit else self.backbone(x)


class OptimizedMultiScaleExtractor(nn.Module):
    """Memory-optimized multi-scale extractor"""
    def __init__(self):
        super().__init__()

        efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=True, features_only=True)
        self.backbone = efficientnet
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Get reduced feature dimensions
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.backbone(test_input)
            # Use only last 3 feature maps to reduce memory
            self.scale_dims = [feat.shape[1] for feat in test_features[-3:]]

        total_dim = sum(self.scale_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),  # Reduced from 512
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        features = self.backbone(x)

        # Use only last 3 feature maps
        features = features[-3:]

        pooled_features = []
        for feat in features:
            pooled = self.global_pool(feat).flatten(1)
            pooled_features.append(pooled)

        combined = torch.cat(pooled_features, dim=1)
        return self.fusion(combined)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for better multi-modal integration"""
    def __init__(self, feature_dims, hidden_dim=128, num_heads=4):
        super().__init__()

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project each modality to same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for dim in feature_dims
        ])

        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, features_list):
        # Process and project each feature
        projected = []
        for feat, proj in zip(features_list, self.projections):
            if feat.dim() > 2:
                feat = feat.flatten(1)
            projected.append(proj(feat).unsqueeze(1))  # [B, 1, hidden_dim]

        # Stack into sequence
        stacked = torch.cat(projected, dim=1)  # [B, num_modalities, hidden_dim]

        # Apply cross-attention (each modality attends to others)
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Flatten and fuse
        attended_flat = attended.flatten(1)  # [B, num_modalities * hidden_dim]
        output = self.fusion(attended_flat)

        return output


class OptimizedAttentionFusion(nn.Module):
    """Memory-optimized attention fusion (original version for backward compatibility)"""
    def __init__(self, feature_dims):
        super().__init__()

        self.feature_dims = feature_dims
        total_dim = sum(feature_dims)

        # Simplified attention
        self.attention = nn.Sequential(
            nn.Linear(total_dim, len(feature_dims)),
            nn.Softmax(dim=1)
        )

        self.projection = nn.Sequential(
            nn.Linear(total_dim, 128),  # Reduced from 256
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, features_list):
        processed_features = []
        for feat in features_list:
            if feat.dim() > 2:
                feat = feat.flatten(1)
            processed_features.append(feat)

        concat_features = torch.cat(processed_features, dim=1)
        attn_weights = self.attention(concat_features)

        # Apply attention weights
        weighted_features = []
        start_idx = 0
        for i, dim in enumerate(self.feature_dims):
            feat = concat_features[:, start_idx:start_idx+dim]
            weight = attn_weights[:, i:i+1]
            weighted_features.append(feat * weight)
            start_idx += dim

        weighted_concat = torch.cat(weighted_features, dim=1)
        return self.projection(weighted_concat)


# ============================================================================
# PART 4: Memory-Optimized Main Model
# ============================================================================

class MemoryOptimizedMIMHybridDetector(nn.Module):
    """Memory-optimized hybrid detector with working MAE"""
    def __init__(
        self,
        ssl_model_type='mae',
        freeze_backbone=True,
        use_frequency=True,
        use_multiscale=True,
        use_mim=True,
        mask_ratio=0.75,
        mim_weight=0.1
    ):
        super().__init__()

        self.use_frequency = use_frequency
        self.use_multiscale = use_multiscale
        self.use_mim = use_mim
        self.mask_ratio = mask_ratio
        self.mim_weight = mim_weight

        # Optimized MAE components
        if use_mim:
            self.patch_embed = MemoryOptimizedPatchEmbedding()
            self.masking_strategy = MemoryEfficientMaskingStrategy(mask_ratio)
            self.mae_decoder = SimplifiedMAEDecoder()

        # Optimized backbones
        self.ssl_backbone = OptimizedSelfSupervisedBackbone(ssl_model_type, freeze_backbone)
        ssl_dim = self.ssl_backbone.feature_dim

        if use_frequency:
            self.freq_extractor = OptimizedFrequencyExtractor()
            freq_dim = 64
        else:
            freq_dim = 0

        if use_multiscale:
            self.multiscale_extractor = OptimizedMultiScaleExtractor()
            multiscale_dim = 256
        else:
            multiscale_dim = 0

        # Fusion
        self.fusion = None
        self._initialized = False

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # Enhanced regularization
        self.mixup = LightweightMixUp(alpha=0.2)
        self.cutmix = CutMix(alpha=1.0)
        self.use_cross_attention = True  # Flag for using cross-attention

    def patchify(self, imgs):
        """Memory-efficient patchify"""
        p = self.patch_embed.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x

    def compute_mim_loss(self, imgs, pred, mask):
        """Memory-efficient MIM loss"""
        try:
            with torch.cuda.amp.autocast(enabled=False):
                target = self.patchify(imgs.float())

                # Normalize patches
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

                # Compute loss only on masked patches
                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)

                # Weight by mask
                mask_sum = mask.sum()
                if mask_sum > 0:
                    loss = (loss * mask).sum() / mask_sum
                else:
                    loss = loss.mean()

            return loss
        except Exception as e:
            print(f"MIM loss error: {e}")
            return torch.tensor(0.0, device=imgs.device)

    def forward(self, x, y=None, training=True):
        """Memory-optimized forward pass"""
        features_list = []
        mim_loss = torch.tensor(0.0, device=x.device)

        # Clear cache periodically
        if training and random.random() < 0.1:
            torch.cuda.empty_cache()

        # Apply enhanced augmentation (MixUp or CutMix)
        y_a, y_b, lam = y, None, 1.0
        if training and y is not None:
            # Randomly choose between MixUp and CutMix
            if random.random() < 0.5:
                x, y_a, y_b, lam = self.mixup(x, y)
            else:
                x, y_a, y_b, lam = self.cutmix(x, y)

        # MAE reconstruction (simplified)
        if training and self.use_mim:
            try:
                # Get patches efficiently
                x_visible, mask, ids_restore = self.masking_strategy(x, self.patch_embed)

                # Get full patches for reconstruction target
                with torch.no_grad():
                    full_patches = self.patch_embed(x)

                # Reconstruction
                pred = self.mae_decoder(full_patches, mask, ids_restore)
                mim_loss = self.compute_mim_loss(x, pred, mask)

                # Clear intermediate tensors
                del x_visible, pred, full_patches

            except Exception as e:
                print(f"MAE error: {e}")
                mim_loss = torch.tensor(0.0, device=x.device)

        # Get features
        ssl_features = self.ssl_backbone(x)
        features_list.append(ssl_features)

        if self.use_frequency:
            freq_features = self.freq_extractor(x)
            features_list.append(freq_features)

        if self.use_multiscale:
            multiscale_features = self.multiscale_extractor(x)
            features_list.append(multiscale_features)

        # Initialize fusion layer
        if not self._initialized:
            actual_dims = []
            for feat in features_list:
                if feat.dim() > 2:
                    actual_dims.append(feat.flatten(1).shape[1])
                else:
                    actual_dims.append(feat.shape[1])

            # Use cross-attention fusion for better multi-modal integration
            if self.use_cross_attention:
                self.fusion = CrossAttentionFusion(actual_dims, hidden_dim=128, num_heads=4).to(x.device)
            else:
                self.fusion = OptimizedAttentionFusion(actual_dims).to(x.device)
            self._initialized = True

        # Fusion and classification
        fused_features = self.fusion(features_list)
        logits = self.classifier(fused_features)

        if training and self.use_mim:
            return logits, mim_loss, y_a, y_b, lam
        else:
            return logits


# ============================================================================
# PART 5: Memory-Optimized Training
# ============================================================================

def train_epoch_memory_optimized(model, dataloader, criterion, optimizer, device, use_mim=True):
    """Memory-optimized training epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_mim_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Training')):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()

        try:
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                if use_mim and model.use_mim:
                    logits, mim_loss, y_a, y_b, lam = model(images, labels, training=True)

                    # Compute classification loss
                    if y_b is not None:
                        cls_loss = model.mixup.compute_loss(criterion, logits.squeeze(), y_a, y_b, lam)
                    else:
                        cls_loss = criterion(logits.squeeze(), labels)

                    # Combined loss
                    total_loss_batch = cls_loss + model.mim_weight * mim_loss
                    total_mim_loss += mim_loss.item()
                else:
                    logits = model(images, training=True)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    cls_loss = criterion(logits.squeeze(), labels)
                    total_loss_batch = cls_loss

            if torch.isnan(total_loss_batch):
                print("NaN loss detected, skipping batch")
                continue

            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()

            # Predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits.squeeze())
                preds = (probs > 0.5).float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except RuntimeError as e:
            print(f"Error in training step: {e}")
            torch.cuda.empty_cache()
            continue

    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.5, 0.0

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_mim_loss = total_mim_loss / len(dataloader) if total_mim_loss > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, avg_cls_loss, avg_mim_loss, accuracy, auc


def get_memory_optimized_transforms(image_size=224, training=True):
    """Memory-optimized transforms"""
    if training:
        return transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Simple Dataset class for evaluation
class SimpleDeepfakeDataset(Dataset):
    """Simple dataset for cross-evaluation"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)


def load_trained_model(checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint"""

    checkpoint_file = Path(checkpoint_path) / 'best_precision_optimized_model.pth'

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    print(f"\nLoading model from: {checkpoint_file}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    # Get model config from checkpoint
    config = {
        'use_frequency': checkpoint.get('use_frequency', True),
        'use_multiscale': checkpoint.get('use_multiscale', True),
        'use_mim': checkpoint.get('use_mim', True),
        'mask_ratio': checkpoint.get('mask_ratio', 0.75),
    }

    print(f"Model configuration:")
    print(f"  - Frequency: {config['use_frequency']}")
    print(f"  - Multi-scale: {config['use_multiscale']}")
    print(f"  - MIM: {config['use_mim']}")

    # Initialize model
    model = MemoryOptimizedMIMHybridDetector(
        use_frequency=config['use_frequency'],
        use_multiscale=config['use_multiscale'],
        use_mim=config['use_mim'],
        mask_ratio=config['mask_ratio']
    ).to(device)

    # Load weights
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if missing:
        print(f"⚠ Missing {len(missing)} keys in model:")
        for k in list(missing)[:5]:
            print(f"   - {k}")
    if unexpected:
        print(f"⚠ Unexpected {len(unexpected)} keys in checkpoint:")
        for k in list(unexpected)[:5]:
            print(f"   + {k}")

    model.eval()

    print(f"✓ Model loaded successfully")

    return model, config


def load_test_dataset(dataset_root, batch_size=16):
    """Load test dataset"""

    dataset_path = Path(dataset_root)

    print(f"\nLoading test dataset from: {dataset_path}")

    # Collect validation images
    val_real_dir = dataset_path / 'original'  # For FF++ datasets
    val_fake_dir = dataset_path / 'manipulated'

    # Check if it's Celeb-DF structure
    if not val_real_dir.exists():
        val_real_dir = dataset_path / 'val' / 'real'
        val_fake_dir = dataset_path / 'val' / 'fake'

    # Check if it's FFHQ+SDFF structure
    if not val_real_dir.exists():
        # For FFHQ+SDFF, we need to create splits
        real_images = list((dataset_path / 'ffhq_cropped_faces').glob('*.jpg'))[:2249]
        fake_images = list((dataset_path / 'ffhq_cropped_faces' / 'SDFF' / 'sdff_cropped').glob('*.png'))

        # Use last 20% for validation
        split_idx_real = int(len(real_images) * 0.8)
        split_idx_fake = int(len(fake_images) * 0.8)

        val_paths = real_images[split_idx_real:] + fake_images[split_idx_fake:]
        val_labels = [0] * len(real_images[split_idx_real:]) + [1] * len(fake_images[split_idx_fake:])

    else:
        # Standard structure
        real_images = list(val_real_dir.rglob('*.png')) + list(val_real_dir.rglob('*.jpg'))
        fake_images = list(val_fake_dir.rglob('*.png')) + list(val_fake_dir.rglob('*.jpg'))

        val_paths = real_images + fake_images
        val_labels = [0] * len(real_images) + [1] * len(fake_images)

    print(f"✓ Found {len([l for l in val_labels if l == 0])} REAL images")
    print(f"✓ Found {len([l for l in val_labels if l == 1])} FAKE images")
    print(f"✓ Total: {len(val_paths)} images")

    # Create dataset
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = SimpleDeepfakeDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    return val_loader, len([l for l in val_labels if l == 0]), len([l for l in val_labels if l == 1])


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test dataset"""

    print(f"\nRunning evaluation...")

    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            # Forward pass
            outputs = model(images, training=False)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert to numpy
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Compute metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Metrics at optimal threshold
    optimal_preds = (all_probs >= optimal_threshold).astype(int)
    optimal_precision = precision_score(all_labels, optimal_preds, zero_division=0)
    optimal_recall = recall_score(all_labels, optimal_preds, zero_division=0)
    optimal_f1 = f1_score(all_labels, optimal_preds, zero_division=0)
    optimal_accuracy = accuracy_score(all_labels, optimal_preds)
    optimal_cm = confusion_matrix(all_labels, optimal_preds)

    results = {
        'default_threshold': {
            'threshold': 0.5,
            'auc': float(auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        },
        'optimal_threshold': {
            'threshold': float(optimal_threshold),
            'auc': float(auc),  # AUC doesn't change with threshold
            'accuracy': float(optimal_accuracy),
            'precision': float(optimal_precision),
            'recall': float(optimal_recall),
            'f1': float(optimal_f1),
            'confusion_matrix': optimal_cm.tolist()
        }
    }

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\nROC AUC: {auc*100:.2f}%")
    print(f"\nDefault Threshold (0.5):")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"\nOptimal Threshold ({optimal_threshold:.3f}):")
    print(f"  Accuracy:  {optimal_accuracy*100:.2f}%")
    print(f"  Precision: {optimal_precision*100:.2f}%")
    print(f"  Recall:    {optimal_recall*100:.2f}%")
    print(f"  F1-Score:  {optimal_f1*100:.2f}%")
    print(f"\nConfusion Matrix (Optimal):")
    print(f"  TN: {optimal_cm[0,0]}, FP: {optimal_cm[0,1]}")
    print(f"  FN: {optimal_cm[1,0]}, TP: {optimal_cm[1,1]}")
    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-Dataset Evaluation')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--test_dataset', type=str, required=True,
                       help='Path to test dataset root directory')
    parser.add_argument('--output', type=str, default='cross_eval_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("="*70)
    print("CROSS-DATASET EVALUATION")
    print("="*70)
    print(f"Model: {args.model_checkpoint}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load model
    model, model_config = load_trained_model(args.model_checkpoint, args.device)

    # Load test dataset
    test_loader, num_real, num_fake = load_test_dataset(args.test_dataset, args.batch_size)

    # Evaluate
    results = evaluate_model(model, test_loader, args.device)

    # Add metadata
    results['metadata'] = {
        'model_checkpoint': args.model_checkpoint,
        'test_dataset': args.test_dataset,
        'test_samples': {
            'real': num_real,
            'fake': num_fake,
            'total': num_real + num_fake
        },
        'model_config': model_config,
        'evaluation_time': datetime.now().isoformat()
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
