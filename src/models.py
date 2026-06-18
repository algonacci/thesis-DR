import torch
import torch.nn as nn
from torchvision import models

from src.config import (
    NUM_CLASSES,
    STAGE1_UNFREEZE_VIT_BLOCKS,
    STAGE2_UNFREEZE_VIT_BLOCKS,
)

RESNET_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1
VIT_WEIGHTS = models.ViT_B_32_Weights.IMAGENET1K_V1


class ResViTFusionNet(nn.Module):
    """Stage 1: ResNet50 + ViT-B/32 feature fusion."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.resnet50 = models.resnet50(weights=RESNET_WEIGHTS)
        resnet_dim = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()

        self.vit = models.vit_b_32(weights=VIT_WEIGHTS)
        vit_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        self.fc_combined = nn.Linear(resnet_dim + vit_dim, num_classes)

    def forward(self, x):
        r_feat = self.resnet50(x)
        v_feat = self.vit(x)
        fused = torch.cat((r_feat, v_feat), dim=1)
        return self.fc_combined(fused)


class ViTClassifier(nn.Module):
    """Stage 2: ViT-B/32 standalone classifier for masked input."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.vit = models.vit_b_32(weights=VIT_WEIGHTS)
        in_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.vit(x)


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_stage1(model: ResViTFusionNet):
    for p in model.resnet50.layer4.parameters():
        p.requires_grad = True
    for p in model.vit.encoder.layers[-STAGE1_UNFREEZE_VIT_BLOCKS:].parameters():
        p.requires_grad = True
    for p in model.fc_combined.parameters():
        p.requires_grad = True


def unfreeze_stage2(model: ViTClassifier):
    for p in model.vit.encoder.layers[-STAGE2_UNFREEZE_VIT_BLOCKS:].parameters():
        p.requires_grad = True
    for p in model.vit.heads.parameters():
        p.requires_grad = True


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
