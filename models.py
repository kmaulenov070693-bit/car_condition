# models.py
import torch
import torch.nn as nn
import torchvision.models as tvm

class MultiHeadCarNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, dropout=0.2):
        super().__init__()

        if backbone == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = m.fc.in_features
            modules = list(m.children())[:-1]  # без final fc
            self.backbone = nn.Sequential(*modules)
        elif backbone == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            feat_dim = m.classifier[-1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
        else:
            raise ValueError("unknown backbone")

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
        )
        # две отдельные головы (по 1 логиту каждая)
        self.dirty_head = nn.Linear(128, 1)
        self.damaged_head = nn.Linear(128, 1)

    def forward(self, x):
        feats = self.backbone(x)
        h = self.head(feats)
        dirty_logits = self.dirty_head(h).squeeze(1)     # [B]
        damaged_logits = self.damaged_head(h).squeeze(1) # [B]
        # выход: [B,2]
        return torch.stack([dirty_logits, damaged_logits], dim=1)

def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    """
    logits: [B,2], targets: [B,2], mask: [B,2]
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, targets)
    # зануляем отсутствующие метки
    loss = loss * mask
    # нормализуем на число присутствующих меток (чтобы батчи с пропусками не завышали/занижали loss)
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom
