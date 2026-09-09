"""Architectures du détecteur CNN.

Choix : backbones convolutifs modernes, légers, exportables ONNX sans opérateur
exotique, disponibles pré-entraînés ImageNet via ``timm`` :

* ``efficientnet_b0``  — 5,3 M paramètres, ~0,4 GFLOPs à 224 : cible par défaut
  (rapidité CPU, empreinte < 25 Mo en ONNX fp32).
* ``efficientnetv2_s`` — plus précis, ~21 M paramètres.
* ``convnext_tiny``    — 28 M paramètres, très bonne généralisation inter-générateurs
  dans la littérature (GenImage, Community Forensics), coût CPU ×3.
* ``resnet50``         — référence historique (CNNDetection, GenImage baseline).
* ``tiny``             — petit CNN sans dépendance timm : tests CPU et placeholder.

La tête est un unique logit (BCEWithLogits) : sortie ONNX ``[N, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ArchPreset:
    timm_name: str | None
    input_size: int
    description: str


ARCH_PRESETS: dict[str, ArchPreset] = {
    "efficientnet_b0": ArchPreset("efficientnet_b0.ra_in1k", 224, "Défaut : rapide sur CPU, ~20 Mo ONNX."),
    "efficientnetv2_s": ArchPreset("tf_efficientnetv2_s.in21k_ft_in1k", 288, "Plus précis, ~80 Mo ONNX."),
    "convnext_tiny": ArchPreset("convnext_tiny.fb_in22k_ft_in1k", 224, "Meilleure généralisation, ~110 Mo ONNX."),
    "resnet50": ArchPreset("resnet50.a1_in1k", 224, "Référence historique."),
    "tiny": ArchPreset(None, 224, "Petit CNN de test / placeholder de signature (pas de pré-entraînement)."),
}


class TinyCNN(nn.Module):
    """CNN minimal (≈ 60 k paramètres) : validation du pipeline et placeholder ONNX."""

    def __init__(self, width: int = 16):
        super().__init__()
        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.features = nn.Sequential(block(3, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, width * 4))
        self.num_features = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(2, 3))


class ForensicClassifier(nn.Module):
    def __init__(self, arch: str, pretrained: bool = True, drop_rate: float = 0.2, input_size: int | None = None):
        super().__init__()
        if arch not in ARCH_PRESETS:
            raise ValueError(f"Architecture inconnue '{arch}'. Choix : {sorted(ARCH_PRESETS)}")
        self.arch = arch
        preset = ARCH_PRESETS[arch]
        # Taille de crop d'entraînement : les backbones sont entièrement convolutifs
        # (global pooling), la taille est donc libre mais doit être conservée à l'inférence.
        self.input_size = int(input_size or preset.input_size)
        if preset.timm_name is None:
            self.backbone = TinyCNN()
            num_features = self.backbone.num_features
        else:
            import timm

            self.backbone = timm.create_model(preset.timm_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
            num_features = self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(self.dropout(feats))

    def freeze_backbone(self, frozen: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not frozen


def build_model(arch: str, pretrained: bool = True, drop_rate: float = 0.2, input_size: int | None = None) -> ForensicClassifier:
    return ForensicClassifier(arch, pretrained=pretrained, drop_rate=drop_rate, input_size=input_size)


def save_checkpoint(model: ForensicClassifier, path: str | Path, **extra: Any) -> None:
    payload = {
        "arch": model.arch,
        "input_size": model.input_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "state_dict": model.state_dict(),
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[ForensicClassifier, dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False, drop_rate=0.0, input_size=ckpt.get("input_size"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
