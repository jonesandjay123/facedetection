"""Pluggable multi-backbone embedder for primate re-identification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BackboneName = Literal["resnet50", "facenet"]


class MultiEmbedder:
    """Unified embedding interface supporting multiple backbones.

    Args:
        backbone: Model backbone name ("resnet50" or "facenet").
        device: Torch device string.
    """

    def __init__(self, backbone: str = "resnet50", device: str = "cpu") -> None:
        self.backbone_name = backbone
        self.device = torch.device(device)
        self.model: torch.nn.Module
        self.transform: transforms.Compose

        if backbone == "resnet50":
            self._init_resnet50()
        elif backbone == "facenet":
            self._init_facenet()
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}. Use 'resnet50' or 'facenet'.")

        self.model.eval()
        self.model.to(self.device)
        logger.info("Loaded backbone=%s on device=%s", backbone, device)

    # ------------------------------------------------------------------
    # Backbone initialisers
    # ------------------------------------------------------------------

    def _init_resnet50(self) -> None:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        # Remove final FC → output is avgpool 2048-d
        model.fc = torch.nn.Identity()
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _init_facenet(self) -> None:
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is required for the 'facenet' backbone. "
                "Install it with: pip install facenet-pytorch"
            )
        self.model = InceptionResnetV1(pretrained="vggface2")
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            # facenet_pytorch expects [0,1] float tensors; no extra norm needed
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, image_path: Path) -> np.ndarray:
        """Return L2-normalised embedding vector for a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor).squeeze(0).cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Return (N, dim) L2-normalised embeddings for a list of images."""
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            vecs = self.model(batch).cpu().numpy()
        # L2 normalise each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms
