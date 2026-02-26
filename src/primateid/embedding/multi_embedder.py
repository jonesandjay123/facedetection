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

BackboneName = Literal["resnet50", "facenet", "arcface", "dinov2"]


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
        elif backbone == "arcface":
            self._init_arcface()
        elif backbone == "dinov2":
            self._init_dinov2()
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone!r}. "
                "Use 'resnet50', 'facenet', 'arcface', or 'dinov2'."
            )

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

    def _init_arcface(self) -> None:
        """ArcFace via insightface — uses ONNX model (buffalo_l package).

        Falls back to a ResNet100 with ArcFace-style training if insightface
        is not installed.
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required for the 'arcface' backbone. "
                "Install it with: pip install insightface onnxruntime"
            )
        # Use insightface's FaceAnalysis to get the recognition model.
        # We'll extract embeddings directly from the rec model.
        import onnxruntime  # noqa: F401 — ensure available

        self._arcface_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._arcface_app.prepare(ctx_id=-1, det_size=(160, 160))

        # We use a lightweight wrapper as self.model so .eval()/.to() don't fail
        self.model = torch.nn.Identity()
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

    def _init_dinov2(self) -> None:
        """DINOv2 ViT-S/14 from Meta — self-supervised, 384-d embeddings."""
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, image_path: Path) -> np.ndarray:
        """Return L2-normalised embedding vector for a single image."""
        if self.backbone_name == "arcface":
            return self._embed_arcface(image_path)

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor).squeeze(0).cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    def embed_batch(self, image_paths: list[Path]) -> np.ndarray:
        """Return (N, dim) L2-normalised embeddings for a list of images."""
        if self.backbone_name == "arcface":
            return np.stack([self._embed_arcface(p) for p in image_paths])

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

    # ------------------------------------------------------------------
    # ArcFace helpers
    # ------------------------------------------------------------------

    def _embed_arcface(self, image_path: Path) -> np.ndarray:
        """Get ArcFace embedding using insightface's recognition model.

        Since insightface's FaceAnalysis expects to detect a face first,
        and our input is already cropped faces, we feed directly into the
        recognition model (rec_model) bypassing detection.
        """
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Try face detection first
        faces = self._arcface_app.get(img)
        if faces:
            # Use the first detected face's embedding
            vec = faces[0].embedding
        else:
            # No face detected — feed the whole crop directly to rec model
            # Resize to 112x112 as expected by ArcFace
            img_resized = cv2.resize(img, (112, 112))
            # insightface rec model expects (1, 3, 112, 112) float32 BGR
            img_input = cv2.dnn.blobFromImage(
                img_resized, 1.0 / 127.5, (112, 112),
                (127.5, 127.5, 127.5), swapRB=True
            )
            rec_model = self._arcface_app.models["recognition"]
            vec = rec_model.forward(img_input).flatten()

        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec
