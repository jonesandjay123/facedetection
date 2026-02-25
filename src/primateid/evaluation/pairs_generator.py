"""Generate genuine/impostor pairs from a crops directory structure."""

from __future__ import annotations

import csv
import logging
import random
from itertools import combinations
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def generate_pairs(crops_dir: Path, output_path: Path | None = None, seed: int = 42) -> list[dict]:
    """Generate balanced genuine/impostor pairs from folder structure.

    Each sub-folder under *crops_dir* represents one individual.
    Genuine pairs = all intra-folder combinations.
    Impostor pairs = random inter-folder sampling (same count as genuine).

    Returns list of dicts with keys: img1, img2, label, split, source.
    """
    rng = random.Random(seed)
    individuals: dict[str, list[Path]] = {}
    for d in sorted(crops_dir.iterdir()):
        if d.is_dir():
            imgs = _list_images(d)
            if imgs:
                individuals[d.name] = imgs

    if len(individuals) < 2:
        raise ValueError(f"Need ≥2 individuals, found {len(individuals)} in {crops_dir}")

    # Genuine pairs
    genuine: list[dict] = []
    for name, imgs in individuals.items():
        for a, b in combinations(imgs, 2):
            genuine.append({
                "img1": str(a), "img2": str(b),
                "label": 1, "split": "test", "source": name,
            })

    # Impostor pairs — sample to match genuine count
    names = list(individuals.keys())
    impostor: list[dict] = []
    attempts = 0
    max_attempts = len(genuine) * 20
    while len(impostor) < len(genuine) and attempts < max_attempts:
        n1, n2 = rng.sample(names, 2)
        a = rng.choice(individuals[n1])
        b = rng.choice(individuals[n2])
        impostor.append({
            "img1": str(a), "img2": str(b),
            "label": 0, "split": "test", "source": f"{n1}/{n2}",
        })
        attempts += 1

    pairs = genuine + impostor
    rng.shuffle(pairs)
    logger.info("Generated %d genuine + %d impostor pairs", len(genuine), len(impostor))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["img1", "img2", "label", "split", "source"])
            writer.writeheader()
            writer.writerows(pairs)
        logger.info("Pairs written to %s", output_path)

    return pairs


def load_pairs(csv_path: Path) -> list[dict]:
    """Load pairs from an existing CSV."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        pairs = list(reader)
    for p in pairs:
        p["label"] = int(p["label"])
    return pairs
