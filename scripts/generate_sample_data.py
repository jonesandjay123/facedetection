#!/usr/bin/env python3
"""Generate small sample crop images for testing the PrimateReID pipeline."""

from pathlib import Path
import numpy as np
from PIL import Image


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "data" / "sample_crops"
    rng = np.random.RandomState(42)

    individuals = {"person_A": 4, "person_B": 3, "person_C": 3}
    for name, count in individuals.items():
        folder = base / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            arr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(folder / f"{i:03d}.jpg")
            print(f"  Created {folder / f'{i:03d}.jpg'}")

    print(f"\nSample data generated at {base}")


if __name__ == "__main__":
    main()
