# PrimateReID

End-to-end pipeline for primate face detection, cropping, and individual re-identification.

> **[繁體中文版 README](README.zh-TW.md)**

## Architecture

```
Raw Photo → Detection (YOLO/SAM3) → Crop (box/mask) → Embedding (FaceNet/ArcFace/primate) → ReID Evaluation
```

**PrimateReID** handles the full pipeline from raw field photos to individual identification, with built-in evaluation metrics (AUC, EER, decidability) and visualisation.

## Quick Start

```bash
git clone https://github.com/jonesandjay123/PrimateReID.git
cd PrimateReID
pip install -r requirements.txt
```

### Option A: Run with included demo data (real chimpanzee faces)

The repo includes `data/demo_chimp_crops/` — 10 chimpanzees × 30 cropped face photos (300 images, ~22MB) from the CTai/CZoo dataset, ready to test out of the box:

```bash
# ResNet50 backbone
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone resnet50

# FaceNet backbone
PYTHONPATH=src python3 -m primateid.run --crops data/demo_chimp_crops --backbone facenet
```

### Option B: Run with synthetic sample data

```bash
# Generate random test images (no download needed)
python3 scripts/generate_sample_data.py

PYTHONPATH=src python3 -m primateid.run --crops data/sample_crops --backbone resnet50
```

### Option C: Run with your own data

Organise your cropped images into `data/your_dataset/<individual_name>/` folders (see [Data Format](#data-format) below), then:

```bash
PYTHONPATH=src python3 -m primateid.run --crops data/your_dataset --backbone resnet50
```

### CLI Options

```
--crops PATH      Path to crops directory (required)
--backbone STR    Embedding backbone: resnet50 | facenet (default: resnet50)
--output PATH     Output directory (default: results/<backbone>_<timestamp>/)
--device STR      Torch device (default: cpu)
```

### Output Structure

```
results/resnet50_20260225_173000/
├── config.json              # Run parameters
├── pairs.csv                # Pairs used for evaluation
├── embeddings.npz           # All embeddings (for re-running eval without recomputing)
├── scores.csv               # img1, img2, label, similarity
├── summary.json             # AUC, EER, d', threshold
├── figures/
│   ├── roc_curve.png        # ROC curve with EER point
│   └── score_distribution.png  # Genuine vs impostor score histograms
└── report.md                # Human-readable summary report
```

### Data Format

Organise crops into sub-folders by individual identity:

```
data/crops/
├── monkey_A/
│   ├── 001.jpg
│   └── 002.jpg
├── monkey_B/
│   └── 001.jpg
```

Pairs are automatically generated from the folder structure (genuine = same folder, impostor = different folders). To use custom pairs, place a `pairs.csv` in the crops directory.

## Pipeline Components

### Detection
Front-end face/body detection using YOLO or SAM3. Locates primate subjects in raw photographs.

### Cropping
Extracts individual primate regions via bounding-box crop or mask-based crop, preparing clean inputs for embedding.

### Embedding
Generates identity-discriminative feature vectors using multiple backbones:
- **ResNet50** — ImageNet-pretrained, 2048-d embeddings
- **FaceNet** — VGGFace2-pretrained InceptionResNetV1, 512-d embeddings

All embeddings are L2-normalised so cosine similarity = dot product.

### Evaluation
Built-in evaluation engine computing:
- **AUC** — Area Under the ROC Curve
- **EER** — Equal Error Rate
- **d' (decidability)** — separation between genuine and impostor distributions
- **Best threshold** — optimal operating point (Youden's J)

## Project Structure

```
PrimateReID/
├── src/primateid/        # Core pipeline modules
│   ├── detection/        # YOLO, SAM3 detection front-ends
│   ├── cropping/         # Box crop, mask crop
│   ├── embedding/        # Multi-backbone embedder
│   ├── evaluation/       # Pairs generation + metrics + plotting
│   └── utils/
├── scripts/              # Utility scripts
├── configs/              # Experiment configuration (YAML)
├── data/                 # Test data
├── results/              # Experiment outputs
└── tests/
```

## Demo Data

The repo ships with two test datasets:

| Dataset | Path | Description |
|---------|------|-------------|
| **Demo chimpanzees** | `data/demo_chimp_crops/` | 10 individuals × 30 real face crops from CTai/CZoo (~22MB) |
| **Synthetic samples** | `data/sample_crops/` | Generated via `scripts/generate_sample_data.py` (random noise, for CI/smoke tests) |

The demo chimpanzee data is provided for quick evaluation — clone the repo and run immediately, no extra downloads needed.

## Status

**v0.1** — Embedding pipeline + evaluation + CLI operational. Detection and cropping modules in progress.

### Baseline Results (2026-02-25)

| Backbone | AUC | EER | d' |
|----------|-----|-----|-----|
| ResNet50 (ImageNet) | 0.688 | 36.3% | 0.67 |
| FaceNet (VGGFace2) | 0.614 | 42.2% | 0.41 |

Both general-purpose models perform poorly on primate individual identification — validating the need for domain-specific embeddings. See [docs/baseline-results.md](docs/baseline-results.md) for full analysis, backbone candidates (ArcFace, DINOv2, CLIP, SphereFace), and next steps.

## Related Repos

- [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) — Evaluation engine for face embedding analysis
- [FacialRecognitionTest](https://github.com/jonesandjay123/FacialRecognitionTest) — Earlier facial recognition experiments

## Contributors

- **Jones** — Project lead, pipeline architecture
- **Eleane (趙以琳)** — SAM3 detection research, field testing

## License

MIT
