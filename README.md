# PrimateReID

End-to-end pipeline for primate face detection, cropping, and individual re-identification.

> **[繁體中文版 README](README.zh-TW.md)**

## Architecture

```
Raw Photo → Detection (YOLO/SAM3) → Crop (box/mask) → Embedding (FaceNet/ArcFace/primate) → ReID Evaluation
```

**PrimateReID** handles the full pipeline from raw field photos to individual identification. For threshold analysis and evaluation metrics, it integrates with [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) as the evaluation engine.

## Pipeline Components

### Detection
Front-end face/body detection using YOLO or SAM3. Locates primate subjects in raw photographs.

### Cropping
Extracts individual primate regions via bounding-box crop or mask-based crop, preparing clean inputs for embedding.

### Embedding
Generates identity-discriminative feature vectors using multiple backbones — FaceNet, ArcFace, or primate-specific models.

### Evaluation
Feeds embeddings into [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) for distance-based analysis, threshold tuning, and re-identification accuracy reporting.

## Status

**Early Development** — Phase 1 detection comparison (SAM3 vs YOLO) completed. Pipeline integration in progress.

### Phase 1: SAM3 Detection Results

Eleane conducted SAM3 zero-shot detection experiments across three scenarios:

| Scenario | Condition | Finding |
|----------|-----------|---------|
| S1 | Single monkey, clean background | Good segmentation quality |
| S2 | Multiple monkeys, moderate occlusion | Acceptable detection, some missed faces |
| S3 | Field conditions (mixed species, clutter) | Prompt engineering alone insufficient for reliable detection |

These results informed the decision to adopt a multi-stage pipeline with dedicated detectors. Full exploration preserved in [`archive/sam3-exploration/`](archive/sam3-exploration/).

## Project Structure

```
PrimateReID/
├── src/primateid/        # Core pipeline modules
│   ├── detection/        # YOLO, SAM3 detection front-ends
│   ├── cropping/         # Box crop, mask crop
│   ├── embedding/        # FaceNet, ArcFace, primate models
│   ├── evaluation/       # FaceThresholdLab integration
│   └── utils/
├── configs/              # Experiment configuration (YAML)
├── data/                 # Test data (gitignored)
├── results/              # Experiment outputs
├── archive/              # Phase 1 SAM3 exploration (preserved)
└── tests/
```

## Quick Start

```bash
git clone https://github.com/jonesandjay123/PrimateReID.git
cd PrimateReID
pip install -r requirements.txt
# Pipeline usage — coming soon
```

## Related Repos

- [FaceThresholdLab](https://github.com/jonesandjay123/FaceThresholdLab) — Evaluation engine for face embedding analysis
- [FacialRecognitionTest](https://github.com/jonesandjay123/FacialRecognitionTest) — Earlier facial recognition experiments

## Contributors

- **Jones** — Project lead, pipeline architecture
- **Eleane (趙以琳)** — SAM3 detection research, field testing

## License

MIT
