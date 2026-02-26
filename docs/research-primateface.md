# PrimateFace — Research Report

> **Repo:** https://github.com/KordingLab/PrimateFace  
> **Paper:** Parodi et al. (2025) — [bioRxiv 2025.08.12.669927](https://www.biorxiv.org/content/10.1101/2025.08.12.669927)  
> **License:** MIT  
> **Date:** 2026-02-25

---

## 1. Summary

PrimateFace is a comprehensive, cross-species platform for primate facial analysis from the Kording Lab (UPenn). It provides:

- **260,000+ images** spanning 60+ genera of primates (tarsiers → gorillas → humans)
- A genus-balanced subset of 60,000 images with bounding boxes and facial landmarks
- Face detection, landmark estimation, individual recognition, gaze analysis
- DINOv2-based feature extraction pipeline with UMAP visualization
- Pseudo-labeling GUI for custom datasets
- Google Colab tutorials including **macaque face recognition** (App2)

The key insight: models trained cross-species on PrimateFace data achieve performance **comparable to species-specific baselines**, demonstrating that primate facial features generalize across the order.

---

## 2. Architecture & Approach

PrimateFace is an **ecosystem** (not a single model) with multiple modules:

### Detection & Landmarks
- **Face Detection:** MMDetection-based, achieving 0.34 mAP cross-species (vs 0.39 mAP human-only baseline)
- **Landmark Estimation:** MMPose-based, 48-keypoint facial landmarks, 0.061 normalized error cross-species
- **Landmark Converter:** 68→48 keypoint converter (including GNN-based option)

### DINOv2 Feature Extraction (most relevant to us)
- Uses **vanilla DINOv2** (`facebook/dinov2-base` by default) — NOT a fine-tuned primate-specific DINOv2
- Provides a `DINOv2Extractor` class for embedding extraction
- Supports all DINOv2 variants:
  - `dinov2-small` (21M params, 384-dim)
  - `dinov2-base` (86M params, 768-dim) ← recommended
  - `dinov2-large` (300M params, 1024-dim)
  - `dinov2-giant` (1.1B params, 1536-dim)
- UMAP visualization, K-means clustering, active learning subset selection
- Attention heatmap visualization

### Individual Recognition
- App2 notebook: "Rapid Macaque Face Recognition" using DINOv2 embeddings
- Likely uses face crops → DINOv2 → cosine similarity / clustering

### Other Applications
- Gaze following (GazeLLE integration)
- Facial action unit extraction (lip-smacking, subtle movements)
- Cross-subject neural decoding (upcoming)

---

## 3. Available Models & Weights

| Component | Where | Format |
|-----------|-------|--------|
| Face detection model | Likely in repo or HuggingFace | MMDetection checkpoint |
| Landmark estimation model | Likely in repo or HuggingFace | MMPose checkpoint |
| DINOv2 | `facebook/dinov2-base` via HuggingFace | PyTorch (standard DINOv2) |
| HuggingFace Space | [fparodi/PrimateFace](https://huggingface.co/spaces/fparodi/PrimateFace) | Gradio demo |
| Dataset | [fparodi/PrimateFace](https://huggingface.co/datasets/fparodi/PrimateFace) | HuggingFace Datasets |

**Important:** The DINOv2 module uses the **standard Facebook DINOv2**, not a primate-fine-tuned version. The value is in:
1. The curated primate face dataset for fine-tuning
2. The extraction/visualization pipeline code
3. The face detection + cropping → DINOv2 pipeline

---

## 4. Relevance to PrimateReID

### 4.1 DINOv2 Comparison

Our baseline result: vanilla DINOv2 → AUC 0.725 for macaque re-identification.

PrimateFace uses the **same vanilla DINOv2** — they did NOT fine-tune it on primate data. Their contribution is:
- Better **face detection** (crop quality matters for downstream embeddings)
- Cross-species **landmark alignment** (could normalize face crops)
- The **dataset** of 260K primate images for potential fine-tuning

### 4.2 Integration Opportunities

**High Value:**
1. **Face detection model** → Replace our YOLO face detector with their cross-species primate face detector for better crops
2. **Landmark-aligned face crops** → Normalize face orientation before DINOv2 embedding, likely improving AUC
3. **260K primate image dataset** → Fine-tune DINOv2 on primate faces (self-supervised or contrastive), which should beat vanilla DINOv2's 0.725 AUC
4. **App2 Macaque Recognition notebook** → Direct reference implementation for our pipeline

**Medium Value:**
5. **Active learning / subset selection** → Use their DiverseImageSelector for efficient labeling of E小姐's data
6. **Attention visualization** → Interpretability for which facial regions drive re-ID

### 4.3 Transfer Learning Feasibility (E小姐's approach)

E小姐 wants: "use front layers of primate model, fine-tune back layers on macaque data"

**Assessment: Highly feasible with PrimateFace's approach**

Strategy:
1. Use PrimateFace's face detector to crop macaque faces from E小姐's data
2. Use DINOv2-base as backbone (frozen early layers)
3. Fine-tune last 2-4 transformer blocks + add a re-ID head on macaque face crops
4. PrimateFace's 260K cross-species dataset could serve as **pretraining data** before macaque-specific fine-tuning

This is essentially what PrimateFace already validates — cross-species pretraining transfers to specific species.

---

## 5. Dependencies & GPU Requirements

- **Python:** 3.8+
- **PyTorch:** 2.1.0 (CUDA 11.8 or 12.1)
- **Framework deps:** MMDetection, MMPose (optional), DeepLabCut/SLEAP (optional)
- **DINOv2 module:** lightweight — just PyTorch + transformers + umap-learn + plotly
- **GPU:** Required for most modules; DINOv2-base runs on any modern GPU (4GB+ VRAM)
- **Install:** conda env + `uv pip install -e ".[dinov2]"`

---

## 6. Strengths & Limitations

### Strengths
- ✅ Cross-species generalization validated across 60+ genera
- ✅ Complete pipeline: detection → landmarks → embeddings → recognition
- ✅ MIT license, well-documented, Colab tutorials
- ✅ Macaque recognition already demonstrated (App2)
- ✅ Large curated dataset available
- ✅ Active development (Kording Lab, UPenn)

### Limitations
- ⚠️ DINOv2 is vanilla (not primate-fine-tuned) — we'd need to do this ourselves
- ⚠️ Re-ID is demonstrated but not the primary focus (face analysis / action units is)
- ⚠️ Performance metrics are for detection/landmarks, not re-ID specifically
- ⚠️ Some modules (vocal-motor coupling, neural decoding) still "coming soon"
- ⚠️ Requires MMDetection/MMPose ecosystem (complex dependency chain)

---

## 7. Recommended Integration Plan

### Phase 1: Quick Win
1. Clone PrimateFace, install `[dinov2]` module only
2. Use their `DINOv2Extractor` on our existing macaque face crops
3. Compare with our current vanilla DINOv2 pipeline (should be equivalent)
4. Try their face detection model on E小姐's raw frames → compare crop quality with YOLO

### Phase 2: Improved Pipeline
1. Replace our face detector with PrimateFace's cross-species detector
2. Add landmark-based face alignment before DINOv2 extraction
3. Re-evaluate AUC — expect improvement from better crops alone

### Phase 3: Fine-tuned DINOv2
1. Download PrimateFace's 260K dataset
2. Fine-tune DINOv2 on primate faces (self-supervised or contrastive)
3. Further fine-tune on E小姐's macaque data (freeze front layers, train back layers)
4. This should significantly exceed the 0.725 AUC baseline

---

## 8. References

- Parodi, F. et al. (2025). "PrimateFace: A Machine Learning Resource for Automated Face Analysis in Human and Non-human Primates." *bioRxiv*. https://doi.org/10.1101/2025.08.12.669927
- Oquab, M. et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR*. https://arxiv.org/abs/2304.07193
- GitHub: https://github.com/KordingLab/PrimateFace
- HuggingFace Space: https://huggingface.co/spaces/fparodi/PrimateFace
- Documentation: https://docs.primateface.studio
