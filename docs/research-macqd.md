# MacqD — Research Report

> **Repo:** https://github.com/C-Poirier-Lab/MacqD  
> **Paper:** "The MacqD deep-learning-based model for automatic detection of socially housed laboratory macaques" — [Nature Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-95180-x)  
> **License:** Not specified (academic)  
> **Date:** 2026-02-25

---

## 1. Summary

MacqD is a **Mask R-CNN + SWIN Transformer** model specifically designed to detect macaques in laboratory home-cage environments. Published in Nature Scientific Reports, it addresses the challenging scenario of socially housed macaques with:

- Occlusions (overlapping macaque bodies)
- Glass reflections
- Light overexposure
- Complex cage environments (shelves, swings, ropes)

Key claim: MacqD outperforms existing methods (including SIPEC) in generalization — it works on unseen individuals and even unseen facilities.

---

## 2. Architecture & Approach

### Model Architecture
- **Base:** Mask R-CNN (instance segmentation — bounding boxes + pixel-level masks)
- **Backbone:** SWIN Transformer (Tiny variant)
  - `embed_dims=96`
  - `depths=[2, 2, 6, 2]`
  - `num_heads=[3, 6, 12, 24]`
  - `window_size=7`
  - Pretrained from `swin_tiny_patch4_window7_224.pth` (ImageNet)
- **Neck:** Feature Pyramid Network (FPN), channels `[96, 192, 384, 768] → 256`
- **Heads:**
  - RPN (Region Proposal Network) with anchor-based proposals
  - Bbox head: `Shared2FCBBoxHead`, 1024 FC, **1 class** (monkey)
  - Mask head: `FCNMaskHead`, 4 conv layers, **1 class**
- **Framework:** MMDetection

### Training Configuration
- **Optimizer:** AdamW, lr=0.0001, weight_decay=0.05
- **Schedule:** Step LR (steps at epoch 8, 11), 200 epochs total, linear warmup (1000 iters)
- **Input:** 1333×800 resolution, ImageNet normalization
- **Batch:** 2 samples/GPU
- **Data format:** COCO-style annotations (bbox + mask + segmentation polygons)
- **Augmentation:** Random horizontal flip (0.5)

### Training Strategies (3 model variants)
1. **MacqD-Single:** Trained only on frames with single macaques
2. **MacqD-Curriculum:** Curriculum learning — first single, then paired macaques
3. **MacqD-Combine:** Trained on mixed single + paired data simultaneously

### Data
- **Source:** Newcastle University macaque facility, 2014–2020
- **Subjects:** 20 focal macaques
- **Camera:** Wall-mounted (Cube HD Y-cam 1080p, Axis M1065-L 1080p)
- **Environment:** 2.1m × 3m × 2.4m cages, pair-housed, enriched environments
- **Annotations:** COCO format with bbox + instance segmentation masks

---

## 3. Available Models & Weights

| Model | Description | Download |
|-------|-------------|----------|
| MacqD-Single | Single macaque detector | [Dropbox](https://www.dropbox.com/scl/fo/8pcy7ey26ocynd4isxyhd/ALRQm1qGBRh-TMW0ouxo5GY?rlkey=lxwqoso39tcpsdz91n41cjbqy&st=qq8u8vc9&dl=0) |
| MacqD-Curriculum | Curriculum-trained (single→paired) | Same Dropbox link |
| MacqD-Combine | Combined single+paired training | Same Dropbox link |
| SWIN-Tiny backbone | ImageNet pretrained | `swin_tiny_patch4_window7_224.pth` (standard) |

**Note:** Weights are hosted on Dropbox, not HuggingFace. All three variants in the same folder.

---

## 4. Relevance to PrimateReID

### 4.1 Could MacqD Replace YOLO in Our Detection Stage?

**Assessment: Yes, with caveats.**

**Advantages over YOLO:**
- ✅ **Instance segmentation masks** — not just bounding boxes. Masks enable cleaner body/face crops
- ✅ **Specifically trained on macaques** in cage environments (exactly E小姐's scenario)
- ✅ **Handles occlusions** — the primary challenge in social housing
- ✅ **Validated generalization** to unseen individuals and facilities
- ✅ SWIN Transformer backbone is generally stronger than YOLO's CNN backbone for complex scenes

**Disadvantages vs YOLO:**
- ❌ **Much slower inference** — Mask R-CNN is a two-stage detector vs YOLO's single-stage
- ❌ **Detects whole body only** (1 class: "monkey") — no face-specific detection
- ❌ **MMDetection dependency** — heavier than ultralytics
- ❌ **Older framework** — Python 3.7, CUDA 11.3, MMDetection v2 era
- ❌ **No tracking built-in** (paper tested adding tracking separately)

### 4.2 Integration Scenario

MacqD fits best as **Stage 1** in a two-stage detection pipeline:

```
Raw Frame → MacqD (whole body detection + mask) → Crop body region
         → PrimateFace face detector (within body crop) → Face crop
         → DINOv2 embedding → Re-ID
```

Benefits:
- MacqD's mask isolates individual macaques even when overlapping
- Reduces search space for face detection
- Segmentation mask could improve re-ID (body shape as auxiliary feature)

### 4.3 Transfer Learning Feasibility (E小姐's approach)

E小姐 wants: "use front layers of primate model, fine-tune back layers on macaque data"

**For detection (MacqD):**
- MacqD is already macaque-specific — minimal fine-tuning needed
- Could fine-tune on E小姐's specific cage environment (different lighting, camera angle)
- Freeze SWIN backbone → fine-tune FPN + heads on ~200-500 annotated frames
- **Feasibility: HIGH** — standard transfer learning, well-documented in MMDetection

**For re-ID (SWIN features):**
- MacqD's SWIN backbone learns macaque-relevant features
- Could extract intermediate SWIN features as an alternative to DINOv2
- However, SWIN-Tiny (96→768 dim) trained on detection may not be optimal for re-ID
- DINOv2 (self-supervised, 768-dim) likely still better for embedding similarity
- **Feasibility: MEDIUM** — possible but DINOv2 is probably the better backbone for re-ID

---

## 5. Dependencies & GPU Requirements

### Environment (from repo)
- **OS:** Ubuntu 22.04 (documented; macOS/Windows untested)
- **GPU:** NVIDIA GTX 1080 Ti (11GB VRAM) — the model runs on mid-range hardware
- **CUDA:** 11.3
- **Python:** 3.7.13 (⚠️ quite old)
- **Framework:** MMDetection (v2 era, based on Config.py style)
- **Package manager:** Miniconda

### Compatibility Concerns
- Python 3.7 is EOL — may need updates for modern environments
- MMDetection v2 → v3 migration may be needed
- CUDA 11.3 is old but SWIN models are widely supported
- The config is standard MMDet format — relatively easy to port

### GPU Requirements
- Training: GTX 1080 Ti level (11GB), batch_size=2
- Inference: Should run on 4-6GB GPUs
- Much more demanding than YOLO (Mask R-CNN is heavier)

---

## 6. Strengths & Limitations

### Strengths
- ✅ Purpose-built for laboratory macaque detection
- ✅ Instance segmentation (masks, not just boxes)
- ✅ Handles occlusions, reflections, overexposure
- ✅ Published in Nature Scientific Reports (peer-reviewed)
- ✅ Three model variants for different scenarios
- ✅ Validated cross-facility generalization
- ✅ COCO format — standard and interoperable

### Limitations
- ⚠️ Whole-body detection only (no face detection)
- ⚠️ Single class ("monkey") — no species differentiation
- ⚠️ Slow compared to YOLO (two-stage detector)
- ⚠️ Outdated dependencies (Python 3.7, old MMDetection)
- ⚠️ Weights on Dropbox (not HuggingFace — less discoverable, may go down)
- ⚠️ Small repo — no community, minimal documentation beyond the paper
- ⚠️ Training data not publicly released (only model weights)
- ⚠️ No license specified in repo

---

## 7. Recommended Integration Plan

### Phase 1: Evaluate
1. Download MacqD-Combine weights from Dropbox
2. Set up MMDetection environment (consider upgrading to MMDet v3)
3. Run inference on E小姐's raw cage footage
4. Compare detection quality vs our current YOLO detector
5. Measure: detection rate, occlusion handling, inference speed

### Phase 2: Integrate (if Phase 1 is positive)
1. Use MacqD as body detector → crop individual macaques
2. Feed crops to PrimateFace face detector (or our existing face detector)
3. Pipeline: MacqD body detection → face crop → DINOv2 → re-ID
4. The segmentation mask can also be used to:
   - Remove background from body crops
   - Provide body-shape features as auxiliary re-ID signal

### Phase 3: Fine-tune on E小姐's Data
1. Annotate ~200-500 frames from E小姐's specific setup (COCO format)
2. Fine-tune MacqD-Combine with frozen SWIN backbone
3. Should achieve very high detection accuracy on the specific environment

### Alternative: Skip MacqD
If inference speed is critical, YOLO may still be preferred. MacqD's main advantage is **occlusion handling** — if E小姐's data has frequent overlapping macaques, MacqD is worth the speed tradeoff. If macaques are usually well-separated, YOLO is simpler and faster.

---

## 8. References

- "The MacqD deep-learning-based model for automatic detection of socially housed laboratory macaques." *Nature Scientific Reports* (2025). https://www.nature.com/articles/s41598-025-95180-x
- Liu, Z. et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV*. https://arxiv.org/abs/2103.14030
- He, K. et al. (2017). "Mask R-CNN." *ICCV*. https://arxiv.org/abs/1703.06870
- GitHub: https://github.com/C-Poirier-Lab/MacqD
- MMDetection: https://github.com/open-mmlab/mmdetection
