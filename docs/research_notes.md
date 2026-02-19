# Research Notes: Primate Face Detection and Identification

*Last updated: 2026-02-19*

---

## Summary of Findings

### 1. SAM 3 Assessment: Honest Evaluation

**SAM 3 (Segment Anything with Concepts)** is Meta's latest segmentation model (Nov 2025) that adds text/concept-based prompting to the SAM family. For Eleane's macaque identification project:

#### What SAM 3 CAN do:
- Segment "faces" or "monkey faces" from images using text prompts (zero-shot)
- Provide pixel-precise face masks (better than bounding boxes for clean crops)
- Track faces across video frames (inherited from SAM 2)

#### What SAM 3 CANNOT do:
- **Distinguish between individual macaques** — this is the core limitation
- Filter macaque faces from human faces automatically (needs separate classifier)
- Run in real-time on edge devices (too computationally expensive)
- Replace the identification pipeline (ArcFace/embedding model still needed)

#### Bottom line:
**SAM 3 is a useful detection/segmentation component but NOT a replacement for the identification pipeline.** It's best used as a preprocessing step that feeds into Eleane's existing ArcFace-based identification system.

### 2. Is SAM 3 the Right Tool?

**For a PhD thesis, SAM 3 exploration is valuable** because:
- Applying SAM 3 to primate face segmentation is **novel** (no published work yet)
- The comparison between SAM 3 zero-shot vs fine-tuned YOLO is publishable
- It demonstrates awareness of cutting-edge foundation models

**But for practical macaque identification, simpler approaches work better:**
- **YOLOv8 + ArcFace** is faster, more accurate, and field-deployable
- This has been validated by Paulet et al. (2023) on Japanese macaques
- Eleane's existing ResNet50 + ArcFace pipeline is architecturally sound

### 3. Comparison: Existing Pipeline vs SAM 3 Approach

| Aspect | Eleane's Pipeline (ResNet50 + ArcFace) | SAM 3 Approach |
|--------|---------------------------------------|----------------|
| **Detection** | YOLO (trained on specific data) | SAM 3 (zero-shot, text prompt) |
| **Identification** | ArcFace embeddings | Still needs ArcFace or similar |
| **Training data needed** | Yes (bbox + identity labels) | Less (for detection part) |
| **Speed** | Fast (real-time possible) | Slow (batch processing only) |
| **Accuracy** | High (with sufficient data) | Lower for detection, same for ID |
| **Novelty** | Established approach | Novel application |
| **Field deployment** | Laptop or Jetson | Desktop GPU required |

---

## Recommended Next Steps for Eleane

### Immediate (Week 1-2)
1. **Test existing pipeline on macaque images**: Take her chimp-trained model and evaluate on macaque faces. This establishes a baseline and reveals the "domain gap" between species.
2. **Try DINOv2 zero-shot**: Extract features from macaque face crops using DINOv2, cluster them, and see if individuals naturally separate. No training needed.
3. **Set up YOLO face detector**: Use Roboflow to annotate ~200 macaque face bounding boxes, train YOLOv8n. This takes 1-2 days.

### Short-Term (Month 1-2)
4. **Fine-tune ArcFace on macaques**: Transfer from MS1MV2 (human) → C-Faces (chimp) → macaque data. This progressive fine-tuning should give best results.
5. **Evaluate SAM 3**: If publicly available, run SAM 3 with text prompt "monkey face" on field images. Compare detection rate with fine-tuned YOLO.
6. **Design evaluation protocols**: Follow MFID/PFID methodology — test under classification, verification, closed-set, and open-set protocols.

### Medium-Term (Month 3-6)
7. **Build comprehensive macaque dataset**: Aim for 50+ individuals, 20+ images each, from field conditions.
8. **Systematic comparison paper**: Compare YOLOv8 vs SAM 3 vs RetinaFace for detection; ArcFace vs DINOv2 vs CLIP for identification; all combinations.
9. **Open-set performance**: This is the most important metric for field deployment. Focus on recall at low false-positive rates.

### For the Thesis
10. **Novel contributions to claim**:
    - First application of SAM 3 to primate face segmentation
    - Systematic comparison of detection + identification pipelines for macaques
    - Field-validated macaque identification system
    - Open-set evaluation on wild macaque data

---

## Key Literature

| Paper | Relevance |
|-------|-----------|
| Shukla et al. 2018 (MFID) | Directly relevant — macaque face ID with evaluation protocols |
| Shukla et al. 2019 (PFID) | State-of-the-art primate face ID in the wild |
| Paulet et al. 2023 | Japanese macaque detection + recognition pipeline |
| Carion et al. 2025 (SAM 3) | Foundation model for concept-based segmentation |
| Sapkota et al. 2025 | Analyzes SAM 2→3 gap, important for realistic expectations |
| Wasmuht et al. 2025 (SA-FARI) | SAM applied to wildlife, directly relevant methodology |
| Čermák et al. 2023 (WildlifeDatasets) | Toolkit for dataset management |

---

## Technical Architecture Recommendation

```
┌─────────────────────────────────────────────────────┐
│                  INPUT: Field Photo/Video            │
└──────────────────────┬──────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │   Face Detection    │
            │   (YOLOv8-macaque)  │
            │   or SAM 3 (text)   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Species Filter     │
            │  (CLIP/ResNet)      │
            │  macaque vs human   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Face Alignment     │
            │  + Normalization    │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Embedding Model    │
            │  (ArcFace/ResNet50) │
            │  Fine-tuned on      │
            │  macaque faces      │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Gallery Matching   │
            │  Cosine similarity  │
            │  + threshold        │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  OUTPUT:            │
            │  Known ID or        │
            │  "Unknown/New"      │
            └─────────────────────┘
```

---

## Related Documents
- `docs/literature_review.md` — Full survey of academic papers
- `docs/sam3_analysis.md` — Deep analysis of SAM 3 capabilities
- `docs/alternative_approaches.md` — Comparison of technical approaches
- `docs/datasets.md` — Available datasets and data collection guide
- `docs/field_deployment.md` — Practical field deployment considerations
