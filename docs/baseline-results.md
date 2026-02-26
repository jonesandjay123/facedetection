# Baseline Results & Backbone Research

> Date: 2026-02-25 | Dataset: demo_chimp_crops (10 chimpanzees × 30 crops)

## Baseline Results

| Backbone | AUC | EER | d' (decidability) | Best Threshold |
|----------|-----|-----|--------------------|----------------|
| **ResNet50** (ImageNet) | 0.6884 | 36.3% | 0.673 | 0.624 |
| **FaceNet** (VGGFace2) | 0.6141 | 42.2% | 0.407 | 0.751 |

### Interpretation

- **Both models perform poorly** — close to random chance (AUC=0.5) for individual chimpanzee identification
- **ResNet50 > FaceNet**: Counter-intuitive but explainable — FaceNet is over-specialized for human facial geometry (interpupillary distance, nose bridge, mouth shape). These features don't transfer well to primate faces. ResNet50's generic visual features (texture, color patterns, shape) are actually more useful.
- **Score distributions heavily overlap** (see figures), confirming low discriminative power
- **These results establish a clear baseline** that domain-specific models should significantly improve upon

### Key Insight

The gap between "category recognition" (is this a chimpanzee?) and "individual recognition" (is this Fredy or Victor?) requires identity-discriminative features that general-purpose models don't learn. This validates the need for primate-specific or fine-tuned embeddings.

---

## Backbone Candidates for Next Steps

### Tier 1: Ready to Integrate (pre-trained weights available)

| Model | Source | Embedding Dim | Training Data | Expected Benefit |
|-------|--------|---------------|---------------|------------------|
| **ArcFace (InsightFace)** | [deepinsight/insightface](https://github.com/deepinsight/insightface) | 512 | MS1MV2 (human faces) | Superior metric learning loss; better intra-class compactness |
| **SphereFace** | [clcarwin/sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch) | 512 | CASIA-WebFace | Angular margin; used in Deb et al. 2018 for lemur ReID |
| **DINOv2** | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | 768/1024 | LVD-142M (diverse) | Self-supervised; strong on fine-grained visual tasks without labels |
| **CLIP (ViT-B/16)** | [openai/clip](https://github.com/openai/CLIP) | 512 | 400M image-text pairs | Zero-shot generalization; vision-language alignment |

### Tier 2: Requires Fine-Tuning (most promising for primate ReID)

| Approach | Description | Key Paper |
|----------|-------------|-----------|
| **ArcFace fine-tuned on primate faces** | Start from human face weights, fine-tune on CTai/CZoo | Deb et al. "Face Recognition: Primates in the Wild" (2018) |
| **PrimNet** | Primate-specific network from Freytag et al. | "Chimpanzee Faces in the Wild" GCPR 2016 |
| **Log-Euclidean CNN** | Covariance-based features for primate identity | Freytag et al. GCPR 2016 |
| **Triplet Loss fine-tuning** | Any backbone + triplet/contrastive loss on primate data | Standard metric learning |

### Tier 3: Research Frontier

| Approach | Description |
|----------|-------------|
| **MegaDescriptor** | Animal re-identification foundation model (WildlifeDatasets project) |
| **SAM3 + embedding** | Open-vocabulary detection → crop → fine-tuned embedding |
| **Self-supervised pretraining on primate data** | DINO/MAE on unlabeled primate photos → fine-tune |

---

## Recommended Next Steps

### Phase 1: Quick Wins (no training required)

1. **Add ArcFace backbone** — InsightFace provides ONNX models, easy to load
2. **Add DINOv2 backbone** — Self-supervised features; likely better than supervised ImageNet for novel domains
3. **Add CLIP backbone** — Strong zero-shot generalization
4. Re-run evaluation on demo_chimp_crops → compare all 5 backbones

### Phase 2: Fine-Tuning (requires GPU)

4. **Fine-tune ArcFace on full CTai/CZoo** (7,187 images, 86 individuals)
   - Use the annotation file to split train/test
   - ArcFace loss + chimpanzee identity labels
   - Expected: dramatic improvement (literature reports >90% accuracy after fine-tuning)
5. **Triplet loss training** on any backbone as alternative

### Phase 3: Cross-Species Generalization

6. Test fine-tuned chimp model on other primate species (orangutan, macaque)
7. Investigate domain adaptation / few-shot learning for new species

---

## Dataset Source

The demo chimpanzee crops come from the **CTai/CZoo dataset** by Freytag et al.:

- **Paper**: "Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates" (GCPR 2016)
- **Repository**: [cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces)
- **Full dataset**: 7,187 cropped face images, 86 individuals, with identity/gender/age annotations
- **License**: Non-commercial research use

---

## References

1. Freytag, A. et al. "Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates." GCPR 2016.
2. Deb, D. et al. "Face Recognition: Primates in the Wild." IEEE BTAS 2018.
3. Guan, Y. et al. "Face recognition of a Lorisidae species based on computer vision." Global Ecology and Conservation, 2023.
4. Deng, J. et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
5. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." 2023.
