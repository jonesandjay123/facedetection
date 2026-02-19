# Alternative Approaches for Macaque Face Detection and Identification

## 1. Overview of the Pipeline

Any primate face identification system requires two stages:
1. **Detection/Segmentation**: Find and localize faces in images
2. **Identification/Embedding**: Determine which individual the face belongs to

This document compares approaches for each stage and recommends the most practical pipeline for field deployment.

---

## 2. Face Detection Approaches

### 2.1 YOLOv8/YOLOv9/YOLOv11 (Recommended for Detection)

**Strengths:**
- Extremely fast inference (real-time on GPU, near-real-time on CPU)
- Well-established training pipeline with Ultralytics
- Easy to fine-tune on custom macaque face dataset (few hundred images sufficient)
- Small model variants (YOLOv8n) suitable for edge deployment
- Already proven for Japanese macaque face detection (Paulet et al. 2023: 83% accuracy)

**Weaknesses:**
- Outputs bounding boxes, not precise masks
- Requires labelled training data (bounding box annotations)
- May miss partially occluded faces

**Code Example:**
```python
from ultralytics import YOLO

# Fine-tune on macaque face dataset
model = YOLO("yolov8n.pt")
model.train(data="macaque_faces.yaml", epochs=100, imgsz=640)

# Inference
results = model("field_photo.jpg")
for box in results[0].boxes:
    if box.conf > 0.5:
        x1, y1, x2, y2 = box.xyxy[0]
        face_crop = image[int(y1):int(y2), int(x1):int(x2)]
```

### 2.2 SAM 3 (Concept-Based Segmentation)

See `sam3_analysis.md` for detailed analysis.

**Best for:** Zero-shot face segmentation when no labelled data is available.
**Not suitable for:** Individual identification (only detects "face", not "which face").

### 2.3 RetinaFace / MTCNN (Human Face Detectors)

**Strengths:**
- Pre-trained, no fine-tuning needed
- Very fast (MTCNN) to moderately fast (RetinaFace)
- Include facial landmark detection (useful for alignment)

**Weaknesses:**
- Trained on **human faces** — may not generalize well to macaques
- Macaque facial proportions differ significantly (wider face, different eye/nose ratios)
- Empirical testing needed: some papers report partial success on primates

**Verdict:** Worth trying as a baseline. If detection rate > 70% on macaque images, may be usable with minimal fine-tuning.

### 2.4 Faster R-CNN (Two-Stage Detector)

**Strengths:**
- Higher accuracy than single-stage detectors for small objects
- Proven for Japanese macaque face detection (Paulet et al. 2023: 82.2%)

**Weaknesses:**
- Slower than YOLO
- More complex training pipeline
- Larger model size

---

## 3. Face Identification / Embedding Approaches

### 3.1 ArcFace (Recommended — Eleane's Current Approach)

**Architecture:** ResNet50 backbone + ArcFace loss (Additive Angular Margin)

**Strengths:**
- State-of-the-art for face recognition (human and primate)
- Produces highly discriminative 512-d embeddings
- Natural support for open-set identification via distance thresholds
- Already validated in Eleane's pipeline on chimpanzees
- Strong transfer learning potential: pre-train on human faces, fine-tune on macaques

**Strategy for Macaques:**
```python
# 1. Start with ArcFace pre-trained on MS1MV2 (human faces)
# 2. Replace last classification layer
# 3. Fine-tune on macaque dataset with ArcFace loss
# 4. Use embedding distance for open-set identification

import torch
from insightface.recognition.arcface import ArcFaceModel

model = ArcFaceModel(backbone="resnet50", pretrained="ms1mv2")
# Fine-tune on macaque data...

# At inference:
embedding = model.get_embedding(face_crop)
# Compare with gallery embeddings
distances = cosine_distance(embedding, gallery_embeddings)
identity = gallery_ids[distances.argmin()] if distances.min() < threshold else "unknown"
```

### 3.2 DINOv2 (Self-Supervised Visual Features)

**Architecture:** ViT-based self-supervised model by Meta

**Strengths:**
- Pre-trained on massive diverse image dataset (no labels needed)
- Excellent zero-shot feature extraction
- Features are highly discriminative for fine-grained visual differences
- No task-specific training needed — just extract features and compare
- Potentially captures subtle facial differences (scars, fur patterns)

**Weaknesses:**
- Not specifically trained for face recognition
- May require larger embedding dimension for good separation
- Slower than dedicated face models

**Strategy:**
```python
import torch
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("facebook/dinov2-large")
processor = AutoProcessor.from_pretrained("facebook/dinov2-large")

# Extract features from cropped face
inputs = processor(images=face_crop, return_tensors="pt")
features = model(**inputs).last_hidden_state[:, 0]  # CLS token

# Compare features across individuals
similarity = torch.cosine_similarity(features_a, features_b)
```

**Verdict:** Excellent choice for initial prototyping when labelled macaque data is scarce. Can be used as embeddings without any fine-tuning.

### 3.3 CLIP / SigLIP (Vision-Language Models)

**Strengths:**
- Zero-shot classification via text prompts
- Can potentially distinguish "macaque" from "human" from "chimpanzee"
- Useful for **species filtering**, not individual identification

**Weaknesses:**
- Not designed for fine-grained individual identification
- Cannot distinguish between two macaques via text alone
- Image embeddings are less discriminative than ArcFace/DINOv2 for re-ID

**Best Use Case:**
```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Species classification (not individual ID)
texts = ["a macaque face", "a human face", "a chimpanzee face", "not a face"]
inputs = processor(text=texts, images=face_crop, return_tensors="pt")
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
# → Use for filtering, not identification
```

### 3.4 FaceNet (Triplet Loss Embeddings)

**Strengths:**
- Classic face recognition approach
- 128-d embeddings, lightweight
- Triplet loss naturally supports open-set

**Weaknesses:**
- Older architecture (2015), outperformed by ArcFace
- Triplet mining can be tricky with small datasets

**Verdict:** ArcFace is strictly better. Use ArcFace instead.

---

## 4. Complete Pipeline Comparison

### Pipeline A: YOLO + ArcFace (Recommended)
```
Image → YOLOv8 (monkey face detection) → Crop → ArcFace (fine-tuned on macaques) → Embedding → ID
```
- **Pros**: Fast, accurate, proven approach, suitable for field deployment
- **Cons**: Requires labelled bounding boxes and identity labels for training
- **Best for**: When you have or can collect a modest macaque face dataset (50+ individuals)

### Pipeline B: SAM 3 + DINOv2 (Zero-Shot)
```
Image → SAM 3 ("face") → Mask → Crop → DINOv2 → Embedding → ID
```
- **Pros**: No labelled training data needed, works out of the box
- **Cons**: Lower accuracy, SAM 3 may segment human faces too, slower
- **Best for**: Initial exploration, when no macaque data is available yet

### Pipeline C: YOLO + SAM 3 + ArcFace (Maximum Accuracy)
```
Image → YOLOv8 (monkey body) → SAM 3 (box prompt, face mask) → Precise crop → ArcFace → ID
```
- **Pros**: Best segmentation quality, clean face crops improve embedding quality
- **Cons**: Most complex, slowest, requires SAM 3 availability
- **Best for**: Offline batch processing of field photos where accuracy is paramount

### Pipeline D: RetinaFace + CLIP Filter + ArcFace (Transfer Learning)
```
Image → RetinaFace → All faces → CLIP (species filter) → Macaque faces → ArcFace → ID
```
- **Pros**: Uses all pre-trained models, minimal training needed
- **Cons**: RetinaFace may miss some macaque faces
- **Best for**: Quick prototype to test feasibility

---

## 5. Practical Recommendation for Eleane

### Immediate Next Steps (Low Effort, High Impact)
1. **Keep ArcFace pipeline** — it's the right choice for identification
2. **Replace/add YOLO face detector** fine-tuned on macaque faces
3. **Test DINOv2 embeddings** as a baseline without any training
4. **Use CLIP for species filtering** to exclude human faces

### Medium-Term (For PhD Thesis)
1. **Collect macaque face dataset** from field photos (see `datasets.md`)
2. **Fine-tune ArcFace on macaque data** (transfer from human face model)
3. **Evaluate SAM 3** for face segmentation quality (novel contribution)
4. **Compare all pipelines** systematically (publishable results)

### What NOT to Spend Time On
- ❌ Training SAM 3 from scratch (not feasible, not necessary)
- ❌ Using CLIP for individual identification (wrong tool for the job)
- ❌ Building a custom segmentation model (SAM/YOLO are good enough)
- ❌ Real-time processing on Raspberry Pi (not needed for PhD research)
