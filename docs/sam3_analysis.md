# SAM 3 Analysis: Applicability for Primate Face Detection

## 1. What is SAM 3?

**SAM 3 (Segment Anything with Concepts)** is Meta AI's third iteration of the Segment Anything Model family, released in November 2025.

### 1.1 Key Innovation: Concept-Level Segmentation
Unlike SAM 1 (point/box prompts) and SAM 2 (+ video support), SAM 3 introduces **Promptable Concept Segmentation (PCS)**:
- Segment objects using **text/concept prompts** (e.g., "monkey", "face", "tree")
- No need for manual point clicks or bounding boxes
- Bridges the gap between spatial prompting and semantic understanding

### 1.2 Architecture
- Built on top of SAM 2's streaming memory architecture
- Adds a **concept encoder** that maps text/concept descriptions to the prompt space
- Maintains backward compatibility with point and box prompts
- Supports both image and video segmentation

### 1.3 Training Data
- Trained on SA-1B (images) + SA-V (videos) + additional concept-annotated data
- SA-Co (Segment Anything with Concepts) benchmark released alongside

### 1.4 Paper & Resources
- **Paper**: Carion, Gustafson, Hu et al. "SAM 3: Segment Anything with Concepts" (Nov 2025)
- **GitHub**: Expected at `facebookresearch/sam3` (check for availability)
- **HuggingFace**: Look for `facebook/sam3-*` model variants

---

## 2. SAM 3 vs SAM 2 vs SAM 1

| Feature | SAM 1 | SAM 2 | SAM 3 |
|---------|-------|-------|-------|
| **Release** | Apr 2023 | Jul 2024 | Nov 2025 |
| **Input** | Images only | Images + Video | Images + Video |
| **Prompts** | Point, Box, Mask | Point, Box, Mask | Point, Box, Mask, **Text/Concept** |
| **Semantic understanding** | None | None | **Yes (category-level)** |
| **Individual identification** | No | No | **No** |
| **Video tracking** | No | Yes (streaming memory) | Yes |
| **Zero-shot capability** | Spatial only | Spatial + temporal | Spatial + temporal + **semantic** |
| **Model sizes** | ViT-B/L/H | Tiny/Small/Base+/Large | TBD |

### Critical Distinction
SAM 3 can segment **"a monkey face"** but **cannot distinguish between individual monkeys**. This is category-level segmentation, not instance-level identification.

---

## 3. Can SAM 3 Segment Primate Faces? (Zero-Shot)

### 3.1 What Should Work
- **Text prompt "monkey face"**: SAM 3's concept encoder should recognize "monkey" and "face" as concepts
- **Text prompt "primate"**: Likely to segment entire primate bodies
- **Box prompt around face region**: Should produce clean face segmentation (inherited from SAM 1/2)
- **Point prompt on face**: Should work well for face segmentation

### 3.2 What Won't Work
- **"Macaque #7 face"**: SAM 3 has no individual identity understanding
- **Distinguishing species**: Unlikely to differentiate macaque from chimpanzee from human at segmentation level
- **Excluding human faces while keeping monkey faces**: Would need post-processing or a separate classifier

### 3.3 Realistic Expectations
SAM 3 is best used as a **face detection/segmentation module** in a larger pipeline:

```
Image → SAM 3 ("face") → All faces segmented → Species classifier → Macaque faces only → Embedding model → Individual ID
```

It replaces the **detection** step, not the **identification** step.

### 3.4 The SAM2-to-SAM3 Gap
Sapkota et al. (2025) highlighted that prompt-based expertise (SAM 2) does not transfer well to concept-driven tasks (SAM 3). This means:
- SAM 3's text prompting may be inconsistent for fine-grained categories
- For primate faces specifically, **box/point prompts may still outperform text prompts**
- The concept encoder was not trained specifically for primate faces

---

## 4. Prompt Engineering Strategies

### 4.1 Point Prompts (Most Reliable)
```python
# Best for: Processing images where approximate face location is known
# e.g., after a coarse YOLO detector finds monkey bodies
point_coords = np.array([[face_center_x, face_center_y]])
point_labels = np.array([1])  # 1 = foreground
masks = predictor.predict(point_coords=point_coords, point_labels=point_labels)
```

### 4.2 Box Prompts (Recommended for Pipelines)
```python
# Best for: Using output from a pre-trained face/body detector
# YOLO detects monkey → SAM 3 refines face segmentation
box = np.array([x1, y1, x2, y2])  # bounding box around face region
masks = predictor.predict(box=box)
```

### 4.3 Text/Concept Prompts (SAM 3 Specific)
```python
# Best for: Zero-shot face detection without any prior detector
# WARNING: May segment all faces including human faces
# API is speculative — check actual SAM 3 release for exact syntax
masks = predictor.predict(text="monkey face")
```

### 4.4 Combined Prompts
```python
# Most robust: Text concept + spatial hint
masks = predictor.predict(
    text="face",
    box=approximate_face_region,  # from body detector
)
```

---

## 5. Proposed Pipeline Using SAM 3

### Option A: SAM 3 as Zero-Shot Detector
```
Raw Image
  → SAM 3 (text: "face") → All face masks
  → Species classifier (ResNet/CLIP) → Filter to macaque faces only
  → Crop + Normalize
  → Embedding model (ArcFace/FaceNet) → Feature vector
  → Nearest neighbor / threshold → Individual ID
```

### Option B: YOLO + SAM 3 Hybrid (Recommended)
```
Raw Image
  → YOLO (monkey body detection) → Bounding boxes
  → SAM 3 (box prompt per detection) → Precise face masks
  → Crop + Normalize
  → Embedding model (ArcFace fine-tuned on macaques) → Feature vector
  → Individual ID
```

### Option C: SAM 3 for Video Tracking
```
Video Stream
  → Frame 1: SAM 3 (text: "monkey face") → Initial masks
  → Frames 2-N: SAM 3 video tracking (streaming memory) → Track faces
  → Per-track: Sample best frames → Embedding → ID
```

---

## 6. Code Snippets

### 6.1 Loading SAM 2 (Current Stable)
```python
# SAM 2 is currently available; SAM 3 may follow similar patterns
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[500, 375]]),
        point_labels=np.array([1]),
    )
```

### 6.2 Expected SAM 3 Usage (Speculative)
```python
# Check facebookresearch/sam3 for actual API when released
from sam3.build_sam import build_sam3
from sam3.sam3_image_predictor import SAM3ImagePredictor

predictor = SAM3ImagePredictor(build_sam3(model_cfg, checkpoint))
predictor.set_image(image)

# Text-based concept segmentation
masks, scores = predictor.predict(text="monkey face")

# Or hybrid: text + spatial hint
masks, scores = predictor.predict(
    text="face",
    box=np.array([100, 100, 300, 300]),
)
```

### 6.3 Post-Processing: Filtering Non-Macaque Faces
```python
import torch
from torchvision import models, transforms

# Simple species classifier to filter SAM 3 outputs
species_model = models.resnet18(pretrained=True)
# Fine-tune last layer for: macaque / human / other_primate / background
species_model.fc = torch.nn.Linear(512, 4)
species_model.load_state_dict(torch.load("species_classifier.pth"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for mask, score in zip(masks, scores):
    face_crop = apply_mask_and_crop(image, mask)
    pred = species_model(transform(face_crop).unsqueeze(0))
    if pred.argmax() == 0:  # macaque
        # Proceed with individual identification
        embedding = id_model(face_crop)
```

---

## 7. Limitations and Honest Assessment

### 7.1 SAM 3 Cannot Do Individual Identification
This is the most critical limitation. SAM 3 is a **segmentation** model, not an **identification** model. It can find and segment faces but cannot tell individuals apart.

### 7.2 Availability Concerns
- As of early 2026, SAM 3's code/model availability on GitHub and HuggingFace should be verified
- SAM 2.1 is fully available and well-tested
- If SAM 3 is not yet publicly available, SAM 2 with box prompts is a practical alternative

### 7.3 Computational Cost
- SAM models are large (ViT-H ~630M params for SAM 1; SAM 2 similar)
- SAM 3 likely even larger due to concept encoder
- Not suitable for edge deployment without quantization/distillation
- For field use, consider running on a laptop GPU or cloud, not on Raspberry Pi

### 7.4 Overkill for Face Detection?
- A fine-tuned **YOLOv8** face detector may be faster and more accurate for monkey face detection
- SAM 3's strength is in **precise segmentation masks**, not bounding boxes
- If you only need bounding boxes, YOLO is the better choice
- SAM 3 adds value when you need **pixel-perfect face masks** (e.g., for removing background before embedding)

### 7.5 When SAM 3 Makes Sense
- **Zero-shot deployment**: When you have no macaque-specific training data at all
- **Precise segmentation**: When background removal improves identification accuracy
- **Video tracking**: When tracking individuals across video frames
- **Research contribution**: Demonstrating SAM 3's capability on primate faces is novel and publishable

---

## 8. Recommendation

**For Eleane's PhD project**, the most practical approach is:

1. **Use SAM 3 for face segmentation** (if available) or **SAM 2 with box prompts**
2. **Keep the existing ArcFace pipeline** for individual identification
3. **Add a species classifier** to filter out human faces
4. **Fine-tune on macaque data** rather than relying purely on zero-shot

SAM 3 is a valuable **component** of the pipeline, not a complete solution. The novel contribution would be demonstrating that SAM 3's concept-level segmentation can serve as an effective zero-shot face detector for primates, reducing the need for annotated bounding box training data.
