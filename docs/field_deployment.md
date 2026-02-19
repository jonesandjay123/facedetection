# Field Deployment: Practical Considerations for Wild Macaque Face Identification

## 1. Image Quality Challenges

### 1.1 Lighting Variation
- **Forest canopy**: Dappled light creates high-contrast patches on faces
- **Dawn/dusk**: Low light, high noise, color shifts
- **Direct sunlight**: Harsh shadows obscure facial features
- **Mitigation**: Strong color jitter augmentation during training; histogram equalization at inference

### 1.2 Pose Variation
- Macaques rarely look directly at cameras
- Extreme profile views lose discriminative facial features
- **Mitigation**: Multi-pose training data; accept only near-frontal faces (±45°) for identification; use multiple frames from video to find best angle

### 1.3 Occlusion
- Vegetation (branches, leaves) partially covering faces
- Other animals blocking the view
- Self-occlusion (hands near face during grooming)
- **Mitigation**: Occlusion-aware augmentation (random erasing); train detector to output confidence scores → reject low-confidence detections

### 1.4 Distance and Resolution
- Camera traps: fixed position, animals at variable distances
- Handheld: variable zoom and focus
- **Minimum face size**: ~64×64 pixels for reliable identification; ~32×32 for detection only
- **Mitigation**: Super-resolution preprocessing for distant subjects; multi-scale detection

### 1.5 Motion Blur
- Fast-moving animals, especially juveniles
- Handheld camera shake
- **Mitigation**: High shutter speed (>1/500s) for handheld; motion blur augmentation during training

---

## 2. Camera Setup Options

### 2.1 Camera Traps
**Pros:**
- Unattended, 24/7 monitoring
- Minimal disturbance to animals
- Standardized camera position and field of view

**Cons:**
- Fixed angle — may miss faces
- Triggered by motion — may capture body only
- Limited resolution at distance
- IR images at night (face features less visible)

**Recommendations:**
- Position at **macaque eye height** (0.5-1.5m for terrestrial macaques)
- Use **video mode** (3-10 sec clips) rather than single photos for better face capture
- Pair cameras at different angles for multi-view capture

### 2.2 Handheld Photography
**Pros:**
- Flexible angle and distance
- High resolution possible
- Can actively seek frontal face views

**Cons:**
- Requires researcher presence (disturbs animals)
- Inconsistent quality
- Labor intensive

**Recommendations:**
- Use burst mode (5-10 fps)
- Telephoto lens (200-400mm) for safe distance
- Shoot raw for maximum post-processing flexibility

### 2.3 Drone Photography
**Pros:**
- Overhead perspective, covers large area
- Can follow troops

**Cons:**
- Noise disturbs primates
- Top-down angle poor for face detection
- Regulatory restrictions

**Verdict:** Not recommended for face identification; useful for population surveys.

---

## 3. Edge Deployment Options

### 3.1 Deployment Scenarios

| Scenario | Hardware | Use Case |
|----------|----------|----------|
| Lab/office | Desktop GPU (RTX 3090+) | Batch processing field photos, training |
| Field laptop | Laptop GPU (RTX 3060 Mobile) | On-site processing, quick results |
| Camera trap + edge | NVIDIA Jetson Orin Nano | Real-time detection at trap |
| Mobile | iPhone/Android with Neural Engine | Quick field identification |
| Minimal | Raspberry Pi 5 | Detection only (no ID) |

### 3.2 NVIDIA Jetson (Recommended for Edge)

**Jetson Orin Nano (8GB)**
- ~40 TOPS AI performance
- Can run YOLOv8n at ~30 FPS
- Can run ArcFace (ResNet50) at ~10 FPS
- Cannot run SAM 3 (too large)
- Power: 7-15W
- Cost: ~$250

**Jetson AGX Orin (64GB)**
- ~275 TOPS
- Can run full pipeline including SAM 2
- Power: 15-60W
- Cost: ~$2,000

### 3.3 Model Optimization for Edge

```python
# Export YOLO to TensorRT for Jetson
from ultralytics import YOLO
model = YOLO("macaque_face_detector.pt")
model.export(format="engine", device=0, half=True)  # FP16 TensorRT

# Export ArcFace to ONNX
import torch
model = ArcFaceModel()
model.load_state_dict(torch.load("arcface_macaque.pth"))
torch.onnx.export(model, dummy_input, "arcface_macaque.onnx",
                   opset_version=17, dynamic_axes={"input": {0: "batch"}})
```

### 3.4 Mobile Deployment

**iOS (CoreML)**
- Convert PyTorch → CoreML using `coremltools`
- iPhone 15 Pro Neural Engine: ~35 TOPS
- Suitable for YOLOv8n + lightweight embedding model

**Android (TFLite)**
- Convert via ONNX → TFLite
- Pixel 8 Tensor G3: ~10 TOPS
- Suitable for detection; identification may be slow

---

## 4. Processing Modes

### 4.1 Real-Time Processing
- **Use case**: Live monitoring at feeding stations, guided tours
- **Requirements**: <100ms per frame (>10 FPS)
- **Feasible pipeline**: YOLOv8n (detection) + lightweight ArcFace (ID)
- **Not feasible**: SAM 3 (too slow for real-time)

### 4.2 Batch Processing (Recommended for Research)
- **Use case**: Process day's camera trap footage overnight
- **Requirements**: Throughput matters more than latency
- **Pipeline**: Full SAM 3 + DINOv2/ArcFace
- **Hardware**: Desktop GPU or cloud (Google Colab Pro, AWS)
- **Typical throughput**: 1,000-5,000 images/hour on RTX 3090

### 4.3 Hybrid: Detect Now, Identify Later
```
Field (edge device):
  → YOLO face detection
  → Crop and save face images with metadata (GPS, timestamp)
  → Store on SD card

Lab (GPU workstation):
  → Load face crops
  → Run ArcFace embedding
  → Match to gallery
  → Update population database
```

This is the most practical approach for most field research.

---

## 5. Annotation Tools and Workflows

### 5.1 Bounding Box Annotation

#### CVAT (Computer Vision Annotation Tool)
- **URL**: [https://github.com/opencv/cvat](https://github.com/opencv/cvat)
- Self-hosted, open source
- Supports images and video
- AI-assisted annotation (auto-labeling)
- Team collaboration features

#### Label Studio
- **URL**: [https://labelstud.io/](https://labelstud.io/)
- Open source, web-based
- Supports multiple annotation types
- ML backend integration for pre-annotation

#### Roboflow
- **URL**: [https://roboflow.com/](https://roboflow.com/)
- Cloud-based, free tier available
- Built-in augmentation pipeline
- Direct export to YOLO format
- **Recommended for quick start**

### 5.2 Identity Annotation Workflow

```
1. Collect field images
2. Run face detector → extract face crops
3. Run DINOv2/ArcFace → extract embeddings
4. Cluster embeddings (DBSCAN/HDBSCAN) → auto-group similar faces
5. Expert review: confirm/correct clusters, assign identity labels
6. Export labelled dataset
```

### 5.3 Recommended Tool Stack
| Task | Tool | Why |
|------|------|-----|
| Face bbox annotation | Roboflow | Easy, exports to YOLO |
| Identity labelling | Custom script + spreadsheet | Requires expert knowledge |
| Clustering | scikit-learn (HDBSCAN) | Auto-groups similar faces |
| Dataset management | WildlifeDatasets toolkit | Standardized format |
| Version control | DVC (Data Version Control) | Track dataset versions with Git |

---

## 6. Complete Field Workflow

### Phase 1: Data Collection (Field)
1. Set up camera traps at known macaque sites
2. Collect 2-4 weeks of video/photo data
3. Supplement with handheld photos of known individuals

### Phase 2: Initial Processing (Lab)
1. Extract frames from video (1-2 fps)
2. Run face detector → crop all faces
3. Filter: remove blurry, too small, non-macaque faces
4. Cluster and label with expert assistance

### Phase 3: Model Training
1. Fine-tune YOLO on macaque face bboxes
2. Fine-tune ArcFace on identity-labelled faces
3. Evaluate on held-out test set (open-set protocol)

### Phase 4: Deployment
1. Deploy YOLO + ArcFace on field laptop or Jetson
2. Process new images → identify individuals
3. Flag unknown individuals for expert review
4. Continuously expand gallery with new individuals

### Phase 5: Iteration
1. Collect more data from hard cases (misidentifications)
2. Retrain models periodically
3. Track population over time

---

## 7. Power and Connectivity in the Field

### Power Options
- **Solar panel + battery**: 100W panel + 500Wh battery supports Jetson for 8-12 hours
- **Power bank**: 20,000mAh for mobile/RPi for ~6 hours
- **Vehicle power**: Inverter from 12V car battery for laptop processing

### Connectivity
- **Assume offline**: Design for fully offline operation
- **Periodic sync**: Transfer data via USB/SD card to connected device
- **Satellite internet**: Starlink for remote sites (if budget allows)
- **Mobile data**: Upload thumbnails/results for remote monitoring

---

## 8. Summary: What's Practical Now

| Capability | Feasibility | Hardware Needed |
|------------|-------------|-----------------|
| Face detection (YOLO) | ✅ Ready now | Laptop CPU or Jetson |
| Face ID (ArcFace) | ✅ Ready now | Laptop GPU |
| SAM 3 face segmentation | ⚠️ Experimental | Desktop GPU |
| Real-time tracking | ⚠️ Limited | Jetson AGX Orin |
| Mobile app | 🔧 Needs development | iPhone/Android |
| Fully automated field system | 🔬 Research stage | Custom hardware |
