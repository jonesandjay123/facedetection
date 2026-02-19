# Datasets for Primate Face Detection and Identification

## 1. Primate-Specific Datasets

### 1.1 Macaque Datasets

#### MFID Dataset (Shukla et al., 2018)
- **Species**: Rhesus macaque (Macaca mulatta)
- **Size**: 93 individuals
- **Content**: Face images with identity labels
- **Protocols**: Closed-set, open-set, verification
- **Availability**: Contact authors (not publicly hosted)
- **Paper**: arXiv:1811.00743

#### PFID Dataset (Shukla et al., 2019)
- **Species**: Rhesus macaque + chimpanzee
- **Size**: Multi-species, extends MFID
- **Content**: Wild/field face images
- **Availability**: Contact authors
- **Paper**: arXiv:1907.02642

#### Kōjima Japanese Macaque Dataset (Paulet et al., 2023)
- **Species**: Japanese macaque (Macaca fuscata)
- **Size**: Population of Kōjima island
- **Content**: Video footage with face detections and individual IDs
- **Availability**: Contact authors
- **Paper**: arXiv:2310.06489

#### MacaqueFaces (Neuroscience Context)
- Several macaque face datasets exist in **neuroscience research** (for studying face perception in macaques), not for macaque identification
- These contain controlled stimuli images, not field photos
- Less relevant for wildlife monitoring but potentially useful for pre-training

### 1.2 Chimpanzee Datasets

#### Chimpanzee Faces in the Wild (C-Faces / CVJena)
- **Species**: Chimpanzee (Pan troglodytes)
- **Maintainer**: Computer Vision Group, Friedrich Schiller University Jena
- **Size**: ~6,000 face images, ~90 individuals
- **Content**: Faces detected in the wild, identity labels
- **Availability**: Public — [https://github.com/cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces)
- **Relevance**: Eleane's existing model was trained on this dataset
- **Note**: Includes train/test splits for benchmarking

#### CTai Chimpanzee Dataset
- **Species**: Chimpanzee
- **Source**: Taï National Park, Côte d'Ivoire
- **Content**: Camera trap and researcher-taken images
- **Size**: Varies by study

### 1.3 Multi-Species Primate Datasets

#### PrimNet Dataset (Deb & Jain, 2020)
- **Species**: Chimpanzee, golden monkey, lemur
- **Content**: Face images with identity labels
- **Availability**: Contact authors

#### SA-FARI Dataset (Wasmuht et al., 2025)
- **Focus**: Multi-species animal segmentation in video
- **Relevance**: Uses SAM for wildlife; includes primate species
- **Availability**: Check paper for access (Nov 2025)

---

## 2. General Animal Re-ID Datasets

### 2.1 WildlifeDatasets Toolkit
- **GitHub**: [WildlifeDatasets/wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets)
- Unified Python toolkit providing access to multiple wildlife datasets
- Includes standardized data loading, evaluation protocols
- **Recommended**: Use this as a framework for managing Eleane's macaque dataset

### 2.2 Other Notable Datasets
| Dataset | Species | Size | Task |
|---------|---------|------|------|
| ATRW | Amur Tiger | 8,076 images, 92 individuals | Re-ID |
| Happywhale | Humpback Whale | 50,000+ | Fin matching |
| NDD20 | Cattle | 203 cattle, 3,120 images | Nose print ID |
| FriesianCattle2017 | Holstein-Friesian | 89 cattle | Re-ID |
| Whale Shark ID | Whale Shark | Thousands | Pattern matching |

---

## 3. Human Face Datasets (For Transfer Learning)

### 3.1 Pre-Training Sources

#### MS1MV2 / MS-Celeb-1M (Cleaned)
- **Size**: ~5.8M images, 85K identities
- **Use**: Pre-train ArcFace backbone before fine-tuning on macaques
- **Status**: Standard pre-training dataset for face recognition

#### VGGFace2
- **Size**: 3.31M images, 9,131 identities
- **Use**: Alternative pre-training source
- **Strength**: Large pose and age variation

#### LFW (Labeled Faces in the Wild)
- **Size**: 13,233 images, 5,749 identities
- **Use**: Benchmark for verification (not training)
- **Note**: Too small for training, useful for evaluation protocol design

### 3.2 Transfer Learning Strategy
```
Human face dataset (MS1MV2) → Pre-train ResNet50 + ArcFace
    ↓
Macaque face dataset (50-100 individuals) → Fine-tune last layers
    ↓
Field-ready macaque face recognition model
```

This is the most validated approach in the literature. Shukla et al., Deb & Jain, and Paulet et al. all use variants of this strategy.

---

## 4. Data Augmentation Strategies for Small Datasets

Macaque face datasets are inevitably small (typically 50-100 individuals, 10-50 images per individual). Augmentation is critical.

### 4.1 Standard Augmentations
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 4.2 Field-Specific Augmentations
- **Occlusion simulation**: Random erasing / CutOut to simulate vegetation occlusion
- **Lighting variation**: Strong brightness/contrast jitter to simulate forest canopy lighting
- **Motion blur**: Simulates movement during capture
- **Rain/fog overlay**: Simulates adverse weather

### 4.3 Generative Augmentation
- **StyleGAN for macaque faces**: Igaue et al. (2025) demonstrated this approach
- **Diffusion models**: Stable Diffusion with ControlNet could generate realistic macaque face variants
- **Caution**: Generated images must be validated to avoid training on artifacts

### 4.4 Cross-Species Pre-Training
- Train on chimpanzee faces (larger dataset) → fine-tune on macaque faces
- The shared primate facial structure provides useful initialization
- Combined with human face pre-training: Human → Chimpanzee → Macaque

---

## 5. Building a Custom Macaque Dataset from Field Photos

### 5.1 Data Collection Guidelines

#### Photography Protocol
- Capture **frontal and 3/4 profile** views of each individual
- Minimum **10-20 images per individual** for reliable identification
- Include variation: different lighting, distances, seasons
- Document identifying features: scars, ear damage, facial hair patterns

#### Minimum Viable Dataset
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Individuals | 20 | 50-100+ |
| Images/individual | 10 | 30-50 |
| Total images | 200 | 1,500-5,000 |
| Image resolution | 640×640 | 1024×1024+ |

### 5.2 Annotation Workflow

#### Step 1: Face Detection Annotations
- Use [CVAT](https://github.com/opencv/cvat) or [Label Studio](https://labelstud.io/) for bounding box annotation
- Annotate: face bounding box + species label
- Semi-automate: Run pre-trained YOLO → correct predictions manually

#### Step 2: Identity Labels
- Each individual needs a unique ID
- Requires expert knowledge (primatologist) for initial identification
- Group images by individual → verify with multiple experts

#### Step 3: Quality Control
- Remove blurry / heavily occluded images
- Ensure identity labels are correct (most critical)
- Split into train/val/test by **individual** (not by image) for open-set evaluation

### 5.3 Semi-Automated Pipeline
```
Field Photos
  → YOLOv8 (pre-trained) → Auto-detect faces → Manual correction
  → DINOv2 → Extract embeddings → Cluster by similarity
  → Expert review → Assign identity labels to clusters
  → Export as dataset
```

This reduces manual annotation effort by 60-80%.

---

## 6. Dataset Recommendations for Eleane

### Immediate
1. **Use C-Faces (chimpanzee)** for initial testing and pipeline validation
2. **Download WildlifeDatasets toolkit** as infrastructure
3. **Start collecting macaque field photos** systematically

### For Training
1. **Pre-train on MS1MV2** (human faces) → available via InsightFace
2. **Fine-tune on C-Faces** (chimpanzee) → establish cross-species baseline
3. **Fine-tune on macaque data** → final model

### For Evaluation
1. Design **all four protocols**: classification, verification, closed-set, open-set
2. Follow MFID/PFID evaluation methodology (Shukla et al.)
3. Report per-protocol metrics for PhD thesis
