# Literature Review: Primate Face Detection and Identification

## 1. Overview

This review surveys the academic literature on non-human primate (NHP) face detection and individual identification, with a focus on macaques in wild/field conditions. We cover deep learning-based approaches, relevant datasets, segmentation foundation models (SAM family), and the distinction between open-set and closed-set identification paradigms.

---

## 2. Macaque Face Recognition

### 2.1 Macaque Face Identification (MFID)
**Shukla et al. (2018)** — *"Unique Identification of Macaques for Population Monitoring and Control"*
- Proposed MFID, an image-based non-invasive tool for rhesus macaque (Macaca mulatta) facial recognition
- Evaluated on a dataset of **93 individual monkeys** under closed-set, open-set, and verification protocols
- Used deep CNN features with pairwise loss functions
- Demonstrated extensibility to other primate species
- **Key contribution**: First systematic macaque face identification system with multiple evaluation protocols
- arXiv: [1811.00743](https://arxiv.org/abs/1811.00743)

### 2.2 Primate Face Identification in the Wild (PFID)
**Shukla et al. (2019)** — *"Primate Face Identification in the Wild"* (PRICAI 2019)
- Extended MFID work to handle wild/uncontrolled conditions
- Proposed PFID loss: cross-entropy + pairwise loss for learning discriminative representations robust to pose, lighting, and occlusion
- Achieved state-of-the-art on both **rhesus macaques** and **chimpanzees**
- Evaluated under all four protocols: classification, verification, closed-set ID, open-set recognition
- **Key insight**: Pairwise learning is critical for small-dataset primate identification
- arXiv: [1907.02642](https://arxiv.org/abs/1907.02642)

### 2.3 Japanese Macaque Detection and Recognition
**Paulet et al. (2023)** — *"Deep Learning for Automatic Detection and Facial Recognition in Japanese Macaques: Illuminating Social Networks"*
- Developed a two-stage pipeline for Japanese macaques (Macaca fuscata):
  1. **Face detection**: Faster R-CNN achieving 82.2% accuracy
  2. **Individual recognition**: YOLOv8n achieving 83% accuracy
- Applied to social network analysis of the Kōjima island macaque population
- **Practical relevance**: Demonstrates end-to-end pipeline from video to social network construction
- arXiv: [2310.06489](https://arxiv.org/abs/2310.06489)

### 2.4 Macaque Facial Expression Generation
**Igaue et al. (2025)** — *"Motion Transfer-Enhanced StyleGAN for Generating Diverse Macaque Facial Expressions"*
- Uses generative AI (StyleGAN) with motion transfer for data augmentation
- Addresses the critical problem of **limited training data** for macaque faces
- Relevant for augmenting small field datasets
- arXiv: [2511.xxxxx](https://arxiv.org/search/?query=macaque+face+dataset) (Nov 2025)

---

## 3. Chimpanzee Face Recognition

### 3.1 Chimpanzee Faces Dataset (C-Faces)
**Freytag et al. / CVJena** — Chimpanzee Faces in the Wild
- One of the most established primate face datasets
- Contains labelled face images of individual chimpanzees
- Used as a benchmark in multiple primate recognition studies
- Available at: [https://github.com/cvjena](https://github.com/cvjena)
- **Relevance to Eleane's work**: Her existing pipeline (ResNet50 + ArcFace) was trained on this dataset

### 3.2 ChimpFace and Related Systems
- **Loos & Ernst (2013)**: Early work on automated chimpanzee face recognition using local features
- **Deb et al. (2018)**: "Face Recognition: Primates in the Wild" — applied deep learning (VGGFace-based) to chimpanzee identification
- **Brust et al. (2017)**: "Towards Automated Visual Monitoring of Individual Gorillas in the Wild" — extended to gorillas, demonstrating cross-species applicability

### 3.3 PrimNet
**Deb & Jain (2020)** — *"PrimNet: Primate Face Identification"*
- Multi-species primate face recognition system
- Architecture: Modified ResNet with metric learning
- Evaluated on chimpanzees, golden monkeys, and lemurs
- Key finding: **Transfer learning from human face models significantly boosts performance**
- This validates the approach of starting from human face recognition models (like ArcFace) and fine-tuning for primates

---

## 4. Deep Learning for Animal Re-Identification

### 4.1 WildlifeDatasets Toolkit
**Čermák et al. (2023)** — *"WildlifeDatasets: An open-source toolkit for animal re-identification"*
- Open-source Python toolkit for standardized wildlife re-ID research
- Provides unified access to multiple animal datasets
- GitHub: [WildlifeDatasets/wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets)
- **Highly relevant**: Could serve as a framework for Eleane's macaque dataset

### 4.2 General Animal Re-ID Approaches
- **Schneider et al. (2019)**: "Similarity Learning Networks for Animal Individual Re-Identification" — triplet networks for animal re-ID
- **Li et al. (2019)**: "ATRW: A Benchmark for Amur Tiger Re-identification in the Wild" — established protocols for wildlife re-ID
- **Rogers et al. (2024)**: "Recurrence over Video Frames (RoVF) for the Re-identification of Meerkats" — temporal information improves re-ID accuracy

### 4.3 Transfer Learning from Human Face Recognition
A consistent finding across the literature:
- **Human face models (VGGFace, ArcFace, FaceNet) provide excellent starting points** for primate face recognition
- The shared facial structure between humans and NHPs enables effective transfer learning
- Fine-tuning the last few layers on species-specific data is typically sufficient
- This approach outperforms training from scratch, especially with limited data

---

## 5. SAM Family: Segmentation Foundation Models

### 5.1 SAM 1 (Segment Anything Model)
**Kirillov et al. (2023)** — Meta AI
- Foundation model for image segmentation
- Promptable: point, box, and mask prompts
- Trained on SA-1B dataset (11M images, 1.1B masks)
- **No semantic understanding** — segments "something" at prompted location but doesn't know what it is
- Architecture: ViT encoder + prompt encoder + mask decoder
- Paper: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

### 5.2 SAM 2 (Segment Anything in Images and Videos)
**Ravi et al. (2024)** — Meta AI, FAIR
- Extended SAM to video with streaming memory architecture
- Real-time video segmentation with temporal consistency
- Trained on SA-V dataset (largest video segmentation dataset)
- SAM 2.1 update (Sept 2024): Improved checkpoints and training code released
- GitHub: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **Still prompt-based**: Requires explicit prompts (points/boxes), no semantic understanding

### 5.3 SAM 3 (Segment Anything with Concepts)
**Carion, Gustafson, Hu et al. (2025)** — Meta AI
- **Major paradigm shift**: Adds concept-level understanding to SAM
- Supports **Promptable Concept Segmentation (PCS)**: segment by text/concept, not just spatial prompts
- Can segment objects by name (e.g., "monkey face") without manual point/box prompts
- Includes SA-Co (Segment Anything with Concepts) benchmark
- arXiv: November 2025
- **Critical limitation**: "Concept" means category-level, not individual-level identification

### 5.4 SAM for Wildlife Applications
**Wasmuht et al. (2025)** — *"The SA-FARI Dataset: Segment Anything in Footage of Animals for Recognition and Identification"*
- Directly applies SAM to wildlife segmentation and re-ID
- Multi-species animal segmentation in video footage
- Demonstrates SAM's potential as a **preprocessing step** for wildlife identification pipelines
- arXiv: November 2025

### 5.5 The SAM2-to-SAM3 Gap
**Sapkota et al. (2025)** — *"The SAM2-to-SAM3 Gap in the Segment Anything Model Family"*
- Analyzes the fundamental discontinuity between SAM 2 and SAM 3
- SAM 2 excels at prompt-based segmentation; SAM 3 targets concept-driven segmentation
- **Key finding**: Prompt-based expertise does not automatically transfer to concept-driven tasks
- arXiv: December 2025

---

## 6. Open-Set vs Closed-Set Identification

### 6.1 Definitions
- **Closed-set**: All test individuals are seen during training. Standard classification problem.
- **Open-set**: Test set may contain individuals **not seen during training**. Must detect "unknown" individuals.
- **Verification**: One-to-one comparison — "Are these two images the same individual?"

### 6.2 Relevance to Wildlife Monitoring
- Field deployment **requires open-set capability**: New individuals will appear over time
- Closed-set is useful for monitoring **known populations** (e.g., captive groups, well-studied troops)
- Eleane's existing system supports both, which is architecturally sound

### 6.3 Approaches
- **Metric learning** (ArcFace, triplet loss): Naturally supports open-set via embedding distance thresholds
- **Classification + threshold**: Softmax output below threshold → "unknown"
- **Prototypical networks**: Few-shot learning approach, well-suited for adding new individuals
- **PFID approach**: Combined cross-entropy + pairwise loss specifically designed for all four protocols

---

## 7. Key Gaps and Opportunities

1. **Limited macaque-specific datasets**: Most work uses small, private datasets (50-100 individuals). No large-scale, public macaque face benchmark exists comparable to human face datasets.

2. **Species-specific challenges**: Macaques have less facial variation than humans/chimpanzees, making individual identification harder. Fur patterns, scars, and facial proportions are key discriminative features.

3. **Field conditions remain challenging**: Occlusion (by vegetation, other animals), variable lighting, motion blur, and extreme poses are common in field settings but underrepresented in training data.

4. **Integration of detection + identification**: Most papers treat detection and identification separately. End-to-end systems that work on raw field images/video are rare.

5. **SAM 3 for primate face segmentation**: No published work directly applies SAM 3 to primate face segmentation. This represents a **novel contribution opportunity** but requires careful validation.

---

## 8. Summary Table

| Paper | Species | Method | Accuracy | Dataset Size |
|-------|---------|--------|----------|-------------|
| Shukla et al. 2018 (MFID) | Rhesus macaque | CNN + pairwise loss | SOTA (multiple protocols) | 93 individuals |
| Shukla et al. 2019 (PFID) | Macaque + chimp | CNN + PFID loss | SOTA | Multi-species |
| Paulet et al. 2023 | Japanese macaque | Faster R-CNN + YOLOv8n | 82-83% | Kōjima population |
| Freytag/CVJena | Chimpanzee | Various | Benchmark | Public dataset |
| Deb & Jain 2020 (PrimNet) | Multi-primate | Modified ResNet + metric learning | SOTA | Multi-species |
| Čermák et al. 2023 | Multi-species | Toolkit (various methods) | N/A | Toolkit |

---

## References

1. Shukla, A., Cheema, G.S., Anand, S., Qureshi, Q., & Jhala, Y. (2018). Unique Identification of Macaques for Population Monitoring and Control. arXiv:1811.00743
2. Shukla, A., Cheema, G.S., Anand, S., Qureshi, Q., & Jhala, Y. (2019). Primate Face Identification in the Wild. PRICAI 2019. arXiv:1907.02642
3. Paulet, J., Molina, A., Beltzung, B., Suzumura, T., Yamamoto, S., & Sueur, C. (2023). Deep Learning for Automatic Detection and Facial Recognition in Japanese Macaques. arXiv:2310.06489
4. Kirillov, A., et al. (2023). Segment Anything. arXiv:2304.02643
5. Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. Meta AI.
6. Carion, N., Gustafson, L., Hu, Y.-T., et al. (2025). SAM 3: Segment Anything with Concepts. Meta AI.
7. Wasmuht, D.F., et al. (2025). The SA-FARI Dataset: Segment Anything in Footage of Animals. arXiv (Nov 2025).
8. Sapkota, R., et al. (2025). The SAM2-to-SAM3 Gap in the Segment Anything Model Family. arXiv (Dec 2025).
9. Čermák, V., et al. (2023). WildlifeDatasets: An open-source toolkit for animal re-identification. arXiv.
10. Igaue, T., et al. (2025). Motion Transfer-Enhanced StyleGAN for Generating Diverse Macaque Facial Expressions. arXiv (Nov 2025).
