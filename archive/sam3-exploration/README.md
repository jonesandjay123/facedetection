# SAM3 Exploration — Phase 1

This directory contains the original SAM3-based detection exploration conducted by Eleane (趙以琳) and Jones.

## Summary

Phase 1 explored whether Meta's Segment Anything Model 3 (SAM3) could be used for zero-shot primate face detection. Key findings across three test scenarios:

| Scenario | Description | Result |
|----------|-------------|--------|
| S1 | Single monkey, clean background | Good segmentation quality |
| S2 | Multiple monkeys, moderate occlusion | Acceptable detection, some missed faces |
| S3 | Field conditions (mixed species, clutter) | Challenging — prompt engineering insufficient for reliable detection |

## Conclusion

SAM3 provides strong general segmentation but lacks the precision needed for consistent primate face detection without fine-tuning. This motivated the pivot to a multi-stage pipeline approach combining dedicated detectors (YOLO) with specialized embedding models.

## Files

- `detector.py` — Core detection pipeline
- `sam3_wrapper.py` — SAM3 model loading & inference
- `face_filter.py` — Human vs monkey face classification
- `visualizer.py` — Result visualization
- `notebooks/` — Interactive exploration notebooks
- `docs/` — Research notes and literature review
