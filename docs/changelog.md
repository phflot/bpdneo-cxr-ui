# Changelog

All notable changes to BPDneo-CXR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive Sphinx documentation with MyST Parser
- API reference for models, core, and UI modules
- User guides for preprocessing, dataset evaluation, and single image workflows
- Read the Docs configuration
- NumPy-style docstrings across all modules

## [0.1.0] - 2025-01-15

### Added
- Initial release of BPDneo-CXR
- Desktop GUI with three workflow tabs:
  - Preprocessing: Semi-automated ROI extraction with operator grading
  - Dataset Evaluation: Batch inference with binary classification metrics
  - Single Image Evaluation: End-to-end prediction on individual images
- Four pretrained binary classification models:
  - `bpd_xrv_progfreeze_lp_cutmix` (AUROC 0.783)
  - `bpd_xrv_progfreeze` (AUROC 0.775)
  - `bpd_xrv_fullft` (AUROC 0.761)
  - `bpd_rgb_progfreeze` (AUROC 0.717)
- Support for PNG, JPEG, and DICOM image formats
- Excel-based manifest for dataset management
- Automatic model weight downloading from Google Drive
- ROI extraction with GrabCut and Random Walker algorithms
- Binary classification metrics (AUROC, AUPRC, sensitivity, specificity, etc.)
- ROC curve and confusion matrix visualizations
- Model caching via `ModelManager`
- Asynchronous task execution with Qt thread pool
- DICOM metadata handling with auto-inversion
- Perspective correction for scanned images

### Model Backend
- `BPDModel` class for unified ResNet50-based architecture
- Support for TorchXRayVision (XRV) and ImageNet pretraining
- Progressive freezing training strategy
- Binary classification head with BCEWithLogitsLoss
- XRV and ImageNet preprocessing pipelines

### Core Features
- Manifest operations: `read_manifest()`, `upsert_manifest()`
- Image I/O: `load_image_for_display()`, `load_image_for_model()`
- ROI service: `auto_chest_mask()`, `refine_with_scribbles()`
- Metrics computation: `compute_binary_metrics()`, `metrics_to_text()`
- Task system for background operations

### UI Components
- `MainWindow` with three-tab interface
- `PreprocessTab` for guided ROI extraction workflow
- `DatasetEvalTab` for batch evaluation with metrics
- `SingleImageEvalTab` for instant single-image prediction
- Image viewer with zoom, pan, and annotation overlay
- Annotation tools for seed and border drawing

## [0.0.1] - 2025-01-10

### Added
- Project repository initialized
- Model code and architecture implementation
- README with project overview
- Basic requirements.txt
- DICOM support for image loading

---

## Version Naming Convention

- **Major version** (X.0.0): Incompatible API changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

## Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

[Unreleased]: https://github.com/FlowRegSuite/bpdneo-cxr-ui/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FlowRegSuite/bpdneo-cxr-ui/releases/tag/v0.1.0
[0.0.1]: https://github.com/FlowRegSuite/bpdneo-cxr-ui/releases/tag/v0.0.1
