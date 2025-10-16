# BPDneo-CXR Documentation

**Early BPD Prediction from Chest X-rays**

BPDneo-CXR is a desktop GUI application for early prediction of bronchopulmonary dysplasia (BPD) in extremely low birth weight infants using chest X-rays obtained within 24 hours of birth.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/preprocessing
user_guide/dataset_evaluation
user_guide/single_image
user_guide/data_format
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/models
api/core
api/ui
```

```{toctree}
:maxdepth: 1
:caption: Additional Resources

changelog
```

## Features

- **Three workflow modes**:
  - Preprocessing: Semi-automated ROI extraction with operator grading
  - Dataset Evaluation: Batch inference with comprehensive metrics
  - Single Image: End-to-end prediction on individual images

- **Pretrained models**: Four validated models with AUROC up to 0.783
- **Multiple formats**: Support for PNG, JPEG, and DICOM images
- **Binary classification**: Predicts Moderate/Severe vs No/Mild BPD

## Quick Links

- {doc}`installation` - Get started with installation
- {doc}`quickstart` - Your first BPD prediction
- {doc}`api/index` - Complete API reference
- [GitHub Repository](https://github.com/FlowRegSuite/bpdneo-cxr-ui)

## Methodology

The pretrained models in BPDneo-CXR were developed using site-level fine-tuning with progressive layer freezing {cite}`goedicke2025site`. This approach enables robust prediction of BPD from day-1 chest radiographs in extremely preterm infants (gestational age < 28 weeks) while maintaining excellent generalization across multiple clinical sites.

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

## Bibliography

```{bibliography}
```
