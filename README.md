# <img src="https://raw.githubusercontent.com/phflot/bpdneo-cxr-ui/77d3e0839be2403be5cbf37b72f4f4b12f9a5ed8/img/icon.png" alt="BPDneo logo" height="64"> BPDneo‑CXR UI

**Graphical user interface and batch tools for evaluating BPDneo chest‑X‑ray models.**
The app enables non‑expert users to prepare datasets with guided ROI steps and to run reproducible, CPU‑only inference on single images or entire patient cohorts.

![Abstract](https://raw.githubusercontent.com/phflot/bpdneo-cxr-ui/77d3e0839be2403be5cbf37b72f4f4b12f9a5ed8/img/abstract.png)

> **Model output** is binary as defined by the current checkpoints: probability of **Moderate/Severe BPD** vs **No/Mild BPD**. The UI records four‑grade operator labels during preprocessing for downstream use.

## Why this UI

- **Stepwise, "no‑wrong‑moves" preprocessing.** Seed‑first ROI extraction with optional border points, explicit grade entry, and manifest‑only persistence.
- **Deterministic inference.** The app calls the exact transforms and loaders used by the research code (TorchXRayVision or ImageNet pipelines), so results match the reference examples.
- **Portable.** CPU‑only, PyInstaller bundles (Win/macOS) and a Docker CLI for batch runs.

## Modes

1. **Preprocessing**
   - Select root directory and a patient ID (dropdown, free text, or random ID).
   - Import DICOM/PNG/JPG images.
   - Extract ROI with **seed points** (mandatory) and **border points** (optional).
   - Assign grade: `no_bpd`, `mild`, `moderate`, `severe`.
   - Save: writes `prepared/<patient_id>/images/`, `prepared/<patient_id>/masks/`, and `prepared/manifest.xlsx`.

2. **Dataset Evaluation**
   - Read `prepared/manifest.xlsx`, filter rows with ROI.
   - For each image: apply the model's training‑consistent preprocessing, run inference, and write `predictions.csv`.
   - Save basic metrics and plots (binary scope).
   - Runs off‑thread with progress reporting.

3. **Single Image Evaluation**
   - Load one X‑ray (DICOM/PNG/JPG).
   - Select a model from the built‑in registry.
   - Run inference and view probability + binary label; export results.

## Data layout

```
<root>/
  prepared/
    <patient_id>/
      images/  # input copies
      masks/   # ROI masks (png)
  prepared/manifest.xlsx
```

**Manifest columns:**
`patient_id, random_id, image_relpath, roi_relpath, grade_label, preproc_json, eval_split, timestamp`

## Models and preprocessing

- Models are loaded by name from a registry and run in `eval()` mode.
- Preprocessing strictly follows the model's training recipe:
  - **XRV-backed models:** grayscale normalization → tensor → resize to 512×512.
  - **ImageNet-backed models:** resize → RGB → tensor → ImageNet mean/std.
- Inference: forward pass → `sigmoid` → probability of Moderate/Severe.
- The Single‑image and Dataset modes both call the same code path.

### Available Models

Pre-trained models for BPD prediction. All models use ResNet-50 architecture with different initialization and training strategies. AUROC was computed using repeated 5-fold cross-validation.

| Model | Description | AUROC | Download |
|-------|-------------|-------|----------|
| **bpd_xrv_progfreeze_lp_cutmix** | Best performing model with XRV pretraining, progressive freezing, linear probing, and CutMix | 0.783 | [Download](https://cloud.hiz-saarland.de/public.php/dav/files/nLYMSE8jRSg3j8j) |
| **bpd_xrv_progfreeze** | Baseline with XRV pretraining and progressive freezing (no augmentation) | 0.775 | [Download](https://cloud.hiz-saarland.de/public.php/dav/files/SRxGJzLSpEMMAD4) |
| **bpd_xrv_fullft** | XRV pretraining with full fine-tuning (no freezing) | 0.761 | [Download](https://cloud.hiz-saarland.de/public.php/dav/files/w2czAo4oYxFaAGi) |
| **bpd_rgb_progfreeze** | ImageNet baseline with progressive freezing (for comparison) | 0.717 | [Download](https://cloud.hiz-saarland.de/public.php/dav/files/W7EmnFDSFwoFSBL) |


## Install (development)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -U pip
pip install -e .
```

Run the GUI:

```bash
python apps/gui_app.py
```

## Bundles and Docker

- **PyInstaller bundles:** see GitHub Releases (Win and macOS). Double‑click to run.
- **Docker (CLI only):**

```bash
docker run --rm -v "$PWD":/data ghcr.io/<owner>/bpdneo-cxr-ui:latest \
  eval-dataset --root /data --model bpd_xrv_progfreeze_lp_cutmix
```

## Notes

- This software is for research use. It is not a medical device.
- Output is **binary** per current checkpoints (Moderate/Severe vs No/Mild).

## Citation

If you use this application or the associated models in research, please cite:

> Goedicke‑Fritz S., Bous M., Engel A., Flotho M., Hirsch P., Wittig H., Milanovic D., Mohr D., Kaspar M., Nemat S., Kerner D., Bücker A., Keller A., Meyer S., Zemlin M., Flotho P.
> *Site‑Level Fine‑Tuning with Progressive Layer Freezing: Towards Robust Prediction of Bronchopulmonary Dysplasia from Day‑1 Chest Radiographs in Extremely Preterm Infants.* arXiv:2507.12269 (2025).

BibTeX entry:
```bibtex
@article{goedicke2025site,
  title={Site-Level Fine-Tuning with Progressive Layer Freezing: Towards Robust Prediction of Bronchopulmonary Dysplasia from Day-1 Chest Radiographs in Extremely Preterm Infants},
  author={Goedicke-Fritz, Sybelle and Bous, Michelle and Engel, Annika and Flotho, Matthias and Hirsch, Pascal and Wittig, Hannah and Milanovic, Dino and Mohr, Dominik and Kaspar, Mathias and Nemat, Sogand and Kerner, Dorothea and Bücker, Arno and Keller, Andreas and Meyer, Sascha and Zemlin, Michael and Flotho, Philipp},
  journal={arXiv preprint arXiv:2507.12269},
  year={2025}
}
```
