# Preprocessing Workflow

This guide covers the preprocessing workflow for preparing chest X-ray datasets with semi-automated ROI extraction and operator grading.

## Overview

The preprocessing workflow enables you to:

1. Import chest X-ray images (PNG, JPEG, or DICOM)
2. Extract regions of interest (ROIs) using interactive segmentation
3. Assign operator grades (4-level: no_bpd, mild, moderate, severe)
4. Build a structured dataset for batch evaluation

**Output**: Excel manifest + ROI masks in standardized folder structure.

## Prerequisites

- BPDneo-CXR installed ({doc}`../installation`)
- Source images organized by patient (any folder structure)
- Prepared directory created: `<root>/prepared/`

## Step-by-Step Guide

### 1. Launch the GUI

```bash
python apps/gui_app.py
```

Navigate to the **Preprocessing** tab.

### 2. Set Root Directory

1. Click **"Select Root Directory..."**
2. Choose your prepared folder (e.g., `C:/data/bpd_study/prepared/`)
3. The application will:
   - Create `prepared/` if it doesn't exist
   - Initialize `manifest.xlsx` if not present
   - Load existing manifest entries

### 3. Select or Create Patient

**Option A: New Patient**
1. Click **"New Patient"**
2. Enter patient ID (e.g., `PT001`)
3. Click **"Create"**

**Option B: Existing Patient**
1. Select from dropdown (populated from manifest)
2. Click **"Load"**

The patient folder structure will be created automatically:
```
prepared/
  PT001/
    images/
    masks/
```

### 4. Import Images

1. Click **"Import Images..."**
2. Select one or more image files:
   - Supported formats: PNG, JPEG, DCM (DICOM)
   - Multi-select enabled (Ctrl+Click or Shift+Click)
3. Images are copied to `PT001/images/` with standardized names

The image list updates to show all imported images for this patient.

### 5. Extract ROI for Each Image

For each image in the list:

#### 5.1 View Image

- Click image in the list to display in the viewer
- Use mouse wheel to zoom
- Click and drag to pan
- Double-click to fit to window

#### 5.2 Draw Seeds (Required)

1. Select **"Seed Mode"** from the toolbar
2. Click on the lung region to place foreground seeds
3. Place 3-5 seeds covering the chest area
4. Seeds appear as green circles

**Tips**:
- Place seeds in clearly visible lung tissue
- Avoid edges and artifact regions
- More seeds = more robust segmentation

#### 5.3 Draw Border (Optional)

1. Select **"Border Mode"** from the toolbar
2. Click to place polyline vertices around the chest
3. Double-click to close the polyline
4. Border appears as a blue line

**Use border when**:
- Automatic segmentation includes too much background
- Image has poor contrast at lung edges
- Need to exclude artifacts or tubes

#### 5.4 Run Segmentation

1. Click **"Extract ROI"**
2. Algorithm runs (GrabCut or Random Walker)
3. Resulting mask overlays on the image (semi-transparent)

**If the mask is incorrect**:
- Click **"Clear"** to remove seeds/border
- Redraw seeds with better placement
- Click **"Extract ROI"** again

#### 5.5 Assign Grade

1. Select grade from dropdown:
   - `no_bpd` - No bronchopulmonary dysplasia
   - `mild` - Mild BPD
   - `moderate` - Moderate BPD
   - `severe` - Severe BPD
2. This is the operator's annotation (ground truth)

**Note**: Model inference outputs binary classification (Moderate/Severe vs No/Mild), but operator grades are preserved in the manifest for analysis.

#### 5.6 Save

1. Click **"Save"**
2. The application:
   - Saves ROI mask to `PT001/masks/roi_<image_name>.png`
   - Updates `manifest.xlsx` with new entry:
     - `patient_id`, `image_relpath`, `roi_relpath`, `grade_label`, `timestamp`

3. Status bar confirms save: "Saved PT001/roi_001.png to manifest"

### 6. Assign Evaluation Split (Optional)

After processing all images for a patient:

1. Open `prepared/manifest.xlsx` in Excel
2. Add `eval_split` column if not present
3. Assign split labels:
   - `train` - Training set
   - `val` - Validation set
   - `test` - Test set

This enables filtering during dataset evaluation.

## ROI Extraction Methods

BPDneo-CXR supports two segmentation algorithms:

### GrabCut (Default)

**How it works**:
- Seeds define definite foreground
- Border (if provided) defines trimap boundary
- Iterative graph-cut refinement

**Best for**:
- Clear lung boundaries
- Moderate contrast images
- When you can draw a rough border

**Usage**:
```python
from bpd_ui.core import refine_with_scribbles

mask = refine_with_scribbles(
    image=img_array,
    seed_mask=seed_points,
    seed_points=seeds,
    method="grabcut"
)
```

### Random Walker

**How it works**:
- Seeds define labeled regions
- Random walk probabilities computed
- Border enforced as hard constraint

**Best for**:
- Low contrast images
- Complex boundaries
- When automatic detection fails

**Usage**:
```python
from bpd_ui.core import refine_with_scribbles

mask = refine_with_scribbles(
    image=img_array,
    seed_mask=seed_points,
    seed_points=seeds,
    method="random_walker"
)
```

## Automatic Chest Detection

For quick preprocessing of high-quality images:

1. Click **"Auto-Mask"** button
2. Automatic detection runs (Otsu thresholding + morphology)
3. Review the mask
4. If satisfactory, assign grade and save
5. If incorrect, clear and draw manual seeds

**Limitations**:
- Assumes centered chest X-ray
- May fail on rotated or cropped images
- Not robust to artifacts or overlay text

## Troubleshooting

### Issue: Segmentation includes too much background

**Solutions**:
- Draw a tighter border polyline
- Add more seeds in the center of the lung region
- Try the Random Walker method

### Issue: Segmentation excludes lung regions

**Solutions**:
- Add more seeds in excluded areas
- Remove or redraw the border
- Reduce the number of seeds (over-seeding can cause issues)

### Issue: DICOM images appear inverted or dark

**Solutions**:
- Use the "Invert" checkbox in the viewer
- Check DICOM PhotometricInterpretation tag
- Report issue with DICOM metadata

### Issue: Images are skewed or rotated

**Solution**:
- Enable **"Auto-Deskew"** checkbox
- Automatically corrects perspective distortion
- Uses Hough line detection to find edges

## Data Organization

After preprocessing, your directory structure should look like:

```
<root>/
  prepared/
    PT001/
      images/
        img_001.png
        img_002.png
      masks/
        roi_001.png
        roi_002.png
    PT002/
      images/
        img_001.png
      masks/
        roi_001.png
    manifest.xlsx
```

**Manifest columns**:
- `patient_id`: Patient identifier
- `random_id`: Anonymized ID (optional)
- `image_relpath`: Relative path from `prepared/` (e.g., `PT001/images/img_001.png`)
- `roi_relpath`: Relative path to mask
- `grade_label`: Operator annotation (`no_bpd`, `mild`, `moderate`, `severe`)
- `eval_split`: Dataset split (`train`, `val`, `test`)
- `timestamp`: Last modification timestamp

## Next Steps

- {doc}`dataset_evaluation` - Run batch inference on your dataset
- {doc}`single_image` - Test individual images
- {doc}`data_format` - Detailed data format specification
