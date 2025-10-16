# UI Module

The `bpd_ui.ui` module provides the PyQt6-based graphical user interface for BPDneo-CXR.

## Overview

The GUI is organized into three main tabs:

1. **Preprocessing Tab** - Semi-automated ROI extraction with operator grading
2. **Dataset Evaluation Tab** - Batch inference with metrics and visualizations
3. **Single Image Evaluation Tab** - End-to-end prediction on individual images

## Design Philosophy

The interface follows a "no wrong moves" guided workflow:

- **Clear state transitions**: Each step must complete before the next is enabled
- **Immediate validation**: Invalid inputs are caught before execution
- **Informative feedback**: Progress bars, status messages, and error dialogs
- **Undo-friendly**: Operations write to manifest/disk, allowing manual rollback

## Components

### Main Window

```{eval-rst}
.. autoclass:: bpd_ui.ui.main_window.MainWindow
   :members:
   :undoc-members:
   :show-inheritance:
```

### Preprocessing Tab

```{eval-rst}
.. autoclass:: bpd_ui.ui.preprocess_tab.PreprocessTab
   :members:
   :undoc-members:
   :show-inheritance:
```

**Workflow**:
1. Select root directory (`<root>/prepared/`)
2. Enter patient ID or select from manifest
3. Import images (multi-select: PNG, JPEG, DICOM)
4. For each image:
   - View image with zoom/pan
   - Draw ROI seeds (foreground points)
   - Optionally draw border polyline
   - Click "Extract ROI" to run segmentation
   - Select grade label from dropdown (`no_bpd`, `mild`, `moderate`, `severe`)
   - Click "Save" to write mask and update manifest
5. Move to next image

**ROI Tools**:
- Seed mode: Click to add foreground points
- Border mode: Click to add polyline vertices, double-click to close
- Clear: Remove current annotation
- Auto-mask: Run automatic chest detection

### Dataset Evaluation Tab

```{eval-rst}
.. autoclass:: bpd_ui.ui.dataset_eval_tab.DatasetEvalTab
   :members:
   :undoc-members:
   :show-inheritance:
```

**Workflow**:
1. Load manifest (`<root>/prepared/manifest.xlsx`)
2. Select evaluation split (`train`, `val`, `test`, or `all`)
3. Select model from dropdown
4. Click "Run Evaluation"
5. Progress bar updates during batch inference
6. Results saved to `<root>/prepared/eval/<timestamp>/`:
   - `predictions.csv` - Per-image predictions
   - `metrics.json` - Computed metrics (AUROC, accuracy, etc.)
   - `metrics.txt` - Human-readable report
   - `plots/roc_curve.png` - ROC curve
   - `plots/confusion_matrix.png` - Confusion matrix

**Metrics Displayed**:
- AUROC, AUPRC
- Accuracy, sensitivity, specificity
- Positive/negative predictive values
- Confusion matrix
- ROC curve

### Single Image Evaluation Tab

```{eval-rst}
.. autoclass:: bpd_ui.ui.single_eval_tab.SingleEvalTab
   :members:
   :undoc-members:
   :show-inheritance:
```

**Workflow**:
1. Click "Select Image..." to open file dialog
2. Select model from dropdown (default: `bpd_xrv_progfreeze_lp_cutmix`)
3. (Optional) Draw ROI seeds to restrict inference region
4. Click "Run Inference"
5. View results:
   - Binary label: "Moderate/Severe BPD" or "No/Mild BPD"
   - Probability scores for both classes

**Features**:
- Supports PNG, JPEG, and DICOM images
- Automatic preprocessing (resize, normalize)
- Instant prediction (< 1 second on CPU)
- No dataset setup required

### Overlay Canvas

```{eval-rst}
.. autoclass:: bpd_ui.ui.overlay_canvas.OverlayCanvas
   :members:
   :undoc-members:
   :show-inheritance:
```

**Features**:
- Interactive image canvas with mask overlay
- Annotation tools for ROI extraction
- Zoom/pan with mouse and keyboard shortcuts

## Styling

The application uses QSS (Qt Style Sheets) for consistent theming:

```python
from PySide6.QtWidgets import QApplication
from pathlib import Path

app = QApplication([])

# Load custom stylesheet
qss_path = Path(__file__).parent / "resources" / "qss" / "main.qss"
with open(qss_path) as f:
    app.setStyleSheet(f.read())
```

## Usage Examples

### Launching the GUI

```python
from bpd_ui.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
```

### Programmatic Tab Access

```python
from bpd_ui.ui.main_window import MainWindow

window = MainWindow()

# Access tabs
preprocess_tab = window.tab_widget.widget(0)
dataset_tab = window.tab_widget.widget(1)
single_tab = window.tab_widget.widget(2)

# Set active tab
window.tab_widget.setCurrentIndex(2)  # Single image tab
```

### Using Overlay Canvas

```python
from bpd_ui.ui.overlay_canvas import OverlayCanvas
from PySide6.QtGui import QPixmap

# Create canvas
canvas = OverlayCanvas()
canvas.set_image(QPixmap("xray.jpg"))

# Enable annotation mode
canvas.set_annotation_mode(True)
```

## Threading

All long-running operations (inference, batch evaluation, ROI extraction) run in background threads using the task system from `bpd_ui.core.tasks`:

```python
from bpd_ui.core import submit

def run_batch_inference():
    def work(progress_callback):
        for i, image_path in enumerate(image_paths):
            # ... process image ...
            progress_callback.emit(int(100 * i / len(image_paths)))
        return results

    task = submit(
        work,
        on_success=self.display_results,
        on_error=self.handle_error
    )
```

**Benefits**:
- Responsive UI during long operations
- Progress reporting via signals
- Graceful error handling
- Clean separation of UI and business logic
