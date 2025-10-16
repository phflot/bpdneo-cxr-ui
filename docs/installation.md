# Installation

## Requirements

- Python 3.10 or later
- Windows, macOS, or Linux
- (Optional) CUDA-compatible GPU for faster training

## Installation Methods

### From Source (Recommended for Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FlowRegSuite/bpdneo-cxr-ui.git
   cd bpdneo-cxr-ui
   ```

2. **Create a conda environment** (recommended):
   ```bash
   conda create --name bpdneo python=3.10
   conda activate bpdneo
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

### Using pip (Future)

Once published to PyPI:

```bash
pip install bpdneo-cxr-ui
```

## Verify Installation

Test that the package is correctly installed:

```python
import bpd_ui
from bpd_ui.models import list_available_models

# List available models
models = list_available_models()
for name, info in models.items():
    print(f"{name}: AUROC {info['auroc']:.3f}")
```

Expected output:
```
bpd_xrv_progfreeze_lp_cutmix: AUROC 0.783
bpd_xrv_progfreeze: AUROC 0.775
bpd_xrv_fullft: AUROC 0.761
bpd_rgb_progfreeze: AUROC 0.717
```

## Troubleshooting

### ImportError: No module named 'bpd_ui'

Make sure you've installed the package:
```bash
pip install -e .  # From the repository root
```

### PyQt/PySide6 Issues

If you encounter GUI-related errors, reinstall PySide6:
```bash
pip uninstall PySide6
pip install PySide6
```

### CUDA/GPU Issues

The application runs on CPU by default. GPU is not required for inference.

## Next Steps

- {doc}`quickstart` - Run your first prediction
- {doc}`user_guide/preprocessing` - Learn about the preprocessing workflow
