from pathlib import Path
import numpy as np
from PIL import Image


def load_image_for_display(path: str | Path):
    p = str(path)
    if p.lower().endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        return img
    return Image.open(p).convert("RGB")


def load_image_for_model(path: str | Path):
    p = str(path)
    if p.lower().endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(p)
        arr = ds.pixel_array
        return Image.fromarray(arr)
    return Image.open(p)
