"""
Manifest State Management
==========================

This module provides functions for managing the Excel manifest file that tracks
all preprocessed images and their metadata in the BPDneo-CXR dataset structure.

The manifest serves as the single source of truth for:
- Which images have been preprocessed
- Location of images and ROI masks
- Operator grade labels (4-level: no_bpd, mild, moderate, severe)
- Preprocessing metadata (rectification, ROI method)
- Dataset split assignments

Functions
---------
manifest_path
    Construct full path to manifest.xlsx
read_manifest
    Load manifest as DataFrame (creates empty if missing)
upsert_manifest
    Insert or update a manifest row by (patient_id, image_relpath) key

Notes
-----
The manifest uses an append/update pattern keyed by (patient_id, image_relpath).
Updates are in-place, maintaining row order when possible.

See Also
--------
bpd_ui.ui.preprocess_tab : UI that writes to manifest
bpd_ui.ui.dataset_eval_tab : UI that reads from manifest
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

MANIFEST = "prepared/manifest.xlsx"


def manifest_path(root: Path) -> Path:
    """
    Construct the full path to the manifest Excel file.

    Parameters
    ----------
    root : Path
        Root directory of the dataset

    Returns
    -------
    Path
        Full path to prepared/manifest.xlsx

    Examples
    --------
    >>> from pathlib import Path
    >>> from bpd_ui.core.state import manifest_path
    >>> root = Path("/data/bpdneo")
    >>> path = manifest_path(root)
    >>> print(path)
    /data/bpdneo/prepared/manifest.xlsx
    """
    return Path(root) / MANIFEST


def read_manifest(root: Path) -> pd.DataFrame:
    """
    Read the manifest Excel file, creating an empty DataFrame if missing.

    This function loads the manifest file if it exists, or returns an empty
    DataFrame with the correct schema if the manifest doesn't exist yet.

    Parameters
    ----------
    root : Path
        Root directory of the dataset

    Returns
    -------
    pd.DataFrame
        Manifest DataFrame with columns:
        - patient_id : str, patient identifier
        - random_id : str, randomized ID (for de-identification)
        - image_relpath : str, relative path to image (from root/prepared/)
        - roi_relpath : str, relative path to mask (from root/prepared/)
        - grade_label : str, operator grade (no_bpd, mild, moderate, severe)
        - preproc_json : str, JSON-encoded preprocessing metadata
        - eval_split : str, dataset split (e.g., 'train', 'test', 'holdout')
        - timestamp : str, ISO 8601 timestamp of last update

    Notes
    -----
    If the manifest file does not exist, returns an empty DataFrame with the
    correct column schema. This allows code to proceed without special cases
    for the first write.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bpd_ui.core.state import read_manifest
    >>> root = Path("/data/bpdneo")
    >>> df = read_manifest(root)
    >>> print(df.columns.tolist())
    ['patient_id', 'random_id', 'image_relpath', 'roi_relpath',
     'grade_label', 'preproc_json', 'eval_split', 'timestamp']
    >>>
    >>> # Filter by grade
    >>> severe_cases = df[df['grade_label'] == 'severe']

    See Also
    --------
    upsert_manifest : Update or insert manifest rows
    manifest_path : Get path to manifest file
    """
    p = manifest_path(root)
    if p.exists():
        return pd.read_excel(p)
    cols = [
        "patient_id",
        "random_id",
        "image_relpath",
        "roi_relpath",
        "grade_label",
        "preproc_json",
        "eval_split",
        "timestamp",
    ]
    return pd.DataFrame(columns=cols)


def upsert_manifest(root: Path, row: dict) -> None:
    """
    Insert or update a row in the manifest by (patient_id, image_relpath) key.

    This function implements an upsert (update or insert) pattern. If a row
    with the same (patient_id, image_relpath) key already exists, it updates
    that row. Otherwise, it appends a new row. The timestamp is automatically
    set to the current UTC time.

    Parameters
    ----------
    root : Path
        Root directory of the dataset
    row : dict
        Row data dictionary with at least 'patient_id' and 'image_relpath'.
        Should contain all manifest columns:
        - patient_id : str
        - random_id : str (can be empty string)
        - image_relpath : str
        - roi_relpath : str
        - grade_label : str
        - preproc_json : str (JSON-encoded metadata)
        - eval_split : str

    Notes
    -----
    This function:
    1. Reads the existing manifest
    2. Looks for existing row by (patient_id, image_relpath)
    3. Automatically sets timestamp to datetime.utcnow().isoformat()
    4. Updates existing row or appends new row
    5. Creates parent directory if needed
    6. Writes manifest back to Excel

    The composite key (patient_id, image_relpath) allows multiple images per
    patient while ensuring each image is tracked uniquely.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bpd_ui.core.state import upsert_manifest
    >>> import json
    >>>
    >>> root = Path("/data/bpdneo")
    >>> row = {
    ...     "patient_id": "P001",
    ...     "random_id": "",
    ...     "image_relpath": "P001/images/xray_001.jpg",
    ...     "roi_relpath": "P001/masks/xray_001.png",
    ...     "grade_label": "moderate",
    ...     "preproc_json": json.dumps({"rectified": True, "roi": "auto"}),
    ...     "eval_split": "train"
    ... }
    >>> upsert_manifest(root, row)
    >>>
    >>> # Update grade for same image
    >>> row["grade_label"] = "severe"
    >>> upsert_manifest(root, row)  # Updates existing row

    See Also
    --------
    read_manifest : Read the manifest DataFrame
    """
    df = read_manifest(root)
    key = (row["patient_id"], row["image_relpath"])
    mask = (df["patient_id"] == key[0]) & (df["image_relpath"] == key[1])
    row["timestamp"] = datetime.utcnow().isoformat()
    if mask.any():
        df.loc[mask, list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(p, index=False)
