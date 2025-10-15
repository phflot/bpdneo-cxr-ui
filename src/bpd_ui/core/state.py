from pathlib import Path
from datetime import datetime
import pandas as pd

MANIFEST = "prepared/manifest.xlsx"


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST


def read_manifest(root: Path) -> pd.DataFrame:
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
