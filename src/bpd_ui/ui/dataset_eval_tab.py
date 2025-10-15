from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QGroupBox,
    QComboBox,
    QProgressBar,
    QCheckBox,
)
from bpd_ui.core.state import read_manifest
from bpd_ui.core.model_manager import ModelManager
from bpd_ui.core.image_io import load_image_for_model
from bpd_ui.core.tasks import submit
from bpd_ui.core.roi_service import bbox_from_mask
from bpd_ui.core.metrics import compute_binary_metrics, metrics_to_text


class DatasetEvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = None
        self.mm = ModelManager("cpu")
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        g = QGroupBox("Dataset")
        h = QHBoxLayout()
        self.root_label = QLabel("No root")
        b = QPushButton("Choose Root…")
        b.clicked.connect(self._choose_root)
        h.addWidget(self.root_label, 1)
        h.addWidget(b)
        g.setLayout(h)
        v.addWidget(g)

        c = QGroupBox("Run")
        vv = QVBoxLayout()
        hh = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItem("bpd_xrv_progfreeze_lp_cutmix")
        self.run = QPushButton("Evaluate")
        self.run.clicked.connect(self._run_eval)
        self.pb = QProgressBar()
        self.pb.setRange(0, 0)
        self.pb.hide()
        hh.addWidget(QLabel("Model:"))
        hh.addWidget(self.model_combo, 1)
        hh.addWidget(self.run)
        hh.addWidget(self.pb)
        vv.addLayout(hh)
        self.use_roi_check = QCheckBox("Use ROI (crop to mask bbox)")
        self.use_roi_check.setChecked(True)
        vv.addWidget(self.use_roi_check)
        c.setLayout(vv)
        v.addWidget(c)

        self.out_label = QLabel("")
        v.addWidget(self.out_label)

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Root")
        if d:
            self.root = Path(d)
            self.root_label.setText(d)

    def _run_eval(self):
        if not self.root:
            return
        self.run.setEnabled(False)
        self.pb.show()
        model_name = self.model_combo.currentText()
        use_roi = self.use_roi_check.isChecked()
        sigs = submit(self._eval_job, self.root, model_name, use_roi)
        sigs.finished.connect(self._done)
        sigs.error.connect(self._err)

    def _eval_job(self, root: Path, model_name: str, use_roi: bool):
        df = read_manifest(root)
        df = df[df["roi_relpath"].notnull()]
        rows = []
        for _, r in df.iterrows():
            ip = (
                root
                / "prepared"
                / r["patient_id"]
                / "images"
                / Path(str(r["image_relpath"])).name
            )
            img = load_image_for_model(ip)
            if use_roi and pd.notna(r["roi_relpath"]):
                mp = (
                    root
                    / "prepared"
                    / r["patient_id"]
                    / "masks"
                    / Path(str(r["roi_relpath"])).name
                )
                if mp.exists():
                    m = Image.open(mp).convert("L")
                    box = bbox_from_mask(np.array(m))
                    img = img.crop((box[0], box[1], box[2], box[3]))
            prob, logit = self.mm.predict(model_name, img)
            row_data = {
                "patient_id": r["patient_id"],
                "image_relpath": r["image_relpath"],
                "prob": prob,
                "logit": logit,
                "label_bin": int(prob >= 0.5),
            }
            if "grade_label" in r and pd.notna(r["grade_label"]):
                grade = r["grade_label"]
                gt_bin = (
                    1
                    if grade in ["moderate", "severe"]
                    else 0
                    if grade in ["no_bpd", "mild"]
                    else None
                )
                row_data["grade_label"] = grade
                if gt_bin is not None:
                    row_data["ground_truth_bin"] = gt_bin
            rows.append(row_data)
        outd = root / "prepared" / "eval"
        outd.mkdir(parents=True, exist_ok=True)
        pcsv = outd / "predictions.csv"
        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(pcsv, index=False)

        metrics_data = {"n": len(rows)}
        if "ground_truth_bin" in pred_df.columns:
            valid = pred_df[pred_df["ground_truth_bin"].notna()]
            if len(valid) > 0:
                y_true = valid["ground_truth_bin"].values
                y_prob = valid["prob"].values
                metrics_data = compute_binary_metrics(y_true, y_prob)

        mj = outd / "metrics.json"
        with open(mj, "w") as f:
            json.dump(metrics_data, f, indent=2)

        mt = outd / "metrics.txt"
        with open(mt, "w") as f:
            f.write(metrics_to_text(metrics_data))

        return str(pcsv)

    def _done(self, path):
        self.pb.hide()
        self.run.setEnabled(True)
        self.out_label.setText(f"Saved: {path}")

    def _err(self, msg):
        self.pb.hide()
        self.run.setEnabled(True)
        self.out_label.setText(f"Error: {msg}")
