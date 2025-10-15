from pathlib import Path
import json
import pandas as pd
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
)
from bpd_ui.core.state import read_manifest
from bpd_ui.core.model_manager import ModelManager
from bpd_ui.core.image_io import load_image_for_model
from bpd_ui.core.tasks import submit


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
        c.setLayout(hh)
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
        sigs = submit(self._eval_job, self.root, model_name)
        sigs.finished.connect(self._done)
        sigs.error.connect(self._err)

    def _eval_job(self, root: Path, model_name: str):
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
            prob, logit = self.mm.predict(model_name, img)
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "image_relpath": r["image_relpath"],
                    "prob": prob,
                    "logit": logit,
                    "label_bin": int(prob >= 0.5),
                }
            )
        outd = root / "prepared" / "eval"
        outd.mkdir(parents=True, exist_ok=True)
        pcsv = outd / "predictions.csv"
        pd.DataFrame(rows).to_csv(pcsv, index=False)
        mj = outd / "metrics.json"
        with open(mj, "w") as f:
            json.dump({"n": len(rows)}, f)
        return str(pcsv)

    def _done(self, path):
        self.pb.hide()
        self.run.setEnabled(True)
        self.out_label.setText(f"Saved: {path}")

    def _err(self, msg):
        self.pb.hide()
        self.run.setEnabled(True)
        self.out_label.setText(f"Error: {msg}")
