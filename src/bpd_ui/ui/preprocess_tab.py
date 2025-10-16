from pathlib import Path
from shutil import copy2
import json
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QGroupBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QDialog,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from bpd_ui.core.image_io import load_image_for_display
from bpd_ui.core.state import upsert_manifest
from bpd_ui.core.roi_service import auto_chest_mask, refine_with_scribbles
from bpd_ui.core.deskew import rectify_if_scan
from bpd_ui.core.tasks import submit
from .overlay_canvas import OverlayCanvas


class PreprocessTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = None
        self.pid = None
        self.paths = []
        self.idx = -1
        self.mask_np = None
        self.display_pix = None
        self.current_pil = None
        self.current_path = None
        self._rectified = False
        self._build()

    def _build(self):
        L = QHBoxLayout(self)
        self.canvas = OverlayCanvas(self)
        self.canvas.setMinimumSize(700, 700)
        L.addWidget(self.canvas, 3)

        R = QVBoxLayout()
        g = QGroupBox("Dataset")
        h = QHBoxLayout()
        self.root_lbl = QLabel("No root")
        b = QPushButton("Choose…")
        b.clicked.connect(self._choose_root)
        h.addWidget(self.root_lbl, 1)
        h.addWidget(b)
        g.setLayout(h)
        R.addWidget(g)

        g2 = QGroupBox("Patient")
        h2 = QHBoxLayout()
        self.pid_combo = QComboBox()
        self.pid_combo.setEditable(True)
        self.pid_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_patient_id)
        h2.addWidget(QLabel("ID:"))
        h2.addWidget(self.pid_combo, 1)
        h2.addWidget(add_btn)
        g2.setLayout(h2)
        R.addWidget(g2)

        g3 = QGroupBox("Files")
        v3 = QVBoxLayout()
        self.listw = QListWidget()
        v3.addWidget(self.listw)
        h3 = QHBoxLayout()
        folder_btn = QPushButton("Choose Folder…")
        folder_btn.clicked.connect(self._choose_folder)
        add_btn = QPushButton("Add Images…")
        add_btn.clicked.connect(self._add_files)
        h3.addWidget(folder_btn)
        h3.addWidget(add_btn)
        v3.addLayout(h3)
        g3.setLayout(v3)
        R.addWidget(g3, 1)

        g4 = QGroupBox("Grade")
        h4 = QHBoxLayout()
        self.grade = QComboBox()
        [
            self.grade.addItem(x)
            for x in ["no_bpd", "mild", "moderate", "severe", "unknown"]
        ]
        h4.addWidget(self.grade, 1)
        g4.setLayout(h4)
        R.addWidget(g4)

        ops = QHBoxLayout()
        self.auto_btn = QPushButton("Auto‑ROI")
        self.auto_btn.clicked.connect(self._auto)
        self.seed_btn = QPushButton("Seed")
        self.seed_btn.clicked.connect(lambda: self.canvas.set_mode("seed"))
        self.border_btn = QPushButton("Border")
        self.border_btn.clicked.connect(lambda: self.canvas.set_mode("border"))
        self.refine_btn = QPushButton("Refine")
        self.refine_btn.clicked.connect(self._refine)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        for b in [
            self.auto_btn,
            self.seed_btn,
            self.border_btn,
            self.refine_btn,
            self.clear_btn,
            self.save_btn,
        ]:
            ops.addWidget(b)
        R.addLayout(ops)

        R.addStretch(1)
        L.addLayout(R, 2)
        self.listw.currentRowChanged.connect(self._show_idx)

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Root")
        if d:
            self.root = Path(d)
            self.root_lbl.setText(d)
            self._load_existing_patient_ids()

    def _load_existing_patient_ids(self):
        if not self.root:
            return
        manifest_path = self.root / "prepared" / "manifest.xlsx"
        if manifest_path.exists():
            try:
                from bpd_ui.core.state import read_manifest

                df = read_manifest(self.root)
                if "patient_id" in df.columns:
                    existing_ids = df["patient_id"].unique().tolist()
                    existing_ids.sort()
                    self.pid_combo.clear()
                    self.pid_combo.addItems([str(pid) for pid in existing_ids])
            except Exception:
                pass

    def _get_next_numeric_id(self):
        max_id = 0
        for i in range(self.pid_combo.count()):
            pid = self.pid_combo.itemText(i)
            try:
                num = int(pid)
                max_id = max(max_id, num)
            except ValueError:
                continue
        return str(max_id + 1)

    def _add_patient_id(self):
        next_id = self._get_next_numeric_id()
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Patient ID")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Patient ID:"))
        id_edit = QLineEdit(next_id)
        layout.addWidget(id_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_id = id_edit.text().strip()
            if new_id:
                idx = self.pid_combo.findText(new_id)
                if idx == -1:
                    self.pid_combo.addItem(new_id)
                    self.pid_combo.setCurrentText(new_id)
                else:
                    self.pid_combo.setCurrentIndex(idx)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        folder_path = Path(folder)
        image_extensions = {".png", ".jpg", ".jpeg", ".dcm"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if not image_files:
            return

        self.listw.clear()
        for f in image_files:
            it = QListWidgetItem(f.name)
            it.setData(Qt.ItemDataRole.UserRole, str(f))
            self.listw.addItem(it)

        if self.listw.count() > 0:
            self.listw.setCurrentRow(0)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.dcm)"
        )
        if not files:
            return
        for f in files:
            it = QListWidgetItem(Path(f).name)
            it.setData(Qt.ItemDataRole.UserRole, f)
            self.listw.addItem(it)
        if self.listw.count() > 0 and self.idx < 0:
            self.listw.setCurrentRow(0)

    def _show_idx(self, i):
        self.idx = i
        if i < 0:
            return
        p = self.listw.item(i).data(Qt.ItemDataRole.UserRole)
        self.current_path = Path(p)
        disp = load_image_for_display(p)
        disp, self._rectified = self._rectify_if_needed(disp, p)
        self.current_pil = disp
        arr = np.array(disp)
        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.display_pix = QPixmap.fromImage(qimg)
        self.canvas.set_image(self.display_pix)
        self.canvas.set_mask(None)
        self.mask_np = None

    def _rectify_if_needed(self, pil_img, path):
        changed = False
        if str(path).lower().endswith((".png", ".jpg", ".jpeg")):
            import cv2

            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            warp = rectify_if_scan(bgr)
            changed = warp is not bgr and warp.shape != bgr.shape
            return Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)), changed
        return pil_img, False

    def _auto(self):
        if self.idx < 0 or self.current_pil is None:
            return
        self.auto_btn.setEnabled(False)
        self.auto_btn.setText("Processing...")
        img = self.current_pil

        def job(img_arg):
            return auto_chest_mask(img_arg)

        sigs = submit(job, img)
        sigs.finished.connect(self._on_auto_done)
        sigs.error.connect(self._on_error)

    def _on_auto_done(self, m):
        self.mask_np = m
        m = np.ascontiguousarray(m)
        h, w = m.shape
        qmask = QImage(m.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self.canvas.set_mask(qmask)
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("Auto‑ROI")

    def _refine(self):
        if self.idx < 0 or self.mask_np is None or self.current_pil is None:
            return
        self.refine_btn.setEnabled(False)
        self.refine_btn.setText("Processing...")
        gray = np.array(ImageOps.grayscale(self.current_pil))
        sx = self.canvas.seeds_img
        bx = self.canvas.border_img if len(self.canvas.border_img) > 2 else None

        def job(gray_arg, seeds, border, init_mask):
            return refine_with_scribbles(gray_arg, seeds, border, init_mask)

        sigs = submit(job, gray, sx, bx, self.mask_np)
        sigs.finished.connect(self._on_refine_done)
        sigs.error.connect(self._on_error)

    def _on_refine_done(self, m):
        self.mask_np = m
        m = np.ascontiguousarray(m)
        h, w = m.shape
        qmask = QImage(m.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self.canvas.set_mask(qmask)
        self.refine_btn.setEnabled(True)
        self.refine_btn.setText("Refine")

    def _on_error(self, msg):
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("Auto‑ROI")
        self.refine_btn.setEnabled(True)
        self.refine_btn.setText("Refine")

    def _clear(self):
        self.canvas.seeds_img.clear()
        self.canvas.border_img.clear()
        self.canvas.set_mask(None)
        self.mask_np = None
        self.canvas.update()

    def _save(self):
        if (
            not self.root
            or self.idx < 0
            or self.mask_np is None
            or not self.pid_combo.currentText().strip()
            or self.current_pil is None
            or self.current_path is None
        ):
            return
        pid = self.pid_combo.currentText().strip()
        p = self.current_path
        img_name = p.name
        mask_name = p.stem + ".png"
        img_dst = self.root / "prepared" / pid / "images" / img_name
        mask_dst = self.root / "prepared" / pid / "masks" / mask_name
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)

        is_dcm = str(p).lower().endswith(".dcm")
        if is_dcm:
            copy2(p, img_dst)
        else:
            self.current_pil.save(img_dst)

        Image.fromarray(self.mask_np).save(mask_dst)

        roi_method = "auto"
        if self.canvas.seeds_img or self.canvas.border_img:
            roi_method = "auto+refine"
        preproc = {
            "rectified": bool(self._rectified),
            "roi": roi_method,
            "seeds": len(self.canvas.seeds_img),
            "border_pts": len(self.canvas.border_img),
        }
        row = {
            "patient_id": pid,
            "random_id": "",
            "image_relpath": f"{pid}/images/{img_name}",
            "roi_relpath": f"{pid}/masks/{mask_name}",
            "grade_label": self.grade.currentText(),
            "preproc_json": json.dumps(preproc),
            "eval_split": "holdout",
        }
        upsert_manifest(self.root, row)
