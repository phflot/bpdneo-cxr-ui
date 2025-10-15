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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
        self.pid_edit = QLineEdit()
        rb = QPushButton("Random")
        rb.clicked.connect(self._random_pid)
        h2.addWidget(QLabel("ID:"))
        h2.addWidget(self.pid_edit, 1)
        h2.addWidget(rb)
        g2.setLayout(h2)
        R.addWidget(g2)

        g3 = QGroupBox("Files")
        v3 = QVBoxLayout()
        self.listw = QListWidget()
        v3.addWidget(self.listw)
        addb = QPushButton("Add Images…")
        addb.clicked.connect(self._add_files)
        v3.addWidget(addb)
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

    def _random_pid(self):
        import uuid

        self.pid_edit.setText(uuid.uuid4().hex[:8])

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
        disp = self._rectify_if_needed(disp, p)
        self.current_pil = disp
        arr = np.array(disp)
        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.display_pix = QPixmap.fromImage(qimg)
        self.canvas.set_image(self.display_pix)
        self.canvas.set_mask(None)
        self.mask_np = None

    def _rectify_if_needed(self, pil_img, path):
        if str(path).lower().endswith((".png", ".jpg", ".jpeg")):
            import cv2

            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            warp = rectify_if_scan(bgr)
            return Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
        return pil_img

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
            or not self.pid_edit.text()
            or self.current_pil is None
            or self.current_path is None
        ):
            return
        pid = self.pid_edit.text()
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

        preproc = {"rectified": not is_dcm}
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
