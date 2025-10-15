from pathlib import Path
import json
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QFileDialog,
    QStackedWidget,
    QLineEdit,
)
from bpd_ui.core.state import upsert_manifest


class PreprocessTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = None
        self.patient_id = None
        self.image_paths = []
        self.current_idx = -1
        self.roi_mask = None
        self.grade = "unknown"
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        self.stack = QStackedWidget()
        v.addWidget(self._root_step())
        v.addWidget(self.stack)
        self.stack.addWidget(self._patient_step())
        self.stack.addWidget(self._files_step())
        self.stack.addWidget(self._roi_step())
        self.stack.addWidget(self._grade_step())
        self._goto(0)

    def _root_step(self):
        g = QGroupBox("Select Root")
        h = QHBoxLayout()
        self.root_label = QLabel("No root")
        b = QPushButton("Choose…")
        b.clicked.connect(self._choose_root)
        h.addWidget(self.root_label, 1)
        h.addWidget(b)
        g.setLayout(h)
        return g

    def _patient_step(self):
        w = QWidget()
        v = QVBoxLayout(w)
        h = QHBoxLayout()
        self.patient_edit = QLineEdit()
        self.rand_btn = QPushButton("Random ID")
        self.next_btn1 = QPushButton("Next")
        self.rand_btn.clicked.connect(self._random_id)
        self.next_btn1.clicked.connect(lambda: self._goto(1))
        h.addWidget(QLabel("Patient ID:"))
        h.addWidget(self.patient_edit, 1)
        h.addWidget(self.rand_btn)
        h.addWidget(self.next_btn1)
        v.addLayout(h)
        return w

    def _files_step(self):
        w = QWidget()
        v = QVBoxLayout(w)
        h = QHBoxLayout()
        self.files_label = QLabel("No files")
        b = QPushButton("Add Files…")
        b.clicked.connect(self._choose_files)
        n = QPushButton("Next")
        n.clicked.connect(lambda: self._goto(2))
        h.addWidget(self.files_label, 1)
        h.addWidget(b)
        h.addWidget(n)
        v.addLayout(h)
        return w

    def _roi_step(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.roi_info = QLabel("Add seed, optional border, then Accept")
        self.accept_btn = QPushButton("Accept ROI")
        self.accept_btn.clicked.connect(lambda: self._goto(3))
        v.addWidget(self.roi_info)
        v.addWidget(self.accept_btn)
        return w

    def _grade_step(self):
        w = QWidget()
        v = QVBoxLayout(w)
        h = QHBoxLayout()
        self.grade_combo = QComboBox()
        for g in ["no_bpd", "mild", "moderate", "severe", "unknown"]:
            self.grade_combo.addItem(g)
        s = QPushButton("Save")
        s.clicked.connect(self._save_row)
        h.addWidget(QLabel("Grade:"))
        h.addWidget(self.grade_combo, 1)
        h.addWidget(s)
        v.addLayout(h)
        return w

    def _goto(self, i):
        self.stack.setCurrentIndex(i)

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Root")
        if not d:
            return
        self.root = Path(d)
        self.root_label.setText(d)
        self._goto(0)

    def _random_id(self):
        import uuid

        self.patient_edit.setText(uuid.uuid4().hex[:8])

    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.dcm)"
        )
        if not files:
            return
        self.image_paths = [Path(f) for f in files]
        self.files_label.setText(f"{len(self.image_paths)} files selected")

    def _save_row(self):
        if not self.root or not self.patient_edit.text() or not self.image_paths:
            return
        pid = self.patient_edit.text()
        preproc = {}
        for p in self.image_paths:
            rel_img = f"{pid}/images/{p.name}"
            rel_mask = f"{pid}/masks/{p.stem}.png"
            row = {
                "patient_id": pid,
                "random_id": "",
                "image_relpath": rel_img,
                "roi_relpath": rel_mask,
                "grade_label": self.grade_combo.currentText(),
                "preproc_json": json.dumps(preproc),
                "eval_split": "holdout",
            }
            upsert_manifest(self.root, row)
