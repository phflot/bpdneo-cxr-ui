from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtCore import Qt, QRectF, QPointF


class OverlayCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pix = None
        self.mask = None
        self.mode = "view"
        self._img_w = 0
        self._img_h = 0
        self._draw_rect = QRectF()
        self.seeds_img = []
        self.border_img = []

    def set_image(self, qpix: QPixmap):
        self.pix = qpix
        self._img_w = qpix.width()
        self._img_h = qpix.height()
        self.seeds_img.clear()
        self.border_img.clear()
        self.update()

    def set_mask(self, qimg: QImage | None):
        self.mask = qimg
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode

    def _compute_draw_rect(self) -> QRectF:
        r = self.rect()
        if not self.pix:
            self._draw_rect = QRectF()
            return self._draw_rect
        pr = self.pix.rect()
        prf = QRectF(pr)
        prf = prf.scaled(r.width(), r.height(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (r.width() - prf.width()) / 2
        y = (r.height() - prf.height()) / 2
        self._draw_rect = QRectF(x, y, prf.width(), prf.height())
        return self._draw_rect

    def _widget_to_image(self, pt: QPointF) -> tuple[int, int] | None:
        if self._draw_rect.isNull() or self._img_w == 0:
            return None
        sx = self._img_w / self._draw_rect.width()
        sy = self._img_h / self._draw_rect.height()
        ix = int((pt.x() - self._draw_rect.x()) * sx)
        iy = int((pt.y() - self._draw_rect.y()) * sy)
        if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
            return ix, iy
        return None

    def mousePressEvent(self, e):
        if self.mode not in ("seed", "border") or not self.pix:
            return
        img_pt = self._widget_to_image(e.position())
        if img_pt is None:
            return
        if self.mode == "seed":
            self.seeds_img.append(img_pt)
        else:
            self.border_img.append(img_pt)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#f5f5f5"))
        dr = self._compute_draw_rect()
        if self.pix:
            p.drawPixmap(dr, self.pix, self.pix.rect())
            if self.mask:
                p.setOpacity(0.35)
                p.drawImage(dr, self.mask)
                p.setOpacity(1.0)
        pen = QPen(QColor("#ff5252"), 3)
        p.setPen(pen)

        def img_to_widget(pt):
            sx = dr.width() / self._img_w if self._img_w else 1
            sy = dr.height() / self._img_h if self._img_h else 1
            return QPointF(dr.x() + pt[0] * sx, dr.y() + pt[1] * sy)

        for x, y in self.seeds_img:
            wp = img_to_widget((x, y))
            p.drawPoint(int(wp.x()), int(wp.y()))
        if len(self.border_img) > 1:
            for i in range(len(self.border_img) - 1):
                a = img_to_widget(self.border_img[i])
                b = img_to_widget(self.border_img[i + 1])
                p.drawLine(int(a.x()), int(a.y()), int(b.x()), int(b.y()))
