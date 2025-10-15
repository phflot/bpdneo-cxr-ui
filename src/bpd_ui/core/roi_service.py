import torch
import numpy as np
import torchxrayvision as xrv
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
    disk,
)
from skimage.segmentation import random_walker


class SegCache:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = None
        self.targets = None

    def get(self):
        if self.model is None:
            self.model = xrv.baseline_models.chestx_det.PSPNet().to(self.device).eval()
            self.targets = self.model.targets
        return self.model, self.targets


_seg = SegCache("cpu")


def xrv_preprocess(pil_img):
    import torchvision as tv

    img = np.array(pil_img)
    if img.ndim == 3:
        img = img.mean(2)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]
    t = tv.transforms.Compose([xrv.datasets.XRayResizer(512)])
    img = t(img)
    ten = torch.from_numpy(img).float()
    return ten


def auto_chest_mask(pil_img, use_heart=True, use_mediastinum=True):
    model, targets = _seg.get()
    x = xrv_preprocess(pil_img).to(_seg.device)
    with torch.inference_mode():
        y = model(x[None])
    y = y[0].cpu().numpy()
    idx = [targets.index("Left Lung"), targets.index("Right Lung")]
    if use_heart and "Heart" in targets:
        idx.append(targets.index("Heart"))
    if use_mediastinum and "Mediastinum" in targets:
        idx.append(targets.index("Mediastinum"))
    m = (y[idx] > 0.5).any(0)
    m = binary_closing(m, disk(3))
    m = binary_opening(m, disk(3))
    m = remove_small_objects(m, 256)
    m = remove_small_holes(m, 256)
    return m.astype(np.uint8) * 255


def bbox_from_mask(mask, pad_frac=(0.1, 0.08)):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    H, W = mask.shape
    dy_up = int(H * pad_frac[0])
    dx = int(W * pad_frac[1])
    y0 = max(0, y0 - dy_up)
    x0 = max(0, x0 - dx)
    x1 = min(W - 1, x1 + dx)
    return (x0, y0, x1, y1)


def refine_with_scribbles(gray, seeds_xy, border_xy, init_mask=None):
    H, W = gray.shape[:2]
    labels = np.zeros((H, W), dtype=np.uint8)
    for x, y in seeds_xy:
        if 0 <= x < W and 0 <= y < H:
            labels[int(y), int(x)] = 1
    if border_xy is not None and len(border_xy) > 0:
        from matplotlib.path import Path

        yy, xx = np.mgrid[:H, :W]
        inside = (
            Path(border_xy)
            .contains_points(np.vstack([xx.ravel(), yy.ravel()]).T)
            .reshape(H, W)
        )
        labels[~inside] = 2
    lab = random_walker(gray.astype(np.float32), labels, beta=90, mode="bf")
    m = lab == 1
    m = binary_closing(m, disk(3))
    m = binary_opening(m, disk(3))
    m = remove_small_objects(m, 128)
    m = remove_small_holes(m, 128)
    return m.astype(np.uint8) * 255
