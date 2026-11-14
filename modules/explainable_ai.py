import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import UPLOADS_DIR


def generate_gradcam_demo(image_path: str, patient_id: str) -> str:
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    h, w, _ = image.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    radius = min(h, w) * 0.38
    heatmap = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * radius ** 2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = heatmap * 255
    jet_map = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_rgb, 0.75, jet_map, 0.35, 0)

    output_dir = UPLOADS_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{patient_id}_{timestamp}_gradcam_demo.png"
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)
