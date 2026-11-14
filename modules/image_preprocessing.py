from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import SUPPORTED_IMAGE_TYPES, UPLOADS_DIR


def validate_uploaded_image(uploaded_file) -> bool:
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")
    mime_type = getattr(uploaded_file, "type", "")
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image format. Use PNG, JPG, or JPEG.")

    buffer = uploaded_file.getbuffer()
    if hasattr(buffer, "nbytes"):
        size = buffer.nbytes
    elif hasattr(buffer, "getvalue"):
        size = len(buffer.getvalue())
    else:
        size = len(uploaded_file.getvalue())

    if size == 0:
        raise ValueError("Uploaded image is empty.")
    return True


def save_uploaded_image(uploaded_file, patient_id: str) -> str:
    validate_uploaded_image(uploaded_file)
    file_suffix = Path(uploaded_file.name).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{patient_id}_{timestamp}{file_suffix}"
    target_path = UPLOADS_DIR / file_name
    target_path.write_bytes(uploaded_file.getvalue())
    return str(target_path)


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0,1] to match typical model input conventions.
    normalized = rgb_image.astype(np.float32) / 255.0
    return normalized


def create_pil_preview(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.autocontrast(image)
    return image
