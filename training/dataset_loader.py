from pathlib import Path
from typing import Any, Dict, List, Tuple


def list_dataset_files(data_dir: str) -> List[str]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    files = []
    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            files.append(str(item))
    return sorted(files)


def load_apatos_labels(csv_path: str) -> Dict[str, str]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return {}

    labels: Dict[str, str] = {}
    with csv_file.open("r", encoding="utf-8") as handle:
        header = [value.strip() for value in handle.readline().strip().split(",")]
        for line in handle:
            values = [value.strip() for value in line.strip().split(",")]
            if len(values) < 2:
                continue
            record = {header[i]: values[i] for i in range(len(header)) if i < len(values)}
            image_id = record.get("id") or record.get("image_id") or record.get("image")
            diagnosis = record.get("diagnosis") or record.get("label") or record.get("severity")
            if image_id and diagnosis:
                labels[str(image_id)] = str(diagnosis)
    return labels


def build_dataset_manifest(data_dir: str, csv_path: str = "") -> List[Dict[str, Any]]:
    images = list_dataset_files(data_dir)
    label_map = load_apatos_labels(csv_path) if csv_path else {}
    manifest: List[Dict[str, Any]] = []

    for image_path in images:
        image_file = Path(image_path)
        stem = image_file.stem
        label_value = label_map.get(stem) or label_map.get(image_file.name)
        if label_value is None:
            continue
        try:
            grade = int(float(label_value))
        except (TypeError, ValueError):
            continue
        manifest.append({"image_path": image_path, "label": grade})

    return manifest


def train_val_test_split(manifest: List[Dict[str, Any]], val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not manifest:
        raise ValueError("No dataset samples were found to split.")

    import random

    random.Random(seed).shuffle(manifest)
    total = len(manifest)
    val_count = max(1, int(total * val_ratio))
    test_count = max(1, int(total * test_ratio))
    train_count = total - val_count - test_count

    if train_count <= 0:
        train_count = max(1, total - val_count)

    train = manifest[:train_count]
    val = manifest[train_count:train_count + val_count]
    test = manifest[train_count + val_count:]

    return train, val, test
