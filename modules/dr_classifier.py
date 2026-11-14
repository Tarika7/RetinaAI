from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models

from config import DEFAULT_MODEL_PATH, MODEL_DIR, SEVERITY_CLASSES


def build_efficientnet_b0(num_classes: int = 5) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def model_available() -> bool:
    expected_paths = [
        DEFAULT_MODEL_PATH,
        MODEL_DIR / "best_model.pth",
        MODEL_DIR / "dr_model.pth",
    ]
    return any(path.exists() for path in expected_paths)


def load_model_if_available() -> Optional[torch.nn.Module]:
    candidate_paths = [
        DEFAULT_MODEL_PATH,
        MODEL_DIR / "best_model.pth",
        MODEL_DIR / "dr_model.pth",
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            model = build_efficientnet_b0()
            state = torch.load(candidate, map_location="cpu")
            if isinstance(state, dict):
                if all(key.startswith("features") or key.startswith("classifier") for key in state.keys()):
                    model.load_state_dict(state)
                    model.eval()
                    return model
            if isinstance(state, torch.nn.Module):
                state.eval()
                return state
        except Exception:
            continue
    return None


def generate_demo_prediction(image_array: np.ndarray) -> Tuple[int, List[float], float]:
    mean_intensity = float(np.mean(image_array))
    std_dev = float(np.std(image_array))
    signal = (mean_intensity / 255.0) * 0.55 + (std_dev / 255.0) * 0.45

    raw_grade = signal * 4.0
    grade = int(np.clip(round(raw_grade), 0, 4))
    probabilities = np.full(5, 0.08, dtype=float)
    probabilities[grade] += 0.62
    remaining = 1.0 - probabilities[grade]
    weights = np.array([1.0 / (abs(i - grade) + 1.0) for i in range(5)], dtype=float)
    weights[grade] = 1.0
    other_mass = remaining * (weights / weights.sum())
    probabilities = probabilities + other_mass
    probabilities = probabilities / probabilities.sum()

    confidence = float(np.clip(0.68 + (0.24 * (1.0 - abs(signal - (grade / 4.0)))), 0.67, 0.97))
    return grade, [float(p) for p in probabilities], confidence


def predict_dr_severity(image_path: str) -> Dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image, dtype=np.float32)
    model = load_model_if_available()

    if model is not None:
        try:
            image_tensor = torch.tensor(np.transpose(image_array, (2, 0, 1)) / 255.0, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
            values = probabilities.cpu().numpy().tolist()
            index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[index].item())
            return {
                "predicted_grade": index,
                "predicted_severity": SEVERITY_CLASSES[index],
                "prediction_confidence": round(float(confidence), 4),
                "probabilities": [round(float(v), 4) for v in values],
                "analysis_mode": "MODEL INFERENCE",
                "warning": "A compatible model checkpoint was loaded. This output is based on the model checkpoint and should still be reviewed by a qualified clinician.",
                "model_version": "efficientnet-b0",
            }
        except Exception:
            pass

    grade, probabilities, confidence = generate_demo_prediction(image_array)
    severity = SEVERITY_CLASSES[grade]
    return {
        "predicted_grade": int(grade),
        "predicted_severity": severity,
        "prediction_confidence": round(float(confidence), 4),
        "probabilities": [round(float(p), 4) for p in probabilities],
        "analysis_mode": "DEMO MODE",
        "warning": "This is a demonstration prediction engine and is not a medically validated AI model.",
        "model_version": "demo-heuristic-v1",
    }


def get_model_status() -> str:
    model = load_model_if_available()
    if model is not None:
        return "Model checkpoint loaded successfully. The app is using model inference for this session."
    if model_available():
        return "Model file detected, but no compatible checkpoint could be loaded. Falling back to DEMO MODE."
    return "Demo mode active: no trained model is present."
