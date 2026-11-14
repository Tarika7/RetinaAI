from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "database" / "retina_ai.db"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "model"
ASSETS_DIR = PROJECT_ROOT / "assets"

SEVERITY_CLASSES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

SEVERITY_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Mild non-proliferative changes.",
    2: "More evident retinal abnormalities.",
    3: "Advanced non-proliferative changes.",
    4: "Advanced disease with proliferative changes.",
}

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
DEFAULT_MODEL_PATH = MODEL_DIR / "dr_model.pth"
