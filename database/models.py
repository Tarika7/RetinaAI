from dataclasses import dataclass
from typing import Optional


@dataclass
class Patient:
    patient_id: str
    patient_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    diabetes_duration: Optional[str] = None
    registration_date: Optional[str] = None


@dataclass
class Examination:
    examination_id: int
    patient_id: str
    examination_date: str
    image_path: str
    predicted_grade: int
    predicted_severity: str
    prediction_confidence: float
    prediction_probabilities: str
    model_version: str
    analysis_mode: str


@dataclass
class Explanation:
    explanation_id: int
    examination_id: int
    gradcam_image_path: Optional[str]
    explanation_status: str
