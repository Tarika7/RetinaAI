from typing import List, Optional

from database.db import add_patient, list_patients, search_patient_by_id


def generate_patient_id(existing_ids: List[str]) -> str:
    if not existing_ids:
        return "DR001"

    numeric_ids = []
    for patient_id in existing_ids:
        try:
            numeric_ids.append(int(str(patient_id).replace("DR", "")))
        except ValueError:
            continue

    next_number = max(numeric_ids) + 1 if numeric_ids else 1
    return f"DR{next_number:03d}"


def register_patient(
    patient_name: str,
    age: Optional[int],
    gender: Optional[str],
    diabetes_duration: Optional[str],
    registration_date: str,
    patient_id: Optional[str] = None,
) -> str:
    existing_ids = [row["patient_id"] for row in list_patients()]
    assigned_id = patient_id or generate_patient_id(existing_ids)
    add_patient(
        patient_id=assigned_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        diabetes_duration=diabetes_duration,
        registration_date=registration_date,
    )
    return assigned_id


def find_patient(patient_id: str):
    return search_patient_by_id(patient_id)
