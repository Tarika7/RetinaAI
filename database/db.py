import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DB_PATH, REPORTS_DIR, UPLOADS_DIR


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            diabetes_duration TEXT,
            registration_date TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS examinations (
            examination_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            examination_date TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_grade INTEGER NOT NULL,
            predicted_severity TEXT NOT NULL,
            prediction_confidence REAL NOT NULL,
            prediction_probabilities TEXT NOT NULL,
            model_version TEXT,
            analysis_mode TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS explainability (
            explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            examination_id INTEGER NOT NULL,
            gradcam_image_path TEXT,
            explanation_status TEXT,
            FOREIGN KEY(examination_id) REFERENCES examinations(examination_id)
        )
        """
    )
    conn.commit()
    conn.close()


def add_patient(
    patient_id: str,
    patient_name: str,
    age: Optional[int],
    gender: Optional[str],
    diabetes_duration: Optional[str],
    registration_date: str,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO patients (
            patient_id, patient_name, age, gender, diabetes_duration, registration_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (patient_id, patient_name, age, gender, diabetes_duration, registration_date),
    )
    conn.commit()
    conn.close()


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_patients() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM patients ORDER BY registration_date DESC, patient_id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def search_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    return get_patient(patient_id)


def save_examination(
    patient_id: str,
    examination_date: str,
    image_path: str,
    predicted_grade: int,
    predicted_severity: str,
    prediction_confidence: float,
    prediction_probabilities: List[float],
    model_version: str,
    analysis_mode: str,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO examinations (
            patient_id, examination_date, image_path, predicted_grade,
            predicted_severity, prediction_confidence, prediction_probabilities,
            model_version, analysis_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            examination_date,
            image_path,
            predicted_grade,
            predicted_severity,
            float(prediction_confidence),
            json.dumps([float(p) for p in prediction_probabilities]),
            model_version,
            analysis_mode,
        ),
    )
    conn.commit()
    exam_id = cursor.lastrowid
    conn.close()
    return int(exam_id)


def get_patient_examinations(patient_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM examinations
        WHERE patient_id = ?
        ORDER BY examination_date ASC, examination_id ASC
        """,
        (patient_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_examinations(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT e.*, p.patient_name
        FROM examinations e
        JOIN patients p ON p.patient_id = e.patient_id
        ORDER BY e.examination_date DESC, e.examination_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_examinations() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM examinations ORDER BY examination_date DESC, examination_id DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_examination_by_id(examination_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM examinations WHERE examination_id = ?",
        (examination_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_examination_before(patient_id: str, examination_date: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT * FROM examinations
        WHERE patient_id = ? AND examination_date < ?
        ORDER BY examination_date DESC, examination_id DESC
        LIMIT 1
        """,
        (patient_id, examination_date),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_gradcam(examination_id: int, gradcam_image_path: str, explanation_status: str) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO explainability (examination_id, gradcam_image_path, explanation_status)
        VALUES (?, ?, ?)
        """,
        (examination_id, gradcam_image_path, explanation_status),
    )
    conn.commit()
    conn.close()


def get_explainability_for_examination(examination_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM explainability WHERE examination_id = ? ORDER BY explanation_id DESC LIMIT 1",
        (examination_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_patient_stats() -> Dict[str, int]:
    conn = get_connection()
    patients_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    examinations_count = conn.execute("SELECT COUNT(*) FROM examinations").fetchone()[0]
    dr_patients = conn.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM examinations WHERE predicted_grade >= 1"
    ).fetchone()[0]
    close_monitoring = conn.execute(
        "SELECT COUNT(*) FROM examinations WHERE predicted_grade >= 2"
    ).fetchone()[0]
    conn.close()
    return {
        "total_patients": int(patients_count),
        "total_scans": int(examinations_count),
        "dr_patients": int(dr_patients),
        "monitoring": int(close_monitoring),
    }


def get_distribution_counts() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT predicted_grade, COUNT(*) as count FROM examinations GROUP BY predicted_grade ORDER BY predicted_grade ASC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def seed_demo_data() -> None:
    conn = get_connection()
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()

    if patient_count > 0:
        return

    from PIL import Image, ImageDraw

    demo_root = UPLOADS_DIR / "demo"
    demo_root.mkdir(parents=True, exist_ok=True)
    patient_id = "DEMO-001"
    patient_name = "Sample Patient"
    registration_date = "2026-11-01"
    add_patient(
        patient_id=patient_id,
        patient_name=patient_name,
        age=58,
        gender="Female",
        diabetes_duration="8 years",
        registration_date=registration_date,
    )

    image_path = demo_root / "demo_retina.png"
    image = Image.new("RGB", (224, 224), color=(230, 238, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 35, 185, 185), outline=(95, 120, 150), width=4)
    draw.ellipse((60, 60, 160, 160), outline=(140, 160, 185), width=2)
    for x, y in [(90, 80), (120, 105), (95, 140), (130, 138), (105, 160)]:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(180, 70, 70), outline=(180, 70, 70))
    image.save(image_path)

    save_examination(
        patient_id=patient_id,
        examination_date=registration_date,
        image_path=str(image_path),
        predicted_grade=2,
        predicted_severity="Moderate NPDR",
        prediction_confidence=0.82,
        prediction_probabilities=[0.12, 0.18, 0.41, 0.22, 0.07],
        model_version="demo-seeded-v1",
        analysis_mode="DEMO MODE",
    )
