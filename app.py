from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from config import ASSETS_DIR, SEVERITY_CLASSES
from database.db import (
    get_all_examinations,
    get_examination_by_id,
    get_latest_examination_before,
    get_patient,
    get_patient_examinations,
    get_patient_stats,
    get_recent_examinations,
    init_db,
    list_patients,
    save_examination,
    save_gradcam,
)
from modules.dr_classifier import get_model_status, predict_dr_severity
from modules.explainable_ai import generate_gradcam_demo
from modules.image_preprocessing import create_pil_preview, preprocess_image, save_uploaded_image, validate_uploaded_image
from modules.patient_management import register_patient
from modules.progression_monitor import classify_progression, get_patient_progression
from modules.report_generator import generate_pdf_report
from utils.helpers import deserialize_probabilities, today_iso

st.set_page_config(page_title="RETINA-AI", page_icon="👁️", layout="wide")

CSS_PATH = ASSETS_DIR / "styles.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data
def get_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def render_dashboard():
    stats = get_patient_stats()
    recent = get_recent_examinations(5)
    if stats["total_patients"] == 0 and stats["total_scans"] == 0:
        st.info("Sample demo data has been seeded so the dashboard is populated and clearly labeled as demonstration content.")
    distribution = {}
    for i in range(5):
        distribution[str(i)] = 0
    for entry in get_all_examinations():
        distribution[str(entry["predicted_grade"])] = distribution.get(str(entry["predicted_grade"]), 0) + 1

    st.markdown('<div class="section-badge">Clinical intelligence</div>', unsafe_allow_html=True)
    st.title("RETINA-AI")
    st.caption("Intelligent diabetic retinopathy screening and progression monitoring")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(f'''<div class="kpi-card"><div class="kpi-label">Patients</div><div class="kpi-value">{stats['total_patients']}</div><div class="kpi-trend">Registered</div></div>''', unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown(f'''<div class="kpi-card"><div class="kpi-label">Scans</div><div class="kpi-value">{stats['total_scans']}</div><div class="kpi-trend">Analyzed</div></div>''', unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown(f'''<div class="kpi-card"><div class="kpi-label">DR patients</div><div class="kpi-value">{stats['dr_patients']}</div><div class="kpi-trend">Detected</div></div>''', unsafe_allow_html=True)
    with metric_cols[3]:
        st.markdown(f'''<div class="kpi-card"><div class="kpi-label">Monitoring</div><div class="kpi-value">{stats['monitoring']}</div><div class="kpi-trend">Needs review</div></div>''', unsafe_allow_html=True)

    chart_col, alerts_col = st.columns([1.8, 1.1])
    with chart_col:
        st.markdown('<div class="panel-card"><h3>Severity distribution</h3></div>', unsafe_allow_html=True)
        distribution_df = pd.DataFrame(
            {
                "Severity grade": [0, 1, 2, 3, 4],
                "Count": [distribution.get(str(i), 0) for i in range(5)],
            }
        )
        chart_col.bar_chart(distribution_df.set_index("Severity grade"), use_container_width=True)

    with alerts_col:
        st.markdown('<div class="panel-card"><h3>Recent progression alerts</h3></div>', unsafe_allow_html=True)
        if recent:
            for row in recent[:4]:
                patient = get_patient(row["patient_id"])
                previous = get_latest_examination_before(row["patient_id"], row["examination_date"])
                previous_grade = previous["predicted_grade"] if previous else None
                status = classify_progression(previous_grade, int(row["predicted_grade"])) if previous or row["predicted_grade"] is not None else "First Visit"
                badge_class = {
                    "Worsened": "pill-worsened",
                    "Stable": "pill-stable",
                    "Improved": "pill-improved",
                    "First Visit": "pill-first",
                }.get(status, "pill-first")
                st.markdown(f'''<div class="alert-box"><div class="status-pill {badge_class}">{status}</div><div style="margin-top:0.45rem; font-weight:600;">{patient['patient_name']}</div><div style="color:#5b7084; font-size:0.82rem;">{row['examination_date']} · Grade {row['predicted_grade']}</div></div>''', unsafe_allow_html=True)
        else:
            st.info("No examinations recorded yet.")

    st.markdown('<div class="section-badge" style="margin-top:1.3rem;">Recent scans</div>', unsafe_allow_html=True)
    if recent:
        recent_df = pd.DataFrame(
            [
                {
                    "Patient ID": r["patient_id"],
                    "Patient name": get_patient(r["patient_id"])["patient_name"],
                    "Date": r["examination_date"],
                    "Grade": r["predicted_grade"],
                    "Severity": r["predicted_severity"],
                }
                for r in recent
            ]
        )
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("Upload a retinal scan to populate the dashboard.")


def render_register_patient():
    st.header("Register Patient")
    with st.form("patient_registration_form"):
        patient_name = st.text_input("Patient name / anonymized identifier")
        age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
        gender = st.selectbox("Gender (optional)", ["", "Female", "Male", "Other", "Prefer not to say"])
        diabetes_duration = st.text_input("Diabetes history duration (optional)", placeholder="e.g., 6 years")
        registration_date = st.date_input("Registration date", value=date.today())
        submitted = st.form_submit_button("Register patient")

        if submitted:
            if not patient_name.strip():
                st.warning("Patient name is required.")
            else:
                patient_id = register_patient(
                    patient_name=patient_name.strip(),
                    age=int(age) if age else None,
                    gender=gender or None,
                    diabetes_duration=diabetes_duration.strip() or None,
                    registration_date=registration_date.isoformat(),
                )
                st.success(f"Patient registered successfully. Patient ID: {patient_id}")

    st.subheader("Search patient")
    search_id = st.text_input("Enter Patient ID")
    if search_id:
        patient = get_patient(search_id)
        if patient:
            st.write(patient)
        else:
            st.warning("Patient ID not found.")

    st.subheader("Patient registry")
    patients = list_patients()
    if patients:
        st.dataframe(pd.DataFrame(patients), use_container_width=True)
    else:
        st.info("No patients registered yet.")


def render_image_analysis():
    patients = list_patients()
    if not patients:
        st.warning("Register a patient before uploading a retinal scan.")
        return

    st.header("Analyze Retinal Image")
    patient_options = [p["patient_id"] for p in patients]
    selected_patient_id = st.selectbox("Patient ID", patient_options)
    uploaded_file = st.file_uploader("Upload retinal fundus image", type=["jpg", "jpeg", "png"])
    examination_date = st.date_input("Examination date", value=date.today())

    if st.button("Analyze image"):
        if uploaded_file is None:
            st.warning("Please upload a retinal fundus image.")
            return

        try:
            validate_uploaded_image(uploaded_file)
            image_path = save_uploaded_image(uploaded_file, selected_patient_id)
            image = create_pil_preview(image_path)
            result = predict_dr_severity(image_path)
            grade = int(result["predicted_grade"])
            severity = result["predicted_severity"]
            confidence = float(result["prediction_confidence"])
            probabilities = result["probabilities"]
            model_status = get_model_status()
            exam_date_str = examination_date.isoformat()

            previous = get_latest_examination_before(selected_patient_id, exam_date_str)
            previous_grade = previous["predicted_grade"] if previous else None
            progression_status = classify_progression(previous_grade, grade) if previous or previous_grade is not None else "First Visit"

            exam_id = save_examination(
                patient_id=selected_patient_id,
                examination_date=exam_date_str,
                image_path=image_path,
                predicted_grade=grade,
                predicted_severity=severity,
                prediction_confidence=confidence,
                prediction_probabilities=probabilities,
                model_version=result.get("model_version", "demo-heuristic-v1"),
                analysis_mode=result.get("analysis_mode", "DEMO MODE"),
            )

            gradcam_path = generate_gradcam_demo(image_path, selected_patient_id)
            save_gradcam(
                exam_id,
                gradcam_path,
                "Illustrative demo visualization (not model-generated)" if "DEMO" in str(result.get("analysis_mode", "DEMO MODE")).upper() else "Generated using model explainability",
            )

            st.success("Analysis completed successfully.")
            st.caption(f"Model status: {model_status}")
            st.caption(f"Clinical note: {result['warning']}")

            col_original, col_details = st.columns([1.2, 1])
            with col_original:
                st.subheader("Original retinal image")
                st.image(image, use_column_width=True)

            with col_details:
                st.subheader("Prediction")
                st.markdown(f"<h2 style='color:#0b2d4d'>{severity}</h2>", unsafe_allow_html=True)
                st.metric("Prediction confidence", f"{confidence * 100:.2f}%")
                st.write(f"Patient ID: {selected_patient_id}")
                st.write(f"Examination date: {exam_date_str}")
                st.write(f"Previous grade: {previous_grade if previous_grade is not None else 'N/A'}")
                st.write(f"Progression status: {progression_status}")
                st.write("Disclaimer: A change in predicted grade alone is not definitive biological progression. Differences in acquisition settings and model uncertainty can affect predictions.")

            st.subheader("Class probability distribution")
            probabilities_df = pd.DataFrame(
                {"Severity class": [str(i) for i in range(5)], "Probability": probabilities}
            )
            st.bar_chart(probabilities_df.set_index("Severity class"), use_container_width=True)

            st.subheader("Explainability preview")
            col_heatmap, col_merged = st.columns(2)
            col_heatmap.image(Image.open(gradcam_path), use_column_width=True)
            col_merged.image(image, use_column_width=True)
            st.caption("The highlighted regions indicate model attention and should not be interpreted as confirmed clinical lesions.")

        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")


def render_patient_history():
    patients = list_patients()
    if not patients:
        st.info("No patient records available.")
        return

    st.header("Patient History")
    patient_id = st.selectbox("Choose patient", [p["patient_id"] for p in patients])
    examinations = get_patient_examinations(patient_id)
    if not examinations:
        st.info("No examinations found for this patient.")
        return

    history_df = pd.DataFrame(
        [
            {
                "Examination ID": e["examination_id"],
                "Date": e["examination_date"],
                "Grade": e["predicted_grade"],
                "Severity": e["predicted_severity"],
                "Confidence": f"{float(e['prediction_confidence']) * 100:.2f}%",
                "Mode": e["analysis_mode"],
            }
            for e in examinations
        ]
    )
    st.dataframe(history_df, use_container_width=True)


def render_progression_monitoring():
    patients = list_patients()
    if not patients:
        st.info("Register patients to view longitudinal monitoring.")
        return

    st.header("Progression Monitoring")
    patient_id = st.selectbox("Select patient", [p["patient_id"] for p in patients], key="progression_patient")
    progression = get_patient_progression(patient_id)
    if not progression["timeline"]:
        st.info("This patient has no recorded examinations yet.")
        return

    st.plotly_chart(progression["figure"], use_container_width=True)

    st.subheader("Examination timeline")
    for item in progression["timeline"]:
        st.markdown(
            f"- **{item['examination_date']}** | Grade {item['predicted_grade']} | Previous grade: {item['previous_grade'] if item['previous_grade'] is not None else 'N/A'} | Status: {item['status']}"
        )


def render_explainability():
    patients = list_patients()
    if not patients:
        st.info("No patient data available.")
        return

    st.header("Explainable AI")
    patient_id = st.selectbox("Select patient", [p["patient_id"] for p in patients], key="xai_patient")
    examinations = get_patient_examinations(patient_id)
    if not examinations:
        st.info("No examinations exist for this patient.")
        return

    latest = examinations[-1]
    img_path = latest["image_path"]
    gradcam_path = generate_gradcam_demo(img_path, patient_id)

    original = Image.open(img_path)
    heatmap = Image.open(gradcam_path)
    col1, col2 = st.columns(2)
    col1.subheader("Original fundus image")
    col1.image(original, use_container_width=True)
    col2.subheader("Grad-CAM visualization")
    col2.image(heatmap, use_container_width=True)
    st.caption("The highlighted regions indicate areas that contributed more strongly to the model's prediction. These are model attention regions and should not be interpreted as confirmed clinical lesions.")


def render_report_generation():
    patients = list_patients()
    if not patients:
        st.info("Register a patient and analyze an image before generating a report.")
        return

    st.header("Generate Report")
    patient_id = st.selectbox("Patient ID", [p["patient_id"] for p in patients], key="report_patient")
    examinations = get_patient_examinations(patient_id)
    if not examinations:
        st.info("No examinations available for this patient.")
        return

    exam_selection = st.selectbox(
        "Select examination",
        [f"{e['examination_id']} - {e['examination_date']} - Grade {e['predicted_grade']}" for e in examinations],
        key="report_exam",
    )
    exam_data = next(e for e in examinations if f"{e['examination_id']} - {e['examination_date']} - Grade {e['predicted_grade']}" == exam_selection)

    if st.button("Generate PDF report"):
        patient = get_patient(patient_id)
        previous = get_latest_examination_before(patient_id, exam_data["examination_date"])
        previous_grade = previous["predicted_grade"] if previous else None
        progression_state = classify_progression(previous_grade, int(exam_data["predicted_grade"])) if previous or previous_grade is not None else "First Visit"
        probabilities = deserialize_probabilities(exam_data["prediction_probabilities"])
        gradcam_path = None
        if exam_data["analysis_mode"] == "DEMO MODE":
            gradcam_path = generate_gradcam_demo(exam_data["image_path"], patient_id)
        report_path = generate_pdf_report(
            patient=patient,
            examination=exam_data,
            previous_grade=previous_grade,
            progression_status=progression_state,
            gradcam_path=gradcam_path,
            probabilities=probabilities,
        )
        with open(report_path, "rb") as file_obj:
            report_bytes = file_obj.read()
        st.success(f"Report generated successfully at {report_path}")
        st.download_button("Download PDF", data=report_bytes, file_name=Path(report_path).name, mime="application/pdf")


def render_about():
    st.header("About the System")
    st.write("RETINA-AI is a research prototype designed to showcase diabetic retinopathy detection, patient history tracking, model explainability, and clinical report generation.")
    st.write("This prototype is intended for healthcare research and decision support. It is not a medical diagnosis and must be reviewed by a qualified ophthalmologist.")
    st.write("Key features: patient registration, retinal image analysis, longitudinal grade comparison, Grad-CAM-style visualization, and PDF reporting.")
    st.info("The current prototype runs in DEMO MODE when no trained model is available. A real trained model can be integrated into the modular architecture later.")


def main():
    init_db()
    from database.db import seed_demo_data

    seed_demo_data()
    nav = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Register Patient",
            "Analyze Retinal Image",
            "Patient History",
            "Progression Monitoring",
            "Explainable AI",
            "Generate Report",
            "About the System",
        ],
    )

    if nav == "Dashboard":
        render_dashboard()
    elif nav == "Register Patient":
        render_register_patient()
    elif nav == "Analyze Retinal Image":
        render_image_analysis()
    elif nav == "Patient History":
        render_patient_history()
    elif nav == "Progression Monitoring":
        render_progression_monitoring()
    elif nav == "Explainable AI":
        render_explainability()
    elif nav == "Generate Report":
        render_report_generation()
    else:
        render_about()


if __name__ == "__main__":
    main()
