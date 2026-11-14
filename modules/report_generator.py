from pathlib import Path
from typing import Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import REPORTS_DIR


def build_ai_interpretation(predicted_grade: int, progression_status: str) -> str:
    if progression_status == "Worsened":
        return (
            "The current examination shows a higher predicted DR severity than the previous visit. "
            "This pattern should be reviewed with the clinical team and compared against image quality and acquisition consistency."
        )
    if progression_status == "Improved":
        return (
            "The current examination shows a lower predicted severity than the previous visit. "
            "This is encouraging but should be reviewed in context because classification variability can be influenced by image quality."
        )
    if progression_status == "Stable":
        return (
            "The current examination appears similar to the previous visit. "
            "The observed pattern is stable, but clinicians should review for subtle retinal changes and image acquisition differences."
        )
    return (
        "This is the patient's first recorded examination. The predicted grade represents the current model output and should be interpreted as a research-only screening signal."
    )


def generate_pdf_report(
    patient: Dict[str, object],
    examination: Dict[str, object],
    previous_grade: Optional[int],
    progression_status: str,
    gradcam_path: Optional[str],
    probabilities: list,
) -> str:
    output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{patient['patient_id']}_{examination['examination_date']}.pdf"
    output_path = output_dir / filename

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0b2d4d"),
        leading=20,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0b2d4d"),
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
    )

    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    elements.append(Paragraph("RETINA-AI", title_style))
    elements.append(Paragraph("Intelligent Diabetic Retinopathy Screening and Progression Monitoring", styles["Italic"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Clinical Decision-Support Report", heading_style))
    elements.append(Paragraph(f"Patient ID: {patient['patient_id']}", body_style))
    elements.append(Paragraph(f"Examination date: {examination['examination_date']}", body_style))
    elements.append(Paragraph(f"Predicted DR grade: {examination['predicted_grade']}", body_style))
    elements.append(Paragraph(f"Predicted severity: {examination['predicted_severity']}", body_style))
    elements.append(Paragraph(f"Prediction confidence: {examination['prediction_confidence']:.2%}", body_style))
    elements.append(Paragraph(f"Previous examination grade: {previous_grade if previous_grade is not None else 'N/A'}", body_style))
    elements.append(Paragraph(f"Progression status: {progression_status}", body_style))
    elements.append(Spacer(1, 0.2 * inch))

    probability_rows = [["Class", "Probability"]]
    for index, value in enumerate(probabilities):
        probability_rows.append([str(index), f"{float(value):.2%}"])
    probability_table = Table(probability_rows, colWidths=[2.5 * inch, 2.5 * inch])
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF3")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    elements.append(Paragraph("Class probability distribution", heading_style))
    elements.append(probability_table)
    elements.append(Spacer(1, 0.2 * inch))

    if gradcam_path and Path(gradcam_path).exists():
        elements.append(Paragraph("Grad-CAM / explainability visualization", heading_style))
        elements.append(Image(str(gradcam_path), width=2.3 * inch, height=2.3 * inch))
        elements.append(Spacer(1, 0.1 * inch))

    ai_summary = build_ai_interpretation(int(examination["predicted_grade"]), progression_status)
    elements.append(Paragraph("AI-generated interpretation", heading_style))
    elements.append(Paragraph(ai_summary, body_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Suggested follow-up: schedule clinician review based on the current severity, progression pattern, and image quality review.", body_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("This report is generated by an AI research prototype and is intended for clinical decision-support research only. It is not a medical diagnosis and must not replace examination by a qualified eye-care professional.", styles["Italic"]))

    doc.build(elements)
    return str(output_path)
