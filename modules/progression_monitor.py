from typing import Dict, List, Optional

import plotly.graph_objects as go

from database.db import get_patient_examinations


def classify_progression(previous_grade: Optional[int], current_grade: int) -> str:
    if previous_grade is None:
        return "First Visit"
    if current_grade < previous_grade:
        return "Improved"
    if current_grade > previous_grade:
        return "Worsened"
    return "Stable"


def build_progression_figure(examinations: List[Dict[str, object]]):
    if not examinations:
        empty_figure = go.Figure()
        empty_figure.add_annotation(
            text="No examinations available for progression analysis.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16},
        )
        empty_figure.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False, "range": [0, 5]},
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        return empty_figure

    dates = [entry["examination_date"] for entry in examinations]
    grades = [int(entry["predicted_grade"]) for entry in examinations]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=grades,
            mode="lines+markers",
            line={"color": "#0f766e", "width": 3},
            marker={"size": 10, "color": "#0f766e"},
            name="DR grade",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title="Longitudinal DR progression",
        xaxis_title="Examination date",
        yaxis_title="Severity grade",
        yaxis={"range": [-0.1, 4.1], "tickvals": [0, 1, 2, 3, 4]},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return figure


def get_patient_progression(patient_id: str):
    examinations = get_patient_examinations(patient_id)
    timeline = []
    for index, exam in enumerate(examinations):
        previous_grade = None if index == 0 else int(examinations[index - 1]["predicted_grade"])
        current_grade = int(exam["predicted_grade"])
        status = classify_progression(previous_grade, current_grade)
        timeline.append(
            {
                "examination_id": exam["examination_id"],
                "examination_date": exam["examination_date"],
                "predicted_grade": current_grade,
                "previous_grade": previous_grade,
                "status": status,
            }
        )

    return {
        "timeline": timeline,
        "figure": build_progression_figure(examinations),
    }
