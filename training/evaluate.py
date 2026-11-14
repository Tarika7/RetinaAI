from typing import Iterable

from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix


def evaluate_predictions(y_true: Iterable[int], y_pred: Iterable[int]):
    y_true = list(y_true)
    y_pred = list(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "quadratic_weighted_kappa": float(kappa),
        "classification_report": report,
        "confusion_matrix": matrix,
    }
