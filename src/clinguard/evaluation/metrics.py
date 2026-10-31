
"""Evaluation metrics for ClinGuard AI anomaly detectors."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_detector(y_true, y_pred):
    """Calculate binary classification metrics."""

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    actual = pd.Series(y_true).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)

    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Evaluation labels cannot contain missing values.")

    if not set(actual.unique()).issubset({0, 1}):
        raise ValueError("Actual labels must contain only 0 and 1.")

    if not set(predicted.unique()).issubset({0, 1}):
        raise ValueError("Predicted labels must contain only 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": round(accuracy_score(actual, predicted), 4),
        "precision": round(
            precision_score(actual, predicted, zero_division=0), 4
        ),
        "recall": round(
            recall_score(actual, predicted, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(actual, predicted, zero_division=0), 4
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "total_records": len(actual),
    }
