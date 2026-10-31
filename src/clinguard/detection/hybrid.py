
"""Hybrid anomaly detection for ClinGuard AI."""

import pandas as pd


def detect_hybrid_anomalies(dataset):
    if dataset.empty:
        raise ValueError("Cannot process an empty dataset.")

    required_columns = [
        "statistical_anomaly_flag",
        "isolation_forest_anomaly_flag",
        "isolation_forest_input_valid",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = dataset.copy()

    statistical = (
        result["statistical_anomaly_flag"]
        .fillna(False)
        .astype(bool)
    )

    isolation_forest = (
        result["isolation_forest_anomaly_flag"]
        .fillna(False)
        .astype(bool)
    )

    isolation_valid = (
        result["isolation_forest_input_valid"]
        .fillna(False)
        .astype(bool)
    )

    # Strategy A: Either model can flag an anomaly.
    result["hybrid_or_flag"] = (
        statistical | isolation_forest
    )

    # Strategy B: Both models must agree.
    result["hybrid_and_flag"] = (
        statistical & isolation_forest & isolation_valid
    )

    # Record which detector contributed to the decision.
    def identify_reason(row):
        reasons = []

        if row["statistical_anomaly_flag"]:
            reasons.append("Statistical")

        if (
            row["isolation_forest_anomaly_flag"]
            and row["isolation_forest_input_valid"]
        ):
            reasons.append("Isolation Forest")

        if not reasons:
            return "No detector flagged"

        return " + ".join(reasons)

    result["hybrid_detection_reason"] = result.apply(
        identify_reason,
        axis=1,
    )

    return result
