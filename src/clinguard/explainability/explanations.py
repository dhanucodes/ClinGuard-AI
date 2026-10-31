
"""Evidence-based anomaly explanations for ClinGuard AI."""

import pandas as pd

from clinguard.data_generation.generator import VITAL_COLUMNS


def explain_anomaly_records(dataset):

    if dataset.empty:
        raise ValueError("Cannot explain an empty dataset.")

    required_columns = [
        "patient_id",
        "timestamp",
        "hybrid_or_flag",
        "hybrid_and_flag",
        "statistical_anomaly_flag",
        "isolation_forest_anomaly_flag",
        "verification_status",
        "verification_reason",
        "context_issues",
    ]

    for vital in VITAL_COLUMNS:
        required_columns.extend([
            vital,
            f"{vital}_baseline",
            f"{vital}_deviation",
            f"{vital}_z_score",
        ])

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = dataset.copy()

    explanations = []
    deviating_vitals = []
    explanation_summaries = []

    for _, row in result.iterrows():

        vital_evidence = []

        for vital in VITAL_COLUMNS:

            current = row[vital]
            baseline = row[f"{vital}_baseline"]
            deviation = row[f"{vital}_deviation"]
            z_score = row[f"{vital}_z_score"]

            if pd.isna(current) or pd.isna(baseline):
                continue

            if pd.isna(deviation):
                continue

            if pd.notna(z_score) and abs(float(z_score)) >= 3.0:
                direction = (
                    "above" if deviation > 0 else "below"
                )

                vital_evidence.append(
                    f"{vital}: {float(current):.2f}, "
                    f"baseline {float(baseline):.2f}, "
                    f"{abs(float(deviation)):.2f} {direction}, "
                    f"Z-score {float(z_score):.2f}"
                )

        if vital_evidence:
            vital_summary = "; ".join(vital_evidence)
        else:
            vital_summary = (
                "No vital sign had an available absolute Z-score "
                "of at least 3.0."
            )

        detector_evidence = []

        if bool(row["statistical_anomaly_flag"]):
            detector_evidence.append("Statistical Z-score detector")

        if bool(row["isolation_forest_anomaly_flag"]):
            detector_evidence.append("Isolation Forest")

        if detector_evidence:
            detector_summary = ", ".join(detector_evidence)
        else:
            detector_summary = "Neither detector flagged this record"

        explanation = (
            f"Patient: {row['patient_id']}. "
            f"Detectors: {detector_summary}. "
            f"Vital-sign evidence: {vital_summary}. "
            f"Context issues: {row['context_issues']}. "
            f"Verification: {row['verification_status']}. "
            f"Reason: {row['verification_reason']}"
        )

        explanations.append(explanation)
        deviating_vitals.append(vital_summary)

        if row["hybrid_or_flag"]:
            explanation_summaries.append(
                "Anomaly candidate identified. " + detector_summary
            )
        else:
            explanation_summaries.append(
                "No hybrid OR anomaly candidate identified."
            )

    result["deviating_vital_evidence"] = deviating_vitals
    result["explanation_summary"] = explanation_summaries
    result["explainable_reason"] = explanations

    return result
