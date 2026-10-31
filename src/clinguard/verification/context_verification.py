
"""Context-aware anomaly verification for ClinGuard AI."""

import pandas as pd


QUALITY_FLAGS = [
    "has_missing_value",
    "has_noise",
    "has_sensor_error",
    "data_quality_issue",
]

VALIDATION_FLAGS = [
    "invalid_timestamp",
    "timestamp_interval_mismatch",
    "has_implausible_value",
]


def to_boolean(value):
    """Safely convert CSV and Pandas values into Boolean values."""

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes"}


def verify_anomaly_context(dataset, strategy="or"):

    if dataset.empty:
        raise ValueError("Cannot verify an empty dataset.")

    if strategy not in {"or", "and"}:
        raise ValueError("Strategy must be 'or' or 'and'.")

    candidate_column = (
        "hybrid_or_flag" if strategy == "or"
        else "hybrid_and_flag"
    )

    required_columns = [
        candidate_column,
        "baseline_ready",
        *QUALITY_FLAGS,
        *VALIDATION_FLAGS,
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

    for column in required_columns:
        result[column] = result[column].map(to_boolean).astype(bool)

    statuses = []
    reasons = []
    quality_summaries = []

    for _, row in result.iterrows():

        candidate = row[candidate_column]
        baseline_ready = row["baseline_ready"]

        quality_issues = [
            flag for flag in QUALITY_FLAGS
            if row[flag]
        ]

        validation_issues = [
            flag for flag in VALIDATION_FLAGS
            if row[flag]
        ]

        issues = quality_issues + validation_issues

        if not baseline_ready:
            issues.append("baseline_not_ready")

        quality_summaries.append(
            ", ".join(issues) if issues else "No recorded issues"
        )

        if candidate and issues:
            status = "candidate_needs_review"
            reason = (
                "A detector identified an anomaly candidate, "
                "but contextual quality or baseline issues require review."
            )

        elif candidate:
            status = "supported_anomaly_candidate"
            reason = (
                "A detector identified an anomaly candidate "
                "with no recorded quality or validation issues "
                "and an available adaptive baseline."
            )

        elif issues:
            status = "quality_review_without_model_alert"
            reason = (
                "No hybrid anomaly was detected, "
                "but contextual data or baseline issues require review."
            )

        else:
            status = "no_model_alert"
            reason = "No hybrid anomaly or recorded context issue was found."

        statuses.append(status)
        reasons.append(reason)

    result["verification_status"] = statuses
    result["verification_reason"] = reasons
    result["context_issues"] = quality_summaries
    result["verification_strategy"] = strategy.upper()

    return result
