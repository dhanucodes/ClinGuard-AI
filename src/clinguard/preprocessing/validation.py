
"""Timestamp and physiological value validation for ClinGuard AI."""

import pandas as pd

from clinguard.data_generation.generator import VITAL_COLUMNS


# Broad data plausibility bounds for synthetic-data validation.
# These are NOT clinical diagnostic or treatment thresholds.
PLAUSIBILITY_RANGES = {
    "heart_rate": (30, 220),
    "spo2": (50, 100),
    "systolic_bp": (50, 250),
    "diastolic_bp": (30, 150),
    "respiratory_rate": (4, 60),
    "temperature": (30, 43),
}


def validate_dataset(dataset, expected_interval_minutes=5):
    """Identify timestamp inconsistencies and implausible values."""

    if dataset.empty:
        raise ValueError("Cannot validate an empty dataset.")

    if (
        not isinstance(expected_interval_minutes, int)
        or expected_interval_minutes <= 0
    ):
        raise ValueError(
            "expected_interval_minutes must be a positive integer."
        )

    required_columns = ["patient_id", "timestamp", *VITAL_COLUMNS]

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = dataset.copy()

    # Parse timestamps without changing the original timestamp column.
    parsed = pd.to_datetime(
        result["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    result["invalid_timestamp"] = parsed.isna()

    # Check time gaps separately for each patient.
    ordered = pd.DataFrame({
        "patient_id": result["patient_id"],
        "timestamp": parsed,
    }, index=result.index).sort_values(
        ["patient_id", "timestamp"],
        kind="stable",
        na_position="last",
    )

    time_gaps = (
        ordered.groupby("patient_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    ordered["timestamp_interval_mismatch"] = (
        time_gaps.notna()
        & time_gaps.ne(expected_interval_minutes)
    )

    result["timestamp_interval_mismatch"] = (
        ordered["timestamp_interval_mismatch"]
        .reindex(result.index)
        .fillna(False)
        .astype(bool)
    )

    # Check each vital sign against its broad plausibility bounds.
    range_flags = []

    for vital, (lower, upper) in PLAUSIBILITY_RANGES.items():

        values = pd.to_numeric(
            result[vital],
            errors="coerce",
        )

        flag_column = f"{vital}_out_of_range"

        result[flag_column] = (
            values.notna()
            & ((values < lower) | (values > upper))
        )

        range_flags.append(flag_column)

    # A row is flagged if any vital sign falls outside its bounds.
    result["has_implausible_value"] = result[
        range_flags
    ].any(axis=1)

    # Combine the new checks into one validation flag.
    result["has_validation_issue"] = (
        result["invalid_timestamp"]
        | result["timestamp_interval_mismatch"]
        | result["has_implausible_value"]
    )

    report = {
        "total_records": len(result),
        "invalid_timestamps": int(
            result["invalid_timestamp"].sum()
        ),
        "timestamp_interval_mismatches": int(
            result["timestamp_interval_mismatch"].sum()
        ),
        "implausible_value_records": int(
            result["has_implausible_value"].sum()
        ),
        "validation_issue_records": int(
            result["has_validation_issue"].sum()
        ),
    }

    return result, report


def print_validation_report(report):
    """Display timestamp and range validation results."""

    print("\nCLINGUARD AI - VALIDATION REPORT")
    print("=" * 45)

    print("Total records:", report["total_records"])
    print("Invalid timestamps:", report["invalid_timestamps"])
    print(
        "Timestamp interval mismatches:",
        report["timestamp_interval_mismatches"],
    )
    print(
        "Records with implausible values:",
        report["implausible_value_records"],
    )
    print(
        "Records with validation issues:",
        report["validation_issue_records"],
    )

    print("\nValidation completed.")
