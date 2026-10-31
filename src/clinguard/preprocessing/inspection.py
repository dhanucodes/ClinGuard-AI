
"""Dataset loading and initial quality inspection for ClinGuard AI."""

from pathlib import Path

import pandas as pd


VITAL_COLUMNS = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature",
]

REQUIRED_COLUMNS = [
    "patient_id",
    "timestamp",
    *VITAL_COLUMNS,
    "anomaly_label",
    "scenario_type",
    "event_id",
]


def load_dataset(file_path):
    """Load the raw synthetic dataset from a CSV file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError("The dataset is empty.")

    return dataset


def inspect_dataset(dataset):
    """Inspect the structure and basic quality of a dataset."""

    if dataset.empty:
        raise ValueError("Cannot inspect an empty dataset.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Count missing values in each physiological measurement.
    missing_values = (
        dataset[VITAL_COLUMNS]
        .isna()
        .sum()
        .to_dict()
    )

    # Check whether complete rows are duplicated.
    duplicate_rows = int(dataset.duplicated().sum())

    # Check repeated timestamps for the same patient.
    duplicate_patient_timestamps = int(
        dataset.duplicated(
            subset=["patient_id", "timestamp"]
        ).sum()
    )

    # Convert timestamps for validation without changing the input.
    parsed_timestamps = pd.to_datetime(
    dataset["timestamp"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

    invalid_timestamps = int(
        parsed_timestamps.isna().sum()
    )

    report = {
        "total_records": len(dataset),
        "total_patients": dataset["patient_id"].nunique(),
        "total_columns": len(dataset.columns),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_patient_timestamps": duplicate_patient_timestamps,
        "invalid_timestamps": invalid_timestamps,
        "anomaly_records": int(
            (dataset["anomaly_label"] == 1).sum()
        ),
    }

    return report


def print_inspection_report(report):
    """Display the inspection results in a readable format."""

    print("\nCLINGUARD AI - DATASET INSPECTION")
    print("=" * 40)

    print("Total records:", report["total_records"])
    print("Total patients:", report["total_patients"])
    print("Total columns:", report["total_columns"])
    print("Anomaly records:", report["anomaly_records"])

    print("\nMissing values in vital-sign columns:")

    for vital, count in report["missing_values"].items():
        print(f"  {vital}: {count}")

    print("\nDuplicate complete rows:", report["duplicate_rows"])

    print(
        "Duplicate patient-timestamp pairs:",
        report["duplicate_patient_timestamps"]
    )

    print("Invalid timestamps:", report["invalid_timestamps"])

    print("\nInspection completed.")
