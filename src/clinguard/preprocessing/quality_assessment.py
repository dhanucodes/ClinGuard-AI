
"""Data-quality assessment for ClinGuard AI."""

import pandas as pd


QUALITY_FLAGS = [
    "has_missing_value",
    "has_noise",
    "has_sensor_error",
    "has_irregular_timestamp",
    "data_quality_issue",
]


def assess_data_quality(dataset):
    """Calculate record-level data-quality statistics."""

    if dataset.empty:
        raise ValueError("Cannot assess an empty dataset.")

    missing_columns = [
        column
        for column in QUALITY_FLAGS
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing quality columns: {missing_columns}"
        )

    total_records = len(dataset)

    issue_counts = {}

    for flag in QUALITY_FLAGS:

        count = int(dataset[flag].fillna(False).astype(bool).sum())

        issue_counts[flag] = {
            "count": count,
            "percentage": round(
                (count / total_records) * 100, 2
            ),
        }

    # Count originally missing vital-sign measurements.
    original_missing_columns = [
        column
        for column in dataset.columns
        if column.endswith("_was_missing")
    ]

    original_missing_counts = {}

    for column in original_missing_columns:

        vital = column.replace("_was_missing", "")

        original_missing_counts[vital] = int(
            dataset[column].fillna(False).sum()
        )

    affected_records = int(
        dataset["data_quality_issue"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    unaffected_records = total_records - affected_records

    report = {
        "total_records": total_records,
        "affected_records": affected_records,
        "unaffected_records": unaffected_records,
        "affected_percentage": round(
            (affected_records / total_records) * 100, 2
        ),
        "issue_counts": issue_counts,
        "original_missing_counts": original_missing_counts,
    }

    return report


def print_quality_report(report):
    """Print a readable data-quality report."""

    print("\nCLINGUARD AI - DATA QUALITY ASSESSMENT")
    print("=" * 48)

    print("Total records:", report["total_records"])
    print("Records with quality issues:", report["affected_records"])
    print("Records without quality issues:", report["unaffected_records"])
    print("Affected record percentage:", report["affected_percentage"], "%")

    print("\nQuality issue breakdown:")

    for issue, values in report["issue_counts"].items():

        print(
            f"{issue}: "
            f"{values['count']} records "
            f"({values['percentage']}%)"
        )

    print("\nOriginally missing vital readings:")

    for vital, count in report["original_missing_counts"].items():

        print(f"{vital}: {count}")

    print("\nQuality assessment completed.")
