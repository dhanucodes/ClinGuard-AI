
"""Run the complete ClinGuard AI preprocessing pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import (
    load_dataset,
    inspect_dataset,
)
from clinguard.preprocessing.quality_assessment import (
    assess_data_quality,
)
from clinguard.preprocessing.missing_values import (
    handle_missing_values,
)
from clinguard.preprocessing.validation import (
    validate_dataset,
)


def main():

    # Define dataset locations.
    raw_path = (
        ROOT / "data" / "synthetic"
        / "clinguard_synthetic_patients.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_final_patients.csv"
    )

    # STEP 1: Load original dataset.
    raw_data = load_dataset(raw_path)

    print("\nCLINGUARD AI - COMPLETE PREPROCESSING")
    print("=" * 48)

    print("\nSTEP 1: Dataset inspection")

    inspection_report = inspect_dataset(raw_data)

    print("Records:", inspection_report["total_records"])
    print("Patients:", inspection_report["total_patients"])
    print("Duplicate rows:", inspection_report["duplicate_rows"])
    print(
        "Invalid timestamps:",
        inspection_report["invalid_timestamps"],
    )

    # STEP 2: Assess original data quality.
    print("\nSTEP 2: Data-quality assessment")

    quality_report = assess_data_quality(raw_data)

    print(
        "Records with quality issues:",
        quality_report["affected_records"],
    )

    print(
        "Affected percentage:",
        quality_report["affected_percentage"],
        "%",
    )

    # STEP 3: Handle missing values.
    print("\nSTEP 3: Missing-value handling")

    processed_data, remaining_missing = handle_missing_values(
        raw_data,
        max_forward_fill=1,
    )

    print(
        "Original missing readings:",
        sum(
            inspection_report["missing_values"].values()
        ),
    )

    print(
        "Remaining missing readings:",
        sum(remaining_missing.values()),
    )

    # STEP 4: Validate processed data.
    print("\nSTEP 4: Timestamp and range validation")

    validated_data, validation_report = validate_dataset(
        processed_data,
        expected_interval_minutes=5,
    )

    print(
        "Invalid timestamps:",
        validation_report["invalid_timestamps"],
    )

    print(
        "Timestamp interval mismatches:",
        validation_report["timestamp_interval_mismatches"],
    )

    print(
        "Implausible-value records:",
        validation_report["implausible_value_records"],
    )

    print(
        "Records with validation issues:",
        validation_report["validation_issue_records"],
    )

    # STEP 5: Save final dataset.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated_data.to_csv(
        output_path,
        index=False,
    )

    print("\nFINAL SUMMARY")
    print("=" * 48)

    print("Original records:", len(raw_data))
    print("Final records:", len(validated_data))
    print(
        "Patients:",
        validated_data["patient_id"].nunique(),
    )

    print("\nFinal dataset saved successfully!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
