
"""Run the ClinGuard AI preprocessing pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import (
    load_dataset,
)
from clinguard.preprocessing.missing_values import (
    handle_missing_values,
)
from clinguard.data_generation.generator import VITAL_COLUMNS


def main():

    # 1. Define input and output paths.
    raw_path = (
        ROOT
        / "data"
        / "synthetic"
        / "clinguard_synthetic_patients.csv"
    )

    processed_path = (
        ROOT
        / "data"
        / "processed"
        / "clinguard_processed_patients.csv"
    )

    # 2. Load the original dataset.
    dataset = load_dataset(raw_path)

    # 3. Count missing values before processing.
    missing_before = (
        dataset[VITAL_COLUMNS]
        .isna()
        .sum()
    )

    total_missing_before = int(missing_before.sum())

    # 4. Apply missing-value handling.
    processed, remaining_missing = handle_missing_values(
        dataset,
        max_forward_fill=1,
    )

    # 5. Save the processed dataset separately.
    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        processed_path,
        index=False,
    )

    # 6. Calculate processing statistics.
    total_missing_after = sum(
        remaining_missing.values()
    )

    filled_count = (
        total_missing_before - total_missing_after
    )

    # 7. Display the report.
    print("\nCLINGUARD AI - PREPROCESSING REPORT")
    print("=" * 45)

    print("Original records:", len(dataset))
    print("Processed records:", len(processed))
    print("Total patients:", processed["patient_id"].nunique())

    print("\nMissing readings before processing:")

    for vital, count in missing_before.items():
        print(f"  {vital}: {count}")

    print("\nMissing readings after processing:")

    for vital, count in remaining_missing.items():
        print(f"  {vital}: {count}")

    print("\nProcessing summary:")
    print("Missing readings before:", total_missing_before)
    print("Missing readings after:", total_missing_after)
    print("Readings forward-filled:", filled_count)

    print("\nProcessed dataset saved successfully!")
    print("Location:", processed_path)


if __name__ == "__main__":
    main()
