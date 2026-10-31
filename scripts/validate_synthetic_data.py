
"""Run timestamp and physiological range validation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.preprocessing.validation import (
    validate_dataset,
    print_validation_report,
)


def main():

    input_path = (
        ROOT
        / "data"
        / "processed"
        / "clinguard_processed_patients.csv"
    )

    output_path = (
        ROOT
        / "data"
        / "processed"
        / "clinguard_validated_patients.csv"
    )

    dataset = load_dataset(input_path)

    validated, report = validate_dataset(
        dataset,
        expected_interval_minutes=5,
    )

    validated.to_csv(output_path, index=False)

    print_validation_report(report)

    print("\nValidated dataset saved at:")
    print(output_path)


if __name__ == "__main__":
    main()
