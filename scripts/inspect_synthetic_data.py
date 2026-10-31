
"""Run the initial inspection of ClinGuard AI synthetic data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import (
    load_dataset,
    inspect_dataset,
    print_inspection_report,
)


def main():

    dataset_path = (
        ROOT
        / "data"
        / "synthetic"
        / "clinguard_synthetic_patients.csv"
    )

    dataset = load_dataset(dataset_path)

    report = inspect_dataset(dataset)

    print_inspection_report(report)


if __name__ == "__main__":
    main()
