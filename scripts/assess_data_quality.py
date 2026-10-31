
"""Run ClinGuard AI data-quality assessment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.preprocessing.quality_assessment import (
    assess_data_quality,
    print_quality_report,
)


def main():

    dataset_path = (
        ROOT
        / "data"
        / "processed"
        / "clinguard_processed_patients.csv"
    )

    dataset = load_dataset(dataset_path)

    report = assess_data_quality(dataset)

    print_quality_report(report)


if __name__ == "__main__":
    main()
