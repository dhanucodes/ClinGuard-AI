
"""Run the ClinGuard AI synthetic data generator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

# Allow Python to locate our source package.
sys.path.insert(0, str(ROOT / "src"))

from clinguard.data_generation.generator import (
    generate_dataset,
    save_dataset,
)


def main():

    dataset = generate_dataset(
        n_patients=8,
        duration_hours=24,
        interval_minutes=5,
        seed=42,
    )

    output_path = (
        ROOT / "data" / "synthetic"
        / "clinguard_synthetic_patients.csv"
    )

    save_dataset(dataset, output_path)

    print("\nClinGuard AI - Synthetic Data Generation")
    print("----------------------------------------")
    print("Total records:", len(dataset))
    print("Total patients:", dataset["patient_id"].nunique())
    print("Anomaly records:", dataset["anomaly_label"].sum())
    print("Data-quality issue records:", dataset["data_quality_issue"].sum())
    print("\nDataset saved successfully!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
