
"""Run patient-specific adaptive baseline modelling."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.adaptive_baseline.baseline import (
    calculate_adaptive_baselines,
)


def main():

    # Input: final validated dataset.
    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_final_patients.csv"
    )

    # Output: dataset containing patient-specific baselines.
    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_baseline_patients.csv"
    )

    # 1. Load the validated dataset.
    dataset = load_dataset(input_path)

    print("\nCLINGUARD AI - ADAPTIVE BASELINE MODELLING")
    print("=" * 52)

    print("Input records:", len(dataset))
    print("Patients:", dataset["patient_id"].nunique())

    # 2. Calculate patient-specific baselines.
    result = calculate_adaptive_baselines(
        dataset,
        window_size=12,
        min_periods=6,
    )

    # 3. Save the baseline-enhanced dataset.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    # 4. Calculate baseline availability.
    ready_count = int(result["baseline_ready"].sum())

    unavailable_count = len(result) - ready_count

    print("\nBASELINE SUMMARY")
    print("-" * 40)

    print("Total records:", len(result))
    print("Records with ready baselines:", ready_count)
    print("Records without ready baselines:", unavailable_count)

    print("\nBaseline availability by patient:")

    patient_summary = (
        result.groupby("patient_id")["baseline_ready"]
        .sum()
    )

    for patient_id, count in patient_summary.items():

        print(
            f"{patient_id}: {count} records with ready baselines"
        )

    # 5. Display a few baseline examples.
    print("\nSAMPLE BASELINE VALUES")

    display_columns = [
        "patient_id",
        "timestamp",
        "heart_rate",
        "heart_rate_baseline",
        "spo2",
        "spo2_baseline",
        "baseline_ready",
    ]

    print(
        result[display_columns]
        .groupby("patient_id")
        .head(2)
        .to_string(index=False)
    )

    print("\nAdaptive baseline modelling completed!")
    print("Saved at:", output_path)


if __name__ == "__main__":
    main()
