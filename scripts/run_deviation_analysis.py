
"""Run adaptive deviation analysis for all virtual patients."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.adaptive_baseline.deviation import (
    calculate_deviations,
)


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_baseline_patients.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_deviation_patients.csv"
    )

    # 1. Load baseline-enhanced dataset.
    dataset = load_dataset(input_path)

    print("\nCLINGUARD AI - ADAPTIVE DEVIATION ANALYSIS")
    print("=" * 52)

    print("Total records:", len(dataset))
    print("Total patients:", dataset["patient_id"].nunique())

    # 2. Calculate deviations and Z-scores.
    result = calculate_deviations(
        dataset,
        window_size=12,
        min_periods=6,
    )

    # 3. Save the results.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    # 4. Display summary.
    print("\nDEVIATION SUMMARY")

    print(
        "Records with ready baselines:",
        int(result["baseline_ready"].sum()),
    )

    print(
        "Records with unavailable baselines:",
        int((~result["baseline_ready"]).sum()),
    )

    print("\nAvailable Z-scores by vital:")

    for vital in [
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature",
    ]:

        column = f"{vital}_z_score"

        print(
            vital,
            ":",
            int(result[column].notna().sum()),
        )

    # 5. Display sample readings.
    display_columns = [
        "patient_id",
        "timestamp",
        "heart_rate",
        "heart_rate_baseline",
        "heart_rate_deviation",
        "heart_rate_z_score",
    ]

    print("\nSAMPLE DEVIATION RESULTS")

    print(
        result[display_columns]
        .groupby("patient_id")
        .tail(2)
        .to_string(index=False)
    )

    print("\nDeviation analysis completed!")
    print("Saved at:", output_path)


if __name__ == "__main__":
    main()
