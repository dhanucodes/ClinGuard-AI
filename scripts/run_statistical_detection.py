
"""Run statistical Z-score anomaly detection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.detection.statistical import (
    detect_statistical_anomalies,
)


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_deviation_patients.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_statistical_detection.csv"
    )

    # 1. Load deviation-enhanced data.
    dataset = load_dataset(input_path)

    print("\nCLINGUARD AI - STATISTICAL ANOMALY DETECTION")
    print("=" * 55)

    # 2. Run statistical detection.
    result = detect_statistical_anomalies(
        dataset,
        z_threshold=3.0,
    )

    # 3. Calculate detection summary.
    detected = int(result["statistical_anomaly_flag"].sum())
    total = len(result)

    print("Total records:", total)
    print("Statistically flagged records:", detected)
    print(
        "Flagged percentage:",
        round((detected / total) * 100, 2),
        "%",
    )

    print("\nDetections by vital sign:")

    for vital in [
        "heart_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "temperature",
    ]:

        count = int(
            result["statistical_anomaly_vitals"]
            .str.split(", ")
            .apply(lambda values: vital in values)
            .sum()
        )

        print(f"{vital}: {count}")

    # 4. Save results.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\nSample detections:")

    print(
        result.loc[
            result["statistical_anomaly_flag"],
            [
                "patient_id",
                "timestamp",
                "statistical_anomaly_vitals",
                "anomaly_label",
                "scenario_type",
            ],
        ].head(10).to_string(index=False)
    )

    print("\nStatistical detection completed!")
    print("Saved at:", output_path)


if __name__ == "__main__":
    main()
