
"""Run Isolation Forest on ClinGuard AI patient data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.detection.isolation_forest import (
    detect_isolation_forest_anomalies,
)


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_deviation_patients.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_isolation_forest.csv"
    )

    dataset = load_dataset(input_path)

    result = detect_isolation_forest_anomalies(
        dataset,
        contamination=0.05,
        random_state=42,
    )

    total_records = len(result)
    valid_records = int(result["isolation_forest_input_valid"].sum())
    excluded_records = total_records - valid_records
    detected_records = int(
        result["isolation_forest_anomaly_flag"].sum()
    )

    detection_percentage = (
        detected_records / valid_records * 100
        if valid_records else 0
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nCLINGUARD AI - ISOLATION FOREST")
    print("=" * 45)
    print("Total records:", total_records)
    print("Valid records:", valid_records)
    print("Excluded records:", excluded_records)
    print("Detected anomalies:", detected_records)
    print(
        "Detection percentage:",
        round(detection_percentage, 2), "%"
    )

    print("\nSample detected records:")

    samples = result[
        result["isolation_forest_anomaly_flag"]
    ][
        [
            "patient_id",
            "timestamp",
            "isolation_forest_score",
            "anomaly_label",
            "scenario_type",
        ]
    ].head(10)

    print(samples.to_string(index=False))

    print("\nDataset saved successfully!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
