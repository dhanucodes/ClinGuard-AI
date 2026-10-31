
"""Run hybrid anomaly detection for ClinGuard AI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.detection.hybrid import detect_hybrid_anomalies


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_statistical_detection.csv"
    )

    isolation_path = (
        ROOT / "data" / "processed"
        / "clinguard_isolation_forest.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_hybrid_detection.csv"
    )

    statistical = load_dataset(input_path)
    isolation = load_dataset(isolation_path)

    keys = ["patient_id", "timestamp"]

    combined = statistical.merge(
        isolation[
            keys + [
                "isolation_forest_anomaly_flag",
                "isolation_forest_input_valid",
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != len(statistical) or len(combined) != len(isolation):
        raise ValueError("The model datasets do not match.")

    result = detect_hybrid_anomalies(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nCLINGUARD AI - HYBRID DETECTION")
    print("=" * 45)

    print("Total records:", len(result))
    print(
        "Statistical detections:",
        int(result["statistical_anomaly_flag"].sum()),
    )
    print(
        "Isolation Forest detections:",
        int(result["isolation_forest_anomaly_flag"].sum()),
    )
    print(
        "Hybrid OR detections:",
        int(result["hybrid_or_flag"].sum()),
    )
    print(
        "Hybrid AND detections:",
        int(result["hybrid_and_flag"].sum()),
    )

    print("\nDetection reason breakdown:")
    print(result["hybrid_detection_reason"].value_counts())

    print("\nDataset saved successfully!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
