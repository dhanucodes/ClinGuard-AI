
"""Compare statistical and Isolation Forest anomaly detectors."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.evaluation.metrics import evaluate_detector


def main():

    statistical_path = (
        ROOT / "data" / "processed"
        / "clinguard_statistical_detection.csv"
    )

    isolation_path = (
        ROOT / "data" / "processed"
        / "clinguard_isolation_forest.csv"
    )

    statistical = pd.read_csv(statistical_path)
    isolation = pd.read_csv(isolation_path)

    keys = ["patient_id", "timestamp"]

    statistical_columns = keys + [
        "anomaly_label",
        "statistical_anomaly_flag",
    ]

    isolation_columns = keys + [
        "anomaly_label",
        "isolation_forest_anomaly_flag",
        "isolation_forest_input_valid",
    ]

    # Join both model outputs using patient and timestamp.
    combined = statistical[statistical_columns].merge(
        isolation[isolation_columns],
        on=keys,
        how="inner",
        suffixes=("_statistical", "_isolation"),
        validate="one_to_one",
    )

    if len(combined) != len(statistical) or len(combined) != len(isolation):
        raise ValueError(
            "The two model datasets do not contain matching patient records."
        )

    if not (
        combined["anomaly_label_statistical"]
        == combined["anomaly_label_isolation"]
    ).all():
        raise ValueError("Ground-truth labels do not match between datasets.")

    # Use only records with complete Z-score inputs for both models.
    common = combined[
        combined["isolation_forest_input_valid"].astype(bool)
    ].copy()

    if common.empty:
        raise ValueError("No common valid records are available.")

    actual = common["anomaly_label_statistical"].astype(int)

    statistical_predictions = (
        common["statistical_anomaly_flag"].astype(int)
    )

    isolation_predictions = (
        common["isolation_forest_anomaly_flag"].astype(int)
    )

    statistical_metrics = evaluate_detector(
        actual, statistical_predictions
    )

    isolation_metrics = evaluate_detector(
        actual, isolation_predictions
    )

    total_records = len(combined)
    valid_records = len(common)
    coverage = round(valid_records / total_records * 100, 2)

    statistical_metrics["coverage_percentage"] = coverage
    isolation_metrics["coverage_percentage"] = coverage

    comparison = pd.DataFrame(
        [
            {
                "model": "Statistical Z-score",
                **statistical_metrics,
            },
            {
                "model": "Isolation Forest",
                **isolation_metrics,
            },
        ]
    )

    output_path = (
        ROOT / "data" / "processed"
        / "detection_model_evaluation.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    print("\nCLINGUARD AI - MODEL EVALUATION")
    print("=" * 55)
    print("Total records:", total_records)
    print("Common valid records:", valid_records)
    print("Excluded records:", total_records - valid_records)
    print("Evaluation coverage:", coverage, "%")

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    print("\nEvaluation saved successfully!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
