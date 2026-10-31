
"""Evaluate hybrid anomaly detection strategies."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.evaluation.metrics import evaluate_detector


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_hybrid_detection.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "hybrid_detection_evaluation.csv"
    )

    dataset = pd.read_csv(input_path)

    # Use only records with valid Isolation Forest inputs.
    valid = dataset[
        dataset["isolation_forest_input_valid"].astype(bool)
    ].copy()

    if valid.empty:
        raise ValueError("No valid records available for evaluation.")

    actual = valid["anomaly_label"].astype(int)

    strategies = {
        "Hybrid OR": valid["hybrid_or_flag"].astype(int),
        "Hybrid AND": valid["hybrid_and_flag"].astype(int),
    }

    results = []

    for strategy_name, predictions in strategies.items():

        metrics = evaluate_detector(actual, predictions)

        metrics["model"] = strategy_name
        metrics["evaluation_records"] = len(valid)

        results.append(metrics)

    comparison = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    print("\nCLINGUARD AI - HYBRID EVALUATION")
    print("=" * 50)

    print("Total records:", len(dataset))
    print("Valid evaluation records:", len(valid))
    print("Excluded records:", len(dataset) - len(valid))

    print("\nHYBRID STRATEGY COMPARISON")
    print(comparison.to_string(index=False))

    print("\nEvaluation saved successfully!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()

