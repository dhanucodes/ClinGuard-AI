
"""Run the ClinGuard AI explanation engine."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.explainability.explanations import (
    explain_anomaly_records,
)


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_context_verified.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_explainable_results.csv"
    )

    dataset = load_dataset(input_path)

    result = explain_anomaly_records(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nCLINGUARD AI - EXPLAINABLE AI")
    print("=" * 50)

    print("Total records:", len(result))

    print(
        "Anomaly candidates:",
        int(result["hybrid_or_flag"].sum()),
    )

    print("\nSample explanations:")

    samples = result[
        result["hybrid_or_flag"]
    ][
        [
            "patient_id",
            "timestamp",
            "deviating_vital_evidence",
            "verification_status",
            "explanation_summary",
        ]
    ].head(5)

    if samples.empty:
        print("No anomaly candidates found.")
    else:
        print(samples.to_string(index=False))

    print("\nExplainability results saved!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
