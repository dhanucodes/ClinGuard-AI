
"""Run context-aware verification for ClinGuard AI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.verification.context_verification import (
    verify_anomaly_context,
)


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_hybrid_detection.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_context_verified.csv"
    )

    dataset = load_dataset(input_path)

    # Begin with OR strategy so that either detector can raise a candidate.
    result = verify_anomaly_context(
        dataset,
        strategy="or",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nCLINGUARD AI - CONTEXT VERIFICATION")
    print("=" * 50)

    print("Total records:", len(result))

    print("\nVerification status breakdown:")
    print(result["verification_status"].value_counts().to_string())

    print("\nSample anomaly candidates:")

    candidates = result[
        result["hybrid_or_flag"]
    ][
        [
            "patient_id",
            "timestamp",
            "verification_status",
            "context_issues",
            "verification_reason",
        ]
    ].head(10)

    if candidates.empty:
        print("No anomaly candidates found.")

    else:
        print(candidates.to_string(index=False))

    print("\nVerification dataset saved!")
    print("Location:", output_path)


if __name__ == "__main__":
    main()
