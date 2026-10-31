"""Test the ClinGuard AI processing service."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.backend.processing import process_patient_history


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_final_patients.csv"
    )

    dataset = pd.read_csv(input_path)

    # Select one patient's complete time series.
    patient_id = dataset["patient_id"].iloc[0]

    patient_history = dataset[
        dataset["patient_id"] == patient_id
    ].copy()

    print("\nCLINGUARD AI - PROCESSING SERVICE TEST")
    print("=" * 45)
    print("Patient:", patient_id)
    print("Historical records:", len(patient_history))

    result = process_patient_history(patient_history)

    print("\nProcessing status:", result["status"])
    print("\nWorkflow summary:")

    for key, value in result["summary"].items():
        print(f"{key}: {value}")

    print("\nValidation report:")

    for key, value in result["validation_report"].items():
        print(f"{key}: {value}")

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_ai_processing_test.csv"
    )

    result["dataset"].to_csv(output_path, index=False)

    print("\nAI processing test completed.")
    print("Output saved to:", output_path)


if __name__ == "__main__":
    main()