
"""Investigate validation flags and their relationship to simulated events."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from clinguard.preprocessing.validation import PLAUSIBILITY_RANGES


def main():

    file_path = (
        ROOT / "data" / "processed"
        / "clinguard_final_patients.csv"
    )

    dataset = pd.read_csv(file_path)

    print("\nCLINGUARD AI - VALIDATION ISSUE INVESTIGATION")
    print("=" * 55)

    # PART 1: Investigate implausible physiological values.
    range_flag_columns = [
        f"{vital}_out_of_range"
        for vital in PLAUSIBILITY_RANGES
    ]

    implausible_records = dataset[
        dataset[range_flag_columns].any(axis=1)
    ]

    print("\nPART 1: Implausible physiological values")
    print("Affected records:", len(implausible_records))

    if not implausible_records.empty:

        for _, row in implausible_records.iterrows():

            for vital in PLAUSIBILITY_RANGES:

                flag = f"{vital}_out_of_range"

                if row[flag]:

                    print("\nPatient:", row["patient_id"])
                    print("Timestamp:", row["timestamp"])
                    print("Vital:", vital)
                    print("Observed value:", row[vital])
                    print("Allowed engineering range:",
                          PLAUSIBILITY_RANGES[vital])
                    print("Anomaly label:", row["anomaly_label"])
                    print("Scenario:", row["scenario_type"])
                    print("Noise flag:", row["has_noise"])
                    print("Sensor error:", row["has_sensor_error"])
                    print("Data-quality issue:",
                          row["data_quality_issue"])

    # PART 2: Compare timestamp mismatches with injected flags.
    print("\nPART 2: Timestamp mismatch investigation")

    mismatch_records = dataset[
        dataset["timestamp_interval_mismatch"]
    ]

    print(
        "Detected timestamp mismatches:",
        len(mismatch_records),
    )

    print(
        "Mismatches with generator irregular flag:",
        int(mismatch_records["has_irregular_timestamp"].sum()),
    )

    print(
        "Mismatches without generator irregular flag:",
        int((~mismatch_records["has_irregular_timestamp"]).sum()),
    )

    # PART 3: Compare all validation issues with anomaly labels.
    print("\nPART 3: Validation and anomaly overlap")

    validation_records = dataset[
        dataset["has_validation_issue"]
    ]

    print(
        "Validation issue records:",
        len(validation_records),
    )

    print(
        "Also labelled as simulated anomalies:",
        int((validation_records["anomaly_label"] == 1).sum()),
    )

    print(
        "Also flagged with noise:",
        int(validation_records["has_noise"].sum()),
    )

    print(
        "Also flagged with sensor errors:",
        int(validation_records["has_sensor_error"].sum()),
    )

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
