
"""Missing-value handling for ClinGuard AI."""

import pandas as pd

from clinguard.data_generation.generator import VITAL_COLUMNS


def handle_missing_values(dataset, max_forward_fill=1):
    """
    Handle short missing-value gaps using forward filling.

    Original missing-value indicators are preserved.
    Longer gaps remain missing.
    """

    if dataset.empty:
        raise ValueError("Cannot process an empty dataset.")

    if not isinstance(max_forward_fill, int) or max_forward_fill < 0:
        raise ValueError(
            "max_forward_fill must be a non-negative integer."
        )

    missing_columns = [
        column
        for column in ["patient_id", *VITAL_COLUMNS]
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work on a copy so the original data is never modified.
    processed = dataset.copy()

    # Record which readings were missing before filling.
    for vital in VITAL_COLUMNS:

        processed[f"{vital}_was_missing"] = (
            processed[vital].isna()
        )

    # Sort records chronologically within each patient.
    processed = processed.sort_values(
        ["patient_id", "timestamp"]
    ).reset_index(drop=True)

    # Fill short gaps separately for each patient.
    for vital in VITAL_COLUMNS:

        processed[vital] = (
            processed.groupby("patient_id")[vital]
            .transform(
                lambda series: series.ffill(
                    limit=max_forward_fill
                )
            )
        )

    # Count remaining missing values after processing.
    remaining_missing = (
        processed[VITAL_COLUMNS]
        .isna()
        .sum()
        .to_dict()
    )

    return processed, remaining_missing
