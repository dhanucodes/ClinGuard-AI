
"""Patient-specific adaptive baseline modelling for ClinGuard AI."""

import pandas as pd

from clinguard.data_generation.generator import VITAL_COLUMNS


def calculate_adaptive_baselines(
    dataset,
    window_size=12,
    min_periods=6,
):
    """
    Calculate patient-specific rolling median baselines.

    Only previous observations are used.
    The current reading is excluded from its own baseline.
    """

    if dataset.empty:
        raise ValueError("Cannot calculate baselines for an empty dataset.")

    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer.")

    if (
        not isinstance(min_periods, int)
        or min_periods < 1
        or min_periods > window_size
    ):
        raise ValueError(
            "min_periods must be between 1 and window_size."
        )

    required_columns = ["patient_id", "timestamp", *VITAL_COLUMNS]

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work on a copy and arrange each patient's readings chronologically.
    result = dataset.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps.")

    result = result.sort_values(
        ["patient_id", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    # Calculate a separate rolling median for every patient and vital.
    for vital in VITAL_COLUMNS:

        baseline_column = f"{vital}_baseline"

        result[baseline_column] = (
            result.groupby("patient_id")[vital]
            .transform(
                lambda series: (
                    series.rolling(
                        window=window_size,
                        min_periods=min_periods,
                    )
                    .median()
                    .shift(1)
                )
            )
        )

    # A baseline is ready only when every vital has an estimate.
    baseline_columns = [
        f"{vital}_baseline"
        for vital in VITAL_COLUMNS
    ]

    result["baseline_ready"] = (
        result[baseline_columns].notna().all(axis=1)
    )

    return result
