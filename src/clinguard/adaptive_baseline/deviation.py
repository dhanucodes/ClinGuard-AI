
"""Calculate deviations from patient-specific adaptive baselines."""

import pandas as pd

from clinguard.data_generation.generator import VITAL_COLUMNS


def calculate_deviations(
    dataset,
    window_size=12,
    min_periods=6,
):
    """
    Calculate signed deviation, absolute deviation and Z-score.

    The standard deviation is calculated from previous readings only.
    The current reading is excluded from its own reference statistics.
    """

    if dataset.empty:
        raise ValueError("Cannot calculate deviations for an empty dataset.")

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

    required_columns = ["patient_id", "timestamp"]

    for vital in VITAL_COLUMNS:
        required_columns.extend([
            vital,
            f"{vital}_baseline",
        ])

    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

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

    for vital in VITAL_COLUMNS:

        baseline_column = f"{vital}_baseline"

        # Current reading minus its previous-reading baseline.
        deviation_column = f"{vital}_deviation"

        result[deviation_column] = (
            result[vital] - result[baseline_column]
        )

        # Magnitude of the difference, regardless of direction.
        result[f"{vital}_absolute_deviation"] = (
            result[deviation_column].abs()
        )

        # Calculate variation using previous readings only.
        previous_std = (
            result.groupby("patient_id")[vital]
            .transform(
                lambda series: (
                    series.rolling(
                        window=window_size,
                        min_periods=min_periods,
                    )
                    .std()
                    .shift(1)
                )
            )
        )

        result[f"{vital}_baseline_std"] = previous_std

        # Avoid division by zero when previous readings do not vary.
        safe_std = previous_std.where(previous_std > 0)

        result[f"{vital}_z_score"] = (
            result[deviation_column] / safe_std
        )

    return result
