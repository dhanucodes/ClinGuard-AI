
"""Statistical Z-score anomaly detection for ClinGuard AI."""

import math
import pandas as pd

from clinguard.data_generation.generator import VITAL_COLUMNS


def detect_statistical_anomalies(
    dataset,
    z_threshold=3.0,
):
    """
    Flag records with unusually high or low vital-sign Z-scores.

    A record is flagged if at least one vital-sign absolute Z-score
    meets or exceeds the configured threshold.
    """

    if dataset.empty:
        raise ValueError("Cannot detect anomalies in an empty dataset.")

    if (
        isinstance(z_threshold, bool)
        or not isinstance(z_threshold, (int, float))
        or not math.isfinite(z_threshold)
        or z_threshold <= 0
    ):
        raise ValueError("z_threshold must be a positive finite number.")

    z_score_columns = [
        f"{vital}_z_score"
        for vital in VITAL_COLUMNS
    ]

    missing_columns = [
        column
        for column in z_score_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Z-score columns: {missing_columns}"
        )

    result = dataset.copy()

    # Check whether each vital sign crosses the threshold.
    flags = (
        result[z_score_columns]
        .abs()
        .ge(z_threshold)
        .fillna(False)
    )

    # A record is flagged if any vital sign crosses the threshold.
    result["statistical_anomaly_flag"] = flags.any(axis=1)

    # Record which vital signs caused the detection.
    result["statistical_anomaly_vitals"] = flags.apply(
        lambda row: ", ".join(
            column.replace("_z_score", "")
            for column in row.index[row]
        ),
        axis=1,
    )

    return result
