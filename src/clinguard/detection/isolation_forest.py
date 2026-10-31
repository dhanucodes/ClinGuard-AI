
"""Isolation Forest anomaly detection for ClinGuard AI."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from clinguard.data_generation.generator import VITAL_COLUMNS


def detect_isolation_forest_anomalies(
    dataset,
    contamination=0.05,
    random_state=42,
):
    if dataset.empty:
        raise ValueError("Cannot detect anomalies in an empty dataset.")

    if (
        isinstance(contamination, bool)
        or not isinstance(contamination, (int, float))
        or not np.isfinite(contamination)
        or not 0 < contamination <= 0.5
    ):
        raise ValueError(
            "contamination must be greater than 0 and at most 0.5."
        )

    feature_columns = [
        f"{vital}_z_score" for vital in VITAL_COLUMNS
    ]

    missing_columns = [
        column for column in feature_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Z-score columns: {missing_columns}"
        )

    result = dataset.copy()

    features = result[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_mask = pd.Series(
        np.isfinite(features.to_numpy()).all(axis=1),
        index=result.index,
    )

    valid_features = features.loc[valid_mask]

    if len(valid_features) < 2:
        raise ValueError(
            "At least two records with complete Z-scores are required."
        )

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )

    model.fit(valid_features)
    predictions = model.predict(valid_features)

    result["isolation_forest_prediction"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result.loc[
        valid_mask, "isolation_forest_prediction"
    ] = predictions

    result["isolation_forest_anomaly_flag"] = (
    result["isolation_forest_prediction"] == -1
).fillna(False).astype(bool)

    result.loc[
        valid_mask, "isolation_forest_score"
    ] = model.decision_function(valid_features)

    result["isolation_forest_input_valid"] = valid_mask

    return result

