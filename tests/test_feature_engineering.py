"""Tests for transparent feature engineering."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_generation import generate_dataset
from src.feature_engineering import (build_features, compute_case_features, _consecutive_directional,
                                     _robust_z)


def test_consecutive_directional_counts_trailing_run():
    assert _consecutive_directional(np.array([1.0, 2.0, 3.0, 4.0])) == 3
    assert _consecutive_directional(np.array([1.0, 2.0, 1.5, 1.0])) == 2
    assert _consecutive_directional(np.array([1.0, 1.0, 1.0])) == 0


def test_robust_z_zero_for_constant_history_plus_offset():
    hist = np.array([5.0, 5.0, 5.0, 5.0])
    # MAD is zero; falls back to std-based scaling; value equal to median -> 0.
    assert abs(_robust_z(5.0, hist)) < 1e-6


def test_compute_case_features_known_values():
    series = [10.0, 10.0, 10.0, 10.0, 20.0]  # final quarter jumps
    row = {"sample_size": 100, "missing_rate": 0.0}
    f = compute_case_features(series, row)
    assert f["qoq_percent_change"] == pytest.approx(1.0, abs=1e-6)
    assert f["z_score"] > 0
    assert f["historical_percentile_rank"] == 1.0  # current is the max
    assert set(f) >= {"z_score", "robust_z_score", "rolling_slope", "volatility_change"}


def test_build_features_full_dataset_no_nan():
    import json
    from src.utils import DATA_DIR

    df, _, _ = generate_dataset(n_cases=90, seed=9)
    # Build a sidecar in-memory matching this df.
    # Reuse the persisted sidecar requires running main; instead construct directly.
    df_main, _, _ = generate_dataset(n_cases=90, seed=9)
    sidecar = {
        r["case_id"]: {"_series": r["_series"]}
        for r in df_main.to_dict("records")
    }
    feats = build_features(df[[c for c in df.columns if not c.startswith("_")]], sidecar=sidecar)
    assert feats.isna().sum().sum() == 0
    assert feats.shape[0] == len(df)
    assert "cross_segment_deviation" in feats.columns
