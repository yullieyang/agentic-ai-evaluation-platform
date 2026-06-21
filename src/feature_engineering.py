"""Transparent monitoring feature engineering.

All features are deterministic functions of a case's quarterly history and a
small amount of cross-case context. Formulas and assumptions are documented
inline. Ground-truth fields are never read here.

Cross-segment and cross-metric features are computed over peer cases sharing the
same model (and metric, for the cross-segment view). Because the synthetic
dataset is case-structured rather than a dense panel, these peer features are
documented approximations intended to expose contradiction signals, not exact
panel statistics.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .utils import DATA_DIR, get_logger

LOGGER = get_logger("feature_engineering")


def _safe_std(values: np.ndarray) -> float:
    return float(max(np.std(values), 1e-9))


def _robust_z(current: float, hist: np.ndarray) -> float:
    """Robust z-score using median and the median absolute deviation (MAD).

    robust_z = (x - median) / (1.4826 * MAD); the constant scales MAD to be a
    consistent estimator of the standard deviation under normality.
    """
    median = float(np.median(hist))
    mad = float(np.median(np.abs(hist - median)))
    scale = 1.4826 * mad if mad > 0 else _safe_std(hist)
    return (current - median) / (scale + 1e-9)


def _consecutive_directional(series: np.ndarray) -> int:
    """Number of trailing quarters moving in a single, consistent direction."""
    diffs = np.diff(series)
    if len(diffs) == 0:
        return 0
    last_sign = np.sign(diffs[-1])
    if last_sign == 0:
        return 0
    count = 0
    for d in diffs[::-1]:
        if np.sign(d) == last_sign and d != 0:
            count += 1
        else:
            break
    return int(count)


def _rolling_slope(series: np.ndarray, window: int = 4) -> float:
    """OLS slope of the last ``window`` quarters (per-quarter change)."""
    w = series[-window:]
    x = np.arange(len(w))
    if len(w) < 2:
        return 0.0
    return float(np.polyfit(x, w, 1)[0])


def _historical_percentile_rank(current: float, hist: np.ndarray) -> float:
    """Percentile rank of the current value within the historical distribution."""
    return float((np.sum(hist <= current) / len(hist)))


def _recovery_indicator(series: np.ndarray) -> float:
    """1.0 if a prior spike has returned to baseline by the final quarter.

    Heuristic: a value in the historical window exceeded mean + 3*std, and the
    final quarter is within 1.5*std of the pre-final baseline.
    """
    hist = series[:-1]
    if len(hist) < 3:
        return 0.0
    base_mean = float(np.mean(hist))
    base_std = _safe_std(hist)
    had_spike = bool(np.any(np.abs(hist - base_mean) > 3.0 * base_std))
    recovered = abs(series[-1] - base_mean) < 1.5 * base_std
    return 1.0 if (had_spike and recovered) else 0.0


def compute_case_features(series: list[float], row: dict[str, Any]) -> dict[str, float]:
    """Compute per-case features from the quarterly series and row scalars."""
    arr = np.asarray(series, dtype=float)
    current = float(arr[-1])
    hist = arr[:-1]
    hist_mean = float(np.mean(hist))
    hist_std = _safe_std(hist)
    previous = float(arr[-2])
    yoy_ref = float(arr[-5]) if len(arr) >= 5 else float(arr[0])
    rolling = arr[-4:]
    sample_size = float(row.get("sample_size", 1) or 1)
    missing_rate = float(row.get("missing_rate", 0.0))

    return {
        "qoq_percent_change": (current - previous) / (abs(previous) + 1e-9),
        "yoy_percent_change": (current - yoy_ref) / (abs(yoy_ref) + 1e-9),
        "rolling_mean": float(np.mean(rolling)),
        "rolling_std": _safe_std(rolling),
        "z_score": (current - hist_mean) / hist_std,
        "robust_z_score": _robust_z(current, hist),
        "rolling_slope": _rolling_slope(arr),
        "consecutive_directional_moves": float(_consecutive_directional(arr)),
        "sample_size_adjusted_variability": hist_std / np.sqrt(max(sample_size, 1.0)),
        "missingness_level": missing_rate,
        "missingness_trend": missing_rate - 0.01,  # baseline missingness ~1%
        "volatility_change": _safe_std(rolling) / hist_std,
        "historical_percentile_rank": _historical_percentile_rank(current, arr),
        "recovery_indicator": _recovery_indicator(arr),
    }


def _load_sidecar() -> dict[str, dict[str, Any]]:
    path = DATA_DIR / "_series_sidecar.json"
    records = json.loads(path.read_text())
    return {r["case_id"]: r for r in records}


def build_features(df: pd.DataFrame, sidecar: dict | None = None) -> pd.DataFrame:
    """Build a feature frame keyed by ``case_id`` for the full dataset."""
    if sidecar is None:
        sidecar = _load_sidecar()

    feats: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        case_id = row["case_id"]
        series = sidecar[case_id]["_series"]
        f = compute_case_features(series, row)
        f["case_id"] = case_id
        feats.append(f)
    fdf = pd.DataFrame(feats).set_index("case_id")

    # Cross-case context, joined back on (model_id, metric_name).
    meta = df.set_index("case_id")[["model_id", "metric_name", "current_value"]]
    joined = fdf.join(meta)

    # Cross-segment deviation: standardized distance of this case's current value
    # from the mean current value of peer cases sharing (model_id, metric_name).
    grp = joined.groupby(["model_id", "metric_name"])["current_value"]
    peer_mean = grp.transform("mean")
    peer_std = grp.transform("std").fillna(0.0)
    joined["cross_segment_deviation"] = (
        (joined["current_value"] - peer_mean) / (peer_std + 1e-9)
    ).abs()

    # Cross-metric contradiction: 1 if this case's z-score sign is opposite to the
    # mean z-score sign of other cases for the same model (transparent proxy).
    model_mean_z = joined.groupby("model_id")["z_score"].transform("mean")
    joined["cross_metric_contradiction"] = (
        np.sign(joined["z_score"]) != np.sign(model_mean_z)
    ).astype(float)

    joined = joined.drop(columns=["model_id", "metric_name", "current_value"])
    joined = joined.rename(columns={"z_score": "z_score_feature"})
    return joined


def main() -> None:
    df = pd.read_csv(DATA_DIR / "synthetic_monitoring_data.csv")
    feats = build_features(df)
    out = DATA_DIR / "features.csv"
    feats.to_csv(out)
    LOGGER.info("computed %d features for %d cases -> %s", feats.shape[1], len(feats), out)


if __name__ == "__main__":
    main()
