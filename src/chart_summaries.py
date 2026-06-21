"""Deterministic, testable chart-derived summaries.

The agent does not receive raw images. Instead, each case is reduced to a
structured summary describing what a monitoring chart of its quarterly series
would show. The logic is fully deterministic so it can be unit-tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _trend_direction(slope: float, scale: float) -> str:
    rel = slope / (abs(scale) + 1e-9)
    if rel > 0.02:
        return "increasing"
    if rel < -0.02:
        return "decreasing"
    return "flat"


def build_chart_summary(
    series: list[float], row: dict[str, Any], features: dict[str, float]
) -> dict[str, Any]:
    """Build a structured chart-derived summary for one case."""
    arr = np.asarray(series, dtype=float)
    diffs = np.diff(arr)
    largest_move_idx = int(np.argmax(np.abs(diffs))) if len(diffs) else 0
    largest_move = float(diffs[largest_move_idx]) if len(diffs) else 0.0
    change_point_at_final = largest_move_idx == len(diffs) - 1 if len(diffs) else False

    return {
        "latest_value": round(float(arr[-1]), 6),
        "historical_range": [round(float(arr[:-1].min()), 6), round(float(arr[:-1].max()), 6)],
        "historical_mean": round(float(arr[:-1].mean()), 6),
        "trend_direction": _trend_direction(
            float(features.get("rolling_slope", 0.0)), float(arr[:-1].mean())
        ),
        "slope_last_4q": round(float(features.get("rolling_slope", 0.0)), 6),
        "largest_historical_movement": round(largest_move, 6),
        "change_point_at_final_quarter": bool(change_point_at_final),
        "cross_segment_deviation": round(float(features.get("cross_segment_deviation", 0.0)), 4),
        "recent_volatility": round(float(features.get("rolling_std", 0.0)), 6),
        "missingness_pattern": {
            "level": round(float(row.get("missing_rate", 0.0)), 4),
            "trend": round(float(features.get("missingness_trend", 0.0)), 4),
        },
        "contradictory_movement": bool(features.get("cross_metric_contradiction", 0.0) >= 1.0),
        "consecutive_directional_moves": int(features.get("consecutive_directional_moves", 0)),
        "recovery_pattern": bool(features.get("recovery_indicator", 0.0) >= 1.0),
    }
