"""Confidence calibration analysis.

Confidence values are model-reported and are treated cautiously: they express
the agent's stated confidence in its own decision, and calibration here asks
whether a stated confidence of p corresponds to an empirical decision-accuracy
of approximately p. This is a property of self-reported confidence, not a
guarantee of probabilistic meaning. All functions accept a boolean ``correct``
array (whether the agent's binary decision matched ground truth).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_arrays(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(list(confidences), dtype=float)
    corr = np.asarray(list(correct), dtype=float)
    return conf, corr


def brier_score(confidences, correct) -> float:
    """Mean squared error between confidence and the correctness indicator."""
    conf, corr = _as_arrays(confidences, correct)
    if len(conf) == 0:
        return 0.0
    return float(np.mean((conf - corr) ** 2))


def reliability_curve(confidences, correct, n_bins: int = 10) -> dict[str, Any]:
    """Binned reliability data for a reliability diagram."""
    conf, corr = _as_arrays(confidences, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, mean_confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        count = int(np.sum(mask))
        centers.append(float((lo + hi) / 2))
        counts.append(count)
        if count:
            accs.append(float(np.mean(corr[mask])))
            mean_confs.append(float(np.mean(conf[mask])))
        else:
            accs.append(float("nan"))
            mean_confs.append(float("nan"))
    return {
        "bin_centers": centers,
        "bin_accuracy": accs,
        "bin_confidence": mean_confs,
        "bin_count": counts,
        "n_bins": n_bins,
    }


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """ECE: count-weighted average gap between confidence and accuracy per bin."""
    conf, corr = _as_arrays(confidences, correct)
    n = len(conf)
    if n == 0:
        return 0.0
    curve = reliability_curve(conf, corr, n_bins)
    ece = 0.0
    for acc, mconf, count in zip(curve["bin_accuracy"], curve["bin_confidence"], curve["bin_count"]):
        if count:
            ece += (count / n) * abs(acc - mconf)
    return float(ece)


def calibration_summary(confidences, correct, n_bins: int = 10) -> dict[str, Any]:
    conf, corr = _as_arrays(confidences, correct)
    return {
        "expected_calibration_error": expected_calibration_error(conf, corr, n_bins),
        "brier_score": brier_score(conf, corr),
        "average_confidence": float(np.mean(conf)) if len(conf) else 0.0,
        "average_accuracy": float(np.mean(corr)) if len(corr) else 0.0,
        "overconfidence_rate": float(np.mean((conf > 0.5) & (corr < 0.5))) if len(conf) else 0.0,
        "underconfidence_rate": float(np.mean((conf < 0.5) & (corr > 0.5))) if len(conf) else 0.0,
        "reliability_curve": reliability_curve(conf, corr, n_bins),
    }
