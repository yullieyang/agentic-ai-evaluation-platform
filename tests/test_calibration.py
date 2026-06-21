"""Tests for calibration metrics."""

from __future__ import annotations

from src.calibration import (brier_score, calibration_summary, expected_calibration_error,
                             reliability_curve)


def test_brier_perfect_and_worst():
    assert brier_score([1.0, 1.0], [1, 1]) == 0.0
    assert brier_score([1.0, 1.0], [0, 0]) == 1.0


def test_ece_zero_for_perfectly_calibrated():
    # All confidence 1.0 and all correct -> gap 0.
    assert expected_calibration_error([1.0] * 10, [1] * 10, n_bins=10) == 0.0


def test_ece_positive_for_overconfident():
    # Confidence high but accuracy low -> positive ECE.
    ece = expected_calibration_error([0.9] * 10, [0] * 10, n_bins=10)
    assert ece > 0.5


def test_reliability_curve_bins_and_counts():
    curve = reliability_curve([0.05, 0.15, 0.95], [0, 1, 1], n_bins=10)
    assert curve["n_bins"] == 10
    assert sum(curve["bin_count"]) == 3


def test_calibration_summary_keys():
    s = calibration_summary([0.2, 0.8, 0.6, 0.4], [0, 1, 1, 0])
    for key in ["expected_calibration_error", "brier_score", "overconfidence_rate",
                "underconfidence_rate", "reliability_curve"]:
        assert key in s
