"""Tests for deterministic QA checks and the rule-based baseline decision."""

from __future__ import annotations

from src.deterministic_checks import baseline_decision, run_checks
from src.schemas import DeterministicCheckResult


def _row(**kw):
    base = {
        "z_score": 0.5, "percent_change": 0.05, "missing_rate": 0.0, "sample_size": 1000,
        "expected_direction": "higher_is_worse",
    }
    base.update(kw)
    return base


def _features(**kw):
    base = {
        "robust_z_score": 0.5, "consecutive_directional_moves": 0, "volatility_change": 1.0,
        "cross_segment_deviation": 0.0, "cross_metric_contradiction": 0.0, "rolling_slope": 0.0,
        "recovery_indicator": 0.0,
    }
    base.update(kw)
    return base


def test_checks_return_typed_results():
    checks = run_checks(_row(), _features(), [1, 1, 1, 1, 1])
    assert all(isinstance(c, DeterministicCheckResult) for c in checks)
    names = {c.check_name for c in checks}
    assert {"zscore", "robust_zscore", "missing_rate", "small_sample"} <= names


def test_high_zscore_triggers_anomaly():
    checks = run_checks(_row(z_score=4.5), _features(robust_z_score=4.5), [1, 1, 1, 1, 9])
    anomaly, severity = baseline_decision(checks)
    assert anomaly is True
    assert severity in {"medium", "high"}


def test_normal_case_no_anomaly():
    checks = run_checks(_row(z_score=0.4, percent_change=0.03), _features(), [1, 1, 1, 1, 1])
    anomaly, severity = baseline_decision(checks)
    assert anomaly is False
    assert severity == "none"


def test_missing_rate_flagged_but_not_anomaly_decision():
    checks = run_checks(_row(missing_rate=0.2), _features(), [1, 1, 1, 1, 1])
    mr = next(c for c in checks if c.check_name == "missing_rate")
    assert mr.triggered is True
    # Missing-rate alone is a data-quality flag, not part of the anomaly decision.
    anomaly, _ = baseline_decision(checks)
    assert anomaly is False


def test_consecutive_drift_triggers():
    checks = run_checks(_row(z_score=1.0), _features(consecutive_directional_moves=4),
                        [1, 2, 3, 4, 5])
    drift = next(c for c in checks if c.check_name == "consecutive_drift")
    assert drift.triggered is True


def test_definition_change_warning_on_extreme_z():
    checks = run_checks(_row(z_score=6.0), _features(robust_z_score=6.0), [1, 1, 1, 1, 12])
    dc = next(c for c in checks if c.check_name == "metric_definition_change")
    assert dc.triggered is True
