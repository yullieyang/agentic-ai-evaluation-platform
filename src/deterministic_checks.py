"""Deterministic QA checks.

Each check is an independently testable function of a case's scalar fields,
engineered features, and quarterly series. Checks return a
``DeterministicCheckResult`` carrying the observed value, threshold, evidence,
rationale, a self-reported confidence, and explicit limitations.

These rules are transparent *baselines*, not ground truth. They are deliberately
magnitude-driven and unaware of scenario semantics (seasonality, definitional
rebasing, masked shifts), so they make the characteristic errors the study
measures: false positives on expected large movements and false negatives on
masked shifts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import DeterministicCheckResult, Severity
from .utils import load_config

# Checks whose triggering contributes to the baseline anomaly decision.
ANOMALY_CHECKS = {
    "abs_percent_change",
    "zscore",
    "robust_zscore",
    "consecutive_drift",
    "volatility_increase",
    "cross_segment_inconsistency",
}

_SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _result(name, triggered, severity, observed, threshold, evidence, rationale,
            confidence, limitations) -> DeterministicCheckResult:
    return DeterministicCheckResult(
        check_name=name, triggered=bool(triggered), severity=severity,
        observed_value=None if observed is None else float(observed),
        threshold=None if threshold is None else float(threshold),
        evidence=evidence, rationale=rationale, confidence=float(confidence),
        limitations=limitations,
    )


def run_checks(
    row: dict[str, Any],
    features: dict[str, float],
    series: list[float],
    config: dict | None = None,
) -> list[DeterministicCheckResult]:
    """Run all deterministic checks for one case."""
    cfg = (config or load_config("evaluation.yaml"))["deterministic_checks"]
    z = float(row["z_score"])
    rz = float(features.get("robust_z_score", z))
    pct = float(row["percent_change"])
    arr = np.asarray(series, dtype=float)
    results: list[DeterministicCheckResult] = []

    # 1. Absolute percent change
    c = cfg["abs_percent_change"]
    sev = "high" if abs(pct) >= c["high_threshold"] else "medium" if abs(pct) >= c["threshold"] else "none"
    results.append(_result(
        "abs_percent_change", abs(pct) >= c["threshold"], sev, abs(pct), c["threshold"],
        f"|percent_change|={abs(pct):.3f}", "Quarter-over-quarter magnitude exceeds threshold.",
        0.8, "Insensitive to seasonality and to whether a large move is expected."))

    # 2. z-score
    c = cfg["zscore"]
    sev = "high" if abs(z) >= c["high_threshold"] else "medium" if abs(z) >= c["threshold"] else "none"
    results.append(_result(
        "zscore", abs(z) >= c["threshold"], sev, abs(z), c["threshold"],
        f"|z|={abs(z):.2f}", "Standardized deviation from historical mean exceeds threshold.",
        0.8, "Sensitive to non-normal history and to small-sample variance."))

    # 3. robust z-score
    c = cfg["robust_zscore"]
    sev = "high" if abs(rz) >= c["high_threshold"] else "medium" if abs(rz) >= c["threshold"] else "none"
    results.append(_result(
        "robust_zscore", abs(rz) >= c["threshold"], sev, abs(rz), c["threshold"],
        f"|robust_z|={abs(rz):.2f}", "MAD-based standardized deviation exceeds threshold.",
        0.75, "More robust to outliers but still magnitude-only."))

    # 4. missing-rate (data-quality flag)
    c = cfg["missing_rate"]
    mr = float(row["missing_rate"])
    sev = "high" if mr >= c["high_threshold"] else "medium" if mr >= c["threshold"] else "none"
    results.append(_result(
        "missing_rate", mr >= c["threshold"], sev, mr, c["threshold"],
        f"missing_rate={mr:.3f}", "Missing-data rate undermines metric reliability.",
        0.85, "Flags data quality, not necessarily a model anomaly."))

    # 5. small-sample instability (caution flag)
    c = cfg["small_sample"]
    n = int(row["sample_size"])
    sev = "medium" if n < c["critical"] else "low" if n < c["threshold"] else "none"
    results.append(_result(
        "small_sample", n < c["threshold"], sev, n, c["threshold"],
        f"sample_size={n}", "Small sample inflates apparent variability.",
        0.7, "Indicates caution; movement may be noise rather than signal."))

    # 6. consecutive directional drift
    c = cfg["consecutive_drift"]
    cd = int(features.get("consecutive_directional_moves", 0))
    results.append(_result(
        "consecutive_drift", cd >= c["min_consecutive"], "medium" if cd >= c["min_consecutive"] else "none",
        cd, c["min_consecutive"], f"consecutive_moves={cd}",
        "Sustained directional movement may indicate drift.", 0.7,
        "Cannot distinguish drift from a deterministic trend or seasonality."))

    # 7. volatility increase
    c = cfg["volatility_increase"]
    vc = float(features.get("volatility_change", 1.0))
    results.append(_result(
        "volatility_increase", vc >= c["ratio_threshold"], "low" if vc >= c["ratio_threshold"] else "none",
        vc, c["ratio_threshold"], f"volatility_change={vc:.2f}",
        "Recent volatility elevated relative to history.", 0.6,
        "Elevated volatility is not itself a level anomaly."))

    # 8. cross-segment inconsistency
    c = cfg["cross_segment_inconsistency"]
    cs = float(features.get("cross_segment_deviation", 0.0))
    results.append(_result(
        "cross_segment_inconsistency", cs >= c["deviation_threshold"],
        "medium" if cs >= c["deviation_threshold"] else "none", cs, c["deviation_threshold"],
        f"cross_segment_deviation={cs:.2f}", "Segment deviates from peers for the same metric.",
        0.6, "Peer set is approximate in this synthetic dataset."))

    # 9. expected-direction contradiction (QA note, not anomaly)
    direction = row["expected_direction"]
    improving = (direction == "higher_is_worse" and z < -cfg["zscore"]["threshold"]) or (
        direction == "lower_is_worse" and z > cfg["zscore"]["threshold"])
    results.append(_result(
        "expected_direction", bool(improving), "low" if improving else "none", z, None,
        f"z={z:.2f}, expected_direction={direction}",
        "Strong movement in the improving direction; verify data correctness.", 0.5,
        "Improving movements can be genuine; this is a verification prompt."))

    # 10. cross-metric contradiction (QA note)
    cm = float(features.get("cross_metric_contradiction", 0.0))
    results.append(_result(
        "cross_metric_contradiction", cm >= 1.0, "low" if cm >= 1.0 else "none", cm, 1.0,
        f"cross_metric_contradiction={cm:.0f}", "Metric moves against the model's other metrics.",
        0.5, "Proxy contradiction signal; may reflect heterogeneous metrics."))

    # 11. seasonal-adjustment warning (heuristic)
    slope = float(features.get("rolling_slope", 0.0))
    seasonal_like = abs(z) >= cfg["zscore"]["threshold"] and cd < 2 and abs(slope) < 1e-6
    results.append(_result(
        "seasonal_warning", bool(seasonal_like), "low" if seasonal_like else "none", abs(z), None,
        f"|z|={abs(z):.2f}, consecutive_moves={cd}",
        "Large isolated movement without drift may be seasonal.", 0.4,
        "Heuristic; no explicit seasonal decomposition is performed."))

    # 12. metric-definition-change warning (heuristic)
    extreme = abs(z) >= 5.0
    results.append(_result(
        "metric_definition_change", bool(extreme), "low" if extreme else "none", abs(z), 5.0,
        f"|z|={abs(z):.2f}", "Extreme single-quarter step may indicate a definition change.",
        0.4, "Cannot distinguish a true shock from a definitional rebasing."))

    # 13. recovery-versus-anomaly distinction (mitigation note)
    rec = float(features.get("recovery_indicator", 0.0))
    results.append(_result(
        "recovery_distinction", rec >= 1.0, "none", rec, 1.0,
        f"recovery_indicator={rec:.0f}", "Final quarter consistent with recovery from a prior spike.",
        0.5, "A recovery pattern can mask a coincident new shift."))

    return results


def baseline_decision(checks: list[DeterministicCheckResult]) -> tuple[bool, Severity]:
    """Aggregate triggered anomaly-checks into a rule-based anomaly decision.

    Deliberately simple and trap-unaware: anomaly is declared if any
    magnitude-based check triggers; severity is the maximum triggered severity.
    """
    triggered = [c for c in checks if c.triggered and c.check_name in ANOMALY_CHECKS]
    if not triggered:
        return False, "none"
    severity = max((c.severity for c in triggered), key=lambda s: _SEVERITY_ORDER[s])
    return True, severity


def checks_to_evidence_dicts(checks: list[DeterministicCheckResult]) -> list[dict]:
    """Compact dict form of triggered checks for inclusion in agent evidence."""
    return [
        {
            "check_name": c.check_name,
            "triggered": c.triggered,
            "severity": c.severity,
            "observed_value": c.observed_value,
            "threshold": c.threshold,
            "evidence": c.evidence,
        }
        for c in checks
        if c.triggered
    ]
