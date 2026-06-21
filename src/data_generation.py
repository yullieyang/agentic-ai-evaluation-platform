"""Synthetic quarterly model-monitoring dataset generation.

The generator builds, for each case, a short quarterly history for one
(model, metric, segment) and injects a labelled scenario pattern in the final
quarter. Per-case monitoring statistics are then derived from that history.

Design principle: ground-truth labels are *not* trivially recoverable from any
single derived field. Several scenario types are constructed specifically to
break naive single-threshold detectors:

* ``false_positive_trap`` — large percent change / high z-score but the movement
  is expected (seasonal or definitional), so ``ground_truth_anomaly`` is False.
* ``false_negative_trap`` — a genuine level shift masked by a small sample or
  high noise, so the z-score is modest but ``ground_truth_anomaly`` is True.
* ``recovery_after_shock`` — values return to baseline; not an anomaly even
  though a window spanning the prior shock can look unusual.
* ``metric_definition_change`` — a step change from a definition change rather
  than model degradation.

All randomness flows from a single seeded ``numpy`` Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .schemas import DIFFICULTIES, SCENARIO_TYPES
from .utils import DATA_DIR, get_logger, set_seed, write_json

LOGGER = get_logger("data_generation")

N_QUARTERS = 12
MODELS = [f"model_{i:02d}" for i in range(1, 9)]  # 8 fictional models
SEGMENTS = ["retail", "commercial", "small_business"]
# (name, base level, baseline noise fraction, expected_direction)
METRICS: list[tuple[str, float, float, str]] = [
    ("psi", 0.05, 0.35, "higher_is_worse"),
    ("auc", 0.82, 0.03, "lower_is_worse"),
    ("ks_statistic", 0.45, 0.05, "lower_is_worse"),
    ("default_rate", 0.04, 0.12, "higher_is_worse"),
    ("score_mean", 620.0, 0.02, "stable"),
]

# Effect-size multiplier applied to genuine anomalies by difficulty. Lower means
# the true signal is harder to see (masked), not that the truth changes.
DIFFICULTY_EFFECT = {"easy": 1.0, "moderate": 0.7, "ambiguous": 0.5, "adversarial": 0.4}
DIFFICULTY_NOISE = {"easy": 0.8, "moderate": 1.0, "ambiguous": 1.5, "adversarial": 1.8}


@dataclass
class ScenarioOutcome:
    """The injected series plus its ground-truth labels and evidence flags."""

    series: list[float]
    sample_size: int
    missing_rate: float
    ground_truth_anomaly: bool
    ground_truth_severity: str
    ground_truth_anomaly_type: str
    ground_truth_reason: str
    contradictory_evidence: bool = False
    missing_evidence_flag: bool = False
    cross_segment_shift: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _baseline(rng: np.random.Generator, base: float, noise_frac: float, n: int) -> np.ndarray:
    """A stationary baseline series around ``base`` with multiplicative noise."""
    sigma = max(abs(base) * noise_frac, 1e-6)
    return base + rng.normal(0.0, sigma, size=n)


def _std(values: np.ndarray) -> float:
    return float(max(np.std(values), 1e-9))


# --------------------------------------------------------------------------- #
# Scenario generators. Each returns a ScenarioOutcome.
# --------------------------------------------------------------------------- #
def _scn_normal(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    return ScenarioOutcome(list(series), int(rng.integers(800, 5000)), float(rng.uniform(0, 0.02)),
                           False, "none", "normal_variation",
                           "Final quarter within normal historical variation.")


def _scn_level_shift(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    sigma = _std(series[:-1])
    k = rng.uniform(3.0, 5.0) * DIFFICULTY_EFFECT[diff]
    sign = 1.0 if direction != "lower_is_worse" else -1.0
    series[-1] = series[:-1].mean() + sign * k * sigma
    sev = "high" if k >= 3.5 else "medium"
    return ScenarioOutcome(list(series), int(rng.integers(800, 5000)), float(rng.uniform(0, 0.02)),
                           True, sev, "level_shift",
                           "Sudden level shift in the final quarter relative to history.")


def _scn_drift(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    sigma = _std(series[:-1])
    sign = 1.0 if direction != "lower_is_worse" else -1.0
    ramp = np.linspace(0, rng.uniform(2.5, 4.0) * DIFFICULTY_EFFECT[diff] * sigma, 4)
    series[-4:] = series[-4:] + sign * ramp
    return ScenarioOutcome(list(series), int(rng.integers(800, 5000)), float(rng.uniform(0, 0.02)),
                           True, "medium", "gradual_drift",
                           "Monotonic drift over the most recent four quarters.")


def _scn_seasonal(rng, base, noise, direction, diff) -> ScenarioOutcome:
    sigma = max(abs(base) * noise, 1e-6)
    season = np.array([np.sin(2 * np.pi * q / 4) for q in range(N_QUARTERS)])
    amp = rng.uniform(2.0, 3.0) * sigma
    series = base + amp * season + rng.normal(0, sigma * 0.5, N_QUARTERS)
    # Final quarter is a seasonal peak: large apparent move, but expected.
    return ScenarioOutcome(list(series), int(rng.integers(800, 5000)), float(rng.uniform(0, 0.02)),
                           False, "none", "seasonal",
                           "Final-quarter movement matches the established seasonal pattern.",
                           extra={"seasonal": True})


def _scn_missing_spike(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    miss = float(rng.uniform(0.12, 0.30))
    return ScenarioOutcome(list(series), int(rng.integers(300, 1500)), miss,
                           True, "medium", "data_quality",
                           "Sharp rise in missing-data rate undermines metric reliability.")


def _scn_small_sample(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff] * 1.4, N_QUARTERS)
    n = int(rng.integers(25, 80))
    # The swing is instability, not a true change.
    return ScenarioOutcome(list(series), n, float(rng.uniform(0, 0.03)),
                           False, "none", "small_sample_instability",
                           "Apparent movement attributable to small-sample variance.")


def _scn_segment(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    sigma = _std(series[:-1])
    sign = 1.0 if direction != "lower_is_worse" else -1.0
    series[-1] = series[:-1].mean() + sign * rng.uniform(3.0, 4.5) * DIFFICULTY_EFFECT[diff] * sigma
    return ScenarioOutcome(list(series), int(rng.integers(400, 2500)), float(rng.uniform(0, 0.02)),
                           True, "high", "segment_anomaly",
                           "Shift isolated to this segment while peers remain stable.",
                           cross_segment_shift=True)


def _scn_contradiction(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    sigma = _std(series[:-1])
    series[-1] = series[:-1].mean() + rng.uniform(2.0, 3.0) * DIFFICULTY_EFFECT[diff] * sigma
    return ScenarioOutcome(list(series), int(rng.integers(800, 4000)), float(rng.uniform(0, 0.02)),
                           True, "medium", "contradiction",
                           "Metric moves inconsistently with a paired metric; needs review.",
                           contradictory_evidence=True)


def _scn_macro(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    sigma = _std(series[:-1])
    sign = 1.0 if direction != "lower_is_worse" else -1.0
    series[-1] = series[:-1].mean() + sign * rng.uniform(3.5, 5.5) * DIFFICULTY_EFFECT[diff] * sigma
    return ScenarioOutcome(list(series), int(rng.integers(1000, 5000)), float(rng.uniform(0, 0.02)),
                           True, "high", "macro_shock",
                           "Large coordinated movement consistent with a macro shock.")


def _scn_fp_trap(rng, base, noise, direction, diff) -> ScenarioOutcome:
    # Large apparent move that is expected (definitional rebasing) -> NOT anomaly.
    series = _baseline(rng, base, noise, N_QUARTERS)
    sigma = _std(series[:-1])
    series[-1] = series[:-1].mean() + rng.uniform(3.5, 5.0) * sigma
    return ScenarioOutcome(list(series), int(rng.integers(1000, 5000)), float(rng.uniform(0, 0.02)),
                           False, "none", "expected_movement",
                           "Large movement is expected (planned rebasing); not a model anomaly.",
                           extra={"expected_large_move": True})


def _scn_fn_trap(rng, base, noise, direction, diff) -> ScenarioOutcome:
    # Genuine shift masked by small sample + high noise -> modest z but IS anomaly.
    series = _baseline(rng, base, noise * 1.8, N_QUARTERS)
    sigma = _std(series[:-1])
    sign = 1.0 if direction != "lower_is_worse" else -1.0
    series[-1] = series[:-1].mean() + sign * rng.uniform(1.4, 2.2) * sigma
    return ScenarioOutcome(list(series), int(rng.integers(40, 110)), float(rng.uniform(0.02, 0.06)),
                           True, "medium", "masked_shift",
                           "Genuine shift masked by sampling noise; weak single-metric signal.")


def _scn_incomplete(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * DIFFICULTY_NOISE[diff], N_QUARTERS)
    is_anom = bool(rng.random() < 0.5)
    if is_anom:
        sigma = _std(series[:-1])
        series[-1] = series[:-1].mean() + rng.uniform(2.5, 3.5) * sigma
    return ScenarioOutcome(list(series), int(rng.integers(200, 1500)), float(rng.uniform(0, 0.04)),
                           is_anom, "low" if is_anom else "none",
                           "indeterminate", "Key evidence is missing; case is not decidable as presented.",
                           missing_evidence_flag=True)


def _scn_noisy(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise * 2.2, N_QUARTERS)
    is_anom = bool(rng.random() < 0.4)
    if is_anom:
        sigma = _std(series[:-1])
        series[-1] = series[:-1].mean() + rng.uniform(2.0, 3.0) * sigma
    return ScenarioOutcome(list(series), int(rng.integers(300, 2000)), float(rng.uniform(0, 0.03)),
                           is_anom, "low" if is_anom else "none",
                           "noisy_signal" if is_anom else "noise",
                           "Signal is heavily obscured by volatility.")


def _scn_recovery(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise, N_QUARTERS)
    sigma = _std(series)
    # Prior shock two quarters back, recovered by the final quarter.
    series[-3] = series.mean() + rng.uniform(3.0, 4.0) * sigma
    series[-1] = series[:-3].mean() + rng.normal(0, sigma * 0.5)
    return ScenarioOutcome(list(series), int(rng.integers(800, 4000)), float(rng.uniform(0, 0.02)),
                           False, "none", "recovery",
                           "Prior shock has recovered to baseline; final quarter is normal.")


def _scn_definition_change(rng, base, noise, direction, diff) -> ScenarioOutcome:
    series = _baseline(rng, base, noise, N_QUARTERS)
    sigma = _std(series[:-1])
    series[-1] = series[:-1].mean() + rng.uniform(4.0, 6.0) * sigma
    return ScenarioOutcome(list(series), int(rng.integers(1000, 5000)), float(rng.uniform(0, 0.02)),
                           False, "none", "definition_change",
                           "Step change reflects a metric-definition change, not degradation.",
                           extra={"definition_change": True})


SCENARIO_GENERATORS: dict[str, Callable[..., ScenarioOutcome]] = {
    "normal_variation": _scn_normal,
    "sudden_level_shift": _scn_level_shift,
    "gradual_drift": _scn_drift,
    "seasonal_movement": _scn_seasonal,
    "missing_data_spike": _scn_missing_spike,
    "small_sample_instability": _scn_small_sample,
    "segment_specific_anomaly": _scn_segment,
    "contradictory_metric_movement": _scn_contradiction,
    "macro_shock": _scn_macro,
    "false_positive_trap": _scn_fp_trap,
    "false_negative_trap": _scn_fn_trap,
    "incomplete_evidence": _scn_incomplete,
    "noisy_data": _scn_noisy,
    "recovery_after_shock": _scn_recovery,
    "metric_definition_change": _scn_definition_change,
}

# Difficulty distribution per scenario (some scenarios are inherently adversarial).
SCENARIO_DIFFICULTY_WEIGHTS = {
    "normal_variation": {"easy": 0.6, "moderate": 0.3, "ambiguous": 0.1, "adversarial": 0.0},
    "sudden_level_shift": {"easy": 0.5, "moderate": 0.35, "ambiguous": 0.1, "adversarial": 0.05},
    "gradual_drift": {"easy": 0.3, "moderate": 0.4, "ambiguous": 0.25, "adversarial": 0.05},
    "seasonal_movement": {"easy": 0.2, "moderate": 0.4, "ambiguous": 0.3, "adversarial": 0.1},
    "missing_data_spike": {"easy": 0.4, "moderate": 0.4, "ambiguous": 0.2, "adversarial": 0.0},
    "small_sample_instability": {"easy": 0.1, "moderate": 0.3, "ambiguous": 0.5, "adversarial": 0.1},
    "segment_specific_anomaly": {"easy": 0.3, "moderate": 0.4, "ambiguous": 0.25, "adversarial": 0.05},
    "contradictory_metric_movement": {"easy": 0.1, "moderate": 0.35, "ambiguous": 0.4, "adversarial": 0.15},
    "macro_shock": {"easy": 0.4, "moderate": 0.4, "ambiguous": 0.2, "adversarial": 0.0},
    "false_positive_trap": {"easy": 0.0, "moderate": 0.2, "ambiguous": 0.3, "adversarial": 0.5},
    "false_negative_trap": {"easy": 0.0, "moderate": 0.2, "ambiguous": 0.3, "adversarial": 0.5},
    "incomplete_evidence": {"easy": 0.1, "moderate": 0.3, "ambiguous": 0.5, "adversarial": 0.1},
    "noisy_data": {"easy": 0.1, "moderate": 0.3, "ambiguous": 0.4, "adversarial": 0.2},
    "recovery_after_shock": {"easy": 0.2, "moderate": 0.4, "ambiguous": 0.3, "adversarial": 0.1},
    "metric_definition_change": {"easy": 0.1, "moderate": 0.3, "ambiguous": 0.3, "adversarial": 0.3},
}


def _quarters() -> list[str]:
    quarters = []
    year, q = 2021, 1
    for _ in range(N_QUARTERS):
        quarters.append(f"{year}Q{q}")
        q += 1
        if q > 4:
            q, year = 1, year + 1
    return quarters


def _available_evidence(outcome: ScenarioOutcome) -> str:
    parts = ["features"]
    if not outcome.missing_evidence_flag:
        parts.append("deterministic")
    parts.append("chart")
    return ",".join(parts)


def generate_dataset(n_cases: int = 600, seed: int = 13) -> tuple[pd.DataFrame, dict, dict]:
    """Generate the synthetic monitoring dataset.

    Returns the case-level DataFrame, dataset metadata, and per-scenario
    generation documentation.
    """
    rng = set_seed(seed)
    quarters = _quarters()
    rows: list[dict[str, Any]] = []

    # Round-robin scenario assignment guarantees coverage of all 15 scenarios.
    for idx in range(n_cases):
        scenario = SCENARIO_TYPES[idx % len(SCENARIO_TYPES)]
        model_id = MODELS[int(rng.integers(0, len(MODELS)))]
        segment = SEGMENTS[int(rng.integers(0, len(SEGMENTS)))]
        metric_name, base, noise_frac, direction = METRICS[int(rng.integers(0, len(METRICS)))]

        weights = SCENARIO_DIFFICULTY_WEIGHTS[scenario]
        diff = rng.choice(DIFFICULTIES, p=[weights[d] for d in DIFFICULTIES])

        outcome = SCENARIO_GENERATORS[scenario](rng, base, noise_frac, direction, diff)
        series = np.asarray(outcome.series, dtype=float)

        current = float(series[-1])
        previous = float(series[-2])
        hist = series[:-1]
        hist_mean = float(np.mean(hist))
        hist_std = _std(hist)
        rolling = series[-4:]
        percent_change = (current - previous) / (abs(previous) + 1e-9)
        z_score = (current - hist_mean) / hist_std

        # Quarter index: place the case at the final modelled quarter.
        quarter = quarters[-1]

        rows.append(
            {
                "case_id": f"case_{idx:04d}",
                "model_id": model_id,
                "quarter": quarter,
                "segment": segment,
                "metric_name": metric_name,
                "current_value": round(current, 6),
                "previous_value": round(previous, 6),
                "historical_mean": round(hist_mean, 6),
                "historical_std": round(hist_std, 6),
                "percent_change": round(percent_change, 6),
                "z_score": round(z_score, 6),
                "rolling_mean": round(float(np.mean(rolling)), 6),
                "rolling_std": round(_std(rolling), 6),
                "sample_size": int(outcome.sample_size),
                "missing_rate": round(float(outcome.missing_rate), 6),
                "expected_direction": direction,
                "scenario_type": scenario,
                "scenario_difficulty": str(diff),
                "ground_truth_anomaly": bool(outcome.ground_truth_anomaly),
                "ground_truth_severity": outcome.ground_truth_severity,
                "ground_truth_anomaly_type": outcome.ground_truth_anomaly_type,
                "ground_truth_reason": outcome.ground_truth_reason,
                "available_evidence": _available_evidence(outcome),
                "contradictory_evidence": bool(outcome.contradictory_evidence),
                "missing_evidence_flag": bool(outcome.missing_evidence_flag),
                # Internal series stored for chart summaries; not part of schema.
                "_series": [round(v, 6) for v in series.tolist()],
                "_cross_segment_shift": bool(outcome.cross_segment_shift),
                "_extra": outcome.extra,
            }
        )

    df = pd.DataFrame(rows)

    metadata = {
        "dataset_version": "synthetic-v1",
        "seed": seed,
        "n_cases": int(len(df)),
        "n_models": len(MODELS),
        "n_quarters": N_QUARTERS,
        "n_segments": len(SEGMENTS),
        "metrics": [m[0] for m in METRICS],
        "scenario_types": SCENARIO_TYPES,
        "difficulties": DIFFICULTIES,
        "anomaly_prevalence": float(df["ground_truth_anomaly"].mean()),
        "schema_fields": [c for c in df.columns if not c.startswith("_")],
    }

    scenario_docs = {
        scn: {
            "generator": SCENARIO_GENERATORS[scn].__name__,
            "description": (SCENARIO_GENERATORS[scn].__doc__ or "").strip(),
        }
        for scn in SCENARIO_TYPES
    }
    return df, metadata, scenario_docs


def main(n_cases: int = 600, seed: int = 13) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata, scenario_docs = generate_dataset(n_cases=n_cases, seed=seed)
    # Persist the schema columns to CSV; keep the internal series in a sidecar.
    schema_cols = metadata["schema_fields"]
    df[schema_cols].to_csv(DATA_DIR / "synthetic_monitoring_data.csv", index=False)
    df[["case_id", "_series", "_cross_segment_shift", "_extra"]].to_json(
        DATA_DIR / "_series_sidecar.json", orient="records"
    )
    write_json(metadata, DATA_DIR / "dataset_metadata.json")
    write_json(scenario_docs, DATA_DIR / "labeled_scenarios.json")
    LOGGER.info(
        "generated %d cases (anomaly prevalence %.2f) -> %s",
        len(df),
        metadata["anomaly_prevalence"],
        DATA_DIR / "synthetic_monitoring_data.csv",
    )


if __name__ == "__main__":
    main()
