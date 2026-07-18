"""Explicit, ID-addressable evidence store for a single review case.

The rest of the codebase (``agent.py``'s ``build_evidence``) already computes
everything a case's evidence needs — engineered features, deterministic QA
checks, a chart-derived summary. This module does not invent new data; it
*restructures* that same per-case data into discrete items, each with a stable
``evidence_id``, so that:

* a finding can cite exactly which items support it (``evidence_ids``), and
  that citation can be checked against this store rather than trusted at face
  value (see ``schemas.AgentFinding.evidence_ids`` and
  ``output_validation.check_evidence_ids_exist``);
* an agentic (tool-using) run can *retrieve* individual items on demand via
  ``evidence_tools`` instead of receiving the entire case pre-assembled in one
  prompt, which is what makes the agentic condition meaningfully different
  from the single-shot condition (see ``experiment_runner``'s ``architecture``
  axis).

Evidence kinds are deliberately mapped to already-computed, real per-case
values — release version, threshold, and definition-change status are the
only genuinely new fields, and they are derived deterministically from
existing scenario/config data, not fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Scenario type -> a business-friendlier "alert type" label. Purely a relabelling
# of the existing ``scenario_type`` taxonomy in ``schemas.SCENARIO_TYPES`` for a
# more realistic alert-workflow framing; it carries no new ground truth.
ALERT_TYPE_BY_SCENARIO: dict[str, str] = {
    "normal_variation": "routine_monitoring_check",
    "sudden_level_shift": "kpi_threshold_breach",
    "gradual_drift": "metric_drift_alert",
    "seasonal_movement": "kpi_threshold_breach",
    "missing_data_spike": "data_quality_alert",
    "small_sample_instability": "data_quality_alert",
    "segment_specific_anomaly": "segment_deviation_alert",
    "contradictory_metric_movement": "cross_metric_inconsistency_alert",
    "macro_shock": "kpi_threshold_breach",
    "false_positive_trap": "kpi_threshold_breach",
    "false_negative_trap": "kpi_threshold_breach",
    "incomplete_evidence": "data_quality_alert",
    "noisy_data": "data_quality_alert",
    "recovery_after_shock": "kpi_threshold_breach",
    "metric_definition_change": "metric_definition_change_alert",
}


def alert_type_for(scenario_type: str) -> str:
    return ALERT_TYPE_BY_SCENARIO.get(scenario_type, "kpi_threshold_breach")


def release_version_for(row: dict[str, Any]) -> str:
    """A synthetic, deterministic release tag derived from the case's quarter.

    Not a real deployment version — a plausible label so the case model can
    carry a "release_version" field and a release-notes evidence item.
    """
    quarter = str(row.get("quarter", "0000Q0"))
    year, q = quarter[:4], quarter[-1]
    return f"monitoring-pipeline-{year}.{q}"


@dataclass
class EvidenceItem:
    """A single, independently-fetchable piece of case evidence."""

    evidence_id: str
    case_id: str
    kind: str  # metric_snapshot | historical_baseline | deterministic_check |
               # validation_rule | release_note | segment_comparison |
               # seasonality_indicator | recovery_indicator
    title: str
    data: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "title": self.title,
            "data": self.data,
            "summary": self.summary,
        }


def build_case_evidence_items(
    case_id: str,
    row: dict[str, Any],
    features: dict[str, float],
    checks: list,  # list[DeterministicCheckResult]
    chart_summary: dict[str, Any],
    eval_cfg: dict[str, Any],
    include_deterministic: bool = True,
) -> list[EvidenceItem]:
    """Build the full, ID-addressable evidence set for one case.

    Every value here comes from data the rest of the pipeline already
    computes for this case (``row``, ``features``, ``checks``,
    ``chart_summary``) — this function only restructures it into stable,
    citeable items. It does not draw any new randomness and does not see any
    ground-truth field.
    """
    items: list[EvidenceItem] = []
    n = 0

    def _next_id() -> str:
        nonlocal n
        n += 1
        return f"{case_id}-EV-{n:02d}"

    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="metric_snapshot",
        title=f"Current-quarter statistics for {row['metric_name']}",
        data={
            "current_value": row["current_value"], "previous_value": row["previous_value"],
            "percent_change": row["percent_change"], "z_score": row["z_score"],
            "sample_size": row["sample_size"], "missing_rate": row["missing_rate"],
        },
        summary=(f"{row['metric_name']} moved {row['percent_change']:.1%} quarter-over-quarter "
                 f"(z={row['z_score']:.2f}) on a sample of {row['sample_size']}."),
    ))

    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="historical_baseline",
        title=f"Historical baseline for {row['metric_name']} / {row['segment']}",
        data={
            "historical_mean": row["historical_mean"], "historical_std": row["historical_std"],
            "rolling_mean": row["rolling_mean"], "rolling_std": row["rolling_std"],
        },
        summary=(f"Historical mean {row['historical_mean']:.4f} (std {row['historical_std']:.4f}); "
                 f"trailing 4-quarter mean {row['rolling_mean']:.4f}."),
    ))

    z_cfg = eval_cfg["deterministic_checks"]["zscore"]
    # NOTE: this rule is intentionally universal (the same threshold applies to
    # every case) rather than keyed by ``alert_type``. ``alert_type`` is a
    # case/dataset-level label derived from ``scenario_type``, and several
    # scenario types map to it 1:1 with a fixed ground-truth outcome — showing
    # it here would leak the answer. ``alert_type`` is therefore kept out of
    # agent-facing evidence entirely (same treatment as ``scenario_type``
    # itself, which has never been shown to the agent).
    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="validation_rule",
        title="Standard monitoring validation rule",
        data={"zscore_threshold": z_cfg["threshold"], "zscore_high_threshold": z_cfg["high_threshold"]},
        summary=(f"Flagged when |z-score| >= {z_cfg['threshold']} "
                 f"(high severity at >= {z_cfg['high_threshold']})."),
    ))

    # Definition-change signal is deliberately derived from the same imperfect,
    # magnitude-based heuristic the deterministic checks already expose
    # (``metric_definition_change``: |z| >= 5.0), not from the generator's
    # internal ground-truth flag — using the literal flag here would leak the
    # answer for the ``metric_definition_change`` scenario (it is set only by
    # that generator and always co-occurs with ground_truth_anomaly=False).
    release_version = release_version_for(row)
    definition_change_check = next((c for c in checks if c.check_name == "metric_definition_change"), None)
    definition_change_suspected = bool(definition_change_check.triggered) if definition_change_check else False
    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="release_note",
        title=f"Release notes for {release_version}",
        data={"release_version": release_version, "definition_change_suspected": definition_change_suspected},
        summary=(f"Release {release_version}: the step change is large enough that a metric-"
                 f"definition change should be verified against the release log (not confirmed)."
                 if definition_change_suspected else
                 f"Release {release_version}: no release-log flag suggests a metric-definition change."),
    ))

    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="segment_comparison",
        title=f"Cross-segment comparison for {row['segment']}",
        data={"segment": row["segment"], "cross_segment_deviation": chart_summary.get("cross_segment_deviation", 0.0)},
        summary=(f"{row['segment']} deviates {chart_summary.get('cross_segment_deviation', 0.0):.2f} "
                 f"standard deviations from peer segments on the same metric."),
    ))

    seasonal_check = next((c for c in checks if c.check_name == "seasonal_warning"), None)
    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="seasonality_indicator",
        title="Seasonality indicator",
        data={"seasonal_pattern_flagged": bool(seasonal_check.triggered) if seasonal_check else False},
        summary=(seasonal_check.evidence if seasonal_check and seasonal_check.triggered
                 else "No seasonal pattern flagged for this quarter."),
    ))

    items.append(EvidenceItem(
        evidence_id=_next_id(), case_id=case_id, kind="recovery_indicator",
        title="Recovery-effect indicator",
        data={"recovery_pattern": bool(chart_summary.get("recovery_pattern", False))},
        summary=("Final quarter is consistent with recovery from an earlier shock."
                 if chart_summary.get("recovery_pattern") else
                 "No recovery-from-shock pattern detected."),
    ))

    if include_deterministic:
        for c in checks:
            if not c.triggered:
                continue
            items.append(EvidenceItem(
                evidence_id=_next_id(), case_id=case_id, kind="deterministic_check",
                title=f"Deterministic check: {c.check_name}",
                data={"check_name": c.check_name, "severity": c.severity,
                      "observed_value": c.observed_value, "threshold": c.threshold},
                summary=c.evidence,
            ))

    return items


class EvidenceStore:
    """In-memory lookup of evidence items, keyed by case then evidence_id.

    Built once per experiment run (or once per case for interactive/tool-use
    calls) from the same per-case data the rest of the pipeline computes —
    see ``build_case_evidence_items``.
    """

    def __init__(self) -> None:
        self._by_case: dict[str, dict[str, EvidenceItem]] = {}

    def register(self, case_id: str, items: list[EvidenceItem]) -> None:
        self._by_case[case_id] = {item.evidence_id: item for item in items}

    def ids_for_case(self, case_id: str) -> list[str]:
        return list(self._by_case.get(case_id, {}).keys())

    def get(self, case_id: str, evidence_id: str) -> Optional[EvidenceItem]:
        return self._by_case.get(case_id, {}).get(evidence_id)

    def list_for_case(self, case_id: str) -> list[EvidenceItem]:
        return list(self._by_case.get(case_id, {}).values())

    def by_kind(self, case_id: str, kind: str) -> list[EvidenceItem]:
        return [i for i in self.list_for_case(case_id) if i.kind == kind]

    def exists(self, case_id: str, evidence_id: str) -> bool:
        return evidence_id in self._by_case.get(case_id, {})
