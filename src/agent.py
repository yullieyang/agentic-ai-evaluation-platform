"""The LLM-based QA agent.

The agent assembles a ground-truth-free evidence block, renders a prompt from a
named variant, calls a provider with a bounded retry loop, and validates the
returned JSON against the strict ``AgentFinding`` schema. It records latency,
token usage, estimated cost, retries, and schema validity.

A leakage assertion guarantees that no ground-truth field is ever placed in the
evidence sent to a provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm_providers import BaseProvider, ProviderResponse
from .schemas import AgentFinding, assert_no_ground_truth, json_schema, parse_model
from .utils import get_logger, load_config

LOGGER = get_logger("agent")

# Mock behavioural profiles per prompt variant. These let the offline mock
# reflect plausible prompt sensitivity. They are simulation parameters, not
# measured model behaviour.
PROMPT_PROFILES: dict[str, dict[str, float]] = {
    "prompt_a_zero_shot": dict(mitigation_skill=0.30, fn_recovery=0.15, abstain_bias=0.20,
                               unsupported_rate=0.40, miscalibration=0.45, malformed_rate=0.06),
    "prompt_b_few_shot": dict(mitigation_skill=0.55, fn_recovery=0.30, abstain_bias=0.45,
                              unsupported_rate=0.18, miscalibration=0.30, malformed_rate=0.03),
    "prompt_c_evidence_constrained": dict(mitigation_skill=0.70, fn_recovery=0.35, abstain_bias=0.45,
                                          unsupported_rate=0.10, miscalibration=0.25, malformed_rate=0.02),
    "prompt_d_conservative": dict(mitigation_skill=0.65, fn_recovery=0.20, abstain_bias=0.65,
                                  unsupported_rate=0.08, miscalibration=0.20, malformed_rate=0.02),
}

MITIGATION_CHECKS = {"seasonal_warning", "metric_definition_change",
                     "recovery_distinction", "expected_direction"}


@dataclass
class AgentResult:
    case_id: str
    finding: Optional[dict[str, Any]]
    schema_valid: bool
    validation_error: Optional[str]
    response: ProviderResponse
    prompt_version: str
    total_retries: int
    mock_unsupported_injected: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def build_evidence(
    row: dict[str, Any],
    features: dict[str, float],
    checks: list,
    chart_summary: dict[str, Any],
    include_deterministic_evidence: bool,
    baseline_anomaly: bool,
    baseline_severity: str,
    evidence_completeness: str = "complete",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (prompt_evidence, mock_context_evidence).

    ``prompt_evidence`` is the human-readable block embedded in the prompt and is
    asserted free of ground-truth fields. ``mock_context_evidence`` is a machine
    summary used only by the offline mock provider.
    """
    triggered = [c for c in checks if c.triggered]
    mitigation_flags = [c.check_name for c in triggered if c.check_name in MITIGATION_CHECKS]

    # Feature view exposed to the agent (no labels).
    feature_view = {
        k: round(float(v), 4)
        for k, v in features.items()
        if k not in ("case_id",)
    }

    prompt_evidence: dict[str, Any] = {
        "model_id": row["model_id"],
        "metric_name": row["metric_name"],
        "segment": row["segment"],
        "quarter": row["quarter"],
        "expected_direction": row["expected_direction"],
        "current_quarter_statistics": {
            "current_value": row["current_value"],
            "previous_value": row["previous_value"],
            "historical_mean": row["historical_mean"],
            "historical_std": row["historical_std"],
            "percent_change": row["percent_change"],
            "z_score": row["z_score"],
            "sample_size": row["sample_size"],
            "missing_rate": row["missing_rate"],
        },
        "engineered_features": feature_view,
        "chart_summary": chart_summary,
    }

    available_sources = ["features", "chart"]
    if include_deterministic_evidence:
        prompt_evidence["deterministic_qa_findings"] = [
            {
                "check_name": c.check_name, "triggered": c.triggered, "severity": c.severity,
                "observed_value": c.observed_value, "threshold": c.threshold,
                "evidence": c.evidence, "limitations": c.limitations,
            }
            for c in checks if c.triggered
        ]
        available_sources.append("deterministic")

    # Incomplete-evidence ablation: remove engineered features and deterministic
    # findings, leaving only sparse current statistics.
    incomplete = evidence_completeness == "incomplete" or bool(row.get("missing_evidence_flag", False))
    if evidence_completeness == "incomplete":
        prompt_evidence.pop("engineered_features", None)
        prompt_evidence.pop("deterministic_qa_findings", None)
        prompt_evidence["chart_summary"] = {"note": "chart summary unavailable for this case"}
        available_sources = ["features"]

    # Leakage guard: ground-truth fields must never appear in the agent prompt.
    assert_no_ground_truth(prompt_evidence)

    small_sample = int(row["sample_size"]) < 100
    mock_evidence = {
        "metric_name": row["metric_name"],
        "z_score": row["z_score"],
        "robust_z_score": features.get("robust_z_score", row["z_score"]),
        "missing_rate": row["missing_rate"],
        "small_sample": small_sample,
        "baseline_anomaly": baseline_anomaly,
        "baseline_severity": baseline_severity,
        "mitigation_flags": mitigation_flags if include_deterministic_evidence else (
            ["recovery_distinction"] if chart_summary.get("recovery_pattern") else []
        ),
        "incomplete": incomplete,
        "available_sources": available_sources,
    }
    return prompt_evidence, mock_evidence


def render_prompt(prompt_version: str, evidence_block: dict[str, Any],
                  prompts_cfg: dict | None = None) -> tuple[str, str]:
    """Render (system, user) prompts for a named variant."""
    cfg = prompts_cfg or load_config("prompts.yaml")
    variant = cfg["variants"][prompt_version]
    contract = cfg["shared_output_contract"]
    system = variant["system"].strip()
    user = (
        variant["user_template"]
        .replace("{evidence_block}", json.dumps(evidence_block, indent=2))
        .replace("{output_contract}", contract)
    )
    return system, user


class QAAgent:
    """Runs a single QA case through a provider with retry and validation."""

    def __init__(self, provider: BaseProvider, prompts_cfg: dict | None = None,
                 max_retries: int = 2, max_tokens: int = 1500):
        self.provider = provider
        self.prompts_cfg = prompts_cfg or load_config("prompts.yaml")
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._schema = json_schema(AgentFinding)

    def run(self, case: dict[str, Any]) -> AgentResult:
        """Run one case.

        ``case`` must contain: row, features, checks, chart_summary,
        prompt_version, include_deterministic_evidence, reviewer_enabled,
        evidence_completeness, baseline_anomaly, baseline_severity, temperature,
        model.
        """
        prompt_version = case["prompt_version"]
        prompt_evidence, mock_evidence = build_evidence(
            row=case["row"], features=case["features"], checks=case["checks"],
            chart_summary=case["chart_summary"],
            include_deterministic_evidence=case["include_deterministic_evidence"],
            baseline_anomaly=case["baseline_anomaly"],
            baseline_severity=case["baseline_severity"],
            evidence_completeness=case.get("evidence_completeness", "complete"),
        )
        system, user = render_prompt(prompt_version, prompt_evidence, self.prompts_cfg)

        context_base = {
            "case_id": case["row"]["case_id"],
            "prompt_version": prompt_version,
            "include_deterministic_evidence": case["include_deterministic_evidence"],
            "reviewer_enabled": case.get("reviewer_enabled", False),
            "evidence_completeness": case.get("evidence_completeness", "complete"),
            "profile": PROMPT_PROFILES.get(prompt_version, {}),
            "evidence": mock_evidence,
        }

        last: ProviderResponse | None = None
        validation_error: Optional[str] = None
        finding_dict: Optional[dict[str, Any]] = None
        unsupported_injected = False
        total_retries = 0

        for attempt in range(self.max_retries + 1):
            total_retries = attempt
            ctx = dict(context_base, attempt=attempt)
            resp = self.provider.complete(
                system, user, model=case["model"], temperature=case.get("temperature", 0.0),
                max_tokens=self.max_tokens, schema=self._schema, context=ctx,
            )
            last = resp
            if resp.parsed is None:
                validation_error = resp.parsing_error or "no parsed output"
                continue
            payload = dict(resp.parsed)
            unsupported_injected = bool(payload.pop("_mock_unsupported_injected", False))
            for k in [key for key in payload if key.startswith("_")]:
                payload.pop(k)
            try:
                finding = parse_model(AgentFinding, payload)
                finding_dict = json.loads(finding.json()) if hasattr(finding, "json") else payload
                validation_error = None
                break
            except Exception as exc:  # noqa: BLE001
                validation_error = f"{type(exc).__name__}: {exc}"
                finding_dict = None
                continue

        assert last is not None
        return AgentResult(
            case_id=case["row"]["case_id"], finding=finding_dict,
            schema_valid=finding_dict is not None, validation_error=validation_error,
            response=last, prompt_version=prompt_version, total_retries=total_retries,
            mock_unsupported_injected=unsupported_injected,
            extra={"prompt_evidence": prompt_evidence, "mock_evidence": mock_evidence},
        )
