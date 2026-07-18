"""Typed schemas for agent outputs, reviewer outputs, deterministic checks, and
dataset records.

Pydantic is used to enforce strict structured output. The installed Pydantic is
v1.x; the code below is written against the v1 API. A small compatibility shim
(``model_to_dict`` / ``parse_model``) isolates the version-specific calls so the
rest of the codebase does not depend on the Pydantic major version.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pydantic
from pydantic import BaseModel, Field

try:  # Literal lives in typing on 3.8+
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

_PYDANTIC_V2 = pydantic.VERSION.startswith("2")

# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
Severity = Literal["none", "low", "medium", "high"]
EvidenceSufficiency = Literal["insufficient", "limited", "adequate", "strong"]
ClaimRisk = Literal["low", "medium", "high"]
ReviewDecision = Literal["approve", "revise", "reject"]
ClaimSupport = Literal["supported", "partially_supported", "unsupported", "unverifiable"]

SCENARIO_TYPES = [
    "normal_variation",
    "sudden_level_shift",
    "gradual_drift",
    "seasonal_movement",
    "missing_data_spike",
    "small_sample_instability",
    "segment_specific_anomaly",
    "contradictory_metric_movement",
    "macro_shock",
    "false_positive_trap",
    "false_negative_trap",
    "incomplete_evidence",
    "noisy_data",
    "recovery_after_shock",
    "metric_definition_change",
]

DIFFICULTIES = ["easy", "moderate", "ambiguous", "adversarial"]


# --------------------------------------------------------------------------- #
# Compatibility shims
# --------------------------------------------------------------------------- #
def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a plain dict for a model across Pydantic v1/v2."""
    if _PYDANTIC_V2:  # pragma: no cover - environment pins v1
        return model.model_dump()
    return model.dict()


def parse_model(model_cls: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    """Validate ``data`` against ``model_cls`` across Pydantic v1/v2."""
    if _PYDANTIC_V2:  # pragma: no cover - environment pins v1
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema dict for a model across Pydantic v1/v2."""
    if _PYDANTIC_V2:  # pragma: no cover
        return model_cls.model_json_schema()
    return model_cls.schema()


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    """A single piece of evidence cited by the agent."""

    metric_name: str
    observed_value: float
    reference_value: Optional[float] = None
    comparison: str = Field(..., description="e.g. 'above', 'below', 'within range'")
    source: str = Field(..., description="e.g. 'feature', 'deterministic_check', 'chart_summary'")
    relevance: str


# --------------------------------------------------------------------------- #
# Agent output
# --------------------------------------------------------------------------- #
class AgentFinding(BaseModel):
    """Strict structured output required from the QA agent."""

    anomaly_detected: bool
    severity: Severity
    anomaly_type: str
    summary: str
    observations: List[str] = Field(
        default_factory=list,
        description="Observed facts only — what the numbers show. Never a cause.")
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Stable evidence_id values (see evidence_store.EvidenceItem) actually "
                     "used to support this finding. Existence of each ID against the case's "
                     "evidence store is checked by output_validation, not by this schema, "
                     "since that check requires state outside a single model instance.")
    affected_metrics: List[str] = Field(default_factory=list)
    possible_explanations: List[str] = Field(
        default_factory=list,
        description="Hypotheses only — a possible cause, never asserted as fact.")
    recommended_follow_up: List[str] = Field(
        default_factory=list,
        description="Recommended next actions for the human reviewer.")
    confidence_score: float
    evidence_sufficiency: EvidenceSufficiency
    unsupported_claim_risk: ClaimRisk
    requires_human_review: bool
    abstained: bool = False
    abstention_reason: Optional[str] = None

    class Config:
        extra = "forbid"

    if not _PYDANTIC_V2:
        from pydantic import validator

        @validator("confidence_score")
        def _confidence_range(cls, value: float) -> float:  # noqa: N805
            if not 0.0 <= value <= 1.0:
                raise ValueError("confidence_score must be in [0, 1]")
            return value

        @validator("severity")
        def _severity_consistency(cls, value: str, values: dict) -> str:  # noqa: N805
            # When no anomaly is detected, severity must be 'none'.
            if values.get("anomaly_detected") is False and value != "none":
                raise ValueError("severity must be 'none' when anomaly_detected is False")
            return value


# --------------------------------------------------------------------------- #
# Reviewer output
# --------------------------------------------------------------------------- #
class ReviewerOutput(BaseModel):
    """Structured output from the optional second-pass reviewer agent."""

    review_decision: ReviewDecision
    revised_anomaly_detected: bool
    revised_severity: Severity
    revised_anomaly_type: str
    unsupported_claims_found: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    review_summary: str
    revised_confidence: float
    requires_human_review: bool

    class Config:
        extra = "forbid"

    if not _PYDANTIC_V2:
        from pydantic import validator

        @validator("revised_confidence")
        def _confidence_range(cls, value: float) -> float:  # noqa: N805
            if not 0.0 <= value <= 1.0:
                raise ValueError("revised_confidence must be in [0, 1]")
            return value


# --------------------------------------------------------------------------- #
# Deterministic check result
# --------------------------------------------------------------------------- #
class DeterministicCheckResult(BaseModel):
    """Result of one deterministic QA check. A transparent baseline, not truth."""

    check_name: str
    triggered: bool
    severity: Severity
    observed_value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: str
    rationale: str
    confidence: float
    limitations: str


# --------------------------------------------------------------------------- #
# LLM-as-judge rubric output
# --------------------------------------------------------------------------- #
class JudgeRubric(BaseModel):
    """Secondary, model-based evaluation on a 1-5 rubric. Never replaces ground
    truth metrics."""

    groundedness: int
    evidence_consistency: int
    clarity: int
    actionability: int
    severity_appropriateness: int
    uncertainty_communication: int
    unsupported_causal_inference: int
    abstention_quality: int
    rationale: str

    class Config:
        extra = "forbid"


# --------------------------------------------------------------------------- #
# Dataset record (used for validating generated data in tests)
# --------------------------------------------------------------------------- #
class CaseRecord(BaseModel):
    case_id: str
    model_id: str
    quarter: str
    segment: str
    metric_name: str
    alert_type: str
    release_version: str
    threshold: float
    current_value: float
    previous_value: float
    historical_mean: float
    historical_std: float
    percent_change: float
    z_score: float
    rolling_mean: float
    rolling_std: float
    sample_size: int
    missing_rate: float
    expected_direction: str
    scenario_type: str
    scenario_difficulty: str
    ground_truth_anomaly: bool
    ground_truth_severity: Severity
    ground_truth_anomaly_type: str
    ground_truth_reason: str
    ground_truth_expected_escalation: bool
    available_evidence: str
    contradictory_evidence: bool
    missing_evidence_flag: bool


# Fields that encode ground truth and must never be exposed to an agent prompt.
GROUND_TRUTH_FIELDS = [
    "ground_truth_anomaly",
    "ground_truth_severity",
    "ground_truth_anomaly_type",
    "ground_truth_reason",
    "ground_truth_expected_escalation",
]


def assert_no_ground_truth(payload: dict[str, Any]) -> None:
    """Raise if any ground-truth field appears (recursively) in an agent payload."""
    text = json.dumps(payload, default=str)
    leaked = [field for field in GROUND_TRUTH_FIELDS if f'"{field}"' in text]
    if leaked:
        raise ValueError(f"ground-truth fields leaked into agent input: {leaked}")
