"""Explicit, single-function human-review escalation policy.

Previously, "does this case need a human?" was decided ad hoc in several
places (inside the mock agent's decision logic, inside the reviewer's
simulated output, implicitly in the UI). This module makes it one documented,
testable policy so a recruiter — or a test — can point at exactly the rule
set behind every escalation, rather than inferring it from scattered code.

This does not replace ``AgentFinding.requires_human_review`` or
``ReviewerOutput.requires_human_review`` (the agent/reviewer's own opinion on
whether a human should look at the case); it is the final, deterministic
policy layer that also accounts for things neither of them can see on their
own — deterministic-validation failures and the tool-call budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .evaluators import detect_unsupported_claim
from .output_validation import ValidationResult, all_passed


@dataclass(frozen=True)
class EscalationConfig:
    confidence_threshold: float = 0.6  # below this, escalate regardless of severity


@dataclass
class EscalationDecision:
    escalate: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def auto_approved(self) -> bool:
        return not self.escalate

    def to_dict(self) -> dict[str, Any]:
        return {"escalate": self.escalate, "reasons": list(self.reasons),
                "auto_approved": self.auto_approved}


def decide_escalation(
    finding: Optional[dict[str, Any]],
    review: Optional[dict[str, Any]] = None,
    validation_results: Optional[list[ValidationResult]] = None,
    tool_calls: int = 0,
    max_tool_calls: int = 0,
    config: Optional[EscalationConfig] = None,
) -> EscalationDecision:
    """Decide whether a case must be escalated to human review.

    Any one of the following is sufficient to escalate:
    schema-validation failure; severity == "high"; confidence below
    ``config.confidence_threshold``; the agent abstained; evidence_sufficiency
    is "insufficient" or "limited"; an unsupported causal claim is detected;
    the reviewer's decision was not "approve"; any deterministic-validation
    check failed; or the tool-call budget was exceeded. A case with none of
    these can be auto-approved.
    """
    cfg = config or EscalationConfig()
    reasons: list[str] = []

    if finding is None:
        reasons.append("schema_validation_failed")
    else:
        if finding.get("severity") == "high":
            reasons.append("high_severity")
        if float(finding.get("confidence_score", 1.0)) < cfg.confidence_threshold:
            reasons.append("low_confidence")
        if finding.get("abstained"):
            reasons.append("agent_abstained")
        if finding.get("evidence_sufficiency") in ("insufficient", "limited"):
            reasons.append("insufficient_evidence")
        if detect_unsupported_claim(finding):
            reasons.append("unsupported_claim")

    if review is not None and review.get("review_decision") != "approve":
        reasons.append("reviewer_non_approve")

    if validation_results and not all_passed(validation_results):
        reasons.append("deterministic_validation_failed")

    if max_tool_calls > 0 and tool_calls > max_tool_calls:
        reasons.append("excessive_tool_calls")

    return EscalationDecision(escalate=bool(reasons), reasons=reasons)
