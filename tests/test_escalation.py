"""Tests for the explicit human-review escalation policy: one trigger case
each, plus the auto-approve (no trigger) case."""

from __future__ import annotations

from src.escalation import EscalationConfig, decide_escalation
from src.output_validation import ValidationResult


def _finding(**overrides):
    base = {
        "anomaly_detected": False, "severity": "none", "anomaly_type": "none",
        "summary": "s", "observations": [], "supporting_evidence": [], "evidence_ids": [],
        "affected_metrics": [], "possible_explanations": [], "recommended_follow_up": [],
        "confidence_score": 0.9, "evidence_sufficiency": "adequate",
        "unsupported_claim_risk": "low", "requires_human_review": False,
        "abstained": False, "abstention_reason": None,
    }
    base.update(overrides)
    return base


def test_no_trigger_auto_approves():
    d = decide_escalation(_finding())
    assert d.escalate is False
    assert d.auto_approved is True
    assert d.reasons == []


def test_schema_validation_failure_escalates():
    d = decide_escalation(None)
    assert d.escalate is True
    assert "schema_validation_failed" in d.reasons


def test_high_severity_escalates():
    d = decide_escalation(_finding(severity="high"))
    assert "high_severity" in d.reasons


def test_low_confidence_escalates():
    d = decide_escalation(_finding(confidence_score=0.3), config=EscalationConfig(confidence_threshold=0.6))
    assert "low_confidence" in d.reasons


def test_abstention_escalates():
    d = decide_escalation(_finding(abstained=True))
    assert "agent_abstained" in d.reasons


def test_insufficient_evidence_escalates():
    d = decide_escalation(_finding(evidence_sufficiency="insufficient"))
    assert "insufficient_evidence" in d.reasons


def test_unsupported_claim_escalates():
    d = decide_escalation(_finding(possible_explanations=["Caused by a pipeline change."]))
    assert "unsupported_claim" in d.reasons


def test_reviewer_non_approve_escalates():
    d = decide_escalation(_finding(), review={"review_decision": "revise"})
    assert "reviewer_non_approve" in d.reasons


def test_reviewer_approve_does_not_escalate_alone():
    d = decide_escalation(_finding(), review={"review_decision": "approve"})
    assert d.escalate is False


def test_deterministic_validation_failure_escalates():
    results = [ValidationResult("some_check", False, "failed")]
    d = decide_escalation(_finding(), validation_results=results)
    assert "deterministic_validation_failed" in d.reasons


def test_excessive_tool_calls_escalates():
    d = decide_escalation(_finding(), tool_calls=8, max_tool_calls=6)
    assert "excessive_tool_calls" in d.reasons


def test_tool_calls_within_limit_does_not_escalate_alone():
    d = decide_escalation(_finding(), tool_calls=4, max_tool_calls=6)
    assert d.escalate is False


def test_multiple_reasons_all_recorded():
    d = decide_escalation(_finding(severity="high", confidence_score=0.2))
    assert "high_severity" in d.reasons and "low_confidence" in d.reasons
    assert len(d.reasons) >= 2
