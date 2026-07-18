"""Tests for strict structured-output schemas and the leakage guard."""

from __future__ import annotations

import pytest

from src.schemas import (AgentFinding, GROUND_TRUTH_FIELDS, ReviewerOutput,
                         assert_no_ground_truth, json_schema, model_to_dict, parse_model)


def _valid_finding() -> dict:
    return {
        "anomaly_detected": True, "severity": "high", "anomaly_type": "level_shift",
        "summary": "s", "observations": ["o"], "supporting_evidence": [], "affected_metrics": [],
        "possible_explanations": [], "recommended_follow_up": ["f"], "confidence_score": 0.8,
        "evidence_sufficiency": "adequate", "unsupported_claim_risk": "low",
        "requires_human_review": True, "abstained": False, "abstention_reason": None,
    }


def test_valid_agent_finding():
    f = parse_model(AgentFinding, _valid_finding())
    assert model_to_dict(f)["severity"] == "high"


def test_confidence_out_of_range_rejected():
    bad = _valid_finding()
    bad["confidence_score"] = 1.4
    with pytest.raises(Exception):
        parse_model(AgentFinding, bad)


def test_severity_consistency_enforced():
    bad = _valid_finding()
    bad["anomaly_detected"] = False
    bad["severity"] = "high"
    with pytest.raises(Exception):
        parse_model(AgentFinding, bad)


def test_extra_fields_forbidden():
    bad = _valid_finding()
    bad["unexpected"] = 1
    with pytest.raises(Exception):
        parse_model(AgentFinding, bad)


def test_missing_required_field_rejected():
    bad = _valid_finding()
    del bad["confidence_score"]
    with pytest.raises(Exception):
        parse_model(AgentFinding, bad)


def test_reviewer_output_valid_and_invalid():
    good = {
        "review_decision": "approve", "revised_anomaly_detected": False, "revised_severity": "none",
        "revised_anomaly_type": "none", "unsupported_claims_found": [], "missing_evidence": [],
        "review_summary": "ok", "revised_confidence": 0.5, "requires_human_review": False,
    }
    parse_model(ReviewerOutput, good)
    bad = dict(good, revised_confidence=2.0)
    with pytest.raises(Exception):
        parse_model(ReviewerOutput, bad)


def test_assert_no_ground_truth_detects_leak():
    payload = {"a": 1, "ground_truth_severity": "high"}
    with pytest.raises(ValueError):
        assert_no_ground_truth(payload)


def test_assert_no_ground_truth_passes_clean_payload():
    assert_no_ground_truth({"z_score": 3.1, "missing_rate": 0.0})


def test_json_schema_generation():
    schema = json_schema(AgentFinding)
    assert "properties" in schema
    assert "confidence_score" in schema["properties"]


def test_ground_truth_field_list_nonempty():
    assert len(GROUND_TRUTH_FIELDS) >= 4


def test_evidence_ids_defaults_to_empty_list():
    f = parse_model(AgentFinding, _valid_finding())
    assert model_to_dict(f)["evidence_ids"] == []


def test_evidence_ids_accepts_list_of_strings():
    payload = _valid_finding()
    payload["evidence_ids"] = ["case_0001-EV-01", "case_0001-EV-03"]
    f = parse_model(AgentFinding, payload)
    assert model_to_dict(f)["evidence_ids"] == ["case_0001-EV-01", "case_0001-EV-03"]


def test_json_schema_includes_evidence_ids():
    schema = json_schema(AgentFinding)
    assert "evidence_ids" in schema["properties"]
