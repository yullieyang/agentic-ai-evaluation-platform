"""Tests for deterministic output validation: one trigger + one non-trigger
case per check, plus the orchestrator."""

from __future__ import annotations

from src.evidence_store import EvidenceItem, EvidenceStore
from src.output_validation import (
    all_passed,
    check_evidence_ids_exist,
    check_high_severity_requires_review,
    check_numeric_consistency,
    check_reviewer_compatibility,
    check_schema_conformance,
    check_tool_call_limit,
    check_unsupported_causal_wording,
    run_output_validation,
)


def _store_with_snapshot(case_id="c1", z=3.5):
    store = EvidenceStore()
    store.register(case_id, [
        EvidenceItem("c1-EV-01", case_id, "metric_snapshot", "t",
                     {"z_score": z, "current_value": 1.0}, "s"),
    ])
    return store


def _finding(**overrides):
    base = {
        "anomaly_detected": True, "severity": "medium", "anomaly_type": "level_shift",
        "summary": "s", "observations": [], "supporting_evidence": [], "evidence_ids": [],
        "affected_metrics": [], "possible_explanations": [], "recommended_follow_up": [],
        "confidence_score": 0.7, "evidence_sufficiency": "adequate",
        "unsupported_claim_risk": "low", "requires_human_review": True,
        "abstained": False, "abstention_reason": None,
    }
    base.update(overrides)
    return base


def test_schema_conformance_pass_and_fail():
    assert check_schema_conformance(True, None).passed is True
    r = check_schema_conformance(False, "boom")
    assert r.passed is False and "boom" in r.detail


def test_evidence_ids_exist_pass_and_fail():
    store = _store_with_snapshot()
    ok = check_evidence_ids_exist(_finding(evidence_ids=["c1-EV-01"]), store, "c1")
    assert ok.passed is True
    bad = check_evidence_ids_exist(_finding(evidence_ids=["c1-EV-99"]), store, "c1")
    assert bad.passed is False
    assert "c1-EV-99" in bad.detail


def test_evidence_ids_exist_empty_list_passes_trivially():
    store = _store_with_snapshot()
    assert check_evidence_ids_exist(_finding(evidence_ids=[]), store, "c1").passed is True


def test_numeric_consistency_pass_and_fail():
    store = _store_with_snapshot(z=3.5)
    matching = _finding(supporting_evidence=[
        {"metric_name": "auc", "observed_value": 3.5, "reference_value": 0.0,
         "comparison": "above", "source": "feature", "relevance": "r"}])
    assert check_numeric_consistency(matching, store, "c1").passed is True

    fabricated = _finding(supporting_evidence=[
        {"metric_name": "auc", "observed_value": 9.9, "reference_value": 0.0,
         "comparison": "above", "source": "feature", "relevance": "r"}])
    r = check_numeric_consistency(fabricated, store, "c1")
    assert r.passed is False


def test_high_severity_requires_review_pass_and_fail():
    ok = check_high_severity_requires_review(_finding(severity="high", requires_human_review=True))
    assert ok.passed is True
    bad = check_high_severity_requires_review(_finding(severity="high", requires_human_review=False))
    assert bad.passed is False


def test_unsupported_causal_wording_pass_and_fail():
    clean = check_unsupported_causal_wording(_finding(possible_explanations=["A hypothesis to check."]))
    assert clean.passed is True
    bad = check_unsupported_causal_wording(
        _finding(possible_explanations=["This was caused by a pipeline change."]))
    assert bad.passed is False


def test_reviewer_compatibility_pass_and_fail():
    finding = _finding()
    ok_review = {"review_decision": "approve", "unsupported_claims_found": [],
                 "revised_anomaly_detected": True, "revised_severity": "medium"}
    assert check_reviewer_compatibility(finding, ok_review).passed is True

    bad_review = {"review_decision": "approve", "unsupported_claims_found": ["x"],
                  "revised_anomaly_detected": True, "revised_severity": "medium"}
    assert check_reviewer_compatibility(finding, bad_review).passed is False

    reject_no_change = {"review_decision": "reject",
                        "revised_anomaly_detected": finding["anomaly_detected"],
                        "revised_severity": finding["severity"],
                        "unsupported_claims_found": []}
    assert check_reviewer_compatibility(finding, reject_no_change).passed is False

    assert check_reviewer_compatibility(finding, None).passed is True


def test_tool_call_limit_pass_and_fail():
    assert check_tool_call_limit(3, 6).passed is True
    assert check_tool_call_limit(7, 6).passed is False


def test_run_output_validation_orchestrator_all_pass():
    store = _store_with_snapshot(z=3.5)
    finding = _finding(evidence_ids=["c1-EV-01"], supporting_evidence=[
        {"metric_name": "auc", "observed_value": 3.5, "reference_value": 0.0,
         "comparison": "above", "source": "feature", "relevance": "r"}])
    results = run_output_validation(
        finding=finding, schema_valid=True, validation_error=None,
        store=store, case_id="c1", review=None, tool_calls=2, max_tool_calls=6,
    )
    assert all_passed(results)
    assert {r.check_name for r in results} == {
        "schema_conformance", "evidence_ids_exist", "numeric_consistency",
        "high_severity_requires_review", "unsupported_causal_wording",
        "reviewer_compatibility", "tool_call_limit",
    }


def test_run_output_validation_short_circuits_on_schema_failure():
    results = run_output_validation(finding=None, schema_valid=False, validation_error="bad json")
    assert len(results) == 1
    assert results[0].check_name == "schema_conformance"
    assert not all_passed(results)
