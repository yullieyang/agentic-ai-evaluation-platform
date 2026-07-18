"""Tests for evaluation metrics, unsupported-claim detection, and statistical tests."""

from __future__ import annotations

from src.evaluators import (classification_metrics, detect_unsupported_claim, evaluate_records,
                            severity_confusion)
from src.statistical_tests import (benjamini_hochberg, bootstrap_ci, cohens_h, mcnemar_test,
                                   paired_permutation_test)


def test_classification_metrics_known_confusion():
    # tp=2, fp=1, fn=1, tn=2
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert abs(m["f1"] - 2 / 3) < 1e-9
    assert m["count_tp"] == 2 and m["count_fp"] == 1


def test_severity_confusion_accuracy():
    res = severity_confusion(["none", "low", "high"], ["none", "low", "medium"])
    assert abs(res["severity_accuracy"] - 2 / 3) < 1e-9
    assert len(res["severity_confusion_matrix"]) == 4


def test_detect_unsupported_claim():
    assert detect_unsupported_claim({"possible_explanations": ["Caused by a pipeline change."]})
    assert not detect_unsupported_claim(
        {"possible_explanations": ["Investigate distribution shift (hypothesis)."]}
    )


def test_evaluate_records_contains_core_keys():
    records = [
        {"gt_anomaly": True, "gt_severity": "high", "pred_anomaly": True, "pred_severity": "high",
         "abstained": False, "confidence": 0.9, "schema_valid": True, "gt_anomaly_type": "x",
         "pred_anomaly_type": "x", "n_supporting_evidence": 1, "n_followup": 1,
         "has_unsupported_claim": False, "latency_s": 0.1, "input_tokens": 100, "output_tokens": 50,
         "estimated_cost": 0.0, "total_retries": 0, "mock_unsupported_injected": False},
        {"gt_anomaly": False, "gt_severity": "none", "pred_anomaly": False, "pred_severity": "none",
         "abstained": False, "confidence": 0.6, "schema_valid": True, "gt_anomaly_type": "none",
         "pred_anomaly_type": "none", "n_supporting_evidence": 0, "n_followup": 0,
         "has_unsupported_claim": False, "latency_s": 0.1, "input_tokens": 100, "output_tokens": 50,
         "estimated_cost": 0.0, "total_retries": 0, "mock_unsupported_injected": False},
    ]
    m = evaluate_records(records)
    for key in ["precision", "recall", "f1", "expected_calibration_error", "brier_score",
                "schema_compliance_rate", "abstention_rate", "unsupported_claim_rate"]:
        assert key in m
    assert m["accuracy"] == 1.0
    # Records without the new agentic/validation/escalation fields simply
    # don't get those aggregate keys — old-style records stay unaffected.
    assert "avg_tool_calls" not in m
    assert "escalation_rate" not in m


def test_evaluate_records_includes_agentic_and_escalation_aggregates_when_present():
    records = [
        {"gt_anomaly": True, "gt_severity": "high", "pred_anomaly": True, "pred_severity": "high",
         "abstained": False, "confidence": 0.9, "schema_valid": True, "gt_anomaly_type": "x",
         "pred_anomaly_type": "x", "n_supporting_evidence": 1, "n_followup": 1,
         "has_unsupported_claim": False, "latency_s": 0.1, "input_tokens": 100, "output_tokens": 50,
         "estimated_cost": 0.0, "total_retries": 0, "mock_unsupported_injected": False,
         "tool_calls": 4, "evidence_citation_correct": True,
         "deterministic_validation_passed": True, "escalate": True, "gt_expected_escalation": True},
        {"gt_anomaly": False, "gt_severity": "none", "pred_anomaly": False, "pred_severity": "none",
         "abstained": False, "confidence": 0.6, "schema_valid": True, "gt_anomaly_type": "none",
         "pred_anomaly_type": "none", "n_supporting_evidence": 0, "n_followup": 0,
         "has_unsupported_claim": False, "latency_s": 0.1, "input_tokens": 100, "output_tokens": 50,
         "estimated_cost": 0.0, "total_retries": 0, "mock_unsupported_injected": False,
         "tool_calls": 2, "evidence_citation_correct": False,
         "deterministic_validation_passed": False, "escalate": False, "gt_expected_escalation": False},
    ]
    m = evaluate_records(records)
    assert m["avg_tool_calls"] == 3.0
    assert m["evidence_citation_accuracy"] == 0.5
    assert m["deterministic_validation_pass_rate"] == 0.5
    assert m["escalation_rate"] == 0.5
    assert m["escalation_precision"] == 1.0
    assert m["escalation_recall"] == 1.0


def test_bootstrap_ci_brackets_point():
    ci = bootstrap_ci([0, 1, 0, 1, 1, 0, 1, 1], n_boot=500, seed=1)
    assert ci["low"] <= ci["point"] <= ci["high"]


def test_mcnemar_detects_difference():
    # condition a correct everywhere, b wrong on many of the same cases
    corr_a = [True] * 30
    corr_b = [False] * 25 + [True] * 5
    res = mcnemar_test(corr_a, corr_b)
    assert res["p_value"] < 0.05
    assert res["n01"] == 25


def test_mcnemar_no_discordant():
    res = mcnemar_test([True, False], [True, False])
    assert res["p_value"] == 1.0


def test_cohens_h_and_permutation():
    assert cohens_h(0.5, 0.5) == 0.0
    perm = paired_permutation_test([1, 1, 1, 0], [0, 0, 0, 0], n_perm=1000)
    assert 0.0 <= perm["p_value"] <= 1.0


def test_benjamini_hochberg_monotone():
    out = benjamini_hochberg([0.001, 0.04, 0.5])
    assert len(out["adjusted"]) == 3
    assert all(0.0 <= p <= 1.0 for p in out["adjusted"])
