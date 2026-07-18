"""Evaluation metrics for agent findings against labelled synthetic ground truth.

Provides classification metrics, severity metrics, abstention/selective metrics,
schema and evidence-quality metrics, an unsupported-claim analysis, and resource
(latency/token/cost) aggregates. Metrics can be computed over an arbitrary
decision field so the same code evaluates pre-review and post-review outputs.

Limitations are documented per metric. Notably, in mock mode the agent's
``anomaly_type`` vocabulary is intentionally distinct from the ground-truth
anomaly-type vocabulary, so exact ``anomaly_type`` accuracy is near zero by
construction; the metric is retained for completeness and real-model use.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from .calibration import brier_score, expected_calibration_error

SEVERITY_LEVELS = ["none", "low", "medium", "high"]
UNSUPPORTED_PATTERNS = ("caused by", "because of", "due to a", "resulted from")


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: Iterable[bool], y_pred: Iterable[bool]) -> dict[str, float]:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    n = len(yt)
    if n == 0:
        return {}
    c = _confusion(yt, yp)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        **{f"count_{k}": v for k, v in c.items()},
    }


def severity_confusion(true_sev: Iterable[str], pred_sev: Iterable[str]) -> dict[str, Any]:
    idx = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    matrix = np.zeros((4, 4), dtype=int)
    correct = 0
    total = 0
    for t, p in zip(true_sev, pred_sev):
        if t in idx and p in idx:
            matrix[idx[t], idx[p]] += 1
            correct += int(t == p)
            total += 1
    return {
        "severity_accuracy": correct / total if total else 0.0,
        "severity_confusion_matrix": matrix.tolist(),
        "severity_levels": SEVERITY_LEVELS,
    }


def detect_unsupported_claim(finding: dict[str, Any]) -> bool:
    """Pattern-based detector: a possible explanation asserts a cause as fact.

    Limitation: a surface-pattern detector is a coarse proxy. It misses
    paraphrased causal claims and may misfire on quoted hypotheses; it is
    combined with mock-injected labels and (optionally) reviewer/human labels for
    a fuller picture.
    """
    for claim in finding.get("possible_explanations", []) or []:
        text = str(claim).lower()
        if any(p in text for p in UNSUPPORTED_PATTERNS) and "hypothesis" not in text:
            return True
    return False


def classify_claims(finding: dict[str, Any]) -> list[dict[str, str]]:
    """Label each possible explanation as supported/partially/unsupported/unverifiable."""
    out = []
    cited_metrics = {e.get("metric_name") for e in finding.get("supporting_evidence", []) or []}
    for claim in finding.get("possible_explanations", []) or []:
        text = str(claim).lower()
        if any(p in text for p in UNSUPPORTED_PATTERNS) and "hypothesis" not in text:
            label = "unsupported"
        elif "hypothesis" in text or "investigate" in text or "possible" in text:
            label = "partially_supported" if cited_metrics else "unverifiable"
        else:
            label = "unverifiable"
        out.append({"claim": str(claim), "label": label})
    return out


def evaluate_records(
    records: list[dict[str, Any]],
    decision_key: str = "pred_anomaly",
    severity_key: str = "pred_severity",
    abstain_key: str = "abstained",
    confidence_key: str = "confidence",
) -> dict[str, Any]:
    """Aggregate metrics over evaluation records.

    Each record must include: gt_anomaly, gt_severity, <decision_key>,
    <severity_key>, <abstain_key>, <confidence_key>, schema_valid, gt_anomaly_type,
    pred_anomaly_type, n_supporting_evidence, n_followup, has_unsupported_claim,
    latency_s, input_tokens, output_tokens, estimated_cost, total_retries.
    """
    if not records:
        return {}
    y_true = [bool(r["gt_anomaly"]) for r in records]
    y_pred = [bool(r[decision_key]) for r in records]
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(severity_confusion([r["gt_severity"] for r in records],
                                       [r[severity_key] for r in records]))

    # Abstention / selective performance.
    abstained = [bool(r.get(abstain_key, False)) for r in records]
    metrics["abstention_rate"] = float(np.mean(abstained))
    kept = [r for r, a in zip(records, abstained) if not a]
    if kept:
        sel = classification_metrics([r["gt_anomaly"] for r in kept],
                                     [r[decision_key] for r in kept])
        metrics["selective_accuracy"] = sel.get("accuracy", 0.0)
        metrics["selective_f1"] = sel.get("f1", 0.0)
    else:
        metrics["selective_accuracy"] = 0.0
        metrics["selective_f1"] = 0.0

    # Schema / evidence quality.
    metrics["schema_compliance_rate"] = float(np.mean([bool(r["schema_valid"]) for r in records]))
    pos = [r for r in records if bool(r[decision_key])]
    metrics["evidence_completeness_rate"] = (
        float(np.mean([r["n_supporting_evidence"] > 0 for r in pos])) if pos else 1.0
    )
    metrics["recommendation_actionability"] = (
        float(np.mean([r["n_followup"] > 0 for r in pos])) if pos else 1.0
    )

    # Unsupported-claim analysis (pattern-based; mock-injected reference if present).
    metrics["unsupported_claim_rate"] = float(np.mean([bool(r.get("has_unsupported_claim", False))
                                                       for r in records]))
    injected = [r.get("mock_unsupported_injected") for r in records]
    if any(i is not None for i in injected):
        inj = np.asarray([bool(i) for i in injected])
        det = np.asarray([bool(r.get("has_unsupported_claim", False)) for r in records])
        tp = int(np.sum(inj & det))
        metrics["unsupported_claim_rate_reference"] = float(np.mean(inj))
        metrics["unsupported_detection_recall"] = float(tp / inj.sum()) if inj.sum() else None

    # Anomaly-type accuracy on true positives (see module note on vocabulary).
    tps = [r for r in records if bool(r["gt_anomaly"]) and bool(r[decision_key])]
    metrics["anomaly_type_exact_accuracy"] = (
        float(np.mean([r["pred_anomaly_type"] == r["gt_anomaly_type"] for r in tps])) if tps else 0.0
    )

    # Confidence + calibration.
    conf = np.asarray([float(r[confidence_key]) for r in records])
    correct = np.asarray([bool(r["gt_anomaly"]) == bool(r[decision_key]) for r in records])
    metrics["average_confidence"] = float(np.mean(conf))
    metrics["confidence_std"] = float(np.std(conf))
    metrics["expected_calibration_error"] = expected_calibration_error(conf, correct)
    metrics["brier_score"] = brier_score(conf, correct)
    metrics["overconfidence_rate"] = float(np.mean((conf > 0.5) & (~correct)))
    metrics["underconfidence_rate"] = float(np.mean((conf < 0.5) & (correct)))

    # Resource aggregates.
    metrics["avg_latency_s"] = float(np.mean([r["latency_s"] for r in records]))
    metrics["avg_input_tokens"] = float(np.mean([r["input_tokens"] for r in records]))
    metrics["avg_output_tokens"] = float(np.mean([r["output_tokens"] for r in records]))
    costs = [r["estimated_cost"] for r in records if r.get("estimated_cost") is not None]
    metrics["avg_estimated_cost"] = float(np.mean(costs)) if costs else 0.0
    metrics["total_estimated_cost"] = float(np.sum(costs)) if costs else 0.0
    metrics["avg_retries"] = float(np.mean([r["total_retries"] for r in records]))

    # Agentic-architecture-specific aggregates (0 for single-shot records, which
    # always report tool_calls=0 by construction).
    if any("tool_calls" in r for r in records):
        metrics["avg_tool_calls"] = float(np.mean([r.get("tool_calls", 0) for r in records]))
    citation_flags = [r.get("evidence_citation_correct") for r in records
                      if r.get("evidence_citation_correct") is not None]
    if citation_flags:
        metrics["evidence_citation_accuracy"] = float(np.mean([bool(c) for c in citation_flags]))
    if any("deterministic_validation_passed" in r for r in records):
        metrics["deterministic_validation_pass_rate"] = float(
            np.mean([bool(r.get("deterministic_validation_passed", True)) for r in records]))
    if any("escalate" in r for r in records):
        metrics["escalation_rate"] = float(np.mean([bool(r.get("escalate", False)) for r in records]))
    if any("gt_expected_escalation" in r for r in records) and any("escalate" in r for r in records):
        esc_metrics = classification_metrics(
            [bool(r.get("gt_expected_escalation", False)) for r in records],
            [bool(r.get("escalate", False)) for r in records],
        )
        metrics["escalation_precision"] = esc_metrics.get("precision")
        metrics["escalation_recall"] = esc_metrics.get("recall")
    return metrics


def group_metrics(records: list[dict[str, Any]], by: str, **kwargs: Any) -> dict[str, dict]:
    """Compute metrics within each level of a grouping key."""
    levels: dict[str, list[dict]] = {}
    for r in records:
        levels.setdefault(str(r.get(by)), []).append(r)
    return {level: evaluate_records(rs, **kwargs) for level, rs in sorted(levels.items())}
