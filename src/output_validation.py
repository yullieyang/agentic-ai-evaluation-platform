"""Deterministic output validation — independent of the LLM.

These checks run *after* an ``AgentFinding`` has already passed Pydantic
schema validation (structural constraints: types, enums, required fields,
confidence range). They cover the cross-referential and business-rule
constraints Pydantic cannot express on its own, because they need external
state (the evidence store, the tool-call budget, the reviewer's output):

* does every cited ``evidence_id`` actually exist for this case?
* do quoted numeric values in ``supporting_evidence`` match the real evidence?
* is a high-severity finding actually flagged for human review?
* is unsupported causal wording present with no causal evidence to back it?
* is the reviewer's decision compatible with the finding it reviewed?
* did the run stay under the configured tool-call limit?

Every check returns a ``ValidationResult`` — pass or fail is never hidden;
callers (the experiment runner, the Streamlit UI) are expected to surface
all of them, not just the failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .evaluators import detect_unsupported_claim
from .evidence_store import EvidenceStore

NUMERIC_TOLERANCE = 1e-3


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"check_name": self.check_name, "passed": self.passed, "detail": self.detail}


def check_schema_conformance(schema_valid: bool, validation_error: Optional[str]) -> ValidationResult:
    return ValidationResult(
        check_name="schema_conformance", passed=bool(schema_valid),
        detail="Output validated against the AgentFinding schema." if schema_valid
        else f"Schema validation failed: {validation_error}",
    )


def check_evidence_ids_exist(finding: dict[str, Any], store: EvidenceStore, case_id: str) -> ValidationResult:
    cited = list(finding.get("evidence_ids") or [])
    if not cited:
        return ValidationResult("evidence_ids_exist", True, "No evidence_ids cited; nothing to check.")
    unknown = [eid for eid in cited if not store.exists(case_id, eid)]
    if unknown:
        return ValidationResult("evidence_ids_exist", False,
                                f"Cited evidence_id(s) not found in the case's evidence store: {unknown}")
    return ValidationResult("evidence_ids_exist", True, f"All {len(cited)} cited evidence_id(s) exist.")


def check_numeric_consistency(finding: dict[str, Any], store: EvidenceStore, case_id: str) -> ValidationResult:
    """Cross-check ``observed_value`` in each ``supporting_evidence`` entry with
    source=="feature" against the case's real metric_snapshot z-score."""
    snapshot_items = store.by_kind(case_id, "metric_snapshot")
    if not snapshot_items:
        return ValidationResult("numeric_consistency", True, "No metric_snapshot evidence to check against.")
    real_z = snapshot_items[0].data.get("z_score")
    mismatches = []
    for entry in finding.get("supporting_evidence") or []:
        if entry.get("source") != "feature":
            continue
        observed = entry.get("observed_value")
        if observed is None or real_z is None:
            continue
        if abs(float(observed) - float(real_z)) > NUMERIC_TOLERANCE + 0.05 * abs(real_z):
            mismatches.append({"claimed": observed, "actual_z_score": real_z})
    if mismatches:
        return ValidationResult("numeric_consistency", False,
                                f"Quoted feature value(s) do not match the case's real evidence: {mismatches}")
    return ValidationResult("numeric_consistency", True, "Quoted numeric values match the source evidence.")


def check_high_severity_requires_review(finding: dict[str, Any]) -> ValidationResult:
    if finding.get("severity") == "high" and not finding.get("requires_human_review", False):
        return ValidationResult("high_severity_requires_review", False,
                                "severity='high' but requires_human_review is False.")
    return ValidationResult("high_severity_requires_review", True,
                            "High-severity findings are flagged for human review (or severity is not high).")


def check_unsupported_causal_wording(finding: dict[str, Any]) -> ValidationResult:
    has_unsupported = detect_unsupported_claim(finding)
    if has_unsupported:
        return ValidationResult("unsupported_causal_wording", False,
                                "possible_explanations asserts a cause as fact with no causal evidence cited.")
    return ValidationResult("unsupported_causal_wording", True,
                            "No unsupported causal wording detected (surface-pattern check).")


def check_reviewer_compatibility(finding: dict[str, Any], review: Optional[dict[str, Any]]) -> ValidationResult:
    if review is None:
        return ValidationResult("reviewer_compatibility", True, "No reviewer pass for this run.")
    decision = review.get("review_decision")
    if decision == "approve" and review.get("unsupported_claims_found"):
        return ValidationResult("reviewer_compatibility", False,
                                "Reviewer approved the finding but also listed unsupported claims found.")
    if decision == "reject" and review.get("revised_anomaly_detected") == finding.get("anomaly_detected") \
            and review.get("revised_severity") == finding.get("severity"):
        return ValidationResult("reviewer_compatibility", False,
                                "Reviewer rejected the finding but its revised decision is unchanged from the original.")
    return ValidationResult("reviewer_compatibility", True, "Reviewer decision is compatible with its stated findings.")


def check_tool_call_limit(tool_calls: int, max_tool_calls: int) -> ValidationResult:
    if tool_calls > max_tool_calls:
        return ValidationResult("tool_call_limit", False,
                                f"{tool_calls} tool calls exceeds the configured limit of {max_tool_calls}.")
    return ValidationResult("tool_call_limit", True,
                            f"{tool_calls}/{max_tool_calls} tool calls used.")


def run_output_validation(
    *,
    finding: Optional[dict[str, Any]],
    schema_valid: bool,
    validation_error: Optional[str],
    store: Optional[EvidenceStore] = None,
    case_id: Optional[str] = None,
    review: Optional[dict[str, Any]] = None,
    tool_calls: int = 0,
    max_tool_calls: int = 0,
) -> list[ValidationResult]:
    """Run every deterministic output check and return all results (pass and
    fail alike) — never just the failures."""
    results = [check_schema_conformance(schema_valid, validation_error)]
    if not schema_valid or finding is None:
        # Everything past this point needs a valid finding to check against.
        return results

    if store is not None and case_id is not None:
        results.append(check_evidence_ids_exist(finding, store, case_id))
        results.append(check_numeric_consistency(finding, store, case_id))
    results.append(check_high_severity_requires_review(finding))
    results.append(check_unsupported_causal_wording(finding))
    results.append(check_reviewer_compatibility(finding, review))
    if max_tool_calls > 0 or tool_calls > 0:
        results.append(check_tool_call_limit(tool_calls, max_tool_calls))
    return results


def all_passed(results: list[ValidationResult]) -> bool:
    return all(r.passed for r in results)
