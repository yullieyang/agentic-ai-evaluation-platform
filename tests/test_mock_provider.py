"""Tests for the deterministic mock provider."""

from __future__ import annotations

from src.llm_providers import MockProvider, ProviderResponse, get_provider
from src.schemas import AgentFinding, ReviewerOutput, parse_model


def _ctx(attempt=0, mode=None):
    ctx = {
        "case_id": "case_test", "prompt_version": "prompt_c_evidence_constrained",
        "include_deterministic_evidence": True, "reviewer_enabled": False,
        "evidence_completeness": "complete", "attempt": attempt,
        "profile": {"malformed_rate": 0.0, "unsupported_rate": 0.0},
        "evidence": {"metric_name": "auc", "z_score": 4.2, "robust_z_score": 4.0,
                     "missing_rate": 0.0, "small_sample": False, "baseline_anomaly": True,
                     "baseline_severity": "high", "mitigation_flags": [], "incomplete": False,
                     "available_sources": ["features", "deterministic"]},
    }
    if mode:
        ctx["mode"] = mode
    return ctx


def test_mock_is_deterministic():
    p = MockProvider(error_rate=0.1)
    r1 = p.complete("s", "u", model="mock-deterministic", context=_ctx())
    r2 = p.complete("s", "u", model="mock-deterministic", context=_ctx())
    assert r1.raw == r2.raw
    assert r1.is_mock is True


def test_mock_output_validates_against_schema():
    p = MockProvider()
    r = p.complete("s", "u", model="mock-deterministic", context=_ctx())
    payload = {k: v for k, v in r.parsed.items() if not k.startswith("_")}
    parse_model(AgentFinding, payload)


def test_mock_malformed_then_recovers_on_retry():
    p = MockProvider()
    ctx0 = _ctx(attempt=0)
    ctx0["profile"] = {"malformed_rate": 1.0}  # force malformed on attempt 0
    r0 = p.complete("s", "u", model="mock-deterministic", context=ctx0)
    assert r0.parsed is None and r0.parsing_error is not None
    ctx1 = _ctx(attempt=1)
    ctx1["profile"] = {"malformed_rate": 1.0}
    r1 = p.complete("s", "u", model="mock-deterministic", context=ctx1)
    assert r1.parsed is not None  # retries recover


def test_mock_reviewer_mode_shape():
    p = MockProvider()
    ctx = _ctx(mode="reviewer")
    ctx["agent_finding"] = {"anomaly_detected": True, "severity": "high", "anomaly_type": "x",
                            "possible_explanations": ["Caused by a pipeline change."],
                            "confidence_score": 0.8}
    r = p.complete("s", "u", model="mock-deterministic", context=ctx)
    review = parse_model(ReviewerOutput, {k: v for k, v in r.parsed.items()
                                          if not k.startswith("_")})
    # The reviewer should detect the unsupported causal claim.
    assert len(r.parsed["unsupported_claims_found"]) >= 1


def test_error_rate_changes_behaviour_distribution():
    low = MockProvider(error_rate=0.0)
    high = MockProvider(error_rate=1.0)
    # With error_rate 1.0 the decision flips relative to error_rate 0.0 on a
    # non-abstained case.
    ctx = _ctx()
    r_low = low.complete("s", "u", model="mock-deterministic", context=ctx)
    r_high = high.complete("s", "u", model="mock-deterministic", context=ctx)
    assert r_low.parsed["anomaly_detected"] != r_high.parsed["anomaly_detected"]


def test_factory_returns_mock():
    assert isinstance(get_provider("mock"), MockProvider)
