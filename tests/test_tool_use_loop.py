"""Tests for the tool-use loop: the mock's real-executor simulation, and the
live Anthropic loop's control flow against a stubbed client (no network, no
API key). ``anthropic`` is an optional dependency — these tests are skipped
entirely if it is not installed, matching the rest of the test suite's
"mock mode and tests never require anthropic/openai" guarantee.
"""

from __future__ import annotations

import json

import pytest

from src.chart_summaries import build_chart_summary
from src.data_generation import generate_dataset
from src.deterministic_checks import baseline_decision, run_checks
from src.evidence_store import EvidenceStore, build_case_evidence_items
from src.evidence_tools import TOOL_SPECS, ToolExecutor
from src.feature_engineering import build_features
from src.llm_providers import MockProvider
from src.schemas import AgentFinding, json_schema, parse_model
from src.utils import load_config


def _bundle_and_store(seed=5):
    df, _, _ = generate_dataset(n_cases=15, seed=seed)
    feats = build_features(df)
    row = df.iloc[0].to_dict()
    cid = row["case_id"]
    f = feats.loc[cid].to_dict()
    checks = run_checks(row, f, row["_series"])
    base_anom, base_sev = baseline_decision(checks)
    chart = build_chart_summary(row["_series"], row, f)
    eval_cfg = load_config("evaluation.yaml")
    items = build_case_evidence_items(cid, row, f, checks, chart, eval_cfg)
    store = EvidenceStore()
    store.register(cid, items)
    return row, f, checks, chart, base_anom, base_sev, store


# --------------------------------------------------------------------------- #
# Mock provider's tool-use path
# --------------------------------------------------------------------------- #
def test_mock_complete_with_tools_drives_real_executor():
    row, f, checks, chart, base_anom, base_sev, store = _bundle_and_store()
    cid = row["case_id"]
    executor = ToolExecutor(store, max_calls=6)
    provider = MockProvider(error_rate=0.1)
    ctx = {
        "case_id": cid, "prompt_version": "prompt_c_evidence_constrained",
        "attempt": 0, "profile": {},
        "evidence": {"metric_name": row["metric_name"], "z_score": row["z_score"],
                     "robust_z_score": f.get("robust_z_score", row["z_score"]),
                     "missing_rate": row["missing_rate"], "small_sample": row["sample_size"] < 100,
                     "baseline_anomaly": base_anom, "baseline_severity": base_sev,
                     "mitigation_flags": [], "incomplete": False,
                     "available_sources": ["features", "deterministic"]},
    }
    resp = provider.complete_with_tools(
        "sys", "usr", model="mock-deterministic", tool_specs=TOOL_SPECS,
        executor=executor, schema=json_schema(AgentFinding), context=ctx,
    )
    assert resp.tool_calls == executor.calls_made
    assert executor.calls_made >= 2  # at least list_available_evidence + get_case_metrics
    assert resp.parsed is not None
    payload = {k: v for k, v in resp.parsed.items() if not k.startswith("_")}
    parse_model(AgentFinding, payload)  # must still validate against the schema
    # Every cited evidence_id must correspond to something actually fetched.
    fetched_ids = {e.result["evidence_id"] for e in executor.call_log
                   if e.ok and isinstance(e.result, dict) and "evidence_id" in e.result}
    assert set(resp.parsed["evidence_ids"]) <= fetched_ids


def test_mock_complete_with_tools_is_deterministic():
    row, f, checks, chart, base_anom, base_sev, store = _bundle_and_store()
    cid = row["case_id"]

    def _run():
        executor = ToolExecutor(store, max_calls=6)
        provider = MockProvider(error_rate=0.1)
        ctx = {"case_id": cid, "prompt_version": "prompt_c_evidence_constrained",
               "attempt": 0, "profile": {},
               "evidence": {"metric_name": row["metric_name"], "z_score": row["z_score"],
                            "robust_z_score": row["z_score"], "missing_rate": row["missing_rate"],
                            "small_sample": False, "baseline_anomaly": base_anom,
                            "baseline_severity": base_sev, "mitigation_flags": [],
                            "incomplete": False, "available_sources": ["features", "deterministic"]}}
        return provider.complete_with_tools("s", "u", model="mock-deterministic",
                                            tool_specs=TOOL_SPECS, executor=executor,
                                            schema=None, context=ctx)

    r1, r2 = _run(), _run()
    assert r1.raw == r2.raw
    assert r1.tool_calls == r2.tool_calls


def test_mock_complete_with_tools_respects_call_limit():
    row, f, checks, chart, base_anom, base_sev, store = _bundle_and_store()
    cid = row["case_id"]
    executor = ToolExecutor(store, max_calls=1)  # deliberately too low to finish the script
    provider = MockProvider(error_rate=0.1)
    ctx = {"case_id": cid, "prompt_version": "prompt_c_evidence_constrained",
           "attempt": 0, "profile": {},
           "evidence": {"metric_name": row["metric_name"], "z_score": row["z_score"],
                        "robust_z_score": row["z_score"], "missing_rate": row["missing_rate"],
                        "small_sample": False, "baseline_anomaly": True,
                        "baseline_severity": "high", "mitigation_flags": ["seasonal_warning"],
                        "incomplete": False, "available_sources": ["features", "deterministic"]}}
    resp = provider.complete_with_tools("s", "u", model="mock-deterministic",
                                        tool_specs=TOOL_SPECS, executor=executor,
                                        schema=None, context=ctx)
    assert executor.calls_made >= 1
    # Every attempted call beyond the limit is logged as a rejected (not ok) call.
    assert any(not e.ok for e in executor.call_log) or executor.calls_made == 1


# --------------------------------------------------------------------------- #
# Live Anthropic provider's tool-use loop (stubbed client; no network, no key,
# and — since __init__ is bypassed below — no real ``anthropic`` package
# needs to be installed to run these either. They test only this codebase's
# loop control-flow (tool_use -> tool_result -> ... -> final schema call)
# against the request/response shapes verified earlier against the real SDK.
# --------------------------------------------------------------------------- #
class _FakeBlock:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, content, usage):
        self.content = content
        self.usage = usage


class _StubMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _StubClient:
    def __init__(self, responses):
        self.messages = _StubMessages(responses)


def _make_stub_provider(responses):
    from src.llm_providers import AnthropicProvider
    provider = AnthropicProvider.__new__(AnthropicProvider)  # bypass __init__ (no real client)
    provider._anthropic = None  # unused outside __init__; no real `anthropic` package needed here
    provider._client = _StubClient(responses)
    return provider


def test_anthropic_tool_loop_calls_tool_then_forces_final_schema():
    row, f, checks, chart, base_anom, base_sev, store = _bundle_and_store(seed=8)
    cid = row["case_id"]
    executor = ToolExecutor(store, max_calls=6)

    final_finding = {
        "anomaly_detected": True, "severity": "medium", "anomaly_type": "level_shift",
        "summary": "s", "observations": ["o"], "supporting_evidence": [], "evidence_ids": [],
        "affected_metrics": [], "possible_explanations": [], "recommended_follow_up": [],
        "confidence_score": 0.7, "evidence_sufficiency": "adequate",
        "unsupported_claim_risk": "low", "requires_human_review": True,
        "abstained": False, "abstention_reason": None,
    }
    responses = [
        # Turn 1: model requests a tool call.
        _FakeResponse(
            content=[_FakeBlock("tool_use", id="toolu_1", name="get_case_metrics",
                                input={"case_id": cid})],
            usage=_FakeUsage(120, 30),
        ),
        # Turn 2: model stops calling tools.
        _FakeResponse(
            content=[_FakeBlock("text", text="I have enough evidence now.")],
            usage=_FakeUsage(140, 15),
        ),
        # Turn 3: final schema-forced call.
        _FakeResponse(
            content=[_FakeBlock("text", text=json.dumps(final_finding))],
            usage=_FakeUsage(160, 40),
        ),
    ]
    provider = _make_stub_provider(responses)
    resp = provider.complete_with_tools(
        "sys", "usr", model="claude-opus-4-8", tool_specs=TOOL_SPECS, executor=executor,
        schema=json_schema(AgentFinding), context={"attempt": 0},
    )

    assert executor.calls_made == 1
    assert executor.call_log[0].tool_name == "get_case_metrics"
    assert executor.call_log[0].ok is True
    assert resp.parsed is not None
    assert resp.parsed["anomaly_detected"] is True
    assert resp.tool_calls == 1
    assert resp.input_tokens == 120 + 140 + 160
    assert resp.output_tokens == 30 + 15 + 40

    # The final call must have been made without `tools` and with output_config set.
    final_call_kwargs = provider._client.messages.calls[-1]
    assert "tools" not in final_call_kwargs
    assert final_call_kwargs["output_config"]["format"]["type"] == "json_schema"

    # The first call must have offered the tool specs.
    first_call_kwargs = provider._client.messages.calls[0]
    assert first_call_kwargs["tools"] == TOOL_SPECS


def test_anthropic_tool_loop_stops_at_call_limit_and_still_forces_final_answer():
    row, f, checks, chart, base_anom, base_sev, store = _bundle_and_store(seed=9)
    cid = row["case_id"]
    executor = ToolExecutor(store, max_calls=1)

    final_finding = {
        "anomaly_detected": False, "severity": "none", "anomaly_type": "none",
        "summary": "s", "observations": [], "supporting_evidence": [], "evidence_ids": [],
        "affected_metrics": [], "possible_explanations": [], "recommended_follow_up": [],
        "confidence_score": 0.4, "evidence_sufficiency": "limited",
        "unsupported_claim_risk": "low", "requires_human_review": True,
        "abstained": True, "abstention_reason": "tool-call limit reached before enough evidence",
    }
    # With max_calls=1, the loop executes exactly one tool call, then — since
    # the limit is checked *before* requesting another turn — it stops asking
    # the model for more tool calls and moves straight to the forced final
    # answer. Only 2 model calls actually happen: the tool-use turn, then the
    # final schema-forced turn (never a 3rd "still trying to call tools" turn).
    responses = [
        _FakeResponse(
            content=[_FakeBlock("tool_use", id="toolu_1", name="get_case_metrics",
                                input={"case_id": cid})],
            usage=_FakeUsage(100, 20),
        ),
        # Final forced call.
        _FakeResponse(content=[_FakeBlock("text", text=json.dumps(final_finding))],
                      usage=_FakeUsage(90, 25)),
    ]
    provider = _make_stub_provider(responses)
    resp = provider.complete_with_tools(
        "sys", "usr", model="claude-opus-4-8", tool_specs=TOOL_SPECS, executor=executor,
        schema=json_schema(AgentFinding), context={"attempt": 0},
    )
    assert executor.calls_made == 1
    assert len(provider._client.messages.calls) == 2  # tool turn + final forced turn only
    assert resp.parsed is not None
    assert resp.parsed["abstained"] is True


def test_anthropic_complete_wraps_provider_error_distinguishably():
    """A provider/network/auth failure must come back as a structured
    ProviderResponse with parsed=None and a 'provider error' prefix — never
    an uncaught exception, and distinguishable from a JSON-parsing failure."""
    class _RaisingMessages:
        def create(self, **kwargs):
            raise RuntimeError("simulated connection reset")

    class _RaisingClient:
        def __init__(self):
            self.messages = _RaisingMessages()

    provider = _make_stub_provider([])
    provider._client = _RaisingClient()
    resp = provider.complete("sys", "usr", model="claude-opus-4-8", context={"attempt": 0})
    assert resp.parsed is None
    assert resp.parsing_error.startswith("provider error:")
    assert "simulated connection reset" in resp.parsing_error


def test_anthropic_complete_with_tools_wraps_provider_error_distinguishably():
    row, f, checks, chart, base_anom, base_sev, store = _bundle_and_store(seed=13)
    executor = ToolExecutor(store, max_calls=6)

    class _RaisingMessages:
        def create(self, **kwargs):
            raise RuntimeError("simulated rate limit")

    class _RaisingClient:
        def __init__(self):
            self.messages = _RaisingMessages()

    provider = _make_stub_provider([])
    provider._client = _RaisingClient()
    resp = provider.complete_with_tools(
        "sys", "usr", model="claude-opus-4-8", tool_specs=TOOL_SPECS, executor=executor,
        schema=json_schema(AgentFinding), context={"attempt": 0},
    )
    assert resp.parsed is None
    assert resp.parsing_error.startswith("provider error:")
    assert executor.calls_made == 0  # failed before any tool was ever called


def test_block_to_param_round_trips_text_and_tool_use():
    from src.llm_providers import _block_to_param

    text_block = _FakeBlock("text", text="hello")
    tool_block = _FakeBlock("tool_use", id="t1", name="get_case_metrics", input={"case_id": "c"})
    assert _block_to_param(text_block) == {"type": "text", "text": "hello"}
    assert _block_to_param(tool_block) == {"type": "tool_use", "id": "t1",
                                           "name": "get_case_metrics", "input": {"case_id": "c"}}
