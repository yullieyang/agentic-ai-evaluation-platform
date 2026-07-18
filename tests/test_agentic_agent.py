"""Tests for the agentic (tool-using) QA agent."""

from __future__ import annotations

import json

from src.agent import PROMPT_PROFILES, build_evidence
from src.agentic_agent import ToolUsingQAAgent, build_minimal_case_framing
from src.chart_summaries import build_chart_summary
from src.data_generation import generate_dataset
from src.deterministic_checks import baseline_decision, run_checks
from src.feature_engineering import build_features
from src.llm_providers import MockProvider
from src.schemas import GROUND_TRUTH_FIELDS, AgentFinding, assert_no_ground_truth, parse_model


def _real_case(seed=3):
    df, _, _ = generate_dataset(n_cases=20, seed=seed)
    feats = build_features(df)
    row = df.iloc[0].to_dict()
    cid = row["case_id"]
    f = feats.loc[cid].to_dict()
    checks = run_checks(row, f, row["_series"])
    base_anom, base_sev = baseline_decision(checks)
    chart = build_chart_summary(row["_series"], row, f)
    _, mock_evidence = build_evidence(
        row=row, features=f, checks=checks, chart_summary=chart,
        include_deterministic_evidence=True, baseline_anomaly=base_anom,
        baseline_severity=base_sev, evidence_completeness="complete")
    case = {
        "row": row, "features": f, "checks": checks, "chart_summary": chart,
        "prompt_version": "prompt_c_evidence_constrained",
        "include_deterministic_evidence": True,
        "mock_evidence": mock_evidence,
        "mock_profile": PROMPT_PROFILES.get("prompt_c_evidence_constrained", {}),
        "temperature": 0.0, "model": "mock-deterministic",
    }
    return case


def test_minimal_case_framing_excludes_ground_truth_and_leaky_fields():
    df, _, _ = generate_dataset(n_cases=20, seed=3)
    row = df.iloc[0].to_dict()
    framing = build_minimal_case_framing(row)
    assert_no_ground_truth(framing)
    blob = json.dumps(framing)
    for gt_field in GROUND_TRUTH_FIELDS:
        assert f'"{gt_field}"' not in blob
    assert "alert_type" not in framing
    assert "scenario_type" not in framing
    # Safe identity fields ARE present.
    assert framing["case_id"] == row["case_id"]
    assert framing["release_version"] == row["release_version"]


def test_agentic_agent_runs_and_produces_valid_finding():
    case = _real_case()
    agent = ToolUsingQAAgent(MockProvider(error_rate=0.1), max_retries=2, max_tool_calls=6)
    result = agent.run(case)
    assert result.schema_valid is True
    parse_model(AgentFinding, result.finding)
    assert result.tool_calls >= 2  # at least list_available_evidence + get_case_metrics
    assert len(result.tool_call_log) == result.tool_calls


def test_agentic_agent_respects_max_tool_calls():
    case = _real_case()
    agent = ToolUsingQAAgent(MockProvider(error_rate=0.1), max_retries=0, max_tool_calls=1)
    result = agent.run(case)
    # The executor itself enforces the limit; the agent must not silently
    # exceed it even though the scripted mock policy wants to call more tools.
    assert result.tool_calls <= 1 + 1  # allow one rejected/logged attempt beyond the cap


def test_agentic_agent_is_reproducible():
    case = _real_case(seed=11)
    r1 = ToolUsingQAAgent(MockProvider(error_rate=0.1)).run(case)
    r2 = ToolUsingQAAgent(MockProvider(error_rate=0.1)).run(case)
    assert r1.finding == r2.finding
    assert r1.tool_calls == r2.tool_calls


def test_agentic_agent_evidence_ids_traceable_to_store():
    case = _real_case(seed=21)
    agent = ToolUsingQAAgent(MockProvider(error_rate=0.1))
    result = agent.run(case)
    store = result.extra["store"]
    cid = case["row"]["case_id"]
    for eid in result.finding.get("evidence_ids", []):
        assert store.exists(cid, eid)
