"""Tests for evidence-retrieval tools: argument validation, missing evidence,
invalid IDs, and the tool-call limit."""

from __future__ import annotations

from src.chart_summaries import build_chart_summary
from src.data_generation import generate_dataset
from src.deterministic_checks import run_checks
from src.evidence_store import EvidenceStore, build_case_evidence_items
from src.evidence_tools import (
    TOOL_NAMES,
    TOOL_SPECS,
    EvidenceToolError,
    ToolExecutor,
    get_case_metrics,
    get_evidence_item,
)
from src.feature_engineering import build_features
from src.utils import load_config


def _store_with_one_case(seed=1):
    df, _, _ = generate_dataset(n_cases=10, seed=seed)
    feats = build_features(df)
    row = df.iloc[0].to_dict()
    cid = row["case_id"]
    f = feats.loc[cid].to_dict()
    checks = run_checks(row, f, row["_series"])
    chart = build_chart_summary(row["_series"], row, f)
    eval_cfg = load_config("evaluation.yaml")
    items = build_case_evidence_items(cid, row, f, checks, chart, eval_cfg)
    store = EvidenceStore()
    store.register(cid, items)
    return store, cid, items


def test_tool_specs_match_dispatch_names():
    spec_names = {s["name"] for s in TOOL_SPECS}
    assert spec_names == set(TOOL_NAMES)


def test_tool_specs_are_valid_json_schema_shape():
    for spec in TOOL_SPECS:
        assert "name" in spec and "description" in spec and "input_schema" in spec
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert "case_id" in schema["properties"]
        assert "case_id" in schema["required"]


def test_get_case_metrics_direct_call():
    store, cid, _ = _store_with_one_case()
    result = get_case_metrics(store, cid)
    assert result["kind"] == "metric_snapshot"
    assert "current_value" in result["data"]


def test_get_case_metrics_unknown_case_raises():
    store, cid, _ = _store_with_one_case()
    try:
        get_case_metrics(store, "not-a-real-case")
        assert False, "expected EvidenceToolError"
    except EvidenceToolError:
        pass


def test_get_evidence_item_invalid_id_raises():
    store, cid, _ = _store_with_one_case()
    try:
        get_evidence_item(store, cid, "not-a-real-id")
        assert False, "expected EvidenceToolError"
    except EvidenceToolError:
        pass


def test_get_evidence_item_valid_id():
    store, cid, items = _store_with_one_case()
    result = get_evidence_item(store, cid, items[0].evidence_id)
    assert result["evidence_id"] == items[0].evidence_id


def test_executor_call_success_and_logging():
    store, cid, _ = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=5)
    result = executor.call("get_case_metrics", {"case_id": cid})
    assert "error" not in result
    assert executor.calls_made == 1
    assert executor.call_log[0].ok is True


def test_executor_missing_case_returns_structured_error_not_raise():
    store, cid, _ = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=5)
    result = executor.call("get_case_metrics", {"case_id": "bogus"})
    assert "error" in result
    assert executor.calls_made == 1
    assert executor.call_log[0].ok is False


def test_executor_invalid_arguments_returns_structured_error():
    store, cid, _ = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=5)
    result = executor.call("get_case_metrics", {"wrong_arg": "x"})
    assert "error" in result


def test_executor_unknown_tool_name_returns_structured_error():
    store, cid, _ = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=5)
    result = executor.call("delete_everything", {"case_id": cid})
    assert "error" in result
    assert "unknown tool" in result["error"]


def test_executor_enforces_max_tool_call_limit():
    store, cid, _ = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=2)
    executor.call("get_case_metrics", {"case_id": cid})
    executor.call("get_case_metrics", {"case_id": cid})
    result = executor.call("get_case_metrics", {"case_id": cid})
    assert "error" in result
    assert "limit" in result["error"]
    assert executor.calls_made == 3  # the rejected call is still logged
    assert executor.limit_reached() is True


def test_executor_log_as_dicts_serializable():
    import json
    store, cid, _ = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=5)
    executor.call("get_case_metrics", {"case_id": cid})
    executor.call("get_evidence_item", {"case_id": cid, "evidence_id": "bogus"})
    blob = json.dumps(executor.log_as_dicts())
    assert "get_case_metrics" in blob
    assert "get_evidence_item" in blob


def test_list_available_evidence_does_not_leak_full_content():
    store, cid, items = _store_with_one_case()
    executor = ToolExecutor(store, max_calls=5)
    result = executor.call("list_available_evidence", {"case_id": cid})
    assert len(result["items"]) == len(items)
    for entry in result["items"]:
        assert set(entry) == {"evidence_id", "kind", "title"}
