"""Tests for the ID-addressable evidence store and case-model extensions."""

from __future__ import annotations

import json

from src.data_generation import generate_dataset
from src.deterministic_checks import run_checks
from src.chart_summaries import build_chart_summary
from src.evidence_store import (
    EvidenceStore,
    alert_type_for,
    build_case_evidence_items,
    release_version_for,
)
from src.feature_engineering import build_features
from src.schemas import GROUND_TRUTH_FIELDS, CaseRecord, assert_no_ground_truth, parse_model
from src.utils import load_config


def _one_case_bundle(seed=1):
    df, _, _ = generate_dataset(n_cases=20, seed=seed)
    feats = build_features(df)
    row = df.iloc[0].to_dict()
    cid = row["case_id"]
    series = row["_series"]
    f = feats.loc[cid].to_dict()
    checks = run_checks(row, f, series)
    chart = build_chart_summary(series, row, f)
    return row, f, checks, chart


def test_case_record_includes_new_fields():
    df, meta, _ = generate_dataset(n_cases=30, seed=9)
    rec = df.iloc[0][meta["schema_fields"]].to_dict()
    parsed = parse_model(CaseRecord, rec)
    assert hasattr(parsed, "alert_type")
    assert hasattr(parsed, "release_version")
    assert hasattr(parsed, "threshold")
    assert hasattr(parsed, "ground_truth_expected_escalation")


def test_alert_type_mapping_is_total():
    from src.schemas import SCENARIO_TYPES
    for scn in SCENARIO_TYPES:
        assert alert_type_for(scn)  # every scenario type maps to something


def test_release_version_deterministic():
    row = {"quarter": "2022Q3"}
    assert release_version_for(row) == release_version_for(row)
    assert "2022" in release_version_for(row)


def test_build_case_evidence_items_ids_unique_and_stable():
    row, f, checks, chart = _one_case_bundle()
    eval_cfg = load_config("evaluation.yaml")
    items = build_case_evidence_items(row["case_id"], row, f, checks, chart, eval_cfg)
    ids = [i.evidence_id for i in items]
    assert len(ids) == len(set(ids))  # unique
    assert all(i.startswith(row["case_id"]) for i in ids)
    # deterministic: same inputs -> same IDs and same count
    items2 = build_case_evidence_items(row["case_id"], row, f, checks, chart, eval_cfg)
    assert [i.evidence_id for i in items2] == ids


def test_evidence_store_lookup():
    row, f, checks, chart = _one_case_bundle()
    eval_cfg = load_config("evaluation.yaml")
    items = build_case_evidence_items(row["case_id"], row, f, checks, chart, eval_cfg)
    store = EvidenceStore()
    store.register(row["case_id"], items)

    assert store.exists(row["case_id"], items[0].evidence_id)
    assert not store.exists(row["case_id"], "not-a-real-id")
    assert store.get(row["case_id"], items[0].evidence_id) is items[0]
    assert store.get("unknown-case", items[0].evidence_id) is None
    assert set(store.ids_for_case(row["case_id"])) == {i.evidence_id for i in items}


def test_evidence_kinds_present():
    row, f, checks, chart = _one_case_bundle()
    eval_cfg = load_config("evaluation.yaml")
    items = build_case_evidence_items(row["case_id"], row, f, checks, chart, eval_cfg)
    kinds = {i.kind for i in items}
    assert {"metric_snapshot", "historical_baseline", "validation_rule",
            "release_note", "segment_comparison", "seasonality_indicator",
            "recovery_indicator"}.issubset(kinds)


def test_evidence_items_do_not_leak_ground_truth():
    """No evidence item may expose a ground-truth field, scenario_type, or
    alert_type — alert_type is derived 1:1 from scenario_type for several
    scenarios and would leak the answer if shown to the agent."""
    df, _, _ = generate_dataset(n_cases=60, seed=3)
    feats = build_features(df)
    eval_cfg = load_config("evaluation.yaml")
    for row in df.to_dict("records")[:60]:
        cid = row["case_id"]
        series = row["_series"]
        f = feats.loc[cid].to_dict()
        checks = run_checks(row, f, series)
        chart = build_chart_summary(series, row, f)
        items = build_case_evidence_items(cid, row, f, checks, chart, eval_cfg)
        blob = json.dumps([i.to_dict() for i in items])
        assert_no_ground_truth(json.loads(blob))
        for gt_field in GROUND_TRUTH_FIELDS:
            assert f'"{gt_field}"' not in blob
        # alert_type is derived 1:1 from scenario_type for several scenarios
        # (e.g. "segment_deviation_alert" only ever occurs on cases that are
        # always ground_truth_anomaly=True) — it must never appear in
        # agent-facing evidence. Deterministic-check names (e.g.
        # "metric_definition_change") legitimately overlap with scenario-type
        # strings already and are pre-existing, intentionally-shown evidence,
        # so they are not checked here.
        assert alert_type_for(row["scenario_type"]) not in blob
