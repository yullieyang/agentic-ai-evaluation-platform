"""Tests for the experiment runner, agent retry/leakage, config loading, and the
human-review store."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.agent import QAAgent, build_evidence
from src.chart_summaries import build_chart_summary
from src.data_generation import generate_dataset
from src.deterministic_checks import baseline_decision, run_checks
from src.experiment_runner import (build_case_bundle, expand_grid, run_condition)
from src.feature_engineering import build_features
from src.llm_providers import MockProvider
from src.review_store import ReviewStore
from src.schemas import GROUND_TRUTH_FIELDS, assert_no_ground_truth
from src.utils import load_config


@pytest.fixture(scope="module")
def small_bundle():
    df, _, _ = generate_dataset(n_cases=75, seed=21)
    schema_cols = [c for c in df.columns if not c.startswith("_")]
    sidecar = {r["case_id"]: {"_series": r["_series"]} for r in df.to_dict("records")}
    clean = df[schema_cols]
    feats = build_features(clean, sidecar=sidecar)
    bundle = build_case_bundle(clean, feats, sidecar)
    return clean, bundle


def test_expand_grid_cartesian_product():
    grid = {"provider": ["mock"], "model": ["m"], "prompt_version": ["a", "b"],
            "temperature": [0.0], "include_deterministic_evidence": [True, False],
            "reviewer_enabled": [False], "evidence_completeness": ["complete"],
            "scenario_filter": ["all"], "mock_error_rate": [0.1]}
    conditions = expand_grid(grid)
    assert len(conditions) == 2 * 2  # prompt x include_det


def test_no_ground_truth_leakage_in_prompt_evidence(small_bundle):
    clean, bundle = small_bundle
    for cid, b in list(bundle.items())[:25]:
        checks = b["checks"]
        prompt_ev, _ = build_evidence(
            row=b["row"], features=b["features"], checks=checks, chart_summary=b["chart_summary"],
            include_deterministic_evidence=True, baseline_anomaly=b["baseline_anomaly"],
            baseline_severity=b["baseline_severity"], evidence_completeness="complete")
        # Explicit guard plus a string scan.
        assert_no_ground_truth(prompt_ev)
        blob = json.dumps(prompt_ev)
        for field in GROUND_TRUTH_FIELDS:
            assert f'"{field}"' not in blob


def test_run_condition_produces_valid_records(small_bundle):
    clean, bundle = small_bundle
    condition = {"provider": "mock", "model": "mock-deterministic",
                 "prompt_version": "prompt_c_evidence_constrained", "temperature": 0.0,
                 "include_deterministic_evidence": True, "reviewer_enabled": True,
                 "evidence_completeness": "complete", "scenario_filter": "all",
                 "mock_error_rate": 0.1}
    prompts_cfg = load_config("prompts.yaml")
    records, summary = run_condition(condition, clean, bundle, "test__000", prompts_cfg)
    assert len(records) == len(clean)
    assert all(set(["gt_anomaly", "pred_anomaly", "post_anomaly", "schema_valid"]) <= set(r)
               for r in records)
    assert "pre_review" in summary and "post_review" in summary
    assert 0.0 <= summary["pre_review"]["precision"] <= 1.0


def test_agent_retry_recovers_from_malformed(small_bundle):
    clean, bundle = small_bundle
    cid = next(iter(bundle))
    b = bundle[cid]

    class FlakyProvider(MockProvider):
        def complete(self, system, user, **kwargs):
            ctx = dict(kwargs.get("context") or {})
            ctx.setdefault("profile", {})
            ctx["profile"] = dict(ctx["profile"], malformed_rate=1.0)
            kwargs["context"] = ctx
            return super().complete(system, user, **kwargs)

    agent = QAAgent(FlakyProvider(), max_retries=2)
    case = {"row": b["row"], "features": b["features"], "checks": b["checks"],
            "chart_summary": b["chart_summary"], "prompt_version": "prompt_a_zero_shot",
            "include_deterministic_evidence": True, "reviewer_enabled": False,
            "evidence_completeness": "complete", "baseline_anomaly": b["baseline_anomaly"],
            "baseline_severity": b["baseline_severity"], "temperature": 0.0,
            "model": "mock-deterministic"}
    result = agent.run(case)
    # Attempt 0 malforms; later attempts recover -> schema valid with >=1 retry.
    assert result.schema_valid is True
    assert result.total_retries >= 1


def test_config_loading():
    cfg = load_config("experiments.yaml")
    assert "grids" in cfg and "mock_main" in cfg["grids"]
    prompts = load_config("prompts.yaml")
    assert "variants" in prompts and len(prompts["variants"]) >= 4


def test_review_store_roundtrip_and_disagreement(tmp_path):
    store = ReviewStore(tmp_path / "reviews.sqlite")
    store.add_review({"case_id": "c1", "experiment_id": "e1", "reviewer_decision": "approve",
                      "corrected_anomaly_detected": True, "agent_anomaly": True, "rule_anomaly": False})
    store.add_review({"case_id": "c2", "experiment_id": "e1", "reviewer_decision": "reject",
                      "corrected_anomaly_detected": False, "agent_anomaly": True, "rule_anomaly": True,
                      "unsupported_claim_flag": 1})
    rows = store.get_reviews("e1")
    assert len(rows) == 2
    summary = store.disagreement_summary("e1")
    assert summary["n"] == 2
    assert summary["acceptance_rate"] == 0.5
    assert summary["agent_human_disagreement"] == 0.5  # c2: agent True, human False
    store.close()


def test_review_store_rejects_invalid_decision(tmp_path):
    store = ReviewStore(tmp_path / "r.sqlite")
    with pytest.raises(ValueError):
        store.add_review({"case_id": "c", "experiment_id": "e", "reviewer_decision": "bogus"})
    store.close()
