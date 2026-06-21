"""Experiment runner.

Expands an experiment grid into conditions, runs each condition over the
synthetic dataset in (by default) mock mode, evaluates pre- and post-review
outputs against ground truth, and writes reproducible outputs. Every condition
records its configuration, seed, git commit, environment metadata, and resource
usage so it can be reproduced from disk.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
from typing import Any, Iterable

import pandas as pd

from .agent import QAAgent
from .calibration import reliability_curve
from .chart_summaries import build_chart_summary
from .deterministic_checks import baseline_decision, run_checks
from .evaluators import classification_metrics, detect_unsupported_claim, evaluate_records, group_metrics
from .feature_engineering import build_features
from .llm_providers import get_provider
from .reviewer_agent import ReviewerAgent
from .schemas import GROUND_TRUTH_FIELDS
from .utils import (DATA_DIR, OUTPUT_DIR, append_jsonl, environment_metadata, get_logger,
                    load_config, read_jsonl, write_json)

LOGGER = get_logger("experiment_runner")

GRID_AXES = [
    "provider", "model", "prompt_version", "temperature",
    "include_deterministic_evidence", "reviewer_enabled",
    "evidence_completeness", "scenario_filter", "mock_error_rate",
]


# --------------------------------------------------------------------------- #
# Loading and per-case precomputation
# --------------------------------------------------------------------------- #
def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(DATA_DIR / "synthetic_monitoring_data.csv")
    feats = build_features(df)
    sidecar = {r["case_id"]: r for r in json.loads((DATA_DIR / "_series_sidecar.json").read_text())}
    return df, feats, sidecar


def build_case_bundle(df: pd.DataFrame, feats: pd.DataFrame, sidecar: dict,
                      eval_cfg: dict | None = None) -> dict[str, dict]:
    """Precompute checks, chart summaries, and the deterministic baseline once."""
    bundle: dict[str, dict] = {}
    for row in df.to_dict("records"):
        cid = row["case_id"]
        series = sidecar[cid]["_series"]
        f = feats.loc[cid].to_dict()
        checks = run_checks(row, f, series, eval_cfg)
        base_anom, base_sev = baseline_decision(checks)
        chart = build_chart_summary(series, row, f)
        bundle[cid] = {
            "row": row, "features": f, "checks": checks, "chart_summary": chart,
            "baseline_anomaly": base_anom, "baseline_severity": base_sev,
        }
    return bundle


# --------------------------------------------------------------------------- #
# Grid expansion
# --------------------------------------------------------------------------- #
def expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product over the grid axes into a list of condition dicts."""
    axes = {ax: grid[ax] for ax in GRID_AXES if ax in grid}
    keys = list(axes)
    conditions = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        conditions.append(dict(zip(keys, combo)))
    return conditions


def _filter_cases(df: pd.DataFrame, scenario_filter: str) -> list[str]:
    if scenario_filter == "adversarial":
        mask = df["scenario_difficulty"] == "adversarial"
    elif scenario_filter == "traps":
        mask = df["scenario_type"].isin(["false_positive_trap", "false_negative_trap"])
    else:
        mask = pd.Series(True, index=df.index)
    return df.loc[mask, "case_id"].tolist()


# --------------------------------------------------------------------------- #
# Per-case evaluation record
# --------------------------------------------------------------------------- #
def _eval_record(row: dict, agent_res, review_res, condition: dict, experiment_id: str) -> dict:
    finding = agent_res.finding or {}
    pred_anom = bool(finding.get("anomaly_detected", False)) if agent_res.schema_valid else False
    pred_sev = finding.get("severity", "none") if agent_res.schema_valid else "none"
    abstained = bool(finding.get("abstained", False)) if agent_res.schema_valid else False
    confidence = float(finding.get("confidence_score", 0.5)) if agent_res.schema_valid else 0.5
    has_unsupported = detect_unsupported_claim(finding) if agent_res.schema_valid else False

    if review_res is not None and review_res.schema_valid:
        rv = review_res.review
        post_anom = bool(rv["revised_anomaly_detected"])
        post_sev = rv["revised_severity"]
        post_conf = float(rv["revised_confidence"])
        reviewer_unsupported = len(rv.get("unsupported_claims_found", []))
        reviewer_decision = rv["review_decision"]
    else:
        post_anom, post_sev, post_conf = pred_anom, pred_sev, confidence
        reviewer_unsupported = 0
        reviewer_decision = None

    return {
        "experiment_id": experiment_id,
        **{ax: condition.get(ax) for ax in GRID_AXES},
        "case_id": row["case_id"],
        "model_id": row["model_id"],
        "metric_name": row["metric_name"],
        "segment": row["segment"],
        "scenario_type": row["scenario_type"],
        "scenario_difficulty": row["scenario_difficulty"],
        "gt_anomaly": bool(row["ground_truth_anomaly"]),
        "gt_severity": row["ground_truth_severity"],
        "gt_anomaly_type": row["ground_truth_anomaly_type"],
        "baseline_anomaly": bool(agent_res.extra.get("baseline_anomaly", False)),
        "pred_anomaly": pred_anom,
        "pred_severity": pred_sev,
        "pred_anomaly_type": finding.get("anomaly_type", "none") if agent_res.schema_valid else "none",
        "abstained": abstained,
        "confidence": confidence,
        "evidence_sufficiency": finding.get("evidence_sufficiency", "insufficient"),
        "unsupported_claim_risk": finding.get("unsupported_claim_risk", "low"),
        "requires_human_review": bool(finding.get("requires_human_review", True)) if agent_res.schema_valid else True,
        "n_supporting_evidence": len(finding.get("supporting_evidence", []) or []),
        "n_followup": len(finding.get("recommended_follow_up", []) or []),
        "has_unsupported_claim": bool(has_unsupported),
        "mock_unsupported_injected": agent_res.mock_unsupported_injected,
        "schema_valid": bool(agent_res.schema_valid),
        "validation_error": agent_res.validation_error,
        "post_anomaly": post_anom,
        "post_severity": post_sev,
        "post_confidence": post_conf,
        "reviewer_decision": reviewer_decision,
        "reviewer_unsupported_found": reviewer_unsupported,
        "latency_s": agent_res.response.latency_s + (review_res.response.latency_s if review_res else 0.0),
        "input_tokens": agent_res.response.input_tokens + (review_res.response.input_tokens if review_res else 0),
        "output_tokens": agent_res.response.output_tokens + (review_res.response.output_tokens if review_res else 0),
        "estimated_cost": (agent_res.response.estimated_cost or 0.0)
        + ((review_res.response.estimated_cost or 0.0) if review_res else 0.0),
        "total_retries": agent_res.total_retries,
        "is_mock": agent_res.response.is_mock,
    }


# --------------------------------------------------------------------------- #
# Run a single condition
# --------------------------------------------------------------------------- #
def run_condition(condition: dict, df: pd.DataFrame, bundle: dict, experiment_id: str,
                  prompts_cfg: dict) -> tuple[list[dict], dict]:
    provider_kwargs = {}
    if condition["provider"] == "mock":
        provider_kwargs["error_rate"] = float(condition.get("mock_error_rate", 0.12))
    provider = get_provider(condition["provider"], **provider_kwargs)

    agent = QAAgent(provider, prompts_cfg=prompts_cfg, max_retries=2)
    reviewer = ReviewerAgent(provider, prompts_cfg=prompts_cfg) if condition["reviewer_enabled"] else None

    case_ids = _filter_cases(df, condition.get("scenario_filter", "all"))
    records: list[dict] = []
    for cid in case_ids:
        b = bundle[cid]
        case = {
            "row": b["row"], "features": b["features"], "checks": b["checks"],
            "chart_summary": b["chart_summary"],
            "prompt_version": condition["prompt_version"],
            "include_deterministic_evidence": bool(condition["include_deterministic_evidence"]),
            "reviewer_enabled": bool(condition["reviewer_enabled"]),
            "evidence_completeness": condition.get("evidence_completeness", "complete"),
            "baseline_anomaly": b["baseline_anomaly"], "baseline_severity": b["baseline_severity"],
            "temperature": float(condition.get("temperature", 0.0)), "model": condition["model"],
        }
        agent_res = agent.run(case)
        agent_res.extra["baseline_anomaly"] = b["baseline_anomaly"]
        review_res = None
        if reviewer is not None and agent_res.schema_valid:
            review_res = reviewer.run(
                case, agent_finding=agent_res.finding,
                prompt_evidence=agent_res.extra["prompt_evidence"],
                mock_evidence=agent_res.extra["mock_evidence"],
            )
        records.append(_eval_record(b["row"], agent_res, review_res, condition, experiment_id))

    pre = evaluate_records(records, decision_key="pred_anomaly", severity_key="pred_severity")
    baseline = classification_metrics([r["gt_anomaly"] for r in records],
                                      [r["baseline_anomaly"] for r in records])
    summary = {"config": condition, "experiment_id": experiment_id, "pre_review": pre,
               "baseline": baseline,
               "by_scenario": group_metrics(records, "scenario_type"),
               "by_difficulty": group_metrics(records, "scenario_difficulty")}
    if condition["reviewer_enabled"]:
        summary["post_review"] = evaluate_records(
            records, decision_key="post_anomaly", severity_key="post_severity",
            confidence_key="post_confidence")
    return records, summary


# --------------------------------------------------------------------------- #
# Run a full grid
# --------------------------------------------------------------------------- #
def _summary_to_csv_row(summary: dict, grid_name: str, timestamp: str, seed: int) -> dict:
    condition = summary["config"]
    pre = summary["pre_review"]
    csv_row = {"experiment_id": summary["experiment_id"], "grid": grid_name,
               "timestamp": timestamp, "seed": seed, **condition}
    for key in ["n", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
                "balanced_accuracy", "severity_accuracy", "abstention_rate", "selective_accuracy",
                "schema_compliance_rate", "unsupported_claim_rate", "expected_calibration_error",
                "brier_score", "average_confidence", "avg_latency_s", "avg_estimated_cost",
                "total_estimated_cost", "avg_retries"]:
        csv_row[f"pre_{key}"] = pre.get(key)
    csv_row["baseline_f1"] = summary["baseline"].get("f1")
    csv_row["baseline_precision"] = summary["baseline"].get("precision")
    csv_row["baseline_recall"] = summary["baseline"].get("recall")
    if "post_review" in summary:
        post = summary["post_review"]
        for key in ["precision", "recall", "f1", "false_positive_rate", "unsupported_claim_rate",
                    "expected_calibration_error", "avg_latency_s", "avg_estimated_cost"]:
            csv_row[f"post_{key}"] = post.get(key)
    return csv_row


def run_experiments(grid_names: Iterable[str], seed: int | None = None) -> dict[str, Any]:
    """Run one or more named grids and persist a single combined output set."""
    exp_cfg = load_config("experiments.yaml")
    prompts_cfg = load_config("prompts.yaml")
    eval_cfg = load_config("evaluation.yaml")
    seed = seed if seed is not None else exp_cfg.get("default_seed", 13)
    n_bins = eval_cfg["calibration"]["n_bins"]

    df, feats, sidecar = load_dataset()
    bundle = build_case_bundle(df, feats, sidecar, eval_cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "case_level_results.jsonl"
    jsonl_path.write_text("")

    aggregate: dict[str, Any] = {}
    rows_for_csv: list[dict] = []
    calib_rows: list[dict] = []
    timestamp = _dt.datetime.utcnow().isoformat() + "Z"
    grid_names = list(grid_names)
    all_mock = True

    for grid_name in grid_names:
        conditions = expand_grid(exp_cfg["grids"][grid_name])
        LOGGER.info("running grid '%s' with %d conditions over %d cases",
                    grid_name, len(conditions), len(df))
        for i, condition in enumerate(conditions):
            all_mock = all_mock and condition["provider"] == "mock"
            experiment_id = f"{grid_name}__{i:03d}"
            records, summary = run_condition(condition, df, bundle, experiment_id, prompts_cfg)
            for rec in records:
                append_jsonl(rec, jsonl_path)
            aggregate[experiment_id] = summary
            rows_for_csv.append(_summary_to_csv_row(summary, grid_name, timestamp, seed))

            conf = [r["confidence"] for r in records]
            corr = [bool(r["gt_anomaly"]) == bool(r["pred_anomaly"]) for r in records]
            curve = reliability_curve(conf, corr, n_bins)
            for c_center, c_conf, c_acc, c_count in zip(
                curve["bin_centers"], curve["bin_confidence"], curve["bin_accuracy"], curve["bin_count"]
            ):
                calib_rows.append({"experiment_id": experiment_id,
                                   "prompt_version": condition["prompt_version"],
                                   "bin_center": c_center, "bin_confidence": c_conf,
                                   "bin_accuracy": c_acc, "bin_count": c_count})

    pd.DataFrame(rows_for_csv).to_csv(OUTPUT_DIR / "experiment_results.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(OUTPUT_DIR / "calibration_results.csv", index=False)
    write_json(aggregate, OUTPUT_DIR / "aggregate_metrics.json")
    _write_sample_outputs(jsonl_path)

    manifest = {
        "grids": grid_names, "timestamp": timestamp, "seed": seed,
        "n_conditions": len(rows_for_csv), "n_cases": int(len(df)),
        "dataset_version": exp_cfg.get("dataset_version"),
        "mock_mode": all_mock,
        "ground_truth_excluded_from_prompts": True,
        "ground_truth_fields": GROUND_TRUTH_FIELDS,
        "environment": environment_metadata(),
    }
    write_json(manifest, OUTPUT_DIR / "run_manifest.json")
    LOGGER.info("completed %d grid(s) -> outputs/", len(grid_names))
    return {"manifest": manifest, "aggregate": aggregate}


def run_grid(grid_name: str, seed: int | None = None) -> dict[str, Any]:
    """Run a single named grid and persist outputs."""
    return run_experiments([grid_name], seed=seed)


def _write_sample_outputs(jsonl_path) -> None:
    records = read_jsonl(jsonl_path)
    if not records:
        return
    first_exp = records[0]["experiment_id"]
    seen: dict[str, dict] = {}
    for r in records:
        if r["experiment_id"] != first_exp:
            continue
        seen.setdefault(r["scenario_type"], r)
    write_json(
        {"note": "MOCK-MODE SIMULATED OUTPUTS — not measurements of any real model.",
         "experiment_id": first_exp, "samples": list(seen.values())},
        OUTPUT_DIR / "sample_agent_outputs.json",
    )


def main(grids: Iterable[str] = ("mock_main", "mock_ablation")) -> None:
    run_experiments(list(grids))


if __name__ == "__main__":
    main()
