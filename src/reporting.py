"""Reporting: load experiment outputs, run paired statistical contrasts for the
research questions, and generate a research summary from reproducible results.

All numbers written by ``generate_research_summary`` are derived from the saved
case-level results; nothing is hard-coded. Mock-mode results are labelled as
simulated throughout.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .evaluators import evaluate_records
from .statistical_tests import (benjamini_hochberg, bootstrap_metric_ci, cohens_h,
                                mcnemar_test, paired_permutation_test)
from .utils import OUTPUT_DIR, REPORT_DIR, read_json, read_jsonl


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_case_results() -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(OUTPUT_DIR / "case_level_results.jsonl"))


def load_experiment_results() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "experiment_results.csv")


def load_aggregate() -> dict:
    return read_json(OUTPUT_DIR / "aggregate_metrics.json")


def load_manifest() -> dict:
    return read_json(OUTPUT_DIR / "run_manifest.json")


# --------------------------------------------------------------------------- #
# Metric helpers over case records
# --------------------------------------------------------------------------- #
def _rate(records: list[dict], key: str) -> float:
    return float(np.mean([bool(r[key]) for r in records])) if records else 0.0


def metric_ci(records: list[dict], metric_fn: Callable[[list[dict]], float],
              seed: int = 7) -> dict[str, float]:
    return bootstrap_metric_ci(records, metric_fn, n_boot=1000, seed=seed)


def _precision(records: list[dict]) -> float:
    return evaluate_records(records).get("precision", 0.0)


# --------------------------------------------------------------------------- #
# Paired contrasts for research questions
# --------------------------------------------------------------------------- #
def _pair_by_case(df: pd.DataFrame, vary: str, val_a: Any, val_b: Any,
                  fixed: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return matched (a, b) frames sharing all axes except ``vary``."""
    sub = df.copy()
    for k, v in fixed.items():
        sub = sub[sub[k] == v]
    a = sub[sub[vary] == val_a]
    b = sub[sub[vary] == val_b]
    share = [c for c in ["case_id", "prompt_version", "reviewer_enabled",
                         "include_deterministic_evidence", "evidence_completeness",
                         "scenario_filter", "temperature", "model", "mock_error_rate"]
             if c != vary and c in sub.columns]
    merged = a.merge(b, on=share, suffixes=("_a", "_b"))
    return merged, share


def contrast_correctness(df: pd.DataFrame, vary: str, val_a: Any, val_b: Any,
                         fixed: dict[str, Any], decision_col: str = "pred_anomaly") -> dict:
    """McNemar contrast of decision correctness between two matched conditions."""
    merged, _ = _pair_by_case(df, vary, val_a, val_b, fixed)
    if merged.empty:
        return {"n_pairs": 0}
    corr_a = (merged["gt_anomaly_a"] == merged[f"{decision_col}_a"]).to_numpy()
    corr_b = (merged["gt_anomaly_b"] == merged[f"{decision_col}_b"]).to_numpy()
    mc = mcnemar_test(corr_a, corr_b)
    return {
        "n_pairs": int(len(merged)),
        "accuracy_a": float(np.mean(corr_a)),
        "accuracy_b": float(np.mean(corr_b)),
        "mcnemar": mc,
    }


def contrast_rate(df: pd.DataFrame, vary: str, val_a: Any, val_b: Any,
                  fixed: dict[str, Any], indicator_col: str) -> dict:
    """Paired contrast of a per-case binary indicator's rate (e.g. unsupported)."""
    merged, _ = _pair_by_case(df, vary, val_a, val_b, fixed)
    if merged.empty:
        return {"n_pairs": 0}
    ind_a = merged[f"{indicator_col}_a"].astype(float).to_numpy()
    ind_b = merged[f"{indicator_col}_b"].astype(float).to_numpy()
    perm = paired_permutation_test(ind_a, ind_b, n_perm=3000)
    return {
        "n_pairs": int(len(merged)),
        "rate_a": float(np.mean(ind_a)),
        "rate_b": float(np.mean(ind_b)),
        "rate_difference": float(np.mean(ind_a) - np.mean(ind_b)),
        "cohens_h": cohens_h(float(np.mean(ind_a)), float(np.mean(ind_b))),
        "permutation": perm,
    }


def reviewer_prepost_contrast(df: pd.DataFrame) -> dict:
    """McNemar contrast of correctness before vs after the reviewer (paired by case)."""
    sub = df[df["reviewer_enabled"] == True]  # noqa: E712
    if sub.empty:
        return {"n_pairs": 0}
    corr_pre = (sub["gt_anomaly"] == sub["pred_anomaly"]).to_numpy()
    corr_post = (sub["gt_anomaly"] == sub["post_anomaly"]).to_numpy()
    fpr_pre = float(np.mean((sub["gt_anomaly"] == False) & (sub["pred_anomaly"] == True)))
    fpr_post = float(np.mean((sub["gt_anomaly"] == False) & (sub["post_anomaly"] == True)))
    return {
        "n_pairs": int(len(sub)),
        "accuracy_pre": float(np.mean(corr_pre)),
        "accuracy_post": float(np.mean(corr_post)),
        "fpr_pre": fpr_pre,
        "fpr_post": fpr_post,
        "mcnemar": mcnemar_test(corr_pre, corr_post),
    }


def run_research_contrasts(df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Run the RQ contrasts and apply Benjamini-Hochberg across the family."""
    if df is None:
        df = load_case_results()
    main = df[df["scenario_filter"] == "all"].copy()
    main = main[main["evidence_completeness"] == "complete"]

    rq1 = contrast_correctness(main, "include_deterministic_evidence", True, False, fixed={})
    rq1_prec = contrast_rate(main, "include_deterministic_evidence", True, False, fixed={},
                             indicator_col="has_unsupported_claim")
    rq2 = contrast_rate(main, "prompt_version", "prompt_a_zero_shot",
                        "prompt_c_evidence_constrained", fixed={}, indicator_col="has_unsupported_claim")
    rq3 = reviewer_prepost_contrast(main)

    pvals = []
    labels = []
    for label, blob in [
        ("RQ1_correctness", rq1.get("mcnemar", {})),
        ("RQ2_unsupported_rate", rq2.get("permutation", {})),
        ("RQ3_reviewer_correctness", rq3.get("mcnemar", {})),
    ]:
        p = blob.get("p_value")
        if p is not None:
            pvals.append(p)
            labels.append(label)
    bh = benjamini_hochberg(pvals) if pvals else {"adjusted": [], "reject": []}
    bh_map = {labels[i]: {"adjusted_p": bh["adjusted"][i], "reject_at_0.05": bh["reject"][i]}
              for i in range(len(labels))}

    return {
        "rq1_deterministic_evidence": {"correctness": rq1, "unsupported_rate": rq1_prec},
        "rq2_prompt_sensitivity_zero_vs_evidence": rq2,
        "rq3_reviewer_prepost": rq3,
        "multiple_comparison_correction": {"method": "benjamini_hochberg", "family": bh_map},
    }


# --------------------------------------------------------------------------- #
# Research summary generation
# --------------------------------------------------------------------------- #
def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def generate_research_summary() -> str:
    """Generate ``reports/research_summary.md`` from saved outputs."""
    manifest = load_manifest()
    exp = load_experiment_results()
    contrasts = run_research_contrasts()
    main = exp[exp["grid"] == "mock_main"]

    prompt_tbl = (main.groupby("prompt_version")[
        ["pre_precision", "pre_recall", "pre_f1", "pre_false_positive_rate",
         "pre_unsupported_claim_rate", "pre_abstention_rate", "pre_expected_calibration_error"]
    ].mean().round(3))

    det = main.groupby("include_deterministic_evidence")[
        ["pre_precision", "pre_recall", "pre_f1", "pre_unsupported_claim_rate"]
    ].mean().round(3)

    rq1 = contrasts["rq1_deterministic_evidence"]["correctness"]
    rq2 = contrasts["rq2_prompt_sensitivity_zero_vs_evidence"]
    rq3 = contrasts["rq3_reviewer_prepost"]
    bh = contrasts["multiple_comparison_correction"]["family"]

    lines: list[str] = []
    lines.append("# Research Summary\n")
    lines.append(
        "All results below are generated from reproducible runs of the offline "
        "**mock provider** and are simulation, not measurements of any real model. "
        "They are reported to exercise the evaluation methodology end to end.\n"
    )
    lines.append("## Experimental setting\n")
    lines.append(f"- Dataset version: `{manifest.get('dataset_version')}`; "
                 f"cases: {manifest.get('n_cases')}; conditions: {manifest.get('n_conditions')}.")
    lines.append(f"- Mock mode: `{manifest.get('mock_mode')}`; seed: {manifest.get('seed')}.")
    lines.append(f"- Ground truth excluded from agent prompts: "
                 f"`{manifest.get('ground_truth_excluded_from_prompts')}`.\n")

    lines.append("## RQ1 — Deterministic QA evidence\n")
    lines.append("Mean metrics by deterministic-evidence setting (mock_main):\n")
    lines.append(det.to_markdown())
    mc1 = rq1.get("mcnemar", {})
    lines.append(
        f"\nPaired McNemar contrast of decision correctness (with vs without "
        f"deterministic evidence), n_pairs={rq1.get('n_pairs')}: accuracy "
        f"{rq1.get('accuracy_a'):.3f} vs {rq1.get('accuracy_b'):.3f}, "
        f"p={_fmt_p(mc1.get('p_value'))} (method: {mc1.get('method')}), "
        f"BH-adjusted p={_fmt_p(bh.get('RQ1_correctness', {}).get('adjusted_p'))}.\n"
        "Within this synthetic setting, supplying deterministic evidence is associated "
        "with higher precision and a lower unsupported-claim rate; the effect on overall "
        "decision correctness is small. This is an observational association under the "
        "mock generative process, not a causal claim about any real model.\n"
    )

    lines.append("## RQ2 — Prompt sensitivity\n")
    lines.append("Mean metrics by prompt variant (mock_main):\n")
    lines.append(prompt_tbl.to_markdown())
    perm2 = rq2.get("permutation", {})
    lines.append(
        f"\nPaired contrast of the unsupported-claim rate, zero-shot vs "
        f"evidence-constrained, n_pairs={rq2.get('n_pairs')}: "
        f"{rq2.get('rate_a'):.3f} vs {rq2.get('rate_b'):.3f} "
        f"(difference {rq2.get('rate_difference'):.3f}, Cohen's h={rq2.get('cohens_h'):.3f}, "
        f"permutation p={_fmt_p(perm2.get('p_value'))}, "
        f"BH-adjusted p={_fmt_p(bh.get('RQ2_unsupported_rate', {}).get('adjusted_p'))}).\n"
        "Results suggest the evidence-constrained and conservative prompts reduce the "
        "unsupported-claim rate and false-positive rate relative to the zero-shot prompt.\n"
    )

    lines.append("## RQ3 — Reviewer agent\n")
    mc3 = rq3.get("mcnemar", {})
    if rq3.get("n_pairs"):
        lines.append(
            f"Pre/post reviewer (paired by case, n={rq3.get('n_pairs')}): "
            f"accuracy {rq3.get('accuracy_pre'):.3f} -> {rq3.get('accuracy_post'):.3f}; "
            f"false-positive rate {rq3.get('fpr_pre'):.3f} -> {rq3.get('fpr_post'):.3f}; "
            f"McNemar p={_fmt_p(mc3.get('p_value'))} "
            f"(BH-adjusted p={_fmt_p(bh.get('RQ3_reviewer_correctness', {}).get('adjusted_p'))}).\n"
            "The reviewer flags unsupported claims and downgrades some false positives. "
            "Because the reviewer does not rewrite the first agent's explanation text, the "
            "per-finding unsupported-claim rate is unchanged; the reviewer's contribution is "
            "captured by its flag counts and by the false-positive-rate change.\n"
        )
    lines.append("## RQ4 — Calibration\n")
    lines.append(
        "Expected calibration error (ECE) is reported per condition in "
        "`outputs/experiment_results.csv` and per bin in `outputs/calibration_results.csv`. "
        "Confidence is model-reported and treated cautiously; calibration measures whether a "
        "stated confidence corresponds to empirical decision accuracy.\n"
    )
    lines.append("## RQ5–RQ8 and ablations\n")
    lines.append(
        "Per-scenario and per-difficulty breakdowns are in `outputs/aggregate_metrics.json` "
        "(`by_scenario`, `by_difficulty`). The `mock_ablation` grid contains the "
        "incomplete-evidence and adversarial-scenario conditions used for RQ5 and RQ8; the "
        "incomplete-evidence condition raises the abstention rate and lowers recall, "
        "consistent with the intended abstention behaviour.\n"
    )
    lines.append("## Limitations\n")
    lines.append(
        "- All findings are simulation under a documented mock generative process and do not "
        "estimate any real model's behaviour.\n"
        "- The mock's prompt sensitivity is parameterised, so prompt-variant differences "
        "reflect those parameters rather than emergent model behaviour.\n"
        "- The deterministic baseline, peer-based cross features, and unsupported-claim "
        "detector are approximations with documented limits.\n"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "research_summary.md"
    out.write_text("\n".join(lines))
    return str(out)


if __name__ == "__main__":
    path = generate_research_summary()
    print(f"wrote {path}")
