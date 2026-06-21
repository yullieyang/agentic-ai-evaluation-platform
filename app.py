"""Streamlit research dashboard for inspecting evaluation results.

The dashboard is for research inspection. It reads the reproducible outputs in
``outputs/`` and presents study-level metrics, per-case review, experiment
comparisons, failure analysis, calibration, human review, and reproducibility
metadata. Mock-mode results are labelled as simulation throughout.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.review_store import ReviewStore
from src.utils import DATA_DIR, OUTPUT_DIR, read_json, read_jsonl

st.set_page_config(page_title="Agentic AI Evaluation and Monitoring Platform", layout="wide")

RESEARCH_QUESTIONS = [
    "RQ1 Does deterministic QA evidence improve precision, recall, and grounding?",
    "RQ2 How do prompt variants affect detection, unsupported claims, and consistency?",
    "RQ3 Does a reviewer agent reduce false positives and unsupported claims?",
    "RQ4 How well calibrated are the agent's confidence scores?",
    "RQ5 Which scenarios produce the highest false-positive/negative rates?",
    "RQ6 How often do human reviewers disagree with rules and agents?",
    "RQ7 What are the quality/latency/token/cost tradeoffs?",
    "RQ8 How does missing/contradictory evidence affect reliability and abstention?",
]


@st.cache_data
def load_outputs():
    exp = pd.read_csv(OUTPUT_DIR / "experiment_results.csv") if (OUTPUT_DIR / "experiment_results.csv").exists() else pd.DataFrame()
    cases = pd.DataFrame(read_jsonl(OUTPUT_DIR / "case_level_results.jsonl")) if (OUTPUT_DIR / "case_level_results.jsonl").exists() else pd.DataFrame()
    calib = pd.read_csv(OUTPUT_DIR / "calibration_results.csv") if (OUTPUT_DIR / "calibration_results.csv").exists() else pd.DataFrame()
    agg = read_json(OUTPUT_DIR / "aggregate_metrics.json") if (OUTPUT_DIR / "aggregate_metrics.json").exists() else {}
    manifest = read_json(OUTPUT_DIR / "run_manifest.json") if (OUTPUT_DIR / "run_manifest.json").exists() else {}
    return exp, cases, calib, agg, manifest


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_DIR / "synthetic_monitoring_data.csv") if (DATA_DIR / "synthetic_monitoring_data.csv").exists() else pd.DataFrame()
    sidecar_path = DATA_DIR / "_series_sidecar.json"
    sidecar = {r["case_id"]: r for r in json.loads(sidecar_path.read_text())} if sidecar_path.exists() else {}
    return df, sidecar


def _mock_banner(manifest):
    if manifest.get("mock_mode", True):
        st.warning("Results shown are from the offline **mock provider** — simulation, "
                   "not measurements of any real model.")


def page_overview(exp, cases, manifest):
    st.title("Study Overview")
    _mock_banner(manifest)
    st.subheader("Research questions")
    for rq in RESEARCH_QUESTIONS:
        st.markdown(f"- {rq}")
    if exp.empty:
        st.info("No experiment outputs found. Run `python -m src.experiment_runner`.")
        return
    st.subheader("Active experiment configuration")
    exp_id = st.selectbox("Experiment", exp["experiment_id"].tolist())
    row = exp[exp["experiment_id"] == exp_id].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Cases", int(row.get("pre_n", 0)))
    cols[1].metric("Precision", f"{row.get('pre_precision', float('nan')):.3f}")
    cols[2].metric("Recall", f"{row.get('pre_recall', float('nan')):.3f}")
    cols[3].metric("F1", f"{row.get('pre_f1', float('nan')):.3f}")
    cols = st.columns(4)
    cols[0].metric("False-positive rate", f"{row.get('pre_false_positive_rate', float('nan')):.3f}")
    cols[1].metric("False-negative rate", f"{row.get('pre_false_negative_rate', float('nan')):.3f}")
    cols[2].metric("Unsupported-claim rate", f"{row.get('pre_unsupported_claim_rate', float('nan')):.3f}")
    cols[3].metric("Schema compliance", f"{row.get('pre_schema_compliance_rate', float('nan')):.3f}")
    cols = st.columns(4)
    cols[0].metric("Calibration error (ECE)", f"{row.get('pre_expected_calibration_error', float('nan')):.3f}")
    cols[1].metric("Abstention rate", f"{row.get('pre_abstention_rate', float('nan')):.3f}")
    cols[2].metric("Avg latency (s)", f"{row.get('pre_avg_latency_s', float('nan')):.3f}")
    cols[3].metric("Total est. cost ($)", f"{row.get('pre_total_estimated_cost', 0.0):.4f}")
    st.caption(f"Baseline (rule) F1 for the same cases: {row.get('baseline_f1', float('nan')):.3f}")


def page_case_review(exp, cases, df, sidecar, agg, manifest):
    st.title("Case Review")
    _mock_banner(manifest)
    if cases.empty or df.empty:
        st.info("No case-level outputs found.")
        return
    exp_id = st.selectbox("Experiment", sorted(cases["experiment_id"].unique()))
    sub = cases[cases["experiment_id"] == exp_id]
    c1, c2, c3, c4 = st.columns(4)
    model = c1.selectbox("Model", ["(all)"] + sorted(sub["model_id"].unique()))
    metric = c2.selectbox("Metric", ["(all)"] + sorted(sub["metric_name"].unique()))
    scenario = c3.selectbox("Scenario", ["(all)"] + sorted(sub["scenario_type"].unique()))
    research_mode = c4.checkbox("Research mode (reveal ground truth)", value=False)
    f = sub.copy()
    if model != "(all)":
        f = f[f["model_id"] == model]
    if metric != "(all)":
        f = f[f["metric_name"] == metric]
    if scenario != "(all)":
        f = f[f["scenario_type"] == scenario]
    if f.empty:
        st.info("No cases match the filters.")
        return
    case_id = st.selectbox("Case", f["case_id"].tolist())
    rec = f[f["case_id"] == case_id].iloc[0].to_dict()

    # Metric history chart.
    series = sidecar.get(case_id, {}).get("_series", [])
    if series:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series, mode="lines+markers", name="metric"))
        fig.update_layout(title="Quarterly metric history", height=300)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Evidence available to the agent")
        drow = df[df["case_id"] == case_id].iloc[0].to_dict()
        st.json({k: drow[k] for k in ["current_value", "previous_value", "historical_mean",
                                      "historical_std", "percent_change", "z_score",
                                      "sample_size", "missing_rate", "expected_direction"]})
        st.subheader("Agent finding")
        st.json({k: rec[k] for k in ["pred_anomaly", "pred_severity", "abstained", "confidence",
                                     "evidence_sufficiency", "unsupported_claim_risk",
                                     "requires_human_review", "has_unsupported_claim"]})
        st.subheader("Reviewer-agent output")
        st.json({k: rec.get(k) for k in ["reviewer_decision", "post_anomaly", "post_severity",
                                         "post_confidence", "reviewer_unsupported_found"]})
    with right:
        st.subheader("Deterministic baseline")
        st.json({"baseline_anomaly": rec["baseline_anomaly"]})
        if research_mode:
            st.subheader("⚠ Ground truth (research mode only — never shown to the agent)")
            st.json({k: rec[k] for k in ["gt_anomaly", "gt_severity", "gt_anomaly_type"]})
        else:
            st.info("Enable research mode to reveal ground-truth labels. Ground truth is "
                    "never part of the agent's evidence.")

    st.subheader("Human review")
    decision = st.radio("Decision", ["approve", "reject", "revise", "uncertain"], horizontal=True)
    notes = st.text_area("Reviewer notes")
    if st.button("Save review"):
        store = ReviewStore()
        store.add_review({
            "case_id": case_id, "experiment_id": exp_id, "reviewer_decision": decision,
            "corrected_anomaly_detected": rec["pred_anomaly"], "reviewer_notes": notes,
            "agent_anomaly": rec["pred_anomaly"], "rule_anomaly": rec["baseline_anomaly"],
            "unsupported_claim_flag": int(bool(rec.get("has_unsupported_claim"))),
        })
        store.close()
        st.success("Saved review.")


def page_comparison(exp, manifest):
    st.title("Experiment Comparison")
    _mock_banner(manifest)
    if exp.empty:
        st.info("No experiment outputs found.")
        return
    main = exp[exp["grid"] == "mock_main"] if "grid" in exp.columns else exp
    metric = st.selectbox("Metric", ["pre_precision", "pre_recall", "pre_f1",
                                     "pre_false_positive_rate", "pre_unsupported_claim_rate",
                                     "pre_expected_calibration_error", "pre_avg_latency_s",
                                     "pre_avg_estimated_cost"])
    by = st.selectbox("Group by", ["prompt_version", "include_deterministic_evidence",
                                   "reviewer_enabled", "evidence_completeness"])
    agg = main.groupby(by)[metric].mean().reset_index()
    st.plotly_chart(px.bar(agg, x=by, y=metric, title=f"{metric} by {by}"), use_container_width=True)
    st.dataframe(main[["experiment_id", "prompt_version", "include_deterministic_evidence",
                       "reviewer_enabled", metric]].round(3))


def page_failure(cases, manifest):
    st.title("Failure Analysis")
    _mock_banner(manifest)
    if cases.empty:
        st.info("No case-level outputs found.")
        return
    exp_id = st.selectbox("Experiment", sorted(cases["experiment_id"].unique()))
    sub = cases[cases["experiment_id"] == exp_id].copy()
    sub["false_positive"] = (~sub["gt_anomaly"]) & sub["pred_anomaly"]
    sub["false_negative"] = sub["gt_anomaly"] & (~sub["pred_anomaly"])
    sub["high_conf_error"] = (sub["confidence"] > 0.7) & (sub["gt_anomaly"] != sub["pred_anomaly"])
    counts = {
        "false_positives": int(sub["false_positive"].sum()),
        "false_negatives": int(sub["false_negative"].sum()),
        "schema_failures": int((~sub["schema_valid"]).sum()),
        "unsupported_claims": int(sub["has_unsupported_claim"].sum()),
        "high_confidence_errors": int(sub["high_conf_error"].sum()),
        "rule_agent_disagreements": int((sub["baseline_anomaly"] != sub["pred_anomaly"]).sum()),
    }
    st.json(counts)
    fp_by_scn = sub[sub["false_positive"]].groupby("scenario_type").size().reset_index(name="false_positives")
    if not fp_by_scn.empty:
        st.plotly_chart(px.bar(fp_by_scn, x="scenario_type", y="false_positives",
                               title="False positives by scenario"), use_container_width=True)
    st.subheader("Example failing cases")
    st.dataframe(sub[sub["false_positive"] | sub["false_negative"]][
        ["case_id", "scenario_type", "scenario_difficulty", "gt_anomaly", "pred_anomaly",
         "confidence", "has_unsupported_claim"]].head(50))


def page_calibration(exp, calib, manifest):
    st.title("Calibration")
    _mock_banner(manifest)
    if calib.empty:
        st.info("No calibration outputs found.")
        return
    exp_id = st.selectbox("Experiment", sorted(calib["experiment_id"].unique()))
    sub = calib[calib["experiment_id"] == exp_id].dropna(subset=["bin_accuracy"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=sub["bin_confidence"], y=sub["bin_accuracy"], mode="markers+lines",
                             name="observed", marker=dict(size=sub["bin_count"] / max(sub["bin_count"].max(), 1) * 20 + 5)))
    fig.update_layout(title="Reliability diagram (confidence vs accuracy)",
                      xaxis_title="confidence", yaxis_title="accuracy", height=420)
    st.plotly_chart(fig, use_container_width=True)
    if not exp.empty:
        row = exp[exp["experiment_id"] == exp_id]
        if not row.empty:
            st.metric("Expected calibration error",
                      f"{row.iloc[0].get('pre_expected_calibration_error', float('nan')):.3f}")
            st.metric("Brier score", f"{row.iloc[0].get('pre_brier_score', float('nan')):.3f}")


def page_human_review(manifest):
    st.title("Human Review")
    _mock_banner(manifest)
    store = ReviewStore()
    summary = store.disagreement_summary()
    rows = store.get_reviews()
    store.close()
    if summary.get("n", 0) == 0:
        st.info("No human reviews recorded yet. Use the Case Review page to add some.")
        return
    st.json(summary)
    st.dataframe(pd.DataFrame(rows))


def page_reproducibility(manifest):
    st.title("Reproducibility")
    if not manifest:
        st.info("No run manifest found.")
        return
    st.json(manifest)
    st.caption("Outputs are written to the `outputs/` directory and regenerated by "
               "`python -m src.experiment_runner`.")


def main():
    exp, cases, calib, agg, manifest = load_outputs()
    df, sidecar = load_dataset()
    page = st.sidebar.radio("Page", ["Study Overview", "Case Review", "Experiment Comparison",
                                     "Failure Analysis", "Calibration", "Human Review",
                                     "Reproducibility"])
    if page == "Study Overview":
        page_overview(exp, cases, manifest)
    elif page == "Case Review":
        page_case_review(exp, cases, df, sidecar, agg, manifest)
    elif page == "Experiment Comparison":
        page_comparison(exp, manifest)
    elif page == "Failure Analysis":
        page_failure(cases, manifest)
    elif page == "Calibration":
        page_calibration(exp, calib, manifest)
    elif page == "Human Review":
        page_human_review(manifest)
    elif page == "Reproducibility":
        page_reproducibility(manifest)


if __name__ == "__main__":
    main()
