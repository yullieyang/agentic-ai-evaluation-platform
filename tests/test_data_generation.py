"""Tests for synthetic data generation: reproducibility, coverage, label quality."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.data_generation import MODELS, SCENARIO_GENERATORS, generate_dataset
from src.schemas import CaseRecord, SCENARIO_TYPES, parse_model


def test_reproducible_same_seed():
    df1, _, _ = generate_dataset(n_cases=90, seed=5)
    df2, _, _ = generate_dataset(n_cases=90, seed=5)
    schema_cols = [c for c in df1.columns if not c.startswith("_")]
    assert df1[schema_cols].equals(df2[schema_cols])


def test_different_seed_changes_data():
    df1, _, _ = generate_dataset(n_cases=90, seed=5)
    df2, _, _ = generate_dataset(n_cases=90, seed=6)
    assert not df1["current_value"].equals(df2["current_value"])


def test_scenario_and_dimension_coverage():
    df, meta, docs = generate_dataset(n_cases=300, seed=11)
    assert set(df["scenario_type"]) == set(SCENARIO_TYPES)
    assert df["model_id"].nunique() == len(MODELS)
    assert df["metric_name"].nunique() >= 5
    assert df["segment"].nunique() >= 3
    assert set(docs) == set(SCENARIO_GENERATORS)
    assert meta["n_cases"] == len(df)


def test_no_missing_values_and_schema_valid():
    df, meta, _ = generate_dataset(n_cases=120, seed=3)
    schema_cols = meta["schema_fields"]
    assert df[schema_cols].isna().sum().sum() == 0
    for rec in df.sample(30, random_state=0)[schema_cols].to_dict("records"):
        parse_model(CaseRecord, rec)


def test_prevalence_reasonable():
    df, meta, _ = generate_dataset(n_cases=300, seed=2)
    assert 0.2 < meta["anomaly_prevalence"] < 0.8


def test_labels_not_trivially_recoverable_from_zscore():
    df, _, _ = generate_dataset(n_cases=400, seed=4)
    best = 0.0
    for t in np.arange(1.5, 5.0, 0.25):
        pred = df["z_score"].abs() > t
        best = max(best, f1_score(df["ground_truth_anomaly"], pred))
    # A single z-score threshold should not perfectly recover the labels.
    assert best < 0.9


def test_traps_have_expected_label_polarity():
    df, _, _ = generate_dataset(n_cases=450, seed=7)
    fp = df[df["scenario_type"] == "false_positive_trap"]
    fn = df[df["scenario_type"] == "false_negative_trap"]
    # False-positive traps are not anomalies despite large movements.
    assert fp["ground_truth_anomaly"].mean() == 0.0
    # False-negative traps are anomalies despite weak single-metric signals.
    assert fn["ground_truth_anomaly"].mean() == 1.0
    assert fp["z_score"].abs().mean() > fn["z_score"].abs().mean()
