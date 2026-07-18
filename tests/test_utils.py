"""Tests for shared utilities: cost estimation, JSON helpers, environment metadata."""

from __future__ import annotations

import json

from src.utils import (
    COST_TABLE,
    dependency_versions,
    estimate_cost,
    extract_json_block,
    read_json,
    write_json,
)


def test_estimate_cost_known_model():
    cost = estimate_cost("claude-opus-4-8", input_tokens=1_000_000, output_tokens=1_000_000)
    rate = COST_TABLE["claude-opus-4-8"]
    assert cost == rate.input_per_mtok + rate.output_per_mtok


def test_estimate_cost_zero_for_mock_model():
    assert estimate_cost("mock-deterministic", 5000, 2000) == 0.0


def test_estimate_cost_unknown_model_returns_none():
    assert estimate_cost("some-model-not-in-the-table", 100, 100) is None


def test_estimate_cost_scales_linearly_with_tokens():
    a = estimate_cost("claude-sonnet-4-6", 100_000, 50_000)
    b = estimate_cost("claude-sonnet-4-6", 200_000, 100_000)
    assert abs(b - 2 * a) < 1e-9


def test_extract_json_block_handles_surrounding_prose():
    text = 'Sure, here it is:\n{"a": 1, "b": [1, 2]}\nHope that helps.'
    assert json.loads(extract_json_block(text)) == {"a": 1, "b": [1, 2]}


def test_extract_json_block_raises_on_no_json():
    import pytest
    with pytest.raises(ValueError):
        extract_json_block("no json here at all")


def test_write_and_read_json_round_trip(tmp_path):
    path = tmp_path / "out.json"
    write_json({"a": 1, "b": [1, 2, 3]}, path)
    assert read_json(path) == {"a": 1, "b": [1, 2, 3]}


def test_dependency_versions_reports_python():
    versions = dependency_versions()
    assert "python" in versions
    assert versions["pandas"] != "not installed"
