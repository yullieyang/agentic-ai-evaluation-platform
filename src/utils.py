"""Shared utilities: seeding, logging, config loading, JSON helpers, cost tables.

This module deliberately avoids importing heavy or optional dependencies
(streamlit, anthropic, openai) so that the core library and the test suite
remain importable in a minimal environment.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
REPORT_DIR = REPO_ROOT / "reports"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> np.random.Generator:
    """Seed the standard-library and NumPy RNGs and return a NumPy Generator.

    Returns a ``numpy.random.Generator`` so callers can thread an explicit,
    independent random stream rather than relying on global state.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single structured stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config from the ``config/`` directory by base name."""
    return load_yaml(CONFIG_DIR / name)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, default=str)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json_block(text: str) -> str:
    """Extract the first balanced top-level JSON object from a text blob.

    LLM providers occasionally wrap JSON in prose or code fences. This scans
    for the first ``{`` and returns the substring up to its matching ``}``.
    Raises ``ValueError`` when no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in text")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("unbalanced JSON object in text")


# --------------------------------------------------------------------------- #
# Environment / provenance metadata
# --------------------------------------------------------------------------- #
def git_commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def dependency_versions() -> dict[str, str]:
    """Return versions of the scientific dependencies actually importable."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod_name in ["pandas", "numpy", "scipy", "sklearn", "pydantic", "plotly"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = "not installed"
    return versions


def environment_metadata() -> dict[str, Any]:
    return {
        "git_commit": git_commit_hash(),
        "dependencies": dependency_versions(),
        "platform": sys.platform,
    }


# --------------------------------------------------------------------------- #
# Cost model (approximate USD per 1M tokens; used only when token counts known)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostRate:
    input_per_mtok: float
    output_per_mtok: float


# These are coarse, documented approximations used to surface order-of-magnitude
# cost tradeoffs in experiments. They are not authoritative pricing and the mock
# provider uses an explicit synthetic rate. Update from provider docs as needed.
COST_TABLE: dict[str, CostRate] = {
    "mock-deterministic": CostRate(0.0, 0.0),
    "claude-opus-4-8": CostRate(5.0, 25.0),
    "claude-sonnet-4-6": CostRate(3.0, 15.0),
    "claude-haiku-4-5": CostRate(1.0, 5.0),
    "gpt-4o": CostRate(2.5, 10.0),
    "gpt-4o-mini": CostRate(0.15, 0.6),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate request cost in USD from token counts, or ``None`` if unknown."""
    rate = COST_TABLE.get(model)
    if rate is None:
        return None
    return (input_tokens / 1e6) * rate.input_per_mtok + (
        output_tokens / 1e6
    ) * rate.output_per_mtok


def ensure_dirs() -> None:
    for directory in (DATA_DIR, OUTPUT_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
