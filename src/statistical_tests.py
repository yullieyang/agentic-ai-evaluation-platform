"""Statistical comparison utilities.

Method selection is documented at the call site in the experiment runner and the
research summary. Each function records its assumptions in its docstring. The
guiding rules:

* Paired binary outcomes (same cases, two conditions) -> McNemar's test, because
  the cases are matched and the outcome is correct/incorrect.
* Differences in a continuous or proportion metric between matched conditions ->
  paired bootstrap confidence interval and a paired permutation test, which make
  minimal distributional assumptions.
* Effect sizes (Cohen's h for proportions, Cohen's d for means) accompany every
  p-value; p-values are never reported alone.
* Multiple related comparisons are corrected with Benjamini-Hochberg.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(values, statistic: Callable = np.mean, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 7) -> dict[str, float]:
    """Nonparametric bootstrap CI for a statistic of a 1-D sample."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    boot = np.array([statistic(arr[rng.integers(0, n, n)]) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {
        "point": float(statistic(arr)),
        "low": float(np.quantile(boot, alpha)),
        "high": float(np.quantile(boot, 1 - alpha)),
        "n": n,
    }


def bootstrap_metric_ci(records: list[dict], metric_fn: Callable[[list[dict]], float],
                        n_boot: int = 2000, ci: float = 0.95, seed: int = 7) -> dict[str, float]:
    """Bootstrap CI for a metric computed over a list of records (case resampling)."""
    rng = np.random.default_rng(seed)
    n = len(records)
    if n == 0:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    point = float(metric_fn(records))
    boot = []
    for _ in range(n_boot):
        sample = [records[i] for i in rng.integers(0, n, n)]
        boot.append(metric_fn(sample))
    boot = np.asarray(boot, dtype=float)
    alpha = (1 - ci) / 2
    return {"point": point, "low": float(np.quantile(boot, alpha)),
            "high": float(np.quantile(boot, 1 - alpha)), "n": n}


def mcnemar_test(correct_a, correct_b) -> dict[str, float]:
    """McNemar's test for two paired binary (correct/incorrect) outcomes.

    Assumptions: the two conditions are evaluated on the *same* matched cases and
    each outcome is binary. Uses the exact binomial test when the number of
    discordant pairs is small (< 25) and the continuity-corrected chi-square
    approximation otherwise.
    """
    a = np.asarray(list(correct_a), dtype=bool)
    b = np.asarray(list(correct_b), dtype=bool)
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    n01 = int(np.sum(a & ~b))   # a correct, b wrong
    n10 = int(np.sum(~a & b))   # a wrong, b correct
    discordant = n01 + n10
    if discordant == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0, "method": "no_discordant"}
    if discordant < 25:
        res = stats.binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided")
        return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)),
                "p_value": float(res.pvalue), "method": "exact_binomial"}
    stat = (abs(n01 - n10) - 1) ** 2 / discordant
    p = float(stats.chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": p,
            "method": "chi2_continuity"}


def paired_permutation_test(values_a, values_b, n_perm: int = 5000, seed: int = 7) -> dict[str, float]:
    """Paired permutation test on the mean difference (sign-flip null)."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    diff = a - b
    observed = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        if abs(float(np.mean(signs * diff))) >= abs(observed):
            count += 1
    return {"observed_mean_diff": observed, "p_value": (count + 1) / (n_perm + 1), "n": len(diff)}


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for the difference between two proportions."""
    return float(2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1))) - 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1))))


def cohens_d_paired(values_a, values_b) -> float:
    """Paired Cohen's d for the mean difference (d_z)."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    diff = a - b
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> dict[str, list]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values and reject flags."""
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return {"adjusted": [], "reject": []}
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return {"adjusted": out.tolist(), "reject": (out <= alpha).tolist()}
