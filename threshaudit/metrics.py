"""Grouped-bootstrap and risk-coverage metrics.

These are the pure statistical building blocks used by :mod:`threshaudit.policy`
and :mod:`threshaudit.audit`. They are domain-agnostic: nothing here knows
about materials, chemistry, or any specific feature representation.
"""

from __future__ import annotations

import numpy as np


def group_bootstrap_upper_bound(
    errors: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int = 300,
    quantile: float = 0.95,
    rng: np.random.Generator | None = None,
) -> float:
    """One-sided bootstrap upper bound on mean error, resampling by group.

    Resampling whole groups (rather than individual rows) prevents
    correlated/near-duplicate rows within the same group from making the
    interval artificially tight. This is the statistical core of the
    "frozen threshold" construction rule.

    Parameters
    ----------
    errors : array of per-row errors (e.g. absolute error) on the
        calibration subset being considered.
    groups : array of group identifiers, same length as ``errors``. Rows
        that must not be split across train/calibration/OOD partitions
        (e.g. duplicate or near-duplicate samples) should share a group id.
    n_bootstrap : number of bootstrap resamples.
    quantile : the one-sided upper quantile to report (0.95 = 95% UCB).
    rng : optional numpy Generator for reproducibility.

    Returns
    -------
    float : the ``quantile``-th percentile of the bootstrap distribution of
        the mean error.
    """
    rng = rng if rng is not None else np.random.default_rng()
    unique_groups = np.unique(groups)
    by_group = {g: errors[groups == g] for g in unique_groups}
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        means[b] = np.concatenate([by_group[g] for g in sampled]).mean()
    return float(np.quantile(means, quantile))


def ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Empirical-CDF rank of ``values`` against a ``reference`` distribution.

    Used to build rank-based hybrid scores from two otherwise incomparable
    score scales (see :class:`threshaudit.scores.EqualRankHybrid`).
    """
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def risk_coverage_curve(scores: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) arrays for a risk-coverage curve.

    Sorting by ``scores`` ascending (lower score = more reliable) and taking
    the cumulative mean error at each coverage level.
    """
    order = np.argsort(scores, kind="stable")
    coverage = np.arange(1, len(order) + 1) / len(order)
    risk = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
    return coverage, risk


def area_under_risk_coverage(scores: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """AURC for the given score, and its excess over the oracle (error-sorted) ranking.

    Returns
    -------
    (aurc, excess_aurc) : the area under the score's own risk-coverage
        curve, and how much larger that area is than the best possible
        (oracle) ranking by true error. ``excess_aurc`` of 0 means the score
        ranked errors as well as an oracle could; larger is worse.
    """
    coverage, risk = risk_coverage_curve(scores, errors)
    aurc = float(np.trapezoid(risk, coverage))
    oracle_coverage, oracle_risk = risk_coverage_curve(errors, errors)
    oracle_aurc = float(np.trapezoid(oracle_risk, oracle_coverage))
    return aurc, aurc - oracle_aurc
