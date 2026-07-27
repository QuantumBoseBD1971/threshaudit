"""Aggregate many :class:`~threshaudit.audit.AuditResult` records into
summary tables and cluster-bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


class AuditReport:
    """Collects audit results (optionally tagged with metadata) for summary.

    Example
    -------
    >>> report = AuditReport()
    >>> report.add(result, task="bulk_modulus", shift="element:O", policy="ensemble")
    >>> report.add(result2, task="bulk_modulus", shift="cluster:1", policy="ensemble")
    >>> report.operational_failure_rate()
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def add(self, result, **metadata) -> None:
        """Add one audit result, with arbitrary tags (task, shift, policy, ...)."""
        payload = result.to_dict() if hasattr(result, "to_dict") else (
            asdict(result) if is_dataclass(result) else dict(result)
        )
        self._records.append({**metadata, **payload})

    def to_frame(self):
        if pd is None:  # pragma: no cover
            raise ImportError("AuditReport.to_frame() requires pandas.")
        return pd.DataFrame(self._records)

    def operational_failure_rate(
        self,
        group_by: list[str] | None = None,
        cluster_col: str | None = None,
        n_bootstrap: int = 3000,
        ci: float = 0.95,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Operational failure rate, optionally with a cluster bootstrap CI.

        Parameters
        ----------
        group_by : optional list of metadata columns to break the rate down
            by (e.g. ``["task", "shift_type"]``). If None, one overall rate
            is returned.
        cluster_col : if given, the bootstrap resamples whole groups of this
            column (e.g. a "holdout" identifier) rather than individual rows
            — appropriate when many rows share a holdout/repetition
            structure and are not independent.
        n_bootstrap : number of cluster-bootstrap resamples for the CI.
        ci : confidence level for the reported interval.
        """
        frame = self.to_frame()
        rng = rng if rng is not None else np.random.default_rng()

        def _rate_and_ci(subframe):
            n = len(subframe)
            failures = int(subframe["operational_failure"].sum())
            point = failures / n if n else float("nan")
            if cluster_col is None or cluster_col not in subframe:
                return {"n": n, "failures": failures, "rate": point}
            clusters = subframe[cluster_col].unique()
            boot = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                sampled = rng.choice(clusters, size=len(clusters), replace=True)
                mask = subframe[cluster_col].isin(sampled)
                boot[b] = subframe.loc[mask, "operational_failure"].mean()
            lower = float(np.nanquantile(boot, (1 - ci) / 2))
            upper = float(np.nanquantile(boot, 1 - (1 - ci) / 2))
            return {
                "n": n,
                "failures": failures,
                "rate": point,
                f"ci{int(ci * 100)}_low": lower,
                f"ci{int(ci * 100)}_high": upper,
            }

        if not group_by:
            return _rate_and_ci(frame)

        out = {}
        for key, subframe in frame.groupby(group_by):
            out[key] = _rate_and_ci(subframe)
        return out
