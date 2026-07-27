"""Frozen-threshold policy construction.

This is the central contribution of the underlying research: selecting an
acceptance threshold on in-distribution (ID) calibration data only, using a
conservative grouped-bootstrap bound, then freezing it before any
out-of-distribution (OOD) labels are observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .metrics import group_bootstrap_upper_bound


@dataclass
class FrozenThreshold:
    """The result of a (possibly failed) threshold construction attempt."""

    tolerance: float
    threshold: float | None
    retained_coverage: float | None
    calibration_ucb: float | None
    constructed: bool

    def accept(self, scores: np.ndarray) -> np.ndarray:
        """Apply this frozen threshold to a new (e.g. OOD) score array."""
        if not self.constructed:
            return np.zeros(len(scores), dtype=bool)
        return scores <= self.threshold


@dataclass
class ThresholdPolicy:
    """Constructs a frozen acceptance threshold from ID calibration data only.

    Parameters
    ----------
    tolerance : the absolute error target (epsilon) the policy must satisfy,
        e.g. a maximum acceptable mean absolute error.
    min_coverage : the smallest fraction of the calibration set a candidate
        threshold is allowed to retain (default 0.20). Prevents "success by
        refusing almost everything."
    n_candidates : number of coverage levels to scan between ``min_coverage``
        and full coverage (default 41, matching the original study).
    n_bootstrap : bootstrap resamples used for each candidate's upper bound.
    confidence : one-sided confidence level for the bootstrap upper bound
        (default 0.95).

    Notes
    -----
    Call :meth:`construct` exactly once per (score, error, group) calibration
    set. The returned :class:`FrozenThreshold` should then be applied,
    unchanged, to OOD data via :meth:`FrozenThreshold.accept` — do not
    re-fit or adjust it after seeing OOD labels. That discipline is the
    entire point of the audit: it tests whether ID calibration alone is
    sufficient, with no OOD-label leakage of any kind.
    """

    tolerance: float
    min_coverage: float = 0.20
    n_candidates: int = 41
    n_bootstrap: int = 300
    confidence: float = 0.95
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be non-negative and finite")
    
        if not np.isfinite(self.min_coverage) or not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in the interval (0, 1]")
    
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")
    
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1")
    
        if not np.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must be in the interval (0, 1)")
    
        self._rng = np.random.default_rng(self.seed)

    def construct(
        self,
        calibration_scores: np.ndarray,
        calibration_errors: np.ndarray,
        calibration_groups: np.ndarray,
    ) -> FrozenThreshold:
        """Construct the frozen threshold from calibration (ID) data only.

        Parameters
        ----------
        calibration_scores : reliability scores on the calibration set;
            larger = less reliable (i.e. more likely to be rejected).
        calibration_errors : true errors (e.g. absolute error) on the same
            calibration rows.
        calibration_groups : group identifiers for grouped bootstrap
            resampling (e.g. to keep duplicate/near-duplicate rows together).

        Returns
        -------
        FrozenThreshold : may have ``constructed=False`` if no candidate
            coverage level satisfied ``tolerance`` at or above
            ``min_coverage`` — this is a legitimate, expected outcome and
            should be reported, not treated as an error.
        """
        calibration_scores = np.asarray(calibration_scores)
        calibration_errors = np.asarray(calibration_errors)
        calibration_groups = np.asarray(calibration_groups)
        
        if calibration_scores.ndim != 1:
            raise ValueError("calibration_scores must be one-dimensional")
        
        if calibration_errors.ndim != 1:
            raise ValueError("calibration_errors must be one-dimensional")
        
        if calibration_groups.ndim != 1:
            raise ValueError("calibration_groups must be one-dimensional")
        
        if len(calibration_scores) == 0:
            raise ValueError("calibration data must not be empty")
        
        if not (
            len(calibration_scores)
            == len(calibration_errors)
            == len(calibration_groups)
        ):
            raise ValueError(
                "calibration scores, errors, and groups must have the same length"
            )
        
        try:
            scores_are_finite = np.isfinite(calibration_scores).all()
        except TypeError as exc:
            raise ValueError("calibration scores must be numeric and finite") from exc
        
        try:
            errors_are_finite = np.isfinite(calibration_errors).all()
        except TypeError as exc:
            raise ValueError("calibration errors must be numeric and finite") from exc
        
        if not scores_are_finite:
            raise ValueError("calibration scores must be finite")
        
        if not errors_are_finite:
            raise ValueError("calibration errors must be finite")
        order = np.argsort(calibration_scores, kind="stable")
        n = len(order)
        candidate_sizes = np.unique(
            np.linspace(
                math.ceil(self.min_coverage * n), n, self.n_candidates
            ).astype(int)
        )

        best = None  # (retained, threshold, ucb) with largest retained satisfying tolerance
        for retained in candidate_sizes:
            chosen = order[:retained]
            ucb = group_bootstrap_upper_bound(
                calibration_errors[chosen],
                calibration_groups[chosen],
                n_bootstrap=self.n_bootstrap,
                quantile=self.confidence,
                rng=self._rng,
            )
            if ucb <= self.tolerance:
                threshold = float(calibration_scores[order[retained - 1]])
                if best is None or retained > best[0]:
                    best = (retained, threshold, ucb)

        if best is None:
            return FrozenThreshold(
                tolerance=self.tolerance,
                threshold=None,
                retained_coverage=None,
                calibration_ucb=None,
                constructed=False,
            )

        retained, threshold, ucb = best
        return FrozenThreshold(
            tolerance=self.tolerance,
            threshold=threshold,
            retained_coverage=retained / n,
            calibration_ucb=ucb,
            constructed=True,
        )
