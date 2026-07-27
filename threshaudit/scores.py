"""Pluggable reliability scores.

A reliability score is any function of the input (and optionally the model)
that produces a per-sample number where *larger means less reliable*. This
module provides a small base class plus a few common built-ins; users are
expected to subclass :class:`ReliabilityScore` for anything model-specific
(e.g. a particular ensemble architecture, an embedding distance, a learned
classifier's output).

threshaudit deliberately does not ship model-fitting code — it audits
*whatever* score and predictor you already have. This keeps the package
dependency-light and usable with any ML stack (sklearn, PyTorch, a custom
foundation-model wrapper, etc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .metrics import ecdf


class ReliabilityScore(ABC):
    """Base class for a reliability score. Larger = less reliable."""

    @abstractmethod
    def score(self, X) -> np.ndarray:
        """Return a 1-D array of reliability scores for rows of ``X``."""
        raise NotImplementedError


class PrecomputedScore(ReliabilityScore):
    """Wraps an already-computed array of scores.

    Useful when scores come from an external model (a foundation model's
    embedding distance, a black-box ensemble, etc) that threshaudit has no
    need to know about directly.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values)

    def score(self, X=None) -> np.ndarray:
        return self._values


class EnsembleDisagreementScore(ReliabilityScore):
    """Standard deviation across an ensemble of point predictions.

    Parameters
    ----------
    predictions : array of shape (n_members, n_samples) — each row is one
        ensemble member's predictions on the same samples.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = np.asarray(predictions)

    def score(self, X=None) -> np.ndarray:
        return self._predictions.std(axis=0, ddof=1)


class QuantileWidthScore(ReliabilityScore):
    """Width between an upper and lower predicted quantile (e.g. 0.9 - 0.1)."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self._lower = np.asarray(lower)
        self._upper = np.asarray(upper)

    def score(self, X=None) -> np.ndarray:
        return self._upper - self._lower


class NearestNeighborDistanceScore(ReliabilityScore):
    """Distance from each query point to its nearest training point.

    Requires scikit-learn (an optional dependency: ``pip install
    threshaudit[sklearn]``).
    """

    def __init__(self, x_train: np.ndarray) -> None:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "NearestNeighborDistanceScore requires scikit-learn. "
                "Install with `pip install threshaudit[sklearn]`."
            ) from exc
        self._model = NearestNeighbors(n_neighbors=1).fit(x_train)

    def score(self, X) -> np.ndarray:
        distances, _ = self._model.kneighbors(X)
        return distances[:, 0]


class EqualRankHybridScore(ReliabilityScore):
    """Mean of empirical-CDF ranks of two or more component scores.

    Combines scores on different scales (e.g. ensemble disagreement in
    prediction units, distance in feature-space units) into one comparable
    [0, 1]-ish scale, without requiring the weights to be tuned on OOD data.

    Parameters
    ----------
    reference_scores : list of arrays, one per component score, computed on
        a *reference* set (typically the calibration set) — used to build
        each component's ECDF.
    weights : optional per-component weights (default: equal weighting).
    """

    def __init__(
        self,
        reference_scores: list[np.ndarray],
        weights: list[float] | None = None,
    ) -> None:
        self._reference_scores = [np.asarray(s) for s in reference_scores]
        n = len(self._reference_scores)
        self._weights = weights if weights is not None else [1.0 / n] * n

    def combine(self, query_scores: list[np.ndarray]) -> np.ndarray:
        """Combine per-component query-set scores into one hybrid score."""
        ranks = [
            ecdf(ref, np.asarray(query))
            for ref, query in zip(self._reference_scores, query_scores, strict=True)
        ]
        return sum(w * r for w, r in zip(self._weights, ranks, strict=True))

    def score(self, X=None) -> np.ndarray:
        raise NotImplementedError(
            "EqualRankHybridScore.combine(query_scores) should be called "
            "directly with the component scores for the set being scored "
            "(reference scores are fixed at construction time)."
        )
