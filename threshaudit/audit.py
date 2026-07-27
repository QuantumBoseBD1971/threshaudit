"""Transfer audit: apply a frozen threshold to shifted (OOD) data and
classify the outcome.

This module implements the three-way outcome distinction that is the key
methodological point of the underlying study: a deployment failure is not
a single undifferentiated thing. It can be:

1. **Construction failure** — no safe policy could even be built on ID data.
2. **Risk breach** — a policy was built and accepted OOD data, but the
   retained error exceeded the target.
3. **Coverage failure** — a policy was built and satisfied the risk target,
   but only by refusing almost everything (below ``min_coverage``).

Conflating these into a single "it worked" / "it didn't" number hides which
failure mode is actually occurring, which matters for deciding what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import area_under_risk_coverage
from .policy import FrozenThreshold


@dataclass
class AuditResult:
    """Outcome of applying one frozen threshold to one OOD partition."""

    constructed: bool
    coverage: float | None
    retained_error: float | None
    risk_violation: float | None
    risk_breach: bool | None
    coverage_failure: bool
    operational_failure: bool
    aurc: float | None
    excess_aurc: float | None

    def to_dict(self) -> dict:
        return {
            "constructed": self.constructed,
            "coverage": self.coverage,
            "retained_error": self.retained_error,
            "risk_violation": self.risk_violation,
            "risk_breach": self.risk_breach,
            "coverage_failure": self.coverage_failure,
            "operational_failure": self.operational_failure,
            "aurc": self.aurc,
            "excess_aurc": self.excess_aurc,
        }


class TransferAudit:
    """Applies a frozen threshold to OOD data and classifies the result.

    Parameters
    ----------
    min_coverage : must match the ``min_coverage`` used when the policy was
        constructed; retransferred coverage below this level is counted as
        a coverage failure regardless of retained risk.
    """

   def __init__(self, min_coverage: float = 0.20) -> None:
       if not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
           raise ValueError("min_coverage must be in the interval (0, 1]")

    self.min_coverage = min_coverage

    def run(
        self,
        frozen: FrozenThreshold,
        ood_scores: np.ndarray,
        ood_errors: np.ndarray,
    ) -> AuditResult:
        """Apply ``frozen`` (built on ID data) to a new OOD score/error set.

        Parameters
        ----------
        frozen : the :class:`~threshaudit.policy.FrozenThreshold` returned by
            :meth:`threshaudit.policy.ThresholdPolicy.construct`. It must not
            be refit or adjusted using any information from this OOD set.
        ood_scores : reliability scores computed on the OOD data.
        ood_errors : true errors (e.g. absolute error) on the same OOD rows.
            Used only for evaluation here — never for threshold selection.
        """
        ood_scores = np.asarray(ood_scores)
         ood_errors = np.asarray(ood_errors)
         
         if ood_scores.ndim != 1:
             raise ValueError("OOD scores must be one-dimensional")
         
         if ood_errors.ndim != 1:
             raise ValueError("OOD errors must be one-dimensional")
         
         if len(ood_scores) == 0:
             raise ValueError("OOD data must not be empty")
         
         if len(ood_scores) != len(ood_errors):
             raise ValueError("OOD scores and errors must have the same length")
         
         try:
             scores_are_finite = np.isfinite(ood_scores).all()
         except TypeError as exc:
             raise ValueError("OOD scores must be numeric and finite") from exc
         
         try:
             errors_are_finite = np.isfinite(ood_errors).all()
         except TypeError as exc:
             raise ValueError("OOD errors must be numeric and finite") from exc
         
         if not scores_are_finite:
             raise ValueError("OOD scores must be finite")
         
         if not errors_are_finite:
             raise ValueError("OOD errors must be finite")
        aurc, excess_aurc = area_under_risk_coverage(ood_scores, ood_errors)

        if not frozen.constructed:
            return AuditResult(
                constructed=False,
                coverage=None,
                retained_error=None,
                risk_violation=None,
                risk_breach=None,
                coverage_failure=False,
                operational_failure=True,  # cannot deploy => operational failure
                aurc=aurc,
                excess_aurc=excess_aurc,
            )

        accepted = frozen.accept(ood_scores)
        coverage = float(accepted.mean())

        if not accepted.any():
            return AuditResult(
                constructed=True,
                coverage=0.0,
                retained_error=None,
                risk_violation=None,
                risk_breach=None,
                coverage_failure=True,
                operational_failure=True,
                aurc=aurc,
                excess_aurc=excess_aurc,
            )

        retained_error = float(ood_errors[accepted].mean())
        risk_violation = retained_error - frozen.tolerance
        risk_breach = bool(retained_error > frozen.tolerance)
        coverage_failure = bool(coverage < self.min_coverage)
        operational_failure = bool(risk_breach or coverage_failure)

        return AuditResult(
            constructed=True,
            coverage=coverage,
            retained_error=retained_error,
            risk_violation=risk_violation,
            risk_breach=risk_breach,
            coverage_failure=coverage_failure,
            operational_failure=operational_failure,
            aurc=aurc,
            excess_aurc=excess_aurc,
        )
