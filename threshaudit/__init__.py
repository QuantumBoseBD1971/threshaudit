"""threshaudit: audit whether a frozen, ID-calibrated reliability threshold
survives distribution shift.

Quickstart
----------
>>> from threshaudit import ThresholdPolicy, TransferAudit
>>> policy = ThresholdPolicy(tolerance=0.5, min_coverage=0.20)
>>> frozen = policy.construct(calibration_scores, calibration_errors, calibration_groups)
>>> audit = TransferAudit(min_coverage=0.20)
>>> result = audit.run(frozen, ood_scores, ood_errors)
>>> result.operational_failure
"""

from .audit import AuditResult, TransferAudit
from .policy import FrozenThreshold, ThresholdPolicy
from .report import AuditReport
from .scores import (
    EnsembleDisagreementScore,
    EqualRankHybridScore,
    NearestNeighborDistanceScore,
    PrecomputedScore,
    QuantileWidthScore,
    ReliabilityScore,
)

__version__ = "0.1.0"

__all__ = [
    "ThresholdPolicy",
    "FrozenThreshold",
    "TransferAudit",
    "AuditResult",
    "AuditReport",
    "ReliabilityScore",
    "PrecomputedScore",
    "EnsembleDisagreementScore",
    "QuantileWidthScore",
    "NearestNeighborDistanceScore",
    "EqualRankHybridScore",
]
