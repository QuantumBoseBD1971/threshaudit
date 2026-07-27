import numpy as np

from threshaudit import ThresholdPolicy, TransferAudit
from threshaudit.policy import FrozenThreshold


def test_construction_failure_is_operational_failure():
    frozen = FrozenThreshold(
        tolerance=0.1, threshold=None, retained_coverage=None,
        calibration_ucb=None, constructed=False,
    )
    audit = TransferAudit(min_coverage=0.20)
    result = audit.run(frozen, np.array([0.1, 0.2]), np.array([0.05, 0.06]))
    assert result.constructed is False
    assert result.operational_failure is True


def test_zero_acceptance_is_coverage_failure_not_undefined_success():
    frozen = FrozenThreshold(
        tolerance=0.1, threshold=-1.0, retained_coverage=0.5,
        calibration_ucb=0.05, constructed=True,
    )
    audit = TransferAudit(min_coverage=0.20)
    ood_scores = np.array([0.5, 0.6, 0.7])  # all above threshold -> rejected
    ood_errors = np.array([0.01, 0.01, 0.01])
    result = audit.run(frozen, ood_scores, ood_errors)
    assert result.coverage == 0.0
    assert result.coverage_failure is True
    assert result.operational_failure is True
    assert result.retained_error is None  # undefined, not "safe"


def test_risk_breach_detected_when_retained_error_exceeds_tolerance():
    frozen = FrozenThreshold(
        tolerance=0.1, threshold=1.0, retained_coverage=1.0,
        calibration_ucb=0.08, constructed=True,
    )
    audit = TransferAudit(min_coverage=0.20)
    ood_scores = np.array([0.1, 0.2, 0.3])
    ood_errors = np.array([0.5, 0.5, 0.5])  # far above tolerance
    result = audit.run(frozen, ood_scores, ood_errors)
    assert result.risk_breach is True
    assert result.operational_failure is True
    assert result.coverage == 1.0


def test_success_case_no_breach_no_coverage_failure():
    frozen = FrozenThreshold(
        tolerance=0.5, threshold=1.0, retained_coverage=1.0,
        calibration_ucb=0.3, constructed=True,
    )
    audit = TransferAudit(min_coverage=0.20)
    ood_scores = np.array([0.1, 0.2, 0.3])
    ood_errors = np.array([0.1, 0.1, 0.1])  # well within tolerance
    result = audit.run(frozen, ood_scores, ood_errors)
    assert result.risk_breach is False
    assert result.coverage_failure is False
    assert result.operational_failure is False


def test_end_to_end_construct_then_audit():
    rng = np.random.default_rng(0)
    n = 400
    cal_scores = rng.uniform(0, 1, n)
    cal_errors = 0.05 + 0.9 * cal_scores + rng.normal(0, 0.01, n)
    cal_groups = np.arange(n)

    policy = ThresholdPolicy(tolerance=0.5, min_coverage=0.20, seed=1)
    frozen = policy.construct(cal_scores, cal_errors, cal_groups)
    assert frozen.constructed

    # OOD set where the score-error relationship has shifted (errors much
    # higher for the same score range) -> should breach risk target
    ood_scores = rng.uniform(0, 1, n)
    ood_errors = 0.05 + 0.9 * ood_scores + 1.0  # shifted up
    audit = TransferAudit(min_coverage=0.20)
    result = audit.run(frozen, ood_scores, ood_errors)
    assert result.operational_failure is True
    assert result.risk_breach is True
