import numpy as np
import pytest

from threshaudit import ThresholdPolicy, TransferAudit


def test_empty_calibration_data_raises_value_error():
    policy = ThresholdPolicy(tolerance=0.5)

    with pytest.raises(ValueError, match="calibration data must not be empty"):
        policy.construct(
            np.array([]),
            np.array([]),
            np.array([]),
        )


def test_mismatched_calibration_lengths_raise_value_error():
    policy = ThresholdPolicy(tolerance=0.5)

    with pytest.raises(ValueError, match="same length"):
        policy.construct(
            np.array([0.1, 0.2]),
            np.array([0.1]),
            np.array([1, 2]),
        )


def test_invalid_tolerance_raises_value_error():
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        ThresholdPolicy(tolerance=-0.1)


@pytest.mark.parametrize("min_coverage", [-0.1, 0.0, 1.1])
def test_invalid_minimum_coverage_raises_value_error(min_coverage):
    with pytest.raises(ValueError, match="min_coverage must be in"):
        ThresholdPolicy(
            tolerance=0.5,
            min_coverage=min_coverage,
        )


def test_non_finite_calibration_scores_raise_value_error():
    policy = ThresholdPolicy(tolerance=0.5)

    with pytest.raises(ValueError, match="finite"):
        policy.construct(
            np.array([0.1, np.nan, 0.3]),
            np.array([0.1, 0.2, 0.3]),
            np.array([1, 2, 3]),
        )


def test_non_finite_calibration_errors_raise_value_error():
    policy = ThresholdPolicy(tolerance=0.5)

    with pytest.raises(ValueError, match="finite"):
        policy.construct(
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, np.inf, 0.3]),
            np.array([1, 2, 3]),
        )


def test_empty_ood_data_raises_value_error():
    audit = TransferAudit(min_coverage=0.2)

    policy = ThresholdPolicy(tolerance=1.0)
    frozen = policy.construct(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1, 2, 3]),
    )

    with pytest.raises(ValueError, match="OOD data must not be empty"):
        audit.run(
            frozen,
            np.array([]),
            np.array([]),
        )


def test_mismatched_ood_lengths_raise_value_error():
    audit = TransferAudit(min_coverage=0.2)

    policy = ThresholdPolicy(tolerance=1.0)
    frozen = policy.construct(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1, 2, 3]),
    )

    with pytest.raises(ValueError, match="same length"):
        audit.run(
            frozen,
            np.array([0.1, 0.2]),
            np.array([0.1]),
        )
