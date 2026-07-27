import numpy as np
import pytest

from threshaudit import ThresholdPolicy


def _make_calibration(seed=0, n=500, n_groups=100):
    """Synthetic calibration set where the score genuinely tracks error,
    so a sensible threshold should be constructible."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, n)
    # errors correlated with score, plus noise
    errors = 0.05 + 0.9 * scores + rng.normal(0, 0.02, n)
    groups = rng.integers(0, n_groups, n)  # several rows can share a group
    return scores, errors, groups


def test_construct_succeeds_when_tolerance_is_generous():
    scores, errors, groups = _make_calibration()
    policy = ThresholdPolicy(tolerance=1.0, min_coverage=0.20, seed=1)
    frozen = policy.construct(scores, errors, groups)
    assert frozen.constructed
    assert frozen.threshold is not None
    assert frozen.retained_coverage >= 0.20


def test_construct_fails_when_tolerance_is_too_strict():
    scores, errors, groups = _make_calibration()
    # even the single best-scoring points have errors ~0.05-0.1, so a
    # tolerance far below that should make construction fail
    policy = ThresholdPolicy(tolerance=0.001, min_coverage=0.20, seed=1)
    frozen = policy.construct(scores, errors, groups)
    assert not frozen.constructed
    assert frozen.threshold is None


def test_frozen_threshold_accept_rejects_when_not_constructed():
    scores, errors, groups = _make_calibration()
    policy = ThresholdPolicy(tolerance=0.001, min_coverage=0.20, seed=1)
    frozen = policy.construct(scores, errors, groups)
    accepted = frozen.accept(np.array([0.1, 0.5, 0.9]))
    assert not accepted.any()


def test_min_coverage_is_respected():
    scores, errors, groups = _make_calibration()
    policy = ThresholdPolicy(tolerance=1.0, min_coverage=0.5, seed=1)
    frozen = policy.construct(scores, errors, groups)
    if frozen.constructed:
        assert frozen.retained_coverage >= 0.5 - 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_reproducible_given_seed(seed):
    scores, errors, groups = _make_calibration(seed=seed)
    p1 = ThresholdPolicy(tolerance=1.0, seed=42).construct(scores, errors, groups)
    p2 = ThresholdPolicy(tolerance=1.0, seed=42).construct(scores, errors, groups)
    assert p1.threshold == p2.threshold
    assert p1.retained_coverage == p2.retained_coverage
