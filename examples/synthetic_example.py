"""A domain-agnostic worked example: no materials science, no chemistry.

Simulates a generic regression deployment where a model is calibrated on
"familiar" data, then deployed on genuinely shifted data, and asks whether a
frozen threshold selected only on the familiar data still meets an absolute
error target after the shift.

Run: python examples/synthetic_example.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from threshaudit import EnsembleDisagreementScore, ThresholdPolicy, TransferAudit


def make_ensemble_predictions(x_train, y_train, x_query, n_members=5, seed=0):
    rng = np.random.default_rng(seed)
    preds = []
    for m in range(n_members):
        idx = rng.integers(0, len(y_train), len(y_train))
        model = RandomForestRegressor(n_estimators=50, random_state=seed + m)
        model.fit(x_train[idx], y_train[idx])
        preds.append(model.predict(x_query))
    return np.vstack(preds)


def main():
    rng = np.random.default_rng(0)

    # "Familiar" domain: x in [0, 5], simple relationship
    n_train, n_cal = 800, 200
    x_train = rng.uniform(0, 5, (n_train, 1))
    y_train = np.sin(x_train[:, 0]) + rng.normal(0, 0.1, n_train)

    x_cal = rng.uniform(0, 5, (n_cal, 1))
    y_cal = np.sin(x_cal[:, 0]) + rng.normal(0, 0.1, n_cal)
    cal_groups = np.arange(n_cal)  # each row its own group here

    # Build ensemble, get disagreement score on calibration data
    cal_preds = make_ensemble_predictions(x_train, y_train, x_cal)
    cal_score = EnsembleDisagreementScore(cal_preds).score()
    cal_point_pred = cal_preds.mean(axis=0)
    cal_error = np.abs(y_cal - cal_point_pred)

    # Freeze a threshold targeting MAE <= 0.15, using calibration data only
    policy = ThresholdPolicy(tolerance=0.15, min_coverage=0.20, seed=1)
    frozen = policy.construct(cal_score, cal_error, cal_groups)
    print(f"Constructed: {frozen.constructed}, "
          f"coverage={frozen.retained_coverage}, "
          f"threshold={frozen.threshold}")

    # "Shifted" domain: x in [10, 15] -- extrapolation region the ensemble
    # never trained on. This is the OOD test.
    n_ood = 200
    x_ood = rng.uniform(10, 15, (n_ood, 1))
    y_ood = np.sin(x_ood[:, 0]) + rng.normal(0, 0.1, n_ood)

    ood_preds = make_ensemble_predictions(x_train, y_train, x_ood)
    ood_score = EnsembleDisagreementScore(ood_preds).score()
    ood_point_pred = ood_preds.mean(axis=0)
    ood_error = np.abs(y_ood - ood_point_pred)

    audit = TransferAudit(min_coverage=0.20)
    result = audit.run(frozen, ood_score, ood_error)

    print("\n--- Transfer audit result (familiar -> shifted extrapolation) ---")
    print(f"Coverage retained OOD: {result.coverage}")
    print(f"Retained OOD error:    {result.retained_error}")
    print(f"Risk breach:           {result.risk_breach}")
    print(f"Coverage failure:      {result.coverage_failure}")
    print(f"Operational failure:   {result.operational_failure}")


if __name__ == "__main__":
    main()
