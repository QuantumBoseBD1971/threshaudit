# threshaudit

**Audit whether a frozen, in-distribution-calibrated reliability threshold survives distribution shift.**

Selective prediction is common practice: a model returns a prediction only
when its uncertainty/reliability score falls below an acceptance threshold,
and abstains otherwise. Usually that threshold is chosen once, using
in-distribution (ID) calibration data, and then frozen for deployment.

`threshaudit` answers a narrower, deployment-relevant question that
calibration plots and uncertainty–error correlation do not: **if you freeze
that threshold and deploy it on genuinely shifted data, does it still meet
your absolute error target?**

It decomposes the answer into three distinguishable outcomes, rather than
one undifferentiated "did it work":

1. **Construction failure** — no threshold could be built on ID data that
   met the target with acceptable coverage.
2. **Risk breach** — a threshold was built and accepted OOD data, but the
   retained error exceeded the target anyway.
3. **Coverage failure** — the target was met, but only by rejecting almost
   everything (below a minimum coverage you specify).

These are easy to conflate in practice — a policy that refuses 99.8% of
out-of-distribution data and reports a great retained-MAE is not a success,
it's a coverage failure. `threshaudit` reports both explicitly.

## Installation

```bash
pip install threshaudit
# optional extras:
pip install "threshaudit[sklearn]"   # for the built-in nearest-neighbor distance score
pip install "threshaudit[pandas]"    # for AuditReport summary tables
```

## Quickstart

```python
from threshaudit import ThresholdPolicy, TransferAudit

# 1. Construct a frozen threshold using ONLY in-distribution calibration data
policy = ThresholdPolicy(tolerance=0.5, min_coverage=0.20)
frozen = policy.construct(calibration_scores, calibration_errors, calibration_groups)

# 2. Apply it, unchanged, to out-of-distribution data
audit = TransferAudit(min_coverage=0.20)
result = audit.run(frozen, ood_scores, ood_errors)

print(result.operational_failure)   # True/False
print(result.risk_breach)           # did retained error exceed tolerance?
print(result.coverage_failure)      # did it retain too little OOD data?
print(result.coverage)              # fraction of OOD data accepted
print(result.retained_error)        # mean error among accepted OOD points
```

See `examples/synthetic_example.py` for a complete, domain-agnostic worked
example (no materials science, no chemistry — just a toy regression problem
with a clear extrapolation shift) showing an ensemble-disagreement score
that stays deceptively low under extrapolation while true error rises
sharply — exactly the kind of silent failure this package is built to catch.

## Bring your own model and score

`threshaudit` does not fit models. It audits whatever reliability score and
point predictions you already have. Built-in score wrappers are provided for
convenience:

- `EnsembleDisagreementScore` — std. dev. across an ensemble of predictions
- `QuantileWidthScore` — width between two predicted quantiles
- `NearestNeighborDistanceScore` — distance to nearest training point (requires scikit-learn)
- `EqualRankHybridScore` — rank-based combination of multiple scores
- `PrecomputedScore` — wraps scores you've already computed elsewhere (e.g.
  from a foundation model's own uncertainty output, a learned reliability
  classifier, or anything else)

Subclass `ReliabilityScore` for anything else.

## Aggregating many audits

```python
from threshaudit import AuditReport

report = AuditReport()
for task, shift, policy_name, result in your_experiment_loop():
    report.add(result, task=task, shift=shift, policy=policy_name)

report.operational_failure_rate(group_by=["task", "shift"])
```
## Scope

`threshaudit` is domain-agnostic and does not perform model fitting, feature
engineering, or application-specific data preparation. It accepts user-supplied
reliability scores, prediction errors, and group identifiers, allowing the same
frozen-threshold audit to be applied across machine-learning deployments and
shift definitions.

## Relationship to the original research

This package generalises the audit protocol introduced in:

> Uddin, M. M. (2026). *Evaluating the Transfer of Frozen Uncertainty-Based
> Error Controls Across Materials Chemistry Shifts.*

The original study applied this protocol to two materials-property
prediction tasks (experimental band gap, computed bulk modulus) using
composition-based features. `threshaudit` strips out everything specific
to that domain (formula parsing, composition clustering, materials
featurization) so the same audit can be applied to any ML deployment
where a frozen reliability threshold is used to decide whether to trust a
prediction — not just materials science.

## Roadmap

Planned development is tracked through GitHub issues and will be delivered through
versioned releases. Current priorities are:

- publish a documented command-line interface for CSV-based audits;
- add structured JSON and CSV export for audit results;
- expand input validation and edge-case test coverage;
- add further domain-agnostic distribution-shift examples;
- publish user and API documentation through GitHub Pages; and
- add runtime and scalability benchmarks for larger audit datasets.

See [`CHANGELOG.md`](CHANGELOG.md) for released changes.

## License

MIT
