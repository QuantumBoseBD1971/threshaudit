---
title: 'threshaudit: A Python package for auditing frozen reliability thresholds under distribution shift'
tags:
  - Python
  - machine learning
  - uncertainty quantification
  - selective prediction
  - distribution shift
  - reliability
authors:
  - name: Mohammed Munir Uddin
    orcid: 0009-0003-0147-2202
    affiliation: 1
affiliations:
  - name: Independent Researcher, United Kingdom
    index: 1
date: 1 August 2026
bibliography: paper.bib
---

# Summary

Many machine learning deployments use selective prediction: a model returns
a prediction only when an accompanying reliability score (e.g. ensemble
disagreement, predictive interval width, or distance to training data) falls
below an acceptance threshold, and abstains otherwise
[@pugnana2023model; @noskov2024selective]. In practice this threshold is
usually chosen once, using in-distribution (ID) calibration data, and then
frozen for deployment on new data that may differ from the training
distribution in composition, environment, or measurement conditions.

`threshaudit` is a small, dependency-light Python package that answers a
specific, deployment-relevant question this workflow leaves unanswered: if a
threshold is frozen using only ID calibration data, does it still satisfy a
predeclared absolute error target once applied, unchanged, to shifted data?
The package separates three outcomes that are easily conflated into a single
pass/fail judgement: (1) construction failure, where no threshold could be
built on ID data that meets the target at an acceptable minimum coverage;
(2) risk breach, where a threshold was built and accepted out-of-distribution
(OOD) data, but the retained error exceeded the target regardless; and (3)
coverage failure, where the target was nominally met, but only by rejecting
almost all OOD inputs. Reporting only an aggregate retained-error number can
hide the difference between a policy that generalises well and one that
merely refuses nearly everything.

# Statement of need

Uncertainty quantification (UQ) methods for machine learning are commonly
evaluated through calibration curves, uncertainty–error correlation, or
ranking metrics such as area under the risk-coverage curve
[@varivoda2023materials; @gruich2023clarifying]. These metrics describe
*ranking quality* or *distributional calibration*, but they do not establish
that a specific, frozen, absolute-error acceptance rule will continue to
hold after a named distribution shift. Recent work in both general machine
learning [@paplham2026evaluating] and applied domains such as
machine-learned interatomic potentials [@ho2026flexible] has begun to
highlight this gap explicitly:
a score can rank errors better than chance, or a calibration procedure can
look well-behaved on held-out data, while a fixed deployment threshold built
from that score still fails to control absolute risk once the input
distribution shifts.

Despite this recognised gap, no general-purpose, reusable software existed
to run this specific audit independent of any one modelling framework or
application domain. Existing selective-regression and conformal-prediction
libraries focus on constructing calibrated intervals or rejection rules
[@noskov2024selective], not on auditing whether a rule already frozen on ID
data transfers to a shift the analyst can define. `threshaudit` fills this
gap: it is domain-agnostic (it does not fit models or engineer features),
requires only numpy as a hard dependency, and accepts arbitrary
user-supplied scores and errors from any modelling stack — scikit-learn
ensembles, deep learning models, or precomputed scores from third-party
foundation models. This makes it straightforward to apply a consistent
audit protocol to a new dataset, model, or application domain without
reimplementing the underlying statistics.

# Functionality

The package exposes three core objects. `ThresholdPolicy` constructs a
frozen threshold from ID calibration scores, errors, and group identifiers,
using a one-sided grouped-bootstrap upper confidence bound on mean error at
each of a range of candidate coverage levels, selecting the largest coverage
level whose bound satisfies a user-specified tolerance. `TransferAudit`
applies a constructed `FrozenThreshold` to OOD scores and errors, and
classifies the outcome into the three-way failure taxonomy described above.
`AuditReport` aggregates many such audits (e.g. across repeated splits,
shift definitions, or candidate scoring methods) into summary tables and
cluster-bootstrap confidence intervals on the overall operational failure
rate. A small set of built-in `ReliabilityScore` subclasses (ensemble
disagreement, quantile interval width, nearest-neighbour distance, and a
rank-based hybrid) are provided for convenience, alongside a
`PrecomputedScore` wrapper for scores computed elsewhere. A fully
domain-agnostic worked example using a synthetic extrapolation scenario is
included to demonstrate the audit workflow.


# References
