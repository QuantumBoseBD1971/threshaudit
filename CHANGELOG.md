# Changelog

All notable changes to `threshaudit` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added validation tests for malformed calibration and OOD inputs.
- Added CI coverage for feature branches and pull requests.

### Changed

- Added explicit validation for tolerance, minimum coverage, array dimensionality, empty inputs, mismatched lengths, and non-finite values.
- Improved `ThresholdPolicy` and `TransferAudit` error handling so invalid inputs raise clear `ValueError` messages.

### Fixed

- Nothing yet.

## [0.1.0] - 2026-07-27

### Added

- Initial public release of the `threshaudit` Python package.
- Construction of frozen reliability thresholds from in-distribution calibration data.
- Transfer auditing on shifted deployment data.
- Separate reporting of construction failure, risk breach, and coverage failure.
- Group-aware confidence-bound support for threshold construction.
- Built-in wrappers for ensemble disagreement, quantile width, nearest-neighbour distance,
  equal-rank hybrid scores, and precomputed reliability scores.
- Aggregation utilities for repeated audits.
- A domain-agnostic synthetic example.
- Automated tests and continuous integration across supported Python versions.
- JOSS manuscript source and BibTeX bibliography.

[Unreleased]: https://github.com/QuantumBoseBD1971/threshaudit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/QuantumBoseBD1971/threshaudit/releases/tag/v0.1.0
