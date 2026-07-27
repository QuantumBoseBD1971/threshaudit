# Contributing to threshaudit

Contributions are welcome. This is a small, focused package, so the
process is intentionally lightweight.

## Reporting a bug or requesting a feature

Please open a GitHub issue describing:
- What you expected to happen
- What actually happened (with a minimal reproducible example if possible)
- Your Python and threshaudit version

## Submitting a change

1. Fork the repository and create a branch for your change.
2. Install the package in editable/dev mode: `pip install -e ".[dev]"`
3. Add or update tests in `tests/` for any behavioural change.
4. Ensure `pytest tests/` passes locally.
5. Open a pull request describing the change and its motivation.

## Scope

`threshaudit` is deliberately narrow: it audits frozen reliability
thresholds under distribution shift. It does not fit models, engineer
features, or provide domain-specific pipelines. Contributions that keep the
core dependency-light (numpy required; pandas/scikit-learn optional) are
preferred over ones that pull in heavier dependencies for niche use cases.

## Code of conduct

Be respectful and constructive. Disagreements about design should be
resolved through discussion on the relevant issue or pull request.
