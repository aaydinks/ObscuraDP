# Changelog

All notable changes to the `obscuradp` library are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until version 1.0.0 any release may change the
public interface.

## [Unreleased]

### Added

- Package skeleton `obscuradp` (`src/` layout, typed, version read from package metadata).
- Tests that run with network access switched off, an ignore-rule guard, and a check that the library never
  imports the research folders.
- Tooling: ruff, mypy, pytest, pre-commit; continuous integration on Python 3.11 to 3.13.
- `LICENSE`: Apache License 2.0.
- Project documents: roadmap, audit of the research code, threat model, target definitions, decision
  records.

- First library modules: `obscuradp.state` (a model's state as one flat vector, buffers included),
  `obscuradp.models.TabularMLP`, `obscuradp.partition` (even and uneven splits among simulated
  participants), `obscuradp.aggregation` (averaging in the clear) and `obscuradp.metrics` (yes/no measures
  with numpy only; run summaries use the sample standard deviation).
- `obscuradp.legacy`: a faithful copy of the Trial 07/27 settings, data preparation, model construction and
  evaluation, kept only so tests can compare old and new code number for number. Not for real data.
- Comparison tests (`tests/golden/`) that run the research code and the library side by side and require
  bit-for-bit equal results, and a fingerprint list that keeps the research folders frozen.

### Changed

- Root `.gitignore` rules are anchored to the repository root so they can no longer hide source folders.

### Security

- Nothing released yet. The research folders must not be used with real data; see `docs/AUDIT.md`.
