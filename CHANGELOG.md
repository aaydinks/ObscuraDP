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
- More library modules: `obscuradp.selection` (masks over a model's flat vector and ways to compare them),
  `obscuradp.crypto.chunking`, `obscuradp.bench.payload` (how large encrypted messages are),
  `obscuradp.seeding` (collision-free seeds for every random stream of a run) and `obscuradp.results`
  (result records that carry the code version, machine, settings fingerprint and seeds).
- `obscuradp.legacy` now holds the complete Trial 07/27 pipeline (accounting, local training, encrypted
  aggregation, mask scoring, the training loop and the benchmarks), compared against the research code:
  bit for bit without encryption, within stated tolerances with it.
- Privacy bookkeeping: `PrivacyPolicy` (one total budget for a whole run), `PrivacyLedger` (every noisy
  release is charged before it happens and refused if it would exceed the budget) and `PrivacyReport`
  (what was promised, what was spent, and the stored releases so anyone can recompute it).
- `obscuradp.accounting`: budgets computed three independent ways and cross-checked, noise calibrated to a
  total budget, and worst-case limits on an attacker's advantage.
- Command `obscuradp report verify FILE` (also `python -m obscuradp`), which recomputes a stored report
  from its recorded releases and exits with status 1 if anything disagrees.
- New dependency: `dp-accounting`.
- The library's own private training step (`obscuradp.mechanisms`, `obscuradp.client.train_private_round`):
  exact Poisson sampling, one joint clipping norm, noise on every step including empty batches, and the
  whole round charged to the ledger before any data is used.
- `obscuradp.audit.bounds`: lower limits on the true privacy budget from the results of an attack, used by
  tests that attack the trainer and catch planted defects.
- `obscuradp.Simulation` with `obscuradp.Plan` and `obscuradp.LocalDPSGD`: a whole federation on one
  machine under one total privacy budget, scored on validation rows, with a privacy report for every run.
- `obscuradp.data`: encoding from declared ranges and categories, and each participant's own training,
  validation and test rows.
- `obscuradp.imbalance.ClassWeighted`: weights from a public figure that keep working under clipping.
- `obscuradp.unsafe`: settings that weaken the guarantee are refused unless `allow_unsafe=True`, and are
  then named in the privacy report.
- A first worked example in the README, run by the tests.

### Changed

- Root `.gitignore` rules are anchored to the repository root so they can no longer hide source folders.

### Security

- Nothing released yet. The research folders must not be used with real data; see `docs/AUDIT.md`.
