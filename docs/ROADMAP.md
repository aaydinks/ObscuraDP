# Roadmap: from research repository to the `obscuradp` library

This is the approved plan (2026-11-10). It follows `docs/NEXT_STEPS_PROMPT.md`, corrects five points in it,
and adds steps it was missing. Progress is tracked in `docs/ROADMAP_STATUS.md`. Findings G1 to G27 are in
`docs/AUDIT.md`.

Standing decisions: standalone core with an optional Flower adapter; about 12 weeks, part-time, CPU only
(reference machine: 4 cores, 15 GB RAM); website copy is proposed first, then edited; package name
`obscuradp`; Apache-2.0 unless the owner objects.

Items marked **(added)** are not in the original prompt.

## Corrections to the original prompt

| Prompt says | Problem | Plan does instead |
|---|---|---|
| G6: with secure aggregation "each client only needs to add a share of the noise" | Only valid when each client sends **one** clipped gradient sum per round at the shared model. With several local steps, a changed record moves every later local step, and those steps were only protected by a fraction of the noise. | Two sound per-record modes: `LocalDPSGD` (several local steps, full noise) and `DistributedDPSGD` (one step per round through a secure sum, noise sized for the honest clients only). |
| Phase 2: "fail if PLD and RDP accountants disagree" | Both give upper limits and RDP is always the looser one, so they always differ. | Test ordering and a band: PLD $\le$ target; PLD lower estimate $\le$ RDP; RDP $\le 1.35 \cdot$ PLD $+ 0.05$ (band pinned after measuring); at sampling rate 1, PLD matches the exact Gaussian formula. |
| API sketch `PairwiseMasking(bits=16)`; TenSEAL CKKS as the default backend | 16-bit values need $16 + \lceil \log_2 N \rceil$ bits each on the wire, which misses the bytes target. CKKS publishes decrypted approximate values every round, the setting of a known key-recovery attack (G15). | Default secure sum is pairwise masking over integers with value width $b \le 14 - \lceil \log_2 N \rceil$ bits. The encryption option is exact integer BFV. CKKS stays as `research_only` behind an unsafe flag. |
| Phase 5: FEMNIST "subsample about 100 to 200 writers" | Per-participant privacy gets its strength from a large population with a small group sampled each round. Subsampling writers throws that away. | Keep all roughly 3,600 writers, sample about 200 per round, 100 rounds. Compute cost depends on the sampled group, not the population. |
| Phase 4: audits at 5 seeds settle the attacker-advantage target | The uncertainty of a measured advantage shrinks like $1/\sqrt{\text{trials}}$; 5 runs cannot support an upper limit of 5%. | About 30 short runs per budget, each with a few hundred test records inserted. |

Two smaller ones: "more local steps" is not a way to gain accuracy at a fixed total budget (dropped), and
Opacus's secure mode depends on an unmaintained package, so it is not the fix for noise quality.

## Sessions

One session is one draft pull request. **GATE** means stop and wait for the owner.

| # | Phase | Weeks | Content | Exit check |
|---|---|---|---|---|
| S1 | 0 | 1 | Audit, baseline reproduction, threat model, target definitions, website wording list, risk list, decision records. **GATE** | Every cited line exists; only `docs/` changed |
| S3 | 0.5 | 2 | **(added)** `LICENSE`; fix website items marked "now" in `docs/WEBSITE_MISMATCHES.md`. Only if approved at the gate | Website build, copy lint, contrast and end-to-end checks pass |
| S4 | 1 | 2 | Packaging, tooling, CI | Clean install; CI green |
| S5 | 1 | 2 to 3 | Comparison harness against the old code, then port the pure parts | Bitwise match on deterministic parts |
| S6 | 1 | 3 | Port the rest; old loop kept verbatim as `sim/legacy.py`; smoke checks become tests | Golden job under 4 minutes; coverage at least 80% |
| S7 | 2 | 3 to 4 | Policy, ledger, report, accountants, `obscuradp report verify` | Accountant ordering tests pass |
| S8 | 2 | 4 | Own DP-SGD step with exact sampling | Three-way gradient test; mutation test |
| S9 | 2 | 4 to 5 | Validation split, declared-bounds preprocessing, class weighting, unsafe flags, new `Simulation` | Unsafe configurations raise |
| S10 | 2 | 5 | Secure-sum interface, pairwise masking, quantizer, key roles, masks | Aggregator cannot decrypt; masked sum is exact |
| S11 | 2 | 5 to 6 | Corrected re-runs, old against new. **GATE** | Numbers regenerate from committed code |
| S12 | 3 | 6 | Distributed and central modes, collusion and dropout settings | Degradation table in the report |
| S13 | 3 | 7 | Integer path: rounding, Skellam noise, its accountant | Bound matches a brute-force check |
| S14 | 3 | 7 to 8 | Equal-budget comparison, measured bytes, per-participant mode, default plan | Results table committed |
| S15 | 4 | 8 | One-run membership audit, three reference numbers, nightly check | Auditor passes its self-test |
| S16 | 4 | 9 | Shadow-model attack, attribute inference, gradient inversion | Audit results committed |
| S17 | 5 | 9 | Dataset downloaders, synthetic generator, resumable runner, timing pilot, written research plan tagged before any run | Tag predates results |
| S18 | 5 | 10 | Benchmark runs | Privacy report beside every metrics file |
| S19 | 5 | 10 | Thin Flower adapter | Report equals the simulator's |
| S20 | 5 | 11 | Analysis, figures, claims registry | Claims check passes |
| S21 | 6 | 11 | Documentation, partner kit | All snippets run in CI |
| S22 | 6 | 12 | Release engineering, website sync. **GATE** before publishing | Fresh-machine install and quickstart under 5 minutes |

S2 is reserved as a second Phase 0 session in case the owner asks for changes at the gate. Two further
sessions are held as buffer.

## Phase 0, audit and freeze (S1). GATE

Baseline reproduction **(added)**: a throwaway environment with the Trial 07/27 pins; run `run_smoke.py` and
`verify_claims.py`; regenerate the seed-42 arm matrix and compare its metrics with the committed
`results/arms.json`; record the outcome and the resolved package list.

Documents: `ROADMAP_STATUS.md`, `AUDIT.md`, `THREAT_MODEL.md`, `KPI_MAPPING.md` (with reference tables for
corrected noise levels, wire bits and attacker-advantage ceilings), `WEBSITE_MISMATCHES.md`,
`RISK_REGISTER.md`, `adr/0001` to `adr/0004`, this file, and one update of `NEXT_STEPS_PROMPT.md`.

Gate decisions: see `docs/ROADMAP_STATUS.md`, "Decisions needed".

## Phase 0.5, stop the known-false claims (S3) (added)

`LICENSE`; items 1 to 12 of `docs/WEBSITE_MISMATCHES.md`. Nothing else.

## Phase 1, package skeleton with zero behaviour change (S4 to S6)

- **S4.** `pyproject.toml` (hatchling, `requires-python >= 3.11`), extras `[he]`, `[openfhe]`, `[flower]`,
  `[audit]`, `[bench]`, `[docs]`, `[dev]`; `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CHANGELOG.md`; version `0.1.0.dev0`. Anchor the `.gitignore` rules and add a CI guard that
  nothing under `src/`, `tests/`, `docs/`, `benchmarks/` is ignored. Finish the root `README.md`. ruff, mypy,
  pytest markers `slow` and `integration`, hypothesis, coverage, pre-commit. GitHub Actions next to the
  existing `website-ci.yml`: lint, types, unit tests on Python 3.11 to 3.13 (Linux), one macOS and one Windows
  job, a nightly job. **(added)** legacy and current lockfiles; Dependabot; `pip-audit`; a no-network test
  backing the no-telemetry promise; a session bootstrap script and a proposed cloud setup script (the CPU-only
  torch download host is blocked in the current cloud environment, see the status file); a TestPyPI dry run.
- **S5.** The old code is the reference: a test fixture adds `Trial 07/27` to `sys.path`, imports `obscura`,
  and runs it inside `contextlib.chdir` because its data paths are relative. An import rule forbids the
  package from importing it; a sha256 manifest freezes the old files. Port config (typed frozen dataclasses),
  models (generic flatten that includes buffers, G20), data, metrics, aggregation.
- **S6.** Port DP, trainer, encryption, selection and bench. Port the old loop **verbatim** as
  `src/obscuradp/sim/legacy.py`; it exists to be compared, improvements go into new modules. Fix G17 and G18
  in new modules only. Results schema v1 with a reproducibility manifest (commit, versions, CPU, threads,
  seeds, configuration hash) (G26). Convert the 7 smoke checks and the Experiment 2 and 3 assertions into
  unit tests; add a weighted-encryption test with unequal weights. A test that compares sha256 of committed
  results with `VERIFICATION.md` (G19).
- **Comparison tiers.** (1) Old and new in one process, bitwise: client splits, model start, noise levels,
  per-round aggregate hashes and metrics for the `plain` and `dp` arms, one thread. (2) Encrypted arms: exact
  mask and chunk counts; feed recorded client states to both aggregators and require agreement within
  $10^{-6}$; one free 3-round run within $10^{-4}$. (3) Frozen values with relative tolerance $10^{-5}$ for
  macOS and Windows.

## Phase 2, privacy correctness (S7 to S11). GATE

- **S7 accounting.** `policy.py`, `ledger.py` (charge before every release, raise when over budget),
  `report.py` (versioned schema that stores the list of charged events),
  `accounting/{pld,rdp,analytic,calibrate,crosscheck}.py` on `dp-accounting`. Calibrate to the **total**
  budget: one shared sampling rate $q$, steps per round $S = \lceil E / q \rceil$, total $T = R \cdot S$,
  bisection that rounds the noise up and certifies the spend is at most the target. **(added)**
  `obscuradp report verify report.json` recomputes the spend from the stored events. Per-participant budgets
  in local mode (G21).
- **S8 own DP-SGD step (added design choice).** `mechanisms/{dpsgd,sampling,noise}.py`, `client/trainer.py`.
  Per-record gradients through `torch.func`; Poisson sampling with a per-participant generator; empty batches
  still take a noise-only step (G27); division by $q \cdot n$ as a real number; joint clipping with a small
  safety margin; normalization layers that mix records are refused; shuffled fixed-size loaders are refused
  in private modes. Reason: one engine then serves local, distributed and central modes by moving where the
  noise is added. Safety net: three independent per-record gradient implementations agree to $10^{-6}$; a
  statistical test of the noise level; a nightly white-box check whose measured lower limit must not exceed
  the ledger; a mutation test in which halved noise must be caught. **Fallback if the session overruns:** keep
  Opacus as the mechanism and replace only the calibration.
- **S9 pipeline hygiene.** Train, validation and test split inside each participant (G13); preprocessing from
  declared feature bounds and categories (G12); `imbalance.ClassWeighted` from a public prior, SMOTE only
  behind `allow_unsafe=True` (G3); the unsafe-flag framework; the new `Simulation`.
- **S10 secure sum and keys.** `aggregation/secure_sum.py`; `crypto/pairwise_masking.py` (key agreement and
  keystream from the `cryptography` package, exact modular integers); `compression/` quantizer and byte
  accounting; `crypto/roles.py` with `KeyHolder`, `ClientEncryptor`, `Aggregator` and a test that the
  aggregator cannot decrypt (G5); CKKS flagged and never returning 64-bit floats (G15), no hand-written noise
  flooding; unused rotation keys removed; `selection.SPE` with data-independent masks by default and the
  batch-size bug fixed (G4, G14). Exact BFV backend if time allows.
- **S11 re-run and gate.** Tune on a synthetic stand-in dataset. Re-run the Trial 07/27 arms and frontier
  with total budgets $\{0.5, 1, 2, 4, 8\}$, no SMOTE, a natural-prevalence test set beside the balanced one
  (G25), 5 seeds, including a logistic-regression model. Publish old and new numbers side by side under
  `benchmarks/results/phase2/`. Gate memo listing every reversed claim.

## Phase 3, better results through better composition (S12 to S14)

- **S12** `DistributedDPSGD(max_colluding=t, max_dropouts=d)` and `CentralDPSGD` on the same engine. Each
  participant's noise share has standard deviation $\sigma C / \sqrt{N - t - d}$. `t` must be given; the
  report prints how the budget degrades with the real number of colluders. Distributed mode with SPE is
  refused. In v0.1 a round with a dropout is retried with fresh randomness and charged only if a sum was
  released.
- **S13 integer path.** Randomized rounding with an exact integer norm check; Skellam noise shares; a
  modulus-headroom planner; Skellam accounting for full-batch rounds only. The Skellam bound is implemented
  from the paper (Agarwal, Kairouz, Liu 2021) and checked against a brute-force computation from
  `scipy.stats.skellam`. This path also removes the floating-point noise issue (G16).
- **S14 comparison at equal total budget.** Local + CKKS (today's design), local + masking, distributed +
  masking, central (upper limit), no privacy. Bytes are measured from serialized messages. `ClientLevelDP`
  with a warning below 1,000 participants. `adr/0005-default-plan.md` chooses the default from the table.

## Phase 4, auditing (S15, S16)

- **S15** `audit.mechanism` (white-box check from S8, made public) and `audit.membership`: the one-run method
  of Steinke, Nasr and Jagielski (2023). About $\min(0.25\,n, 250)$ real records per participant are each
  included by coin flip; scores use only what that stack's attacker can see; the paper's routine gives a
  lower limit on the budget; the advantage is reported at a threshold chosen on a separate half, with a 95%
  interval from resampling whole runs. About 30 runs per budget. Every result shows three numbers: the
  general ceiling, the ceiling for the actual noise mechanism, and the measured value.
- **S16** capped shadow-model attack on the tabular models (32 runs on random halves); attribute inference
  with a baseline that has no model access; gradient inversion against exactly what the server sees in each
  stack, including the exact first-layer reconstruction for small networks at batch size 1. **(added)**
  auditor self-test: it must flag a zero-noise pipeline and must not flag a correct one.
- Audits run on tabular data only in v0.1.

## Phase 5, benchmark and Flower adapter (S17 to S20)

- **S17** downloaders with checksums and a cache; seeded synthetic health generator (10 sites, about 20,000
  rows, about 6% positives, different mix per site); resumable runner; timing pilot on non-private runs with
  seeds not used later. Then `benchmarks/PREREGISTRATION.md` is committed and tagged **before** any benchmark
  run: hypotheses, privacy unit per dataset, budget grid, seeds, metrics, analysis (sample standard
  deviation, resampling intervals, paired comparisons, G22), stopping rules, expected chance-level results at
  small budgets, and the shortcut below.
- **S18 runs**, about 17 CPU hours (estimate, replaced by the pilot):

| Dataset | Population | Per round | Model | Unit | Total budgets | Est. CPU h |
|---|---|---|---|---|---|---|
| Synthetic health | 10 sites | all sites; 30 rounds, or 100 single-step rounds | about 1,300 parameters | record | 0.5, 1, 2, 4, 8, none | 1.5 |
| Stroke, loan | 5 clients | same | 3,137 and 929 | record | same | 1.5 |
| FEMNIST | all of about 3,600 writers | about 200 sampled, 100 rounds | small CNN, about 44,000 | participant | 2, 4, 8, none (one seed at 1) | 11.5 |
| Shakespeare | 1,129 roles | about 50 sampled, 100 rounds | small GRU, about 64,000 | participant | 8, none | 2.75 |

  Cut order if the pilot overruns: Shakespeare; FEMNIST to 75 rounds and 150 sampled; FEMNIST seeds 5 to 3;
  FEMNIST budgets to 2, 8, none. Tabular runs are never cut. Shortcut: tabular runs use real masking and real
  encryption in the loop; FEMNIST and Shakespeare use the plaintext-equivalent sum. This is allowed because a
  CI test shows the integer backends give bitwise-identical sums, one seed per stack is confirmed with real
  cryptography, cost is measured separately on the true model shapes, and every result row records
  `crypto_in_loop`.
- **S19** thin `obscuradp[flower]` adapter taking the same `Plan` and `PrivacyPolicy`; a test that its
  privacy report equals the simulator's; a pinned, tested `flwr` version range.
- **S20** analysis, figures, `benchmarks/claims/` registry migrated from `verify_claims.py`.

## Phase 6, documentation, release, website sync (S21, S22). GATE before publishing

- **S21** mkdocs-material site with API reference; 5-minute CPU quickstart; "Choosing your privacy budget"
  in plain words; threat model; how to read the report; a clinic recipe and an app recipe; low-bandwidth
  settings; "what this does not protect against"; glossary. All snippets run in CI; the website's copy lint
  is applied to the docs. **(added)** partner kit: a template for declaring feature bounds, a one-page
  "privacy label" for a trained model, and `obscuradp doctor`, a dry run on synthetic data that checks a
  partner's setup without touching real records.
- **S22** items 13 to 26 of `docs/WEBSITE_MISMATCHES.md`; extend `website/scripts/sync-data.mjs` with Trial
  07/27 and benchmark results; new `/research` articles and charts; targets shown beside measured values;
  **(added)** `website/src/data/claims.json` and a build check that every number in copy matches its results
  field (G23). Build sdist and wheel, `twine check`, TestPyPI, install test on three operating systems, a
  timed fresh-machine quickstart, then PyPI by Trusted Publishing **only after the owner's go-ahead**. v0.1 is
  labelled "research preview, not independently audited".

## Phase 7, after release (added)

An "Adopters" page and a no-telemetry template for partners to report protected record counts; issue
templates and starter issues for student contributors; a JOSS paper draft; v0.2 scope (threshold encryption
through OpenFHE once its Python wheel is portable, dropout-tolerant secure sums, accounted hyperparameter
tuning, adaptive clipping); moving `Experiment */` and `Trial 07/` under `research/` together with the
roughly 20 website links that point at them.

## Cut order

First to go first: Shakespeare; OpenFHE; JOSS draft; DP-chosen masks; adaptive clipping and server momentum;
Flower secure-aggregation mapping, then the whole adapter (needs the owner's consent, it is a recorded
decision); shadow-model attack; BFV backend; macOS and Windows for the extras; FEMNIST grid.

Never cut: the total-budget ledger, the privacy report, SMOTE refusal, comparison tests against the old
code, the threat model, the website corrections, the one-run audit, the written research plan.

## Verification per phase

- Phase 0: baseline run recorded; every cited `path:line` exists; only `docs/` changed.
- Phase 0.5 and 6: `npm run build && npm run lint:copy && npm run check:contrast`, then `npm run preview` and
  `npm run test:e2e` against it; claims check passes.
- Phase 1: clean `pip install -e ".[dev,he]"`; `pytest -m "not slow"` on three Python versions; comparison
  job equals the old code; ignore-guard passes.
- Phase 2: accountant ordering tests; a 20-round run's ledger total is at most the policy and
  `obscuradp report verify` reproduces it; the aggregator cannot decrypt; unsafe configurations raise;
  white-box check and mutation test behave as designed.
- Phase 3: masked sum equals the plain integer sum exactly (property test); byte counts come from serialized
  messages.
- Phase 4: auditor self-test.
- Phase 5: the research-plan tag predates every result commit; Flower and simulator reports match.
- Phase 6: documentation snippets run in CI; fresh-machine `pip install obscuradp` and quickstart under
  5 minutes.
