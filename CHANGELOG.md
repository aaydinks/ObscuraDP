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
- `obscuradp.tables`: encoding from declared ranges and categories, and each participant's own training,
  validation and test rows.
- `obscuradp.imbalance.ClassWeighted`: weights from a public figure that keep working under clipping.
- `obscuradp.unsafe`: settings that weaken the guarantee are refused unless `allow_unsafe=True`, and are
  then named in the privacy report.
- A first worked example in the README, run by the tests.
- `obscuradp.crypto.PairwiseMasking`: the default secure sum. The server learns the exact total of the
  participants' updates and no single update. Use it with `Plan(secure_sum=...)`.
- `obscuradp.compression`: updates rounded to a few bits per number and packed tightly; every run reports
  its measured upload size (44% of plain for three participants with 12-bit values).
- `obscuradp.crypto.ckks` (research only, refused unless `allow_unsafe=True`) and
  `obscuradp.selection.policy` (which numbers to encrypt, chosen without leaking through the choice).
- New dependency: `cryptography`. `dp-accounting` is limited to versions below 0.7 for now.
- Phase 2 results (`benchmarks/results/phase2/`, `docs/GATE_PHASE2.md`): the research experiments run again
  with one total privacy budget per run, no synthetic rows and held-out rows at their natural mix. They
  replace the earlier Trial 07/27 numbers, whose budgets were per round.

- `obscuradp.DistributedDPSGD(max_colluding=t, max_dropouts=d)`: the participants share the noise. Each
  adds a slice of standard deviation $\sigma C / \sqrt{N - t - d}$, one clipped gradient sum per round goes
  through the secure sum, and the server only sees the total. Refused without a secure sum. The privacy
  report gains a table of the budget for every real number of colluding participants, which
  `obscuradp report verify` recomputes.
- `DistributedDPSGD(noise="skellam")`: whole-number noise for the shared-noise mode. Each record is rounded
  before the noise is added and its length is checked in exact whole-number arithmetic; slices of the noise
  add up exactly. The budget is counted with the formula of Agarwal, Kairouz and Liu (2021), which the
  tests compare with an exact computation. Every row takes part in every round in this path.
- `DistributedDPSGD(sum_bound=...)`: an optional public limit on a participant's sums, for smaller
  messages. Every result reports the share of numbers it cut off.
- `obscuradp.ClientLevelDP` with `PrivacyPolicy(unit="client")`: a budget that protects a whole participant
  with all of its rows. With the secure sum every participant sends a message every round and adds a
  slice of the noise; without it the server adds the noise and the run is flagged. Warns below 1,000
  participants.
- Phase 3 results (`benchmarks/results/phase3/`, `docs/PHASE3_RESULTS.md`): every way of adding the noise
  compared at equal total budgets on the two research tables, 960 runs. A proposal for the default plan
  is in `docs/adr/0005-default-plan.md`.
- `obscuradp.CentralDPSGD`: a trusted server adds the noise. A baseline for comparisons, refused unless
  `allow_unsafe=True` and then flagged in the report.
- `obscuradp.compression.SumGrid`: whole numbers for a secure sum that never cut a value off (values wrap
  around the modulus), with a grid planned from public numbers.
- `Simulation(dropouts=...)` plays missing participants in the shared-noise mode. A round that fails is
  charged to everyone who sent a message.
- `docs/SHARED_NOISE.md` (the argument behind the shared-noise mode and what is not yet proven) and
  `docs/DATASETS.md` (source and license of the two dataset files, as far as they could be found).

- `obscuradp.audit.membership`: an attack test on a whole training run that trains only once (the method
  of Steinke, Nasr and Jagielski, 2023). About a quarter of each participant's rows become test records,
  each kept or left out by a fair coin; two attacks guess the coins from what their attacker can see.
  Every result shows the proven limit for any method, the proven limit for the noise used and the
  measured value side by side, with the most the test could have shown.
- `obscuradp.audit.mechanism`: the strongest-attacker checks of the private training step, the two
  shared-noise modes and the whole-participant mode, which lived in the tests, as public functions.
- `obscuradp.audit.views`: what the server (with any colluding participants) and a model observer may see
  of a round. `obscuradp.audit.one_run_epsilon_lower_bound` and `one_run_p_value`: the paper's routine.
- `obscuradp audit self-test` (`obscuradp.audit.selftest`): the auditor is shown a correct and a broken
  pipeline and must tell them apart.
- `Simulation.run(observer=...)` hands an audit everything that was sent in each round (a `RoundRecord`),
  and `Simulation(declared_rows=...)` fixes the participants' public sizes so that leaving a row out
  changes nothing an observer could read off directly. Runs that use neither are unchanged.
- Phase 4 results (`benchmarks/results/phase4/`, `docs/PHASE4_ATTACK_TESTS.md`): attack tests on the two
  research tables, 960 runs, with the measured attacker advantage at every budget.

### Changed

- Root `.gitignore` rules are anchored to the repository root so they can no longer hide source folders.
- With `secure_noise=True` the choice of rows for every private step now comes from the operating system
  too, not only the noise. Runs with seeded noise are unchanged.

### Security

- Nothing released yet. The research folders must not be used with real data; see `docs/AUDIT.md`.
