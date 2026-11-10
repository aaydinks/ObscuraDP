# Roadmap status

Read this file first in every session, then `docs/ROADMAP.md`. Update it at the end of every session.

**Current state (2026-11-10): S1 and S3 to S7 are done (Phase 0 and Phase 1 complete, Phase 2 started). The
next session is S8** (the library's own private training step and its safety tests). No gate is open. Each
step's pull request is merged by the agent once all its checks are green (owner's decision, 2026-11-10);
work stops at the S11 gate.

## Decisions (Phase 0 gate, answered by the owner on 2026-11-10)

| # | Decision | Answer |
|---|---|---|
| 1 | License | Apache-2.0 |
| 2 | Default way to hide individual updates from the server | Pairwise masking over integers; exact BFV optional; CKKS `research_only` (`adr/0003-secure-sum-default.md`) |
| 3 | Privacy unit and counting | One record; one total budget per run; PLD accountant with RDP cross-check (`adr/0002-privacy-unit-and-budget.md`) |
| 4 | Class imbalance | SMOTE refused inside private training unless forced; class weighting instead (`adr/0004-imbalance-without-smote.md`) |
| 5 | Dataset files in the repository | Stay for now; add source and license notes if they can be found; never shipped in the package |
| 6 | Wording of the attacker-advantage and bytes targets | As proposed in `KPI_MAPPING.md`: a measured target beside two ceilings; bytes counted against 32-bit plain messages |
| 7 | The 12 website statements that are false today | Fixed now, in a separate pull request merged automatically once its checks passed (aaydinks/ObscuraDP#9) |
| 8 | Release label for v0.1 | "Prerun + novel research action" (owner's wording) |

## Session checklist

- [x] **S1, Phase 0.** Audit, baseline reproduction, threat model, target definitions, website wording
      list, risk list, decision records, roadmap. aaydinks/ObscuraDP#8. Gate closed 2026-11-10.
- [ ] S2, spare for changes requested at the gate
- [x] S3, Phase 0.5: `LICENSE` and website items 1 to 12. aaydinks/ObscuraDP#9, merged as `37b924c`
- [x] S4, Phase 1: packaging, tooling, CI. aaydinks/ObscuraDP#8, merged as `fbbf42d`
- [x] S5, Phase 1: comparison harness against the old code; port config, models, data, metrics, aggregation.
      aaydinks/ObscuraDP#21, merged as `544ffba`
- [x] S6, Phase 1: port the rest; old loop kept verbatim in `obscuradp.legacy`; smoke checks as tests;
      results manifest. aaydinks/ObscuraDP#23, merged as `b62ad69`
- [x] S7, Phase 2: policy, ledger, report, accountants, `obscuradp report verify`
- [ ] S8, Phase 2: own DP-SGD step with exact sampling and its safety tests
- [ ] S9, Phase 2: validation split, declared-bounds preprocessing, class weighting, unsafe flags, `Simulation`
- [ ] S10, Phase 2: secure-sum interface, pairwise masking, quantizer, key roles, masks
- [ ] S11, Phase 2: corrected re-runs, old against new. **GATE**
- [ ] S12, Phase 3: distributed and central modes
- [ ] S13, Phase 3: integer noise path
- [ ] S14, Phase 3: equal-budget comparison, measured bytes, default plan
- [ ] S15, Phase 4: membership audit and nightly check
- [ ] S16, Phase 4: shadow models, attribute inference, gradient inversion
- [ ] S17, Phase 5: datasets, synthetic generator, runner, pilot, written research plan (tagged)
- [ ] S18, Phase 5: benchmark runs
- [ ] S19, Phase 5: Flower adapter
- [ ] S20, Phase 5: analysis and claims registry
- [ ] S21, Phase 6: documentation and partner kit
- [ ] S22, Phase 6: release engineering and website sync. **GATE** before publishing

## S7 log (2026-11-10)

### What was done

- **`obscuradp.accounting`**: a record of each noisy release (`GaussianEvent`, `SubsampledGaussianEvent`),
  three independent ways to turn a list of releases into a budget (PLD upper and lower estimates and RDP,
  all on Google's `dp-accounting`, plus the exact formula for plain Gaussian noise), noise calibration to a
  **total** budget ($S = \lceil E / q \rceil$ steps per round, $T = R \cdot S$ for the run; closes G2 and G6
  for new code), the two attacker-advantage ceilings of `docs/KPI_MAPPING.md`, and a cross-check that stops
  a run when the three budget figures contradict each other.
- **`PrivacyPolicy`** (one total budget, what it protects, optional stricter budgets per participant),
  **`PrivacyLedger`** (every release is charged before it happens; a release that would exceed the budget
  raises `BudgetExceeded` and is not recorded; a whole run can be planned in advance so later charges cost
  nothing to check) and **`PrivacyReport`** (plain-words summary, the stored releases, the three budget
  figures, both ceilings, unsafe flags and a fixed list of what no report covers; schema
  `obscuradp.privacy_report/1`).
- **Command line**: `obscuradp report verify FILE` recomputes a stored report from its recorded releases
  (exit status 0 consistent, 1 not, 2 unreadable) and `obscuradp report show FILE`. `python -m obscuradp`
  works too.
- New core dependency `dp-accounting>=0.6` (it brings `absl-py`, `attrs<24` and `dm-tree`).
  Both pin files gained those entries; no earlier pin changed.
- **Tests**: 95 more, for 348 in total.

### What was verified and how

- 348 tests pass with the pinned versions (Python 3.11), coverage of `src/obscuradp` 97%. With the newest
  versions (Python 3.13) 206 pass and the 119 comparison tests skip.
- **Exact answer reproduced**: without subsampling the PLD upper estimate matches the closed formula within
  0.1% at four noise levels, and the formula itself is checked by hand at one point
  ($\delta(\varepsilon = 1, \sigma = 1) = \Phi(-0.5) - e\,\Phi(-1.5) = 0.126937$).
- **The three accountants agree** on a grid of 12 settings (noise multiplier 1 and 3; sampling rate 0.01,
  0.145 and 1; 60 and 412 steps): lower estimate at most upper estimate, both at most the RDP limit, upper
  and lower close.
- **Table 1 of `docs/KPI_MAPPING.md` reproduced**: all 20 PLD cells within 0.02 (slow test, 54 s), and
  three of them on every run. A total budget of 1 for the loan setting needs a noise multiplier of 6.39
  (6.93 with RDP).
- **Table 3 reproduced**: at a total budget of 1 the ceiling for any method is 46.2% and the ceiling for
  these exact releases is 10.6%.
- **Ledger**: a run calibrated to a budget of 1 over 10 rounds spends between 0.99 and 1.0; an eleventh
  round is refused and leaves no trace; scopes are independent; a planned run is charged without a single
  recomputation (counted); the looser RDP accountant refuses a run that PLD accepts.
- **Report**: a stored report verifies; editing the headline, a scope figure, the stored noise level, the
  budget or the within-budget flag is detected, each with its own message.
- The tests can fail: counting the budget for one round only (the old defect) broke 4 tests; letting the
  ledger accept ten times its budget broke 4 tests. Both changes were reverted.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass. The wheel registers the `obscuradp` command.

### Deviations from the plan

- The planned files `accounting/pld.py` and `accounting/rdp.py` are one file, `accounting/accountants.py`,
  because they share the event handling.
- **"RDP inside a band above PLD" is a warning, not an error.** The band (RDP at most 1.35 times PLD) holds
  for the long calibrated runs the library plans (measured 1.08 to 1.24), and a test keeps it there. It is
  not a law: for a short run (60 steps, sampling rate 0.01, noise multiplier 1) RDP is 1.88 times PLD with
  both correct. Raising an error there would have stopped honest runs, so `CrossCheck.warnings()` reports
  it and the report summary prints it as a note. The rules that do raise are the ones that cannot break if
  every accountant is right.
- **Budgets above 200 are reported as unbounded.** The PLD method needs memory in proportion to the budget
  (47 GiB in one trial at a tiny noise level). A budget that large protects nothing, so it is returned as
  infinite without being computed, and any ledger refuses it.
- The lower PLD estimate uses a finer grid for long runs (down to $2 \cdot 10^{-6}$). At the coarse grid its
  error grew with the number of steps and the gap rule failed for 4,660 steps.
- One cell of table 1 changed by 0.02 (stroke, 393 rows, budget 0.5: 15.67 became 15.65). The library's
  search stops closer to the budget than the Phase 0 script did; both values keep the run within budget.
  The document was corrected.
- Calibration searches with RDP first, then PLD. This keeps the PLD method away from tiny noise levels.

### Open items

- The ledger is not yet called by any training code. S8 (the private step) and S9 (the simulation) wire it
  in; until then nothing in the library releases anything.
- A report cannot be written for an unbounded budget (strict JSON has no infinity). No ledger can reach
  that state through `charge`; a ledger rebuilt from stored data could.
- Carried forward: the start-up script of the "ObscuraDP" cloud environment is untested by a fresh session;
  TestPyPI setup by the owner.

## S6 log (2026-11-10)

### What was done

- **`obscuradp.legacy` is now the whole Trial 07/27 pipeline**, copied with the settings passed in and the
  optional libraries imported lazily: `dp` (noise multiplier and both budget figures), `training` (one
  local round, plain or Opacus DP-SGD, with the FedProx term), `he` (CKKS context and aggregation),
  `sensitivity` (mask scoring), `aggregate` (the plain, encrypted and partly encrypted switch), `loop`
  (`run_stack`, line for line) and `bench`. Every known defect is kept and named in the docstrings with
  its finding number.
- **New clean modules** (numpy and torch only): `obscuradp.selection.masks`, `obscuradp.crypto.chunking`,
  `obscuradp.bench.payload`, `obscuradp.seeding` (fixes G17 for new code) and `obscuradp.results`
  (results schema v1 with a reproducibility manifest; closes G26 for new results).
- **Tests**: 134 more, for 253 in total.
- **CI**: the comparison job runs the whole suite once and fails if coverage of the library is under 80%.
- The dependency bot no longer proposes TypeScript major versions for the website (see below).

### What was verified and how

- 253 tests pass with the research code's pinned versions (Python 3.11, CPU-only build), coverage of
  `src/obscuradp` 97%, in under 3 minutes. With the newest versions and no optional packages (Python 3.13)
  111 unit tests pass and the comparison tests skip.
- **Bit for bit equal to the research code** (same process, one thread):
  - noise multipliers, per-round and whole-run budgets and noise level for 6 participant sizes and 4
    budgets;
  - the weights after one local round for both datasets, plain and private, with and without the FedProx
    term;
  - mask scores and masks in the three modes at four fractions, and every mask helper;
  - the complete result record of `run_stack` (every metric of every round, every privacy number, the
    mask and size fields; only wall-clock times excluded) for the `plain` and `dp` stacks on both datasets
    with both algorithms, plus two other budgets, another seed and a wider model.
- **Encrypted stacks** (`he`, `dp_he`, `dp_spe`, both datasets, three mask modes): settings, privacy
  numbers, mask, chunk count and encrypted fraction exactly equal; ciphertext bytes within 0.1%; metrics of
  every round within $10^{-3}$; the encrypted average within $10^{-4}$ of the plain one in a free 3-round
  run. With the same recorded client states fed to both aggregators, results agree within $10^{-6}$ with
  each other and with the plain reference, and entries averaged in the clear are bit for bit equal.
- The research smoke checks are now tests, including two the research code did not have: unequal weights
  really are applied under encryption (the result differs from the unweighted mean by more than $10^{-2}$),
  and a second multiplication on a ciphertext fails with "scale out of bounds", as the write-up states.
- The four sha256 prefixes printed in `Trial 07/27/VERIFICATION.md` are compared with the files by a test
  (closes G19).
- The comparison tests can fail: changing the per-participant seed formula by one digit in the copied loop
  made all 14 selected loop comparisons fail. The change was reverted.
- `ruff check`, `ruff format --check`, strict `mypy` (on Python 3.11 and 3.13), the ignore-rule guard, the
  research fingerprint check, the package build and the release-contents check pass.

### Deviations from the plan

- The copied code lives in the subpackage `obscuradp.legacy` (see the S5 log), with the loop in
  `legacy/loop.py`.
- The copied `make_mask` refuses the random mode without a seed. The research code would silently draw an
  unseeded mask there. The loop always passes a seed, so no result changes.
- The copied `get_sigma` cache also keys on $\delta$, so changing $\delta$ can no longer return a stale
  value. Results for the research $\delta$ are unchanged (tested).
- The third comparison tier (frozen numbers for macOS and Windows) is still not needed: the comparisons
  run live on Linux, and the unit tests of the clean modules run on all three systems.

### Pull requests cleared before this step (2026-11-10)

| PR | What | Outcome |
|---|---|---|
| aaydinks/ObscuraDP#10 to #14 | GitHub Actions version bumps from the dependency bot | Merged in order, each with all checks green. #11 conflicted with #10; a helper agent merged `main` into its branch, keeping both bumps, and its checks passed again before the merge. |
| aaydinks/ObscuraDP#15, #16, #18, #19, #20 | Version bumps that touch the frozen research folders or the pins that reproduce the old results | Closed unmerged (owner's decision). The bot configuration now excludes those paths. |
| aaydinks/ObscuraDP#17 | TypeScript 5.9 to 7.0 for the website | Closed unmerged. A helper agent reproduced the failure: the Astro type checker `@astrojs/check` 0.9.12 (the newest version of any kind on 2026-11-10) accepts TypeScript 5 and 6 only, so `npm ci` cannot resolve. No fix exists without forcing an unsupported combination. The bot now ignores TypeScript major versions; the rule should be removed when the checker supports TypeScript 7. A bump to TypeScript 6 builds cleanly and could be done separately. |
| aaydinks/ObscuraDP#21 | Step 5 | Merged as `544ffba` after `main` was merged in and all 9 checks passed again. |

Left for the owner: the merged remote branch `cu/trusting-pasteur-sms0ou-s03-website` can be deleted.

### Open items

- The start-up script of the "ObscuraDP" cloud environment has still not been exercised by a fresh session.
- TestPyPI setup by the owner (see the S4 log).

## S5 log (2026-11-10)

### What was done

- **Comparison harness** (`tests/golden/conftest.py`): imports the Trial 07/27 package `obscura` from its
  folder and runs its functions with that folder as the working directory. The library never imports it
  (a lint rule and a test enforce this).
- **Frozen reference**: `tools/freeze_research.py` and `tests/golden/research_manifest.json` hold a sha256
  fingerprint of all 125 tracked files in `Experiment 1` to `3` and `Trial 07`. A test and a CI step fail
  if any of them changes without the list being rewritten on purpose.
- **New library modules** (numpy and torch only):
  - `obscuradp.state`: `get_state`, `set_state`, `flatten_state`, `unflatten_state`, `state_layout`.
    Buffers are part of the vector, and integer entries are rounded (closes the buffer part of G20).
  - `obscuradp.models.TabularMLP`: any number of hidden layers, optional probability output.
  - `obscuradp.partition`: `iid_partition`, `dirichlet_partition`, with a typed description of the split.
  - `obscuradp.aggregation`: `fedavg_weights`, `weighted_mean`, `aggregate_states` (in the clear).
  - `obscuradp.metrics`: `binary_metrics`, `roc_auc`, `average_precision`, `evaluate`, `summarize`.
    No scikit-learn needed. `summarize` reports the sample standard deviation (G22).
- **`obscuradp.legacy`** (comparison only, needs the `legacy` extra): `LegacyTrialConfig` (all 46 research
  settings as one frozen object), `data` (loading, pooled scaling, split, resampling, `prepare`), `models`,
  `metrics`. It reproduces the research code's known defects on purpose and says so in its docstrings.
- CI: a new job "Comparison with the research code" installs the research code's pinned versions and runs
  the comparison tests with skips turned into failures.

### What was verified and how

- 119 tests pass locally on Python 3.11 with the research code's pinned versions: 67 unit and property
  tests, 52 comparison tests. Coverage of `src/obscuradp` is 95% overall.
- **Bit-for-bit equal to the research code** (arrays with `array_equal`, states with `torch.equal`,
  numbers with `==`):
  - all 46 settings;
  - uneven and even splits for seeds 42 to 46;
  - loading of both tables; split, scaling, partition and per-participant rows for both datasets at seeds
    42 and 44, with and without resampling;
  - initial weights, state keys, flat vector, layout, rebuild from a flat vector and forward pass for the
    stroke and loan networks and a 4-times-wider one;
  - averaging weights, weighted and unweighted mean, and the averaged state;
  - all 8 evaluation measures on both test sets.
- The numpy-only measures agree with scikit-learn within $10^{-12}$ on 24 random cases, with and without
  tied scores.
- The comparison tests can fail: removing one shuffle from the split made 8 of them fail. A second change
  (computing the weighted mean as an explicit sum) did not change any bit, so it was correctly not flagged.
- `ruff check`, `ruff format --check` and strict `mypy` pass.

### Deviations from the plan

- The plan named a single file `sim/legacy.py`. It is a subpackage, `obscuradp.legacy`, because the copied
  research code spans settings, data, models and evaluation. The old training loop joins it in S6.
- The plan said "port data". The research data preparation is defective by design (pooled scalers,
  resampling), so it was ported only into `obscuradp.legacy`. The library's own data handling comes with
  the declared-bounds preprocessing in S9 and the dataset downloaders in S17.
- The plan's third comparison tier (frozen numbers for macOS and Windows) is not needed yet: everything
  ported so far is compared live against the research code in the same process.
- Coverage of `src/obscuradp` is measured but not yet enforced in CI. The bar is set in S6.

### Open items

- `tools/bootstrap.sh` finished on the cloud machine with exit code 0 after 23 minutes 39 seconds, almost all
  of it downloading GPU libraries through the fallback path (the environment is 5.6 GB). In that
  environment (Python 3.13, newest dependencies, no optional extras) the unit tests pass, the comparison
  tests are skipped as designed, and lint, format and strict types pass. It surfaced one type error that
  only appears with the newest numpy; it is fixed.
- GitHub Actions on aaydinks/ObscuraDP#21 at `87c6155`: all 10 checks green. The comparison job ran 52 tests
  with skips turned into failures.
- The dependency bot configured in S4 opened 11 pull requests when S4 merged, four of them against the
  frozen research folders. Its configuration now excludes those folders and groups updates. The owner
  decides what to do with the pull requests already open (aaydinks/ObscuraDP#10 to #20).
- **Cloud environment "ObscuraDP" created (owner approved, 2026-11-10).** It allows the two hosts the
  CPU-only PyTorch build is served from (`download.pytorch.org` for the index, `download-r2.pytorch.org` for
  the files) and runs `tools/bootstrap.sh` when a session starts. Measured in a fresh clone on this machine
  with both hosts allowed: 1 minute 12 seconds and 1.2 GB, against 23 minutes 39 seconds and 5.6 GB before.
  The setup script has not yet been exercised by a fresh session starting on that environment. The saved
  Cloudflare deploy token stays on the Default environment and must be added again to "ObscuraDP" if the
  website is ever deployed from a session there.

## S4 log (2026-11-10)

### What was done

- `pyproject.toml` (hatchling, Python 3.11 or newer, core dependencies numpy and torch, extras `he`,
  `openfhe`, `flower`, `audit`, `bench`, `legacy`, `docs`, `dev`) and a typed `src/obscuradp` skeleton at
  version `0.1.0.dev0`.
- Tests in `tests/`: version and import; network access switched off for every test (`--disable-socket`),
  with a test that the guard is active; ignore-rule guard; the library never imports the research folders;
  no em dashes in library, tests or documents.
- `tools/`: `bootstrap.sh` (one-command setup), `check_gitignore.py`, `check_release_contents.py` (a release
  holds no data, experiments, results, documents or website), `update_constraints.sh`.
- Root `.gitignore`: 21 directory rules anchored to the repository root.
- CI: `python-ci.yml` (lint, format, types, ignore rules; tests on Python 3.11, 3.12, 3.13 on Linux and 3.12
  on macOS and Windows; package build and content check), `python-nightly.yml` (slow and integration tests
  with the old code's pins, tests against the newest dependencies, `pip-audit`), `release-dry-run.yml`
  (manual; TestPyPI upload needs a one-time setup by the owner), `dependabot.yml`.
- `README.md` finished; `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`,
  `CITATION.cff`, `.pre-commit-config.yaml`.
- `constraints/dev.txt` (pinned development versions, resolved for every supported Python) and
  `constraints/legacy-py311.txt` (the environment that reproduced the Trial 07/27 results in S1).

### What was verified and how

- Locally on Python 3.11, 3.12 and 3.13: 9 tests pass; `ruff check`, `ruff format --check`, `mypy` (strict)
  and `tools/check_gitignore.py` pass. Locally the package was installed without its heavy dependencies to
  save time; the full install is covered by CI.
- Package build: sdist and wheel build, `twine check --strict` passes, and the content check passes (the
  wheel holds the package and its metadata only).
- GitHub Actions on aaydinks/ObscuraDP#8 at commit `0800854`: all 8 checks green, including the full
  install with CPU-only PyTorch and the tests on Linux, macOS and Windows.
- `tools/bootstrap.sh`: see "Open items" below.

### Deviations from the plan

- The planned "legacy and current lockfiles" are constraint files, not hash-pinned lockfiles. Hash pinning
  is deferred to the release step (S22), where it matters for supply-chain checks.
- `LICENSE` came from S3 and was merged into this branch from `main`.
- Coverage is not enforced yet: the package has six lines. The 80% bar applies from S6.

### Open items

- `tools/bootstrap.sh` works on this machine but takes the fallback path (the CPU-only download host is
  blocked here): 23 minutes 39 seconds, 5.6 GB. See the S5 log.
- The cloud environment for this repository now exists. See the S5 log.
- TestPyPI: the owner needs to register the name `obscuradp` on test.pypi.org with this repository's
  `release-dry-run.yml` as a trusted publisher, and create a GitHub environment named `testpypi`, before the
  upload option of the dry run can work.

## S3 log (2026-11-10)

Done by a separate agent in aaydinks/ObscuraDP#9, merged as `37b924c` after its checks passed.

- `LICENSE`: the unmodified Apache License 2.0 text.
- Website items 1 to 12 of `WEBSITE_MISMATCHES.md`. Final wording is listed in the pull request.
- Checks reported: website build (0 errors), copy lint, contrast (17 pairs), 42 behaviour checks against the
  local preview, screenshots at desktop and phone width, and the hosted "Website CI" job.
- Left for S22: two missing spaces in rendered text and a sideways-scrolling table on phones (items 27 and
  28 of `WEBSITE_MISMATCHES.md`); the per-round note for Experiment 1 says the total is "several times
  larger" without a computed figure.

## S1 log (2026-11-10)

### What was done

- Audited `Experiment 1` to `3`, `Trial 07/27`, the repo root and the website against `main` at `34071be`.
  All of G1 to G11 from the prompt are confirmed; G12 to G27 are new. See `AUDIT.md`.
- Reproduced the Trial 07/27 baseline (below).
- Computed three reference tables with `dp-accounting 0.6.0` (see `KPI_MAPPING.md`).
- Wrote `THREAT_MODEL.md`, `KPI_MAPPING.md`, `WEBSITE_MISMATCHES.md`, `RISK_REGISTER.md`, `ROADMAP.md`,
  `adr/0001` to `adr/0004`, and revised `NEXT_STEPS_PROMPT.md` (revision 2, five binding corrections).
- No code, experiment or website file was changed.

### Baseline reproduction

Environment: Python 3.11.18, Linux x86_64, 4 CPU cores, packages exactly as pinned in
`Trial 07/27/requirements.txt`. The full resolved list is in `constraints/legacy-py311.txt`; S4 turns it into
the lockfile for the comparison tests. Everything ran in a scratch copy, so no committed result file was
touched.

| Check | Outcome | Time |
|---|---|---|
| `run_smoke.py` | 7 of 7 checks pass | 14 s |
| `verify_claims.py` | Exit code 0; all 9 claim verdicts equal the documented ones; the four result fingerprints equal those printed in `VERIFICATION.md` | 13 s |
| `run_arms.py` regenerated (20 runs, seed 42) | See below | 7 min 53 s |

Regenerated arm matrix against the committed `Trial 07/27/results/arms.json`:

- Same 20 runs, same `meta` block.
- **All 8 runs without encryption (`plain` and `dp`, both datasets, both algorithms) are bit-for-bit
  identical** in every metric and in every per-client noise multiplier.
- Across all 20 runs, accuracy, recall, precision, F1 and predicted-positive rate are identical.
- 8 of the 12 encrypted runs differ in loss, AUC or average precision by at most $7 \times 10^{-6}$. This is
  expected: the encryption library's randomness cannot be seeded (G17).
- The only structural difference is a field the current code adds, `crypto.wire_payload_ratio` (G26).
- Sum of per-run training time: 451 s now against 430 s recorded.

Conclusion: the committed Trial 07/27 results are reproducible from the committed code and pins. This makes
the old code a usable reference for the Phase 1 comparison tests, with bitwise checks on the unencrypted
runs and a tolerance of about $10^{-5}$ on continuous metrics for the encrypted ones.

Not reproduced in this session (time): the frontier, losslessness, sensitivity, overhead and multi-seed
drivers, and Experiments 1 to 3. `verify_claims.py` re-derives the key numbers of the first four
independently and passed.

### Environment notes for later sessions

- (Resolved on 2026-11-10 by the "ObscuraDP" cloud environment, see the S5 log.) **The CPU-only torch
  download host (`download.pytorch.org`) was blocked** by the Default cloud environment's network rules
  (HTTP 403 from the proxy). `pip install torch` from PyPI works but pulls GPU libraries: the
  environment is 5.1 GB and takes several minutes to build. S4 should propose either allowing that host or a
  setup script that prepares the environment once. This is a request for the owner to approve, not something
  a session should work around.
- TenSEAL 0.3.16, Opacus 1.6.0 and `dp-accounting 0.6.0` all installed from PyPI without problems on
  Python 3.11 and 3.12. `dp-accounting` forces `attrs` 23.2.0.
- The reference environment and the scratch copy lived in the session scratchpad and are gone. Rebuild from
  `constraints/legacy-py311.txt`.

### Dependency versions (PyPI, 2026-11-10)

| Package | Latest | Pinned in Trial 07/27 | Note |
|---|---|---|---|
| torch | 2.15.0 | 2.13.0 | |
| opacus | 1.6.0 | 1.6.0 | |
| tenseal | 0.3.18 | 0.3.16 | Requires Python 3.11 or later |
| dp-accounting | 0.6.0 | not used yet | Pins `attrs<24` |
| flwr | 1.43.0 | not used yet | Privacy features labelled preview |
| flwr-datasets | 0.6.1 | not used yet | |
| openfhe | 1.6.1 | not used yet | Requires Python 3.12 or later; portability unverified |

### What was verified and how

- 152 `path:line` citations across the new documents were checked by script: every cited file exists and
  has the cited lines. A further 22 were checked for the quoted text itself.
- The noise table and both ceiling columns in `KPI_MAPPING.md` are computed, not estimated. The gains
  expected from the distributed mode (2 to 5 times less noise), the BFV message size and the benchmark
  run times in `ROADMAP.md` are estimates and are marked as such.
- `git diff --stat origin/main` shows only files under `docs/`.

### Changes on `main` during this session

- aaydinks/ObscuraDP#6 (the first version of the prompt) and aaydinks/ObscuraDP#7 (website styling and a
  website CI workflow) were merged. The branch was restarted from `main` at `34071be`, and all website line
  references were collected again on that commit. Finding G1 was adjusted: a workflow folder now exists, but
  it only checks the website.

### Open questions carried forward

1. Real row counts per client for the seed-42 stroke split in Trial 07/27 should be recorded in results;
   table 1 in `KPI_MAPPING.md` uses the sizes reported by Experiment 1's split as examples.
2. Whether the one-run audit has enough power on tables of about 4,000 rows (risk 13). Answered by the S15
   pilot.
3. Whether rounding updates to 8 bits costs accuracy at these model sizes. Answered in S14.
4. Whether a tag `research-baseline-v0` should be pushed at `34071be`. Not done: this session was limited to
   its own branch.
