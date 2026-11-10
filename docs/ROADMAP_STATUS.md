# Roadmap status

Read this file first in every session, then `docs/ROADMAP.md`. Update it at the end of every session.

**Current state (2026-11-10): S1, S3, S4 and S5 are done. The next session is S6** (port the private
training step, encryption, masks and benchmarks; keep the old loop verbatim in `obscuradp.legacy`; results
manifest). No gate is open.

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
- [x] S5, Phase 1: comparison harness against the old code; port config, models, data, metrics, aggregation
- [ ] S6, Phase 1: port the rest; `sim/legacy.py`; smoke checks as tests; results manifest
- [ ] S7, Phase 2: policy, ledger, report, accountants, `obscuradp report verify`
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
- The cloud environment proposal (S4) is still to be made. It would cut the 23-minute setup to about a
  minute.

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
- A cloud environment for this repository (allowing the CPU-only PyTorch host and running the bootstrap
  script at start) is proposed to the owner separately. It needs the owner's approval.
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

- **The CPU-only torch download host (`download.pytorch.org`) is blocked** by this cloud environment's
  network rules (HTTP 403 from the proxy). `pip install torch` from PyPI works but pulls GPU libraries: the
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
