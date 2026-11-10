# Roadmap status

Read this file first in every session, then `docs/ROADMAP.md`. Update it at the end of every session.

**Current state (2026-11-10): the Phase 0 gate is closed.** All eight decisions are answered (below). S3
(website statements that are false today, plus the license file) is being handled in its own pull request.
S4 (packaging, tooling, CI) is the session in progress.

## Decisions (Phase 0 gate, answered by the owner on 2026-11-10)

| # | Decision | Answer |
|---|---|---|
| 1 | License | Apache-2.0 |
| 2 | Default way to hide individual updates from the server | Pairwise masking over integers; exact BFV optional; CKKS `research_only` (`adr/0003-secure-sum-default.md`) |
| 3 | Privacy unit and counting | One record; one total budget per run; PLD accountant with RDP cross-check (`adr/0002-privacy-unit-and-budget.md`) |
| 4 | Class imbalance | SMOTE refused inside private training unless forced; class weighting instead (`adr/0004-imbalance-without-smote.md`) |
| 5 | Dataset files in the repository | Stay for now; add source and license notes if they can be found; never shipped in the package |
| 6 | Wording of the attacker-advantage and bytes targets | As proposed in `KPI_MAPPING.md`: a measured target beside two ceilings; bytes counted against 32-bit plain messages |
| 7 | The 12 website statements that are false today | Fixed now, in a separate pull request that is merged automatically once its checks pass |
| 8 | Release label for v0.1 | "Prerun + novel research action" (owner's wording) |

## Session checklist

- [x] **S1, Phase 0.** Audit, baseline reproduction, threat model, target definitions, website wording
      list, risk list, decision records, roadmap. Draft PR aaydinks/ObscuraDP#8. Gate closed 2026-11-10.
- [ ] S2, spare for changes requested at the gate
- [ ] S3, Phase 0.5: `LICENSE` and website items 1 to 12. In progress in its own pull request
- [ ] S4, Phase 1: packaging, tooling, CI
- [ ] S5, Phase 1: comparison harness against the old code; port config, models, data, metrics, aggregation
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
`Trial 07/27/requirements.txt`. The full resolved list is in `docs/legacy-environment.txt`; S4 turns it into
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
  `docs/legacy-environment.txt`.

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
