# Roadmap status

Read this file first in every session, then `docs/ROADMAP.md`. Update it at the end of every session.

**Current state (2026-11-11): S1 and S3 to S14 are done. Phase 3 is complete.** The Phase 2 gate is closed
(the owner answered every decision on 2026-11-10, second table below). One decision waits for the owner
without blocking anything: the default plan, proposed in `adr/0005-default-plan.md`. Phase 4 (attack
tests, S15) is next.
The next stop for the owner is S22; nothing is published before it.

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

## Decisions (Phase 2 gate, answered by the owner on 2026-11-10)

| # | Decision | Answer |
|---|---|---|
| 1 | Reversed and withdrawn claims | Accepted as written in `docs/GATE_PHASE2.md`. The website wording is fixed in S22 (items 29 to 32 of `WEBSITE_MISMATCHES.md`), not earlier |
| 2 | Headline measure for stroke | AUC and average precision on held-out rows at their natural mix. Recall only at a cut-off chosen on validation rows. Recall at a fixed 0.5 cut-off is dropped |
| 3 | Model for small tables | The documentation says logistic regression is the better protected choice for tables of a few thousand rows. The network stays as a comparison |
| 4 | Phase 3 | As planned: S12 shared noise, S13 whole-number noise, S14 comparison at equal budgets |
| 5 | The two dataset files | Stay in the repository, with a note on source and license as far as they can be found (`DATASETS.md`). Never shipped in the package |
| 6 | How far a session may go | Same rules as before: one pull request per step, merged by the session when every check is green, step after step until the next review stop (S22) |

Answers 2 and 3 change wording and report columns when S14 and S21 reach them. The stored Phase 2 results
are a record of what was run and are not rewritten.

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
- [x] S7, Phase 2: policy, ledger, report, accountants, `obscuradp report verify`. aaydinks/ObscuraDP#24,
      merged as `249308b`
- [x] S8, Phase 2: own DP-SGD step with exact sampling and its safety tests. aaydinks/ObscuraDP#25, merged
      as `77071f1`
- [x] S9, Phase 2: validation split, declared-bounds preprocessing, class weighting, unsafe flags,
      `Simulation`. aaydinks/ObscuraDP#26, merged as `c188641`
- [x] S10, Phase 2: secure-sum interface, pairwise masking, quantizer, key roles, masks.
      aaydinks/ObscuraDP#28, merged as `c3d713e`
- [x] S11, Phase 2: corrected re-runs, old against new. aaydinks/ObscuraDP#29, merged as `3a1b0c3`.
      Gate closed 2026-11-10
- [x] S12, Phase 3: distributed and central modes, the owner's answers, the dataset note.
      aaydinks/ObscuraDP#31, merged by the owner as `c4e559d`
- [x] S13, Phase 3: integer noise path. aaydinks/ObscuraDP#32, merged as `5f09379`
- [x] S14, Phase 3: equal-budget comparison, measured bytes, per-participant mode, default plan
      (proposed). PRNUMBER14
- [ ] S15, Phase 4: membership audit and nightly check
- [ ] S16, Phase 4: shadow models, attribute inference, gradient inversion
- [ ] S17, Phase 5: datasets, synthetic generator, runner, pilot, written research plan (tagged)
- [ ] S18, Phase 5: benchmark runs
- [ ] S19, Phase 5: Flower adapter
- [ ] S20, Phase 5: analysis and claims registry
- [ ] S21, Phase 6: documentation and partner kit
- [ ] S22, Phase 6: release engineering and website sync. **GATE** before publishing

## S14 log (2026-11-11)

### What was done

- **`obscuradp.ClientLevelDP`** with `PrivacyPolicy(unit="client")`: a budget that protects a whole
  participant with all of its rows (the owner asked for it inside this step). Each participant takes
  part by its own secret coin, trains without noise and shortens its change to the shared model. With
  the secure sum every participant sends a message every round and adds a slice of the noise; without it
  the server adds the noise and the run needs `allow_unsafe=True`. One ledger entry for the group. It
  warns below 1,000 participants.
- **The comparison** (`benchmarks/phase3/`, results in `benchmarks/results/phase3/`): nine ways of running
  a federation at equal total budgets on both research tables, with the research network and logistic
  regression. 960 runs, ten seeds for stroke and five for loan, about four hours of computing spread
  over four cores. Settings were chosen on made-up tables only (1,728 short runs).
- **`docs/PHASE3_RESULTS.md`**: the results in plain words with their limits.
- **`docs/adr/0005-default-plan.md`**: a proposal for the default plan, **waiting for the owner**. Nothing
  in the library's defaults changed.
- **Tests**: 30 more for the new mode and the comparison scripts, for 734 in total.

### Results in short (details and limits in `docs/PHASE3_RESULTS.md`)

- **Stroke: sharing the noise is better than local noise** at the same total budget, measured against the
  better of two local arms: by 0.01 to 0.03 AUC for logistic regression from a budget of 1, and by 0.04 to
  0.05 for the network from a budget of 2 (ordinary noise) or 4 (whole-number noise). At a budget of 1
  logistic regression reaches 0.800 to 0.806 against 0.774 with local noise and 0.832 without protection.
  At budgets 0.5 and 1 for the network, and 0.5 for logistic regression, the averages favour sharing but
  the difference is not shown.
- **Loan: no difference that matters** (every row-level arm between 0.958 and 0.967).
- **Whole-number noise does about as well as ordinary noise.**
- **The trusted server is not clearly ahead of shared noise.**
- **Messages**: per round, shared ordinary noise is at 0.38 to 0.41 of plain size and whole-number noise
  at 0.57 to 0.69. Over a whole run the shared modes upload 2 to 14 times what local noise uploads,
  because they need 60 to 600 rounds.
- **Whole participants cannot be protected in a group of five** (AUC about 0.5 at a budget of 1).
- **8-bit values** cost 0.022 AUC in one of eight cells; 12-bit values cost nothing anywhere.
- Encrypted (CKKS) updates are about 235 kB each: 17 times plain for the stroke network, 59 times for the
  loan network.

### What was verified and how

- 728 tests pass in the pull-request selection with the pinned versions (Python 3.11), coverage of
  `src/obscuradp` 98%; 566 pass with the newest versions (Python 3.13). The 6 slow tests run nightly; the
  two new ones (the attack on a whole participant, and rebuilding the stored summary from all 960 runs)
  were run once by hand and pass.
- `python -m benchmarks.phase3.report` exits with status 0: 960 records validated, 930 privacy reports
  recomputed from their recorded releases, none over budget, unsafe flags only on the trusted-server
  runs (by design).
- **The new scripts measure what Phase 2 measured**: the local arm and the baseline at the five Phase 2
  seeds (120 runs) give exactly the stored Phase 2 numbers.
- The memo's tables are generated by `report.headline_markdown` and a test checks that the memo contains
  them exactly.
- **Whole-participant mode**: every refusal; no participant moves the sum by more than the clipping norm
  (property test); the opened sum carries $N$ slices and at least the charged noise after the assumed
  colluders remove theirs; a participant that pulls hard arrives shortened to the clipping norm; a
  failed round is charged; who takes part comes from the operating system with `secure_noise=True`. A
  full attack check (1,500 runs with and without the target participant) runs nightly: what it proves
  stays below the ledger's figure.
- The tests can fail: slices of half the size broke 3 tests; an unshortened change broke 2. Both changes
  were reverted.
- **Nothing else moved**: the final weights of five seeded runs of the earlier modes have the same sha256
  as on `main` before this step.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass (the results do not enter the package).

### Deviations from the plan

- **A ninth arm was added: local noise with its rounds chosen like the other arms.** Without it only the
  new modes would have had their number of rounds tuned. It turned out worse than the Phase 2 protocol at
  strict budgets on stroke, so **the yardstick became "the better local arm"** at each table, model and
  budget. It is picked after looking at the held-out rows, which can only favour local noise.
- **The decision record is a proposal, not a decision.** The roadmap says it "chooses the default from
  the table". Which plan the library runs by default is the owner's call; the record gives a
  recommendation (keep local noise as the default, recommend shared noise for groups of few small
  participants) and the alternative.
- The encryption arm is a measurement of message sizes only. What a model learns under it is what the
  local arm learns, so it was not run as an arm.
- The whole-participant mode has ordinary noise only, and with thousands of participants the pairwise
  secure sum will be slow to simulate. S17 has to deal with that.
- The comparison used one assumed colluder throughout.
- Loan has five seeds, not ten: its arms differ by less than 0.01 and it is not the case in doubt.

### Environment

- **A long job only makes progress while the session is working.** The settings search stood still for
  about an hour and a half while the session waited for it idle; the machine had been suspended and
  resumed. Later sessions should stay active (short waits in a loop) while a long computation runs, and
  make such jobs resumable, as `tune.py --cache` and `run.py` are.
- The owner switched planning mode on before S13 and S14; the session presented a plan each time and went
  on once it was approved.

### Open items

- **For the owner: `docs/adr/0005-default-plan.md`.** Keep local noise as the default (proposed) or make
  shared noise the default. Nothing waits for this before S22.
- Several chosen settings are the largest on their grid (600 rounds, a batch of 640, a learning rate of
  3). A wider search might favour the shared modes more, or other arms.
- S15: the attack tests, including the measured attacker advantage at a budget of 1.
- Website wording on message sizes: fewer bytes per round holds for some modes, not fewer bytes in total
  (for S22, with items 29 to 32 of `WEBSITE_MISMATCHES.md`).
- Carried forward for the owner: the license field on the two Kaggle pages (`docs/DATASETS.md`); the
  merged remote branches `cu/trusting-pasteur-sms0ou-s03-website` and `cu/trusting-pasteur-sms0ou`;
  TestPyPI setup before S22.

## S13 log (2026-11-11)

### What was done

- **`DistributedDPSGD(noise="skellam")`**: the shared-noise mode with whole-number noise. Each row's
  clipped gradient is rounded on its own, its length is checked in exact whole-number arithmetic against
  a fixed limit, the rounded rows are added up, and each participant adds a slice of Skellam noise (the
  difference of two Poisson counts). Slices of that noise add up exactly. Nothing is rounded after the
  noise, and no floating-point number carries it. The default noise stays `"gaussian"`; S14 decides.
- **The count** (`obscuradp.accounting.skellam`): the formula of Agarwal, Kairouz and Liu (2021,
  Theorem 3.5 and Corollary 3.6) at whole-number orders, added up over rounds, turned into a budget with
  the conversion of `dp-accounting`. `dp-accounting` below 0.7 has nothing of its own for this noise.
- **`SkellamEvent`** as a release record, `calibrate_skellam` for the noise level of a whole run, and the
  ledger, cross-check, report, collusion table and `obscuradp report verify` all handle it. For these
  releases the report says that one counting method is proven and shows, for comparison, what ordinary
  noise of the same size would cost by the tighter method.
- **`obscuradp.mechanisms.skellam`**: the noise, the rounding with the exact length check, and a planner
  that picks the grid so that rounding lengthens a row by at most 5%.
- **`DistributedDPSGD(sum_bound=...)`** (both noise kinds): an optional public limit on each number of a
  participant's sum, applied before the noise, for smaller messages. Results report the share of numbers
  it cut off.
- **Tests**: 77 more, for 704 in total.

### What was verified and how

- 700 tests pass in the pull-request selection with the pinned versions (Python 3.11), coverage of
  `src/obscuradp` 98%; 538 pass with the newest versions (Python 3.13). The 4 slow tests run nightly.
- **The formula was read in the paper** (arXiv:2110.04995, opened on 2026-11-11 after the owner allowed
  the host). It equals what was planned; the paper states it with the noise variance as the parameter.
- **Exit condition, the formula against brute force**: it is never below the exact value computed from
  the Skellam probabilities (42 pairs of noise level and shift at seven orders; 200 many-dimensional
  cases) and within 1% of it for ample noise. For one release the exact leak at the reported budget is
  below $\delta$.
- **No rounded row is ever longer than the limit**: property test over sizes, scales and rows up to 50
  times too long; rows that cannot be made to fit count for nothing.
- **The noise**: variance as stated; four slices compared with the exact probabilities of their sum; in a
  simulation the opened sum is whole numbers in grid units and carries $N$ slices; after the assumed
  colluders remove theirs, at least the charged variance remains.
- **Budget**: calibration lands between 0.995 and 1.0 of the budget and 2% less noise no longer fits; the
  ledger refuses one release too many; stored reports verify and edits of the variance, the sensitivity
  and the count are each caught.
- **Attack check** (three participants, one colluding, 2,000 runs with and without the target): what the
  attack proves stays below the ledger's figure and comes within 35% of it.
- **Nothing else moved**: the final weights of five seeded runs (local, local with the secure sum,
  unprotected, shared Gaussian noise with a failed round, trusted server) have the same sha256 as on
  `main` before this step.
- The tests can fail: slices a quarter of the needed size broke 2 tests; skipping the length check broke
  3; a formula 10% too small broke 23. Dropping only the formula's second term broke 1 test (see below).
  All changes were reverted.
- End to end: a five-participant run with one failed round prints the table and the whole-number
  sentence, has no "argued, not proven" sentence, and its report file passes `obscuradp report verify`.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass.

### What it costs (measured, five participants, total budget 1)

| | Gaussian noise (S12) | Whole-number noise (S13) |
|---|---|---|
| Counting | tight method (PLD), checked by two others | one proven method (RDP) |
| Rows per round | drawn at random | all |
| Noise for the same budget and the same full rounds | 1.00 | 1.14 (8% from the looser count, 5% from rounding) |
| Bits per number, stroke-sized group | 14 (ratio 0.44) | 19 to 22 (0.59 to 0.69); 15 to 19 with a declared limit of 30 clipping norms |
| Bits per number, loan-sized group | not measured | 18 to 20 (0.56 to 0.62); 15 to 18 with the same limit |
| Open step in the argument | rounding after the noise (argued, checked in one dimension) | none of that kind |

**The whole-number path does not meet the 50% bytes target as it stands** (`docs/KPI_MAPPING.md`). Which
path gives the better model at equal budget is S14's question; nothing here claims either.

### Deviations from the plan

- **Rounding is done per row, not on the sum.** The roadmap says "randomized rounding with an exact
  integer norm check". Checking the sum would bound what a whole participant can change, not what one row
  can. Rounding each row and checking it bounds one row exactly, which is the unit of this library.
- **The brute-force tests cannot see the formula's second term.** On every case tried the exact value
  lies even below the first term (the Gaussian part). The second term is kept because it is what the
  paper proves; a test pins the paper's form.
- **`sum_bound` was added** (not in the roadmap), because without it the message size of the new path
  could not be explored in S14.
- The "modulus-headroom planner" of the roadmap is `plan_integer_grid`: it sets the number of bits from
  the public limit on the sums plus ten standard deviations of a noise slice.
- No sampled rounds with this noise, as the roadmap fixed. A general result for sampled rounds exists in
  the literature and could lift this later; it was not attempted.

### Environment

- `arxiv.org` was added to the allowed hosts of the "ObscuraDP" environment (owner approved). Pages there
  can be fetched with `curl`; the built-in page reader still could not resolve the host in this session.
- The owner merged aaydinks/ObscuraDP#31 before the session did, right after its checks went green. The
  session works on as in the handoff: no plan stops, merge when all nine checks are green.

### Open items

- S14: settings tuned per mode, more than five seeds before any claim about stroke, the report columns
  of answer 2, measured bytes for every arm, and `docs/adr/0005-default-plan.md`. For the whole-number
  path: whether a coarser grid (more noise, fewer bits) or a declared limit on the sums is the better
  trade.
- The Poisson routine of numpy is not a certified exact sampler; the report says so.
- Carried forward for the owner: the license field on the two Kaggle pages (`docs/DATASETS.md`); the
  merged remote branches `cu/trusting-pasteur-sms0ou-s03-website` and `cu/trusting-pasteur-sms0ou`;
  TestPyPI setup before S22.

## S12 log (2026-11-11)

### What was done

- **The owner's answers are in the repository**: `docs/GATE_PHASE2.md` (status and an "Answers" section; the
  result tables are untouched), the second decisions table above, and `docs/DATASETS.md`.
- **`docs/DATASETS.md`** (answer 5): for both files the paths, sizes, row counts and fingerprints, and the
  source the research notes name (two Kaggle entries, fetched from a mirror whose address was never
  recorded). Both licenses are written as **"not found"**: the Kaggle pages cannot be opened from the cloud
  environment and a web search did not show the license field. The owner chose this over waiting.
- **`obscuradp.DistributedDPSGD(max_colluding=t, max_dropouts=d)`**: the participants share the noise. Every
  round is one step. Each participant sends one clipped gradient sum plus a slice of noise with standard
  deviation $\sigma C / \sqrt{N - t - d}$ through the secure sum; the server sees only the total. `t` has
  no default.
- **`obscuradp.CentralDPSGD`**: the same round with a trusted server that adds the noise. Baseline only.
- **The privacy report has a table** of the budget for every real number of colluding participants (the
  exit condition of this step), states $t$ and $d$, and `obscuradp report verify` recomputes the table.
  Reports of the local mode keep exactly their keys; the schema name is unchanged.
- **Budget arithmetic for one step per round**: the number of noisy releases is the number of rounds. One
  noise level serves all participants (every row joins with the same probability
  $q = \min(1, B / n)$, $n$ rows in total) and is sized for the strictest participant budget.
- **`obscuradp.compression.SumGrid`**: whole numbers for the secure sum that are never cut off.
- `obscuradp.client.private_sum_step` and `clipped_sum_step`, `obscuradp.mechanisms.shared_noise`,
  `Simulation(dropouts=...)`, the mode in `obscuradp report show`, `docs/SHARED_NOISE.md`.
- **Tests**: 92 more, for 627 in total.

### What was verified and how

- 623 tests pass in the pull-request selection with the pinned versions (Python 3.11), coverage of
  `src/obscuradp` 98%; 461 pass with the newest versions (Python 3.13). The 4 slow tests run nightly.
- **Refusals**: shared noise without a secure sum is refused when the plan is built, and `allow_unsafe`
  does not change that; so are non-default rounding settings, an impossible group ($N - t - d < 1$), more
  than one local pass, the FedProx term, equal weighting, and a trusted server without `allow_unsafe=True`.
  Partial encryption cannot be reached from a plan at all; a test fails if a field for it is ever added.
- **The opened sum carries the stated noise.** With a model whose gradients are zero, the sum is backed
  out of the model after one round (4,000 numbers): its variance is $N$ slices, and after removing the
  slices of the assumed colluders at least $\sigma^2 C^2$ remains (three settings of $t$ and $d$). The
  trusted server adds exactly $\sigma C$.
- **One budget**: every participant's ledger holds exactly $R$ releases at $(\sigma, q)$, spends between
  0.99 and 1.0 of a budget of 1, and the stored report verifies, also through the command line. With one
  stricter participant, its budget sets the noise for all.
- **The table**: the row for the assumed $t$ equals the report's total, budgets grow with the number of
  colluders, rows without protection are stored as `null` in strict JSON, and 14 different edits of the
  section are each caught by `verify_report`.
- **Measured for the stroke-sized group** (5 participants with 95, 1,589, 54, 1,240 and 456 rows, expected
  batch 160, 100 rounds, total budget 1, $t = 1$): noise multiplier 2.03 for everyone, 14 bits per number
  on the wire (44% of plain size), and budgets of 0.86, 1.00, 1.22, 1.67 and 3.18 for 0 to 4 colluders.
  For comparison only: in the local mode of Phase 2 the same participants needed multipliers of 3.5 to
  28.9, with many steps per round. **Whether this gives better models is not measured yet** (S14).
- **Dropouts**: a failed round leaves the model untouched and is charged to those who sent; every attempt
  uses its own masks; a run with failed rounds ends with fewer updates and within budget.
- **Two attacks with real keys**: a missing participant that works with the server lets the two read the
  others' sum exactly; a missing honest participant leaves everything hidden.
- **Attack check** (three participants, one colluding, full batch, 2,000 runs with and 2,000 without the
  target record): the attack proves no more than the ledger's 1.0 and comes within 30% of it.
- **Rounding**: in one dimension the sum of separately rounded noisy values has the same distribution as
  the true noisy sum plus small independent noise, rounded once, to the limit of double precision (below
  $10^{-10}$) at the planned grid. On a coarse grid the same computation shows a difference above
  $10^{-3}$.
- **The local mode is unchanged**: the final weights of three seeded runs (private, private with the
  secure sum, unprotected) have the same sha256 before and after the change, on Python 3.11 and 3.13.
- The tests can fail: sizing the slice as if everyone were honest broke 10 tests; letting one participant
  add no slice broke 3; not charging a failed round broke 3. All three changes were reverted.
- End to end: a five-participant run with one failed round prints the table, and its report file passes
  `obscuradp report verify`.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass.

### Deviations from the plan

Six points go beyond the roadmap's wording. An independent review by a helper agent reached the first
three as well. The argument is in `docs/SHARED_NOISE.md`.

- **A failed round is charged.** The roadmap said "charged only if a sum was released". If the missing
  participant works with the server, the two can strip the masks from the other messages and read their
  sum, so that rule is unsafe for $t \ge 1$. Now everyone who sent a message is charged, `rounds` is the
  number of rounds started, and nothing is repeated for free. `docs/ROADMAP.md` is corrected.
- **Values are never cut off in the shared-noise mode.** Cutting off a participant's numbers would remove
  noise that the others count on. The clean sum is limited before the noise is added (which cannot raise
  what one record can change), and afterwards numbers wrap around the modulus. The grid is planned by the
  library from public numbers; `PairwiseMasking` must have its default settings in this mode.
- **Rounding after the noise is argued, not machine-proven.** Every report of this mode says so. S13
  removes the gap. One claim from the planning stage was corrected by the numerical check: the sum of
  separately rounded messages is **not** distributed like one rounding of the plain sum (they differ by
  about $10^{-3}$); it is distributed like one rounding of the sum plus small independent noise.
- **A participant on its own counts as one colluder**, because it knows its own slice. With $t = 0$ the
  report says that the table row for 1 applies against a single curious participant.
- **The trusted-server mode needs `allow_unsafe=True`** and is flagged in the report. The roadmap only
  said "upper-bound baseline". It takes no secure sum in this step.
- **`batch_size` means the expected batch of the whole group** in the one-step modes, and equal weighting
  is refused. S14 must tune settings per mode.

Two further changes:

- **A gap in existing code was closed**: with `secure_noise=True` only the noise came from the operating
  system; the choice of rows per step was still drawn from the run's seed, although the budget counts on
  that choice being secret. Both now come from the operating system. Stored results are unaffected.
- Two lines (`compression/packing.py`, `crypto/pairwise_masking.py`) were rewritten from `x.max()` to
  `np.max(x)`: the type checker rejected the first form with numpy 2.5 on Python 3.13 (CI checks types on
  Python 3.11 with numpy 2.4, where both forms pass). No behaviour changed.

### Environment (first session started on the "ObscuraDP" cloud environment)

- **The start-up script works.** `.venv` was ready at the start (Python 3.13, CPU-only torch 2.15.0,
  `dp-accounting` 0.6.0, the dev tools). Call the tools as `.venv/bin/...`; the bare `python` has no torch.
- The Python 3.11 environment for the comparison tests is not prepared by the script. It was built by hand
  in a few minutes as `.venv-legacy` (`python3.11 -m venv`, then the install line of handoff section 8) and is
  ignored by git.
- `www.kaggle.com` is outside the environment's network rules.
- The working branch of this session is `cu/eloquent-bardeen-qlbmdu` (the cloud session assigns it). The
  earlier branch `cu/trusting-pasteur-sms0ou` is no longer used.

### Open items

- S13: whole-number noise. It also closes the rounding gap above and finding G16.
- S14: per-mode settings, more than five seeds before any claim about stroke, and the report columns of
  answer 2.
- Public keys are still not authenticated. In the shared-noise mode a server that swaps keys voids the
  budget, not only the hiding. Planned with the Flower adapter (S19).
- For the owner: read the license field on the two Kaggle pages named in `docs/DATASETS.md` when convenient;
  delete the merged remote branches `cu/trusting-pasteur-sms0ou-s03-website` and
  `cu/trusting-pasteur-sms0ou`; TestPyPI setup before S22.

## S11 log (2026-11-10)

### What was done

- **`benchmarks/phase2/`** (not part of the installed package): declared schemas for the stroke and loan
  tables (`tables.py`), made-up tables with only the public shape of the real ones (`standin.py`), the
  settings search on those (`tune.py`), the run grid (`run.py`) and the report that checks every record
  and sets old numbers beside new ones (`report.py`).
- **Settings** (learning rate, clipping norm) were chosen on the made-up tables: 200 short runs, by mean
  validation AUC. No real row was used. The choices are in `benchmarks/phase2/settings.json`.
- **The grid**: both tables, the research network and logistic regression, no protection and total budgets
  0.5, 1, 2, 4, 8, five seeds, updates in the clear and through the secure sum. 200 runs, 36 minutes of
  computing. One record per run in `benchmarks/results/phase2/runs/`, each with its privacy report.
- **`docs/GATE_PHASE2.md`**: the gate memo. `docs/KPI_MAPPING.md`, `docs/RISK_REGISTER.md` and
  `docs/WEBSITE_MISMATCHES.md` (items 29 to 32) updated to match.

### Results in short (details and limits in the gate memo)

- Loan: AUC 0.960 to 0.962 at a total budget of 1 against 0.964 to 0.976 without protection. Nearly free.
- Stroke: 0.739 (network) and 0.780 (logistic regression) at a total budget of 1 against 0.838 and 0.849
  without protection; 0.816 and 0.819 at a budget of 8. Logistic regression is the better private model.
- The earlier pattern "stricter privacy, better stroke recall" is reversed. It came from synthetic rows and
  per-round labels.
- Secure sum: same AUC as in the clear (within 0.002); uploads 46.9% to 48.2% of plain size.
- With a fixed 0.5 cut-off the private stroke models flip between "all yes" and "all no" across seeds.

### What was verified and how

- `python -m benchmarks.phase2.report` exits with status 0: 200 records validated, 160 privacy reports
  recomputed from their recorded releases, none over budget, no unsafe flag.
- The memo's headline table is generated by `report.headline_markdown` and a test checks that the memo
  contains it exactly; other tests check the stored summary against the stored runs.
- The benchmark pipeline runs end to end on made-up rows in the test suite, on Linux, macOS and Windows.
- 531 tests pass in the pull-request selection with the pinned versions (Python 3.11), coverage of
  `src/obscuradp` 98%; 369 pass with the newest versions (Python 3.13), where a stored report from this
  machine also recomputes exactly. 535 tests in total; the 4 slow ones run nightly, among them the one
  that recomputes all 160 stored reports (86 s).
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass (the results do not enter the package).

### Deviations from the plan

- The benchmark scripts were merged with Step 10 (aaydinks/ObscuraDP#28), not with this step, because a
  repository hook required untracked files to be committed while that pull request was still open.
- The earlier partial-encryption and CKKS arms were not re-run: those backends are research-only now.
  The secure-sum arm is pairwise masking.
- The secure sum was run at every budget for the research network, and at a budget of 1 and without
  protection for logistic regression (200 runs, not 240).
- Averaging is weighted by training rows on both tables. The research code averaged stroke participants
  equally; with each participant adding its own noise, equal weights would let the smallest, noisiest
  participant dominate.
- Stochastic gradient descent with settings from the made-up tables replaces Adam with settings chosen on
  the real test rows.
- Experiment 1 (the paper replication shown on the website) was not re-run; the gate memo's conclusions
  about stroke come from the Trial 07/27 setup on the same table.

### Pull requests handled around this step (2026-11-10)

| PR | What | Outcome |
|---|---|---|
| aaydinks/ObscuraDP#28 | Step 10 and the Step 11 scripts | Merged as `c3d713e`, nine checks green on its final commit. On its first commit the dependency bot attached a configuration check of its own that stayed queued and never reported; the second commit did not carry it. |
| aaydinks/ObscuraDP#27 | Bot: `attrs` 26.1.0 | Unmerged, as recorded in the S10 log. The ignore rule is on `main`; the bot closes its pull request on its next run. |

### Open items for the owner

- The five decisions at the end of `docs/GATE_PHASE2.md`. (Answered on 2026-11-10, see the second
  decisions table at the top.)
- Carried forward: the start-up script of the "ObscuraDP" cloud environment is untested by a fresh session;
  TestPyPI setup; the merged remote branch `cu/trusting-pasteur-sms0ou-s03-website` can be deleted.

## S10 log (2026-11-10)

### What was done

- **`obscuradp.crypto.PairwiseMasking`** (the default secure sum, decision record 0003). Every pair of
  participants agrees on a secret the server never sees (X25519), stretches it into a stream of random
  numbers (HKDF-SHA256, then ChaCha20), and one of the two adds that stream to its update while the other
  subtracts it. A single message then looks like random numbers; in the sum every stream cancels and the
  exact total remains. Arithmetic is on whole numbers modulo $2^k$. A sum index can be used once only.
- **Roles are separate objects** (`crypto/roles.py`): `MaskingClient` runs on a participant's machine and
  holds its secrets; `MaskingAggregator` runs on the server and holds none (closes G5 for new code).
- **`obscuradp.compression`**: `Quantizer` rounds an update to $b$-bit whole numbers at random but right
  on average; `pack_bits` writes them with exactly as many bits as needed. Upload sizes reported by the
  library are the lengths of real messages.
- **`Plan(secure_sum=PairwiseMasking(...))` in the simulation**: each participant sends its weighted
  change to the shared model, rounded and masked; the server sees only the total. The privacy report's
  "who is protected from whom" text changes accordingly, and every result carries the measured upload
  bytes, the plain size, their ratio, the one-time setup bytes and the share of values cut off.
- **`obscuradp.crypto.ckks`**, research only: a key holder separate from the party that adds, no unused
  rotation keys, decrypted sums rounded to four decimals and returned as 32-bit numbers (G15). Refused
  unless `allow_unsafe=True`.
- **`obscuradp.selection.policy`**: the choice of which numbers to encrypt comes from a public seed by
  default; scores from private data are available with noise charged to the ledger (fixes G14 for new
  code); a choice from private data without noise is refused unless forced (G4).
- Dependencies: `cryptography>=42` added. `dp-accounting` limited to below 0.7 (see deviations).
- **Tests**: 38 more, for 525 in total.

### What was verified and how

- 522 tests pass in the pull-request selection with the pinned versions (Python 3.11), coverage of
  `src/obscuradp` 98%. With the newest versions (Python 3.13) 360 pass and 2 skip (no `pandas`, no
  `sklearn`); the comparison tests are not selected there.
- **The masked sum is exact**: property test over 2 to 7 participants, 1 to 62 bits and random values;
  the result equals the plain sum modulo $2^k$ every time.
- **A single message reveals nothing visible**: with all-zero input the 16 possible 4-bit values appear
  equally often in 40,000 numbers (chi-square below 60 on 15 degrees of freedom); the same input in
  another sum, or from another participant, gives a different message.
- **Nothing opens without everyone**: three of four messages added by hand give noise; the aggregator
  refuses an incomplete or unknown set.
- **The server-side object holds no secret**: its only fields are the participant names and the bit
  width. Both ends of a pair derive the same secret; different pairs and fresh setups differ.
- **Bytes** (measured from messages): three participants, 12-bit values, 129-number model: 226 bytes
  against 516 as 32-bit numbers, ratio 0.44. Stroke-sized model, five participants: 5,882 against 12,548,
  ratio 0.47. With 8-bit values and three participants: 0.31. All below the 0.5 target.
- **Same model with and without the secure sum**: with 20-bit values the final weights agree within the
  rounding limit and the test score within 0.001; with 12-bit values a private run's score moves by less
  than 0.02 at the same noise level. Runs are repeatable although the keys are fresh each time.
- **CKKS wrapper** (old-code setup, needs TenSEAL): refused by default; sum within $10^{-4}$ of the plain
  sum; output is 32-bit on a four-decimal grid; neither the aggregator nor a participant can decrypt; the
  public material is more than 20 times smaller than the research code's context with rotation keys; one
  update of the stroke network is 10 to 40 times larger than plain.
- The tests can fail: letting both ends of a pair add the mask broke 4 tests; all-zero masks (correct
  sums, unprotected messages) broke 3 tests. Both changes were reverted.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass.

### Deviations from the plan

- **A flaw found by the tests before it shipped**: the aggregator first inferred the number of values
  from the message length, which is wrong when the last byte is padding. It now takes the length (the
  model size, which is public) as an argument and checks every message against it.
- **`dp-accounting` is limited to below 0.7.** Version 0.7.0, released during this work, requires
  `typing-extensions` 4.9.x, which forces `torch` back from 2.15 to 2.5 when pins are regenerated. The
  limit carries a comment saying when to lift it.
- The CKKS wrapper is not wired into the simulation. It is there for size and accuracy comparisons with
  the role objects directly; the simulation's secure sum is pairwise masking.
- The exact BFV backend ("if time allows" in the roadmap) was not built.
- The update length is passed to `combine` explicitly, a small change to the planned interface.
- No dropout recovery, as decided: a missing message makes the sum unopenable and the round is repeated
  with a new sum index.

### Pull requests handled around this step (2026-11-10)

| PR | What | Outcome |
|---|---|---|
| aaydinks/ObscuraDP#26 | Step 9 | Merged as `c188641`, nine checks green. |
| aaydinks/ObscuraDP#22 | Bot: `mpmath` 1.4.1 | Closed by the bot itself once the ignore rule from Step 9 was on `main`. |
| aaydinks/ObscuraDP#27 | Bot: `attrs` 23.2.0 to 26.1.0 in `constraints/dev.txt` | Not merged. Six of nine checks fail at the install step. A helper agent confirmed the cause from a job log and a local reproduction: `dp-accounting` 0.6.0 declares `attrs<24,>=22`, and so do 0.5.0, 0.5.1 and 0.7.0 (the newest), so the pinned set cannot be installed with `attrs` 26.1.0. `attrs` 23.2.0 is already the newest version below 24. Nothing was changed on that branch. The bot configuration now ignores `attrs` 24 and later, and `dp-accounting` 0.7 and later (which the library's own requirement excludes); each rule says when to remove it. The pull request is left for the bot to close on its next run. |

Owner's rule for such bot pull requests (2026-11-10): the bot keeps watching that file; when a bump is
forbidden by another package, a helper confirms the cause, the pull request stays unmerged and a
documented ignore rule is added; bumps that pass all checks are merged.

### Open items

- Public keys are not authenticated: a server that swaps keys in transit could read updates. The report
  says so. Authenticated key exchange belongs with the Flower adapter (S19).
- With two participants each can work out the other's update from the sum. The report states the number
  of participants that may collude with the server ($N - 2$).
- Carried forward: the start-up script of the "ObscuraDP" cloud environment is untested by a fresh session;
  TestPyPI setup by the owner.

## S9 log (2026-11-10)

### What was done

- **`obscuradp.unsafe`**: settings that weaken the privacy guarantee are refused with
  `UnsafeConfiguration`. With `allow_unsafe=True` they run, and the reason is written into the privacy
  report in plain words. Four reasons so far: synthetic rows made before private training, class weights
  taken from private labels, a failure probability $\delta$ that is not below one over the number of
  records, and statistics taken from private data.
- **`obscuradp.tables`**: `TableSchema` encodes a table from declared ranges and category lists, row by row,
  with no statistic taken from the data (closes G12 for new code). `make_client_data` cuts each
  participant's rows into training, validation and test parts, class by class (closes G13 for new code).
- **`obscuradp.imbalance`**: `ClassWeighted` (weights from a public prevalence figure), `NoRebalancing`,
  and `UnsafeOversampling` for SMOTE-style methods, which a private run refuses (closes G3 for new code).
- **`obscuradp.Plan`, `obscuradp.LocalDPSGD` and `obscuradp.Simulation`**: the new loop. The noise of each
  participant is worked out once for the whole run from the total budget; the whole run is reserved in
  the ledger before any data is touched; every round is charged before it happens; scores during
  training come from validation rows and test rows are scored once at the end; results become records
  with code version, machine, settings fingerprint and seed. Participants can have stricter budgets of
  their own (G21).
- `train_plain_round` (the unprotected baseline), `metrics.best_threshold` (decision threshold chosen on
  validation rows), a first worked example in `README.md`, and the public names at the top level of the
  package.
- **Tests**: 63 more, for 487 in total.

### What was verified and how

- 484 tests pass in the pull-request selection with the pinned versions (Python 3.11), coverage of
  `src/obscuradp` 98%. With the newest versions (Python 3.13) the unit tests pass and the comparison tests
  skip.
- **One budget for the whole run**: for three participants (630, 420 and 280 training rows, 4 rounds) the
  simulation's noise multipliers equal an independent calibration over all rounds, each ledger holds
  exactly the steps that were run, each participant spends between 0.99 and 1.0 of a budget of 1, and the
  stored report passes `verify_report`.
- **A budget that cannot be met stops before training** (no round is run), and a participant with half
  the budget gets more than 1.5 times the noise.
- **Unsafe settings**: oversampling, weights from private labels and a too-large $\delta$ each raise by
  default; when forced, the run completes, the report is marked unsafe and names the reason.
- **Clean evaluation**: the last history entry equals the final validation score, the threshold equals
  the best one on validation rows, and test rows are scored once.
- **Encoding is row by row**: changing or removing other rows never changes a row's encoding (property
  test, extreme values included); every row lands in exactly one part of the split.
- **Class weights work under clipping**: on a toy task with one positive row in five, recall at
  threshold 0.5 is 0.0 without weights and 1.0 with them at the same noise level; at the threshold chosen
  on validation rows the private run reaches F1 0.75 against 0.77 without privacy.
- The README example is run as printed by a test, and its report verifies through the command line.
- The tests can fail: removing the unsafe check from the simulation broke 4 tests; sizing the noise for
  one round only (the old mistake) broke the calibration test. Both changes were reverted.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass.

### Deviations from the plan

- **Per-record clipping limits (new).** Clipping every record's gradient to the same length cancels class
  weights: in the first trial the weighted private run had recall 0.27 where the unprotected one had
  0.97. `clipped_sum` now accepts one factor in (0, 1] per record, and `ClassWeighted` gives the heavier
  class the full clipping norm $C$ and the lighter class $C \cdot w_{\text{light}} / w_{\text{heavy}}$.
  No record can exceed $C$, the noise stays sized for $C$, so the guarantee is unchanged (property test:
  one record still moves the sum by at most its own limit).
- The plan's `sim/` folder holds only the new loop. The old loop stays in `obscuradp.legacy.loop`.
- The data-preparation module is called `obscuradp.tables`, not `data`: the release check forbids any
  folder named `data` in the package (so that no dataset can ever ship), and that rule was left as it is.
- Each participant uses its own sampling rate $q_i = \min(1, B / n_i)$ and its own noise level, as in
  table 1 of `docs/KPI_MAPPING.md`, in place of one shared rate.
- In the unprotected trainer one pass is $\lceil n / B \rceil$ batches, so whole passes see every row.

### Pull requests handled around this step (2026-11-10)

| PR | What | Outcome |
|---|---|---|
| aaydinks/ObscuraDP#25 | Step 8 | Merged as `77071f1`. Its first run failed on macOS only: on GitHub's virtual Macs torch prints a one-time notice ("Skipping device Apple Paravirtual device") the first time `torch.func` runs a backward pass, and the test settings turn warnings into errors. A filter for exactly that message was added; every other warning is still an error. |
| aaydinks/ObscuraDP#22 | Bot: `mpmath` 1.3.0 to 1.4.1 in `constraints/dev.txt` | Not merged. Six of its nine checks fail at the install step. A helper agent confirmed the cause from the job log and a local reproduction: `torch` 2.15.0 requires `sympy>=1.13.3`, and both `sympy` 1.13.3 and 1.14.0 (the newest, no pre-release beyond it) declare `mpmath<1.4`, so the pinned set cannot be installed with `mpmath` 1.4.1. No fix exists without forcing an unsupported combination, and nothing was changed on that branch. The bot configuration now ignores `mpmath` 1.4 and later; the rule should be removed once a released `sympy` accepts it. The pull request is left open for the bot to close on its next run, or for the owner to close. |

### Open items

- The simulation sends updates in the clear (protected by each participant's own noise). Hiding
  individual updates from the server is S10.
- `Simulation` handles yes/no outcomes only (its scoring uses `binary_metrics`).
- Carried forward: the start-up script of the "ObscuraDP" cloud environment is untested by a fresh session;
  TestPyPI setup by the owner.

## S8 log (2026-11-10)

### What was done

- **`obscuradp.mechanisms`**: `sampling` (each record joins each step independently with probability $q$),
  `noise` (Gaussian noise; a stream seeded from the operating system for real data) and `dpsgd` (the
  gradient of every record separately through `torch.func`, one joint clipping norm over all parameters,
  noise on the sum, division by the fixed number $q \cdot n$).
- **`obscuradp.client.train_private_round`**: one participant's private round. It charges the whole round
  to the ledger before it draws a single random number, takes the records as tensors and draws every batch
  itself, takes a noise-only step when a batch is empty (closes G27 for new code), supports the FedProx
  term and any torch optimizer, and offers an observer hook for audits.
- **`obscuradp.audit.bounds`**: turns an attack's results into a lower limit on the true budget, either from
  exact error counts with no assumptions or, when the attacker's score is Gaussian by construction, from
  the separation of the two score distributions.
- Budgets of repeated identical settings are now remembered (many participants and rounds share them).
- `scipy` is declared as a dependency (it was already installed through `dp-accounting`).
- **Tests**: 76 more, for 424 in total.

### What was verified and how

- **Three independent per-record gradient implementations agree to $10^{-6}$**: the library's engine,
  Opacus and a loop with one backward pass per record, on the stroke and loan networks, logistic
  regression and a deeper network, three seeds each (12 cases). The clipped sums agree too, also against
  what Opacus hands its own optimizer.
- **One record never moves the clipped sum by more than the clipping norm**: property test on real
  gradients at input scales from $10^{-3}$ to $10^{3}$ (40 generated cases). Records whose gradient is not
  a finite number contribute nothing.
- **Noise level**: on 61,500 noisy coordinates the spread is $\sigma C / (q n)$ within 1.5%, each
  coordinate is centred on the clipped sum over the fixed divisor, and coordinates are uncorrelated.
- **Sampling**: inclusion frequency, batch-size mean and variance, the share of empty batches and
  independence between steps match Poisson sampling.
- **The trainer replayed from its seeds**: batches, clipped sums, noise and the fixed divisor of every step
  reproduce the trainer's weights to $10^{-6}$. A refused round changes nothing and draws nothing.
- **Attack check, every pull request** (2 full-batch steps, ledger budget 1.0, 2,000 runs with and 2,000
  without the target record, attacker sees every noisy gradient): the attack proves a budget of at least
  0.46 at 99.9% confidence (point estimate 0.87), below the ledger's 1.0 and close enough to show the
  accounting is not loose.
- **Planted defects are caught by that check**: with half the reported noise, and with clipping at three
  times the reported norm, the proven budget exceeds 1.5 times the ledger's figure.
- **Attack check, nightly** (Poisson sampling at rate 0.5, 8 steps, a small network, exact error counts,
  threshold chosen on one half of the runs and scored on the other): proven 2.2 against a ledger figure of
  13.1 (98 s).
- Speed: after start-up, one round of 65 steps on a 683-row table takes 0.16 s (2.4 ms per step) on one
  CPU thread, against 2.0 ms per step for Opacus.
- `ruff check`, `ruff format --check`, strict `mypy`, the ignore-rule guard, the research fingerprint check,
  the package build and the release-contents check pass.

### Deviations from the plan

- The fallback (keep Opacus as the mechanism) was not needed.
- The third gradient implementation is a plain loop in the tests. The loop inside
  `obscuradp.legacy.sensitivity` cannot be called on its own (it clips, averages and adds noise in the same
  function).
- The divisor is $q$ times the **declared** dataset size (`declared_records`, by default the actual size).
  The guarantee treats that number as public. The audits pass the same declared size in both worlds.
- The attack check that runs on every pull request uses full batches, where the best attack and its exact
  Gaussian form are known, so 4,000 runs suffice to catch a factor of two in the noise. The sampled check
  makes no such assumption and is therefore much less sharp; it runs nightly.
- "Shuffled fixed-size loaders are refused" holds by design: the trainer accepts tensors only.
- The three slow tests (the two full reference tables of S7 and the sampled attack of S8, about 2.5 minutes
  together) run in the nightly job only, as their marker says. The comparison job on pull requests now
  leaves them out; it ran the S7 ones once, in aaydinks/ObscuraDP#24.

### Open items

- Floating-point Gaussian noise has known weaknesses against an attacker who can read exact low-order bits
  (G16). The integer noise path is S13; `mechanisms/noise.py` says so.
- Nothing yet calls the trainer for a whole federation. That is the `Simulation` of S9.
- Carried forward: the start-up script of the "ObscuraDP" cloud environment is untested by a fresh session;
  TestPyPI setup by the owner.

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
