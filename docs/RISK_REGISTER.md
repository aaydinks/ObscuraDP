# Risk register (Phase 0)

Likelihood and impact are rated low, medium or high. "Owner" is the session in `docs/ROADMAP.md` that carries
the mitigation. Review this file at every gate.

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| 1 | A bug in the library's own DP-SGD step gives a false privacy guarantee | Medium | High | Three independent per-record gradient implementations must agree; nightly white-box check; mutation test with halved noise; fallback to Opacus as the mechanism | S8 |
| 2 | Steps actually run differ from steps the accountant was told about | Medium | High | Charge the ledger before each release; assert the step count; `obscuradp report verify` recomputes from stored events | S7, S8 |
| 3 | Correct accounting makes results much worse and reverses a published headline | High | Medium | Noise table shown at the Phase 0 gate (`docs/KPI_MAPPING.md`); old and new numbers published side by side; distributed mode as the planned recovery. Outcome (S14): the recovery works in part. On stroke, shared noise is better than local noise by 0.01 to 0.05 AUC from a budget of 1 (logistic regression) or 2 (network) and reaches the unprotected model at a budget of 8; at budgets 0.5 and 1 a gap of 0.03 to 0.15 to the unprotected model remains (`docs/PHASE3_RESULTS.md`) | S11, S14 |
| 4 | Encryption backend exposes keys or has the wrong trust boundary | Medium | High | Pairwise masking is the default; CKKS is `research_only`; role objects with a cannot-decrypt test | S10 |
| 5 | Phases 2 and 3 overrun | High | Medium | Split into 8 sessions; a named fallback per session; cut order in `docs/ROADMAP.md` | S7 to S14 |
| 6 | Fragile dependencies: TenSEAL upkeep; OpenFHE ships one wheel for one Python and platform; `dp-accounting` pins an old `attrs`; Flower's privacy features are labelled preview; torch releases change random streams | High | Medium | Optional extras; lockfiles; a pinned environment for comparison tests; a scheduled job on the latest versions | S4 |
| 7 | The benchmark exceeds the CPU budget | Medium | Medium | Timing pilot; cut order; plaintext-equivalent shortcut with a confirming subset; resumable runner | S17, S18 |
| 8 | A website target cannot be met as worded (attacker advantage, bytes) | High | Medium | Definitions fixed at the Phase 0 gate; measured values shown beside targets | S1, S14, S15 |
| 9 | Context is lost between sessions | Medium | Medium | Status file; short decision records; bootstrap script; small pull requests | every session |
| 10 | No independent security review before release | High | High | Release label "Prerun + novel research action" (owner's wording); README and docs state that there has been no independent audit; `SECURITY.md`; ask for outside review after release | S22 |
| 11 | Dataset licenses are unclear | Medium | Low | Owner decision at the gate (the files stay; `docs/DATASETS.md` says what is known and that both licenses are "not found"); the wheel never ships data; downloaders with checksums | S1, S12, S17 |
| 12 | The cloud environment cannot reach the CPU-only torch download host, so each session downloads several gigabytes of GPU libraries | High | Low | Propose a setup script or an allowed host; cache wheels | S4 |
| 13 | The one-run audit is too weak on tables of about 4,000 rows to say anything useful | Medium | Medium | Pilot first; report the audit's own detection power; shadow-model attack as a second opinion. Outcome (S15): partly true. Thirty pooled runs can prove at most about half of a budget of 1 and a single run next to nothing, so the audit cannot confirm a budget. It does measure the attacker's advantage (to about one point either way) and it catches a pipeline whose noise is missing. Every result carries "the most the test could show" (`docs/PHASE4_ATTACK_TESTS.md`) | S15, S16 |
| 14 | A partner misreads the report and deploys an unsafe configuration | Medium | High | Unsafe flags in plain words; "what this does not protect against" page; `obscuradp doctor` | S9, S21 |

## Changes at the Phase 2 gate (2026-11-10)

- **Risk 3 has happened, in part.** Correct accounting reverses the stroke headline (better recall with
  stricter privacy) and lowers stroke quality at strict budgets (AUC 0.74 to 0.78 at a total budget of 1
  against 0.84 to 0.85 without protection). The loan result holds. Old and new numbers are published side
  by side in `docs/GATE_PHASE2.md`. The planned recovery (participants sharing the noise, S12 and S14) is
  now the main open question; its impact stays medium until that is measured.
- **Risk 8, bytes:** the target is met by the default secure sum (uploads 47% to 48% of plain size,
  measured). **Risk 8, attacker advantage:** unchanged; the proven limit for the added noise is 10.6% at a
  total budget of 1 and 5.7% at 0.5, so "5% at a budget of 1" can only be a measured result (S15).
- **Risk 6 has happened twice:** `dp-accounting` 0.7.0 pins an old `typing-extensions` that forces `torch`
  back to 2.5, and it forbids `attrs` 24 and later; `sympy` forbids `mpmath` 1.4. Handled with a version
  limit and documented ignore rules for the dependency bot.
- **New, low:** the dependency bot's own configuration check stayed queued on one commit of
  aaydinks/ObscuraDP#28 and never reported. The rules were verified by the bot acting on them (it closed
  aaydinks/ObscuraDP#22).

## Changes after S15 (2026-11-11)

- **Risk 8, attacker advantage, has happened for one table.** Measured against the attacker the default
  plan is written for, the advantage at a total budget of 1 is within 5% on the stroke table and not on
  the loan table (4.3% and 5.3%, upper ends 5.6% and 6.3%). The website sentence cannot stay a single
  promise (item 33 of `docs/WEBSITE_MISMATCHES.md`, for S22).
- **Risk 13:** see its row. The single-step check, which attacks the noise itself, remains the test that
  can tell a mechanism with too little noise from a correct one; the whole-run audit cannot.
- **Risk 1** now has a public tool behind it: the strongest-attacker checks are in
  `obscuradp.audit.mechanism`, and `obscuradp audit self-test` shows that the auditor flags a broken
  pipeline and passes a correct one.
- **New, medium: averages hide groups of rows.** Where the rare class is given more weight, rows of the
  common class are shortened to a few percent of the clipping length and the rare rows carry the full
  length. The measured average on the stroke table says little about the rare rows. To be measured
  apart in S16.

## Triggers that reopen a decision

- Any independent finding that the default plan leaks more than its report states: stop feature work, fix,
  and add a regression test first.
- TenSEAL or OpenFHE wheels missing for a supported Python version at release time: ship that extra as
  "unsupported on this platform", do not block the core.
- Timing pilot more than twice the estimate: apply the cut order before running anything else.
