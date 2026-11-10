# Risk register (Phase 0)

Likelihood and impact are rated low, medium or high. "Owner" is the session in `docs/ROADMAP.md` that carries
the mitigation. Review this file at every gate.

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| 1 | A bug in the library's own DP-SGD step gives a false privacy guarantee | Medium | High | Three independent per-record gradient implementations must agree; nightly white-box check; mutation test with halved noise; fallback to Opacus as the mechanism | S8 |
| 2 | Steps actually run differ from steps the accountant was told about | Medium | High | Charge the ledger before each release; assert the step count; `obscuradp report verify` recomputes from stored events | S7, S8 |
| 3 | Correct accounting makes results much worse and reverses a published headline | High | Medium | Noise table shown at the Phase 0 gate (`docs/KPI_MAPPING.md`); old and new numbers published side by side; distributed mode as the planned recovery | S11, S14 |
| 4 | Encryption backend exposes keys or has the wrong trust boundary | Medium | High | Pairwise masking is the default; CKKS is `research_only`; role objects with a cannot-decrypt test | S10 |
| 5 | Phases 2 and 3 overrun | High | Medium | Split into 8 sessions; a named fallback per session; cut order in `docs/ROADMAP.md` | S7 to S14 |
| 6 | Fragile dependencies: TenSEAL upkeep; OpenFHE ships one wheel for one Python and platform; `dp-accounting` pins an old `attrs`; Flower's privacy features are labelled preview; torch releases change random streams | High | Medium | Optional extras; lockfiles; a pinned environment for comparison tests; a scheduled job on the latest versions | S4 |
| 7 | The benchmark exceeds the CPU budget | Medium | Medium | Timing pilot; cut order; plaintext-equivalent shortcut with a confirming subset; resumable runner | S17, S18 |
| 8 | A website target cannot be met as worded (attacker advantage, bytes) | High | Medium | Definitions fixed at the Phase 0 gate; measured values shown beside targets | S1, S14, S15 |
| 9 | Context is lost between sessions | Medium | Medium | Status file; short decision records; bootstrap script; small pull requests | every session |
| 10 | No independent security review before release | High | High | Release label "Prerun + novel research action" (owner's wording); README and docs state that there has been no independent audit; `SECURITY.md`; ask for outside review after release | S22 |
| 11 | Dataset licenses are unclear | Medium | Low | Owner decision at the gate; the wheel never ships data; downloaders with checksums | S1, S17 |
| 12 | The cloud environment cannot reach the CPU-only torch download host, so each session downloads several gigabytes of GPU libraries | High | Low | Propose a setup script or an allowed host; cache wheels | S4 |
| 13 | The one-run audit is too weak on tables of about 4,000 rows to say anything useful | Medium | Medium | Pilot first; report the audit's own detection power; shadow-model attack as a second opinion | S15, S16 |
| 14 | A partner misreads the report and deploys an unsafe configuration | Medium | High | Unsafe flags in plain words; "what this does not protect against" page; `obscuradp doctor` | S9, S21 |

## Triggers that reopen a decision

- Any independent finding that the default plan leaks more than its report states: stop feature work, fix,
  and add a regression test first.
- TenSEAL or OpenFHE wheels missing for a supported Python version at release time: ship that extra as
  "unsupported on this platform", do not block the core.
- Timing pilot more than twice the estimate: apply the cut order before running anything else.
