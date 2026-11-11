# 0005. The default plan

Status: **proposed (waiting for the owner).** Written in step S14 on 2026-11-11 from
`docs/PHASE3_RESULTS.md`. Until the owner decides, nothing changes in the library's defaults.

## Context

A plan says by which means a run stays inside its privacy budget. After steps S12 to S14 the library
offers four ways of adding the noise for a budget that protects single rows, and one for a budget that
protects whole participants. They were compared at equal total budgets on the two research tables
(960 runs, `benchmarks/results/phase3/`). In short:

| Plan | What it needs the user to trust | Measured against local noise | Messages per round | Messages in total |
|---|---|---|---|---|
| `LocalDPSGD` | Nothing beyond the participant itself | the yardstick | 1.00 in the clear, 0.47 through the secure sum | smallest |
| `DistributedDPSGD`, ordinary noise | The secure sum, unchanged public keys, at most the stated number of colluders, and one step that is argued, not machine-proven (rounding) | Stroke: better by 0.01 to 0.05 AUC from a budget of 1 (logistic regression) or 2 (network). Loan: no difference that matters | 0.38 to 0.41 | 4 to 14 times the local plan, because it needs many more rounds |
| `DistributedDPSGD`, whole-number noise | The same without the argued step; the noise sampler is not a certified one | Stroke: better from a budget of 1 (logistic regression) or 4 (network). Loan: no difference that matters | 0.57 to 0.69 | 2 to 7 times the local plan |
| `CentralDPSGD` | The server, completely | Not clearly ahead of shared noise | 1.00 | largest |
| `ClientLevelDP` | As the shared plans | No use with five participants (AUC near 0.5 at a budget of 1) | 0.31 | small |

Two things the comparison could not settle: how the plans compare with many participants or more assumed
colluders, and whether the settings found on made-up tables are near the best for any of them.

## Proposed decision

1. **`Plan()` keeps `LocalDPSGD` as its default.** A default should hold with the fewest assumptions,
   and local noise is the only plan whose guarantee does not depend on a secure sum, on public keys
   being passed on unchanged, or on how many participants work with the server. It is also what every
   stored result and every example so far uses.
2. **The documentation recommends `DistributedDPSGD(max_colluding=t)` with the secure sum for groups of
   few, small participants**, where local noise costs the most, and says what must be trusted in
   exchange. The gain on the stroke table is measured and holds against the better of two local arms.
   The caller still has to choose `t`; there is no default for it.
3. **Ordinary noise stays the default noise of that plan; whole-number noise is the documented choice
   when the argued rounding step or floating-point noise is not acceptable.** The two gave models of
   about the same quality. Whole-number noise misses the message-size target per round and needs every
   row in every round.
4. **Updates through the secure sum keep 12-bit values.** Eight bits cost accuracy in one of eight cells
   (stroke network, budget 1: 0.022 AUC) and nothing in the others; twelve bits cost nothing anywhere
   and already meet the message-size target.
5. **`CentralDPSGD` stays a baseline behind `allow_unsafe=True`.** It bought nothing measurable over
   shared noise here, so there is no reason to recommend trusting the server.
6. **`ClientLevelDP` is documented as a plan for a thousand participants or more.** With five it does not
   work, and the library says so when it is used.

## The alternative the owner may prefer

Make shared noise the default plan. For: it gave the better model on the small table at every budget
where a difference could be shown, and never a worse one. Against: a default that needs a secure sum
would make the simplest call (`Plan()`) refuse to run without further settings, the caller cannot be
spared the choice of `t`, the plan uploads more in total, and its ordinary-noise form still carries the
argued rounding step. If the owner chooses this, the change is one line in `src/obscuradp/plan.py` plus
the examples, and the decision should name the noise kind and whether `t` gets a default.

## Consequences of the proposal

- No stored result and no example changes.
- Step S21 (documentation) gets a short guide: which plan for which situation, with the table above.
- The website's sentence about fewer bytes per round can be supported for local noise through the secure
  sum and for shared ordinary noise, not for whole-number noise, and should not suggest fewer bytes in
  total for the shared plans (to be handled with the other website items in S22).
- If later evidence (the benchmark of steps S17 and S18, with more participants) changes the picture,
  this record is replaced, not edited.
