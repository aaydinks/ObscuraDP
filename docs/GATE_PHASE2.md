# Phase 2 gate: what correct privacy counting does to the results

Date: 2026-11-10. Status: **waiting for the owner's review.** No Phase 3 work starts before the
decisions at the end of this memo are answered.

Every number below comes from `benchmarks/results/phase2/summary.json`, which is built from 200 stored
runs by `python -m benchmarks.phase2.report`. The full tables are in
`benchmarks/results/phase2/summary.md`.

## In one paragraph

Phase 2 rebuilt the privacy part of the project so that one stated budget covers a whole training run and
every run produces a report anyone can recompute. With that in place the two research experiments were
run again. On the loan table the honest budget costs almost nothing: the quality of the model's ranking
(AUC, where 1 is perfect and 0.5 is guessing) stays at 0.96 even at the strictest budget tried. On the
stroke table it costs a lot at strict budgets and little at loose ones: 0.84 without protection, 0.74 to
0.78 at a total budget of 1, about 0.82 at a total budget of 8. The earlier headline that stronger privacy
went together with better stroke recall does not survive. It was produced by two mistakes that the audit
had already found, not by the method.

## What was run

- **Tables:** the same stroke and loan files as the research code, with the same rows.
- **Participants:** five, split as in the research code (uneven class mixes for stroke, an even split for
  loan). New: each participant keeps 70% of its own rows for training, 15% for choosing settings and 15%
  for the final score.
- **Models:** the research networks (two hidden layers) and, new, a plain logistic regression.
- **Protocol kept from the research code:** 20 rounds, 3 passes over the local rows per round, expected
  batch size 32.
- **Budgets:** a total $\varepsilon$ of 0.5, 1, 2, 4 and 8 for the whole run at $\delta = 10^{-5}$, counted
  per record, plus a run with no protection. Five seeds each. 200 runs in all, 36 minutes of computing on
  four CPU cores.
- **Secure sum:** every budget was also run with updates hidden from the server (pairwise masking) for
  the research network, and at budget 1 and without protection for logistic regression.

What changed against the research runs, each change answering an audit finding:

| Earlier runs | These runs | Finding |
|---|---|---|
| Noise sized for one round, reused for 20 rounds; the label was a per-round budget | Noise sized once for the whole run; the label is the total | G2, G6 |
| Synthetic rows made from real ones (SMOTE) before private training | No synthetic rows; rare outcome handled by class weights from a public figure (2.9% of US adults report a stroke, CDC 2024) | G3 |
| Columns scaled with statistics of the pooled data | Columns scaled from ranges declared in advance | G12 |
| Settings chosen while looking at test scores | Settings chosen on made-up tables with the same public shape; validation rows for the cut-off | G13 |
| Stroke scored on 84 balanced rows only | Scored on all held-out rows at their natural mix; the balanced score is kept beside it | G25 |
| One program held the key and saw every update | Server sees only the sum; no key on the server | G5 |

## Results

AUC on held-out rows at their natural mix. Mean of five seeds, plus or minus the sample standard deviation.
Columns are total budgets; "none" is no protection.

| Table, model | none | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| loan, network | 0.976 ± 0.007 | 0.958 ± 0.007 | 0.960 ± 0.006 | 0.961 ± 0.006 | 0.966 ± 0.007 | 0.967 ± 0.007 |
| loan, logistic | 0.964 ± 0.006 | 0.960 ± 0.008 | 0.962 ± 0.007 | 0.962 ± 0.007 | 0.963 ± 0.007 | 0.963 ± 0.007 |
| stroke, network | 0.838 ± 0.025 | 0.644 ± 0.075 | 0.739 ± 0.047 | 0.767 ± 0.051 | 0.792 ± 0.037 | 0.816 ± 0.025 |
| stroke, logistic | 0.849 ± 0.019 | 0.742 ± 0.043 | 0.780 ± 0.036 | 0.799 ± 0.030 | 0.813 ± 0.024 | 0.819 ± 0.020 |

What these numbers say:

1. **Loan: privacy is nearly free.** From no protection to the strictest budget the network loses 0.018
   and logistic regression 0.004. Each participant has about 600 training rows and the outcome is easy to
   predict.
2. **Stroke: privacy has a real price, and it grows as the budget shrinks.** At a total budget of 1 the
   network loses 0.10 and logistic regression 0.07. At 8 the loss is 0.02 to 0.03.
3. **The simpler model is the better private model.** Logistic regression beats the research network on
   stroke at every budget, most clearly at the strict ones (0.742 against 0.644 at 0.5). Fewer numbers to
   protect means less noise in total.
4. **Why stroke is hard.** The uneven split leaves some participants with fewer than 100 training rows.
   A small participant uses each row often, so it needs far more noise for the same budget: at a total
   budget of 1 the noise multiplier ranges from 3.5 for the largest participant to 28.9 for the smallest.
   On loan it is 6.8 for everyone.
5. **A fixed cut-off of 0.5 is not usable for private stroke models.** Across seeds they flip between
   calling nearly everyone positive and nearly no one (recall $0.41 \pm 0.54$ at budget 1 for the
   network). Ranking quality (AUC) is stable; the cut-off has to be chosen on validation rows. Doing so,
   the F1 score at budget 1 is $0.19 \pm 0.07$ against $0.27 \pm 0.04$ without protection. Both are low
   because only 4% of rows are positive.
6. **Hiding updates from the server costs nothing in accuracy.** With the secure sum the AUC equals the
   in-the-clear AUC to within 0.002 at every budget, on both tables and both models.

## Earlier numbers beside the new ones

The stroke table published by the research code (`Trial 07/27/results/multiseed.json`), scored on its
balanced 84-row set, five seeds, with the spread recomputed as a sample standard deviation:

| Label in the earlier tables (per round) | Whole-run figure the old code itself computed | Recall | AUC |
|---|---|---|---|
| 0.5 | 2.5 | 0.881 ± 0.056 | 0.785 ± 0.031 |
| 1 | 4.9 | 0.862 ± 0.070 | 0.792 ± 0.027 |
| 9 | 45.4 | 0.790 ± 0.070 | 0.798 ± 0.026 |
| 50 | 344.9 | 0.771 ± 0.080 | 0.801 ± 0.030 |

Neither column of budgets describes real patients: the training rows were synthetic blends of real ones,
so one patient shaped many rows.

## Claims that change

| Earlier claim | What the corrected runs show | Verdict |
|---|---|---|
| Stroke recall rises as the privacy budget gets stricter (0.88 at "0.5", 0.77 at "50") | With one honest total budget, quality falls as the budget gets stricter (AUC 0.82 at 8, 0.78 at 1, 0.74 at 0.5, logistic regression). The earlier pattern came from the recipe for synthetic rows and from the 0.5 cut-off on a balanced set | **Reversed** |
| The private arm beats the non-private arm on stroke (recall 0.69 against 0.57, seed 42) | The unprotected model is better at every budget (AUC 0.84 to 0.85 against at most 0.82) | **Reversed** |
| "$\varepsilon = 9$", "$\varepsilon = 0.5$" and so on as the privacy level of a run | Those were budgets for one round. The old code's own whole-run figure was 45.4 for "9" and 2.5 for "0.5", and even those did not hold for real records | **Withdrawn** |
| Loan: the private arm is close to the non-private one (AUC 0.980 against 0.995) | Still true, and now at a real total budget: 0.960 at 1 against 0.976 without protection | **Holds** |
| Encryption adds no accuracy loss | Holds for the secure sum (AUC within 0.002). Not re-run for CKKS, which is research-only now | **Holds for the default** |
| Messages are smaller than plain federated learning | True for the default secure sum only: 47% to 48% of plain size with five participants. CKKS messages are 10 to 40 times larger than plain | **Holds for the default only** |

A point that is not a privacy effect: the unprotected loan baseline is lower here than in the research run
(0.976 against 0.995). The new baseline does not scale columns with statistics of the data and does not
choose its settings by looking at real test scores. That is the price of an honest protocol, and it applies
to protected and unprotected runs alike.

## Where the website targets stand

| Target | Now |
|---|---|
| At most 5% attacker advantage at a total budget of 1 | Not measured yet (the attack tests are step S15). The proven limits for these runs: 46.2% for any method with this budget, and 10.6% (loan) to 10.7% (stroke) for the exact noise that was added. Even at a total budget of 0.5 the limit for this noise is 5.7%, still just above 5%, so a guaranteed 5% needs a budget below 0.5 |
| At least 50% fewer bytes per round | **Met by the default secure sum**: uploads are 46.9% to 48.2% of the same update as 32-bit numbers (five participants, 12-bit values), measured from the real messages. Not met by CKKS |
| Mathematically guaranteed privacy | The accounting is now correct, checked three ways and recomputable from each report. All 160 protected runs stayed within budget (largest spend at budget 1: 0.99999) and none raised an unsafe flag. Still no outside audit |
| Records protected, organizations | Unchanged: nothing to count yet |

## Limits of this evidence

- **Small test sets.** Stroke has 736 to 738 held-out rows per seed, with 31 or 32 positive ones. The 95%
  interval on the stroke AUC at budget 1 is 0.68 to 0.80 for the network and 0.74 to 0.83 for logistic
  regression. Differences of a few hundredths between neighbouring budgets are within the noise.
- **Five seeds.** Enough to see the large effects above, not the small ones.
- **Declared ranges.** The column ranges are round numbers a data owner could state in advance, but they
  were written by someone who had seen the column ranges of these two public tables.
- **Settings.** Learning rate and clipping norm were chosen on made-up tables. They are probably not the
  best for the real tables, for protected and unprotected runs alike.
- **Simulation noise.** Noise came from the run's seed so the runs can be repeated. A deployment must use
  the operating system's randomness (`secure_noise=True`), and floating-point noise has a known weakness
  that the planned whole-number noise removes (step S13).
- **Secure sum.** Public keys are not authenticated and a missing participant stalls a round.
- **No independent review** of the code or the proofs it relies on.
- **Not re-run:** the partial-encryption arms and the CKKS arms of the earlier experiments. Partial
  encryption only matters for the encryption backends, which are research-only now.

## What this means for Phase 3

The stroke result is the case the plan anticipated. When every participant adds all of its own noise, small
participants drown. Phase 3's first step (S12) lets participants share the noise through the secure sum:
each adds a share with standard deviation $\sigma C / \sqrt{N - t - d}$ for $N$ participants, of which $t$
may collude and $d$ may drop out, and takes one step per round. With five participants and one possible
colluder that is half the noise per participant. Whether that recovers the stroke loss is the main question
of Phase 3, and step S14 compares the two modes at equal budgets.

## Decisions needed from the owner

1. **Accept the reversed claims.** If yes, the website's research page needs new wording in step S22;
   until then `docs/WEBSITE_MISMATCHES.md` lists the affected lines (items 29 to 32).
2. **Headline measure for stroke.** Recommended: AUC and average precision on natural rows, with recall
   reported only at a cut-off chosen on validation rows. Recall at a fixed 0.5 cut-off should be dropped.
3. **Default model for small tables.** Recommended: say in the documentation that logistic regression is
   the better private choice for tables of a few thousand rows, and keep the network as a comparison.
4. **Proceed to Phase 3 as planned** (shared noise, then whole-number noise, then the equal-budget
   comparison), or change the order.
5. **The two dataset files.** They are still in the repository under the frozen research folder. The
   earlier decision was "stay for now". Nothing in Phase 2 changes that, and the package still ships no
   data.
