# DIAGNOSIS -- the stroke training side of Trial 07/27

Two failures were reported against `results/arms.json` (snapshot preserved as
`results/arms_BROKEN_reference.json`):

* **P1** the non-private stroke arms collapsed -- `plain` reached BCE 7.76 and a
  4.8% predicted-positive rate on a class-balanced test set;
* **P2** the epsilon axis carried no signal -- recall was 0.5238 at every epsilon.

Both are real, both are now fixed, and both trace back to the same place: **the
trial composed Experiment 1's data recipe with a non-private arm and a shortened
schedule, and neither of those two additions was safe against what per-client
SMOTETomek does to the local objective.**

Reference used throughout: `Experiment 1/` re-run from its own code on this box
(`stroke_fl.train.run_experiment`, 20 rounds x 5 local epochs) and its published
`results/results.json` / `results_multiseed.json`.

---

## 0. The two facts everything else follows from

Measured on the seed-42 stroke partition (`obscura.orchestrator.prepare`):

| client | pre-SMOTE size | real stroke cases | post-SMOTE size | FedAvg weight | share of all real cases | synthetic positives per real one |
|---|---|---|---|---|---|---|
| 0 | 259 | 17 | 484 | 6.5% | 10.2% | 14.2 |
| 1 | 220 | **93** | 234 | **3.1%** | **55.7%** | 1.3 |
| 2 | 572 | 12 | 1120 | 14.9% | 7.2% | 46.7 |
| 3 | 2482 | 22 | **4920** | **65.6%** | 13.2% | **111.8** |
| 4 | 393 | 23 | 738 | 9.9% | 13.8% | 16.0 |

**Fact A -- per-client SMOTETomek makes each client's local problem separable, and
the worst offender is the biggest client.** One local round (3 epochs, plain Adam,
lr 1e-3) from the shared init:

```
client 0  n=  484 realpos= 17   LOCAL train acc=0.8533  |logit| mean= 0.53 max= 1.76
client 1  n=  234 realpos= 93   LOCAL train acc=0.8034  |logit| mean= 0.23 max= 0.60
client 2  n= 1120 realpos= 12   LOCAL train acc=0.8795  |logit| mean= 1.58 max= 4.57
client 3  n= 4920 realpos= 22   LOCAL train acc=0.9644  |logit| mean= 5.94 max=16.12
client 4  n=  738 realpos= 23   LOCAL train acc=0.8550  |logit| mean= 1.31 max= 4.19
```

Client 3 interpolates 2460 positives out of 22 real ones, so its local training set
is nearly linearly separable. Plain BCE on a separable set has **no finite
minimiser** -- the optimiser just keeps pushing the logit scale out. 16.12 is the
float32 saturation ceiling (`log((1-1e-7)/1e-7)`), i.e. probabilities have already
hit exactly 0 and 1 after ONE round.

**Fact B -- sample-weighted FedAvg hands that client the model.** Standard FedAvg
weights client c by n_c because n_c counts independent samples. After per-client
resampling it counts *how aggressively the client had to duplicate its minority
class*, so the weighting rewards the most imbalanced client. Client 3 gets 65.6%
of the aggregate while holding 13% of the real stroke cases; client 1, which holds
56% of every real stroke case in the federation, gets 3.1%.

---

## P1 -- why the non-private arms collapsed

Round-by-round trace of the `plain` stroke arm (10 rounds x 3 local epochs,
Adam lr 1e-3, sample-weighted FedAvg -- the shipped configuration). The pooled-train
and logit columns needed instrumentation `run_stack` does not expose, so this trace
comes from a re-derivation of the same loop with an independently drawn global init;
it lands within one test row of the shipped arm (`results/arms_BROKEN_reference.json`
records recall 0.0952, BCE 7.7629, AUC 0.7177 at round 10):

| round | recall | BCE | AUC | test pred-pos rate | pooled-train pred-pos rate | pooled-train acc | mean abs test logit |
|---|---|---|---|---|---|---|---|
| 1 | 0.310 | 1.14 | 0.765 | 0.190 | 0.454 | 0.865 | 3.32 |
| 2 | 0.333 | 2.08 | 0.756 | 0.202 | 0.444 | 0.868 | 6.48 |
| 5 | 0.119 | 5.02 | 0.732 | 0.071 | 0.398 | 0.858 | 10.82 |
| 10 | **0.095** | **8.48** | 0.700 | **0.048** | 0.384 | 0.863 | **12.90** |

Pooled-training accuracy is **flat at 0.86 from round 1 to round 10** while the
mean absolute logit **quadruples**. Nothing is being learned after round 1; the
run is pure margin inflation, exactly the Fact-A failure mode, amplified by
Fact B into the global model. Test AUC decays 0.765 -> 0.700 as the global model
is dragged onto client 3's 22-seed synthetic manifold, and the operating point
slides off the balanced test set (predicted-positive rate 0.19 -> 0.048) because
real held-out positives do not sit where the interpolated ones do.

DP-SGD looked better for exactly the reason stated in the bug report: Opacus's
per-sample clipping bounds each step, so the DP arms never reach the runaway
regime. **Experiment 1 never exposed this because every Experiment 1 run went
through Opacus -- it has no non-private arm at all.** Re-running Experiment 1's
own config (20 rounds x 5 epochs, lr 1e-3, sample-weighted) through this trial's
loop *without* DP gives BCE **11.78** and recall 0.0952: the reference
configuration is itself unusable for a non-private baseline.

## P2 -- why the epsilon axis was flat

Two independent causes, both measured.

### (a) Adam renormalises the DP noise -- CONFIRMED, but not the whole story

One client, one local round, identical seeds, only sigma varied. Pairwise L2
distance between the resulting weight vectors:

```
Adam                                              SGD (same loop, optimiser swapped)
        0.25   0.50   1.00   2.00   5.00  10.00           0.25   0.50   1.00   2.00   5.00  10.00
0.25  0.0000 0.2270 0.3553 0.4230 0.4638 0.4778   0.25  0.0000 0.0032 0.0097 0.0227 0.0617 0.1267
1.00  0.3553 0.1355 0.0000 0.0753 0.1220 0.1381   1.00  0.0097 0.0065 0.0000 0.0130 0.0520 0.1170
10.0  0.4778 0.2681 0.1381 0.0649 0.0175 0.0000   10.0  0.1267 0.1235 0.1170 0.1040 0.0650 0.0000
||w-w0||  0.628  0.503  0.452  0.435  0.430  0.429  ||w-w0||  0.006  0.009  0.014  0.027  0.065  0.130
```

Under Adam the total displacement **saturates at 0.429 and stops depending on
sigma at all**: going from sigma 2 to sigma 10 (a 5x noise increase) moves the
endpoint by 0.065, i.e. 15% of the displacement it had already made. Under SGD the
displacement scales with sigma as it should (0.006 -> 0.130). This is Adam's scale
invariance: once the noise dominates the signal, `m / sqrt(v)` divides the sigma
straight back out, so the step size is `lr` regardless of how much noise was
injected. **Lead (a) is confirmed.**

But sigma is not renormalised away entirely -- the low-sigma end (0.25 -> 1.0)
still separates. Experiment 1 uses the same Adam and *does* get a monotone
frontier (recall 0.4524 -> 0.2857 over eps 0.5 -> 50, single seed). So Adam alone
does not explain a *completely* flat axis.

### (b) The schedule was cut in half, so every epsilon was still at the start of the trajectory

Epsilon acts on utility only through **how far along the fitting trajectory the
noisy run gets**: more noise -> slower progress -> the model stays closer to the
balanced-prior operating point -> more positives predicted -> higher recall. That
is Experiment 1's own documented mechanism, and its AUC/ppr columns confirm it.
`ROUNDS` was cut 20 -> 10 and `LOCAL_EPOCHS` 5 -> 3 to pay for the second dataset
and five stacks -- a 3.3x cut in total local work. At 30 local epochs every arm
from eps 0.5 to eps 50 is still bunched at the beginning of that trajectory:

| config (stroke, FedAvg, plaintext agg) | eps 0.5 | eps 2 | eps 9 | eps 50 | recall spread |
|---|---|---|---|---|---|
| r10 e3 lr1e-3 sample-weighted (**as shipped**) | 0.5238 | 0.5238 | 0.5238 | 0.4286 | 0.095, flat over 0.5-9 |
| r20 e3 lr1e-3 sample-weighted | 0.5000 | 0.5000 | 0.4048 | 0.3810 | 0.119, monotone |
| r20 e5 lr1e-3 sample-weighted (**Experiment 1 exactly**) | 0.4762 | 0.4286 | 0.3810 | 0.3333 | 0.143, monotone |

Rounds is the knob: same lr, same weighting, 10 -> 20 rounds turns a flat axis into
a monotone one. And the trial's dp arm at 10 rounds was *not* wrong -- it was
simply younger than Experiment 1's: BCE 2.05 at eps 9 against Experiment 1's 3.58
at the same epsilon.

The interaction with Fact B made it worse: sigma is solved per client at that
client's own sampling rate, so the biggest client always gets the *smallest* sigma
and therefore the *least* epsilon sensitivity (client 3: sigma 1.50 -> 0.29 across
eps 0.5 -> 50, a 5.2x range, versus client 1's 5.23 -> 0.34, a 15x range). The 66%
aggregation weight was sitting on the least epsilon-sensitive client in the
federation.

---

## Leads tested and rejected

**(d) a plumbing bug making sigma a no-op -- REJECTED, checked first and cheaply.**
`get_sigma` is cached on the full key `(epsilon, n_local, epochs, batch_size)` and
returns distinct values per client and per epsilon (eps 0.5 -> `[3.75, 5.23, 2.54,
1.50, 3.09]`; eps 50 -> `[0.33, 0.34, 0.31, 0.29, 0.32]`). Those values reach
Opacus: training one client at sigma 1e-4 versus sigma 0.5 moves the weight vector
by ||dw|| = 1.374, far above any numerical floor. `optimizer.pre_step()` is not
silently skipping steps either -- the fedprox (mu=0.01) and fedavg (mu=0) DP arms
produce different results (BCE 1.9245 vs 2.0463), which they could not if the
proximal branch never stepped. Per-client seeding (`seed*1000 + round*10 + client`)
does reset the RNG per local round, but that is a common-random-numbers design: the
same standard normals are drawn and then *scaled by sigma*, so the noise differs
between epsilons by construction.

**(e) test-set construction / train-test scaling -- REJECTED.** `data.py`'s stroke
path is a faithful copy of Experiment 1's: same cleaning, same `StandardScaler` on
the three continuous columns fit on train only, same class-balanced test set. It
reproduces Experiment 1's partition *exactly* -- sizes `[259, 220, 572, 2482, 393]`,
positive counts `[17, 93, 12, 22, 23]`, 84 test rows with 42 positives. Running the
trial's own loop at Experiment 1's exact hyperparameters (20 rounds, 5 local
epochs, lr 1e-3, sample-weighted, FedAvg) lands on recall 0.4762 at eps 0.5,
within one held-out positive of Experiment 1's published 0.4524 (FedAvg) / 0.4762
(FedProx) -- so the data pipeline is not the defect.

**Weight decay as the cure for the logit blow-up -- REJECTED empirically.** Adam's
update is ~`lr` per step regardless of gradient magnitude, so decoupled decay would
need `wd` of order 1 to balance it. Measured on the `plain` arm: `wd=1e-3` gives
BCE 4.10, AdamW `wd=0.01` gives 7.50, `wd=0.05` gives 6.69, against 7.76 with no
decay. It does not bite at any sane value.

**Gradient-norm clipping in the plain path as a non-private analogue of DP-SGD's
clipping -- REJECTED on theory and consistent with the above.** Adam is invariant
to a rescaling of the gradient, which is precisely why the DP arms drift too, only
more slowly. Clipping the norm before Adam is close to a no-op.

**(b) "too much local work" as the sole cause of P1 -- PARTLY REJECTED.**
Experiment 1 does 3.3x *more* local work and does not collapse, because clipping
bounds it. Reducing local epochs alone (`e1`, 10 rounds) only gets BCE from 7.76
to 3.39. The work matters, but only in combination with the missing bound.

---

## The fix

Three changes, all in `obscura/config.py` plus a two-line wiring change in
`obscura/orchestrator.py`. No public API, no signatures, no returned keys, and no
crypto code was touched.

**1. `ROUNDS` 10 -> 20** (`config.py`). Restores Experiment 1's round count. This
is what gives the epsilon axis a utility axis to act on. `LOCAL_EPOCHS` deliberately
stays at 3 rather than Experiment 1's 5: per-round sigma is solved for
`LOCAL_EPOCHS`, so more local epochs buy the same epsilon at higher sigma and
measurably *compress* the spread (0.119 recall spread at 3 local epochs versus
0.071 at 5), besides costing 60% more wall-clock.

**2. `LR["stroke"]` 1e-3 -> 1e-4** (`config.py`). The one deviation from
Experiment 1, and the one that makes a non-private baseline possible at all.
Experiment 1 could use 1e-3 because Opacus bounded every step; the `plain` and `he`
arms have no such bound and the local objective has no finite minimiser (Fact A).
1e-4 keeps every arm inside the calibrated regime without changing the optimiser,
the schedule, the model or the loss. Isolated effect, holding rounds and weighting
fixed: plain BCE 2.485 -> 0.806.

**3. Uniform aggregation weights for resampled datasets** -- new
`config.AGG_WEIGHTED = {"stroke": False, "loan": True}`, read once in
`run_stack` and passed to **both** `aggregate.aggregate_states` and
`aggregate.plaintext_reference` from the same variable so the losslessness
reference can never disagree with the aggregate it is checking. Rationale is Fact
B: after per-client SMOTETomek, `n_c` is not a count of evidence. Loan is not
resampled and keeps standard sample-weighted FedAvg. Isolated effect, holding
rounds and lr fixed: plain BCE 2.501 -> 0.806, plain recall 0.3333 -> 0.5714.

Nothing was changed in `he.py`, in the `lossless` / `float32_floor` /
`dev_over_float32_floor` bookkeeping, or in any module's signature. `run_smoke.py`
passes all 7 checks.

---

## Before / after

### P1 -- the non-private stroke arms (eps 9.0 where applicable, seed 42)

| | | recall | accuracy | pred-pos rate | BCE | AUC | AP |
|---|---|---|---|---|---|---|---|
| fedavg `plain` | before | 0.0952 | 0.5476 | 0.0476 | 7.7629 | 0.7177 | - |
| | **after** | **0.5714** | **0.6667** | **0.4048** | **0.8062** | **0.7744** | 0.7629 |
| fedprox `plain` | before | 0.1429 | 0.5595 | 0.0833 | 4.9873 | 0.7171 | - |
| | **after** | **0.5714** | **0.6667** | **0.4048** | **0.7957** | **0.7761** | 0.7648 |
| fedavg `dp` | before | 0.5238 | 0.6548 | 0.3690 | 2.0463 | 0.7812 | - |
| | **after** | 0.6905 | 0.7262 | 0.4643 | 0.5816 | 0.8050 | 0.8008 |

BCE is in the requested 0.5-1.0 band, the predicted-positive rate on a
class-balanced test set is 0.40 (against 0.50 for the balanced local training
data), and recall is 6x the broken value.

### P2 -- recall across `config.EPSILON_GRID`, stroke, seed 42

| eps | mean sigma | recall (fedprox) | recall (fedavg) | pred-pos rate | BCE | AUC | AP |
|---|---|---|---|---|---|---|---|
| 0.5 | 3.223 | **0.7857** | 0.7857 | 0.5357 | 0.6000 | 0.7959 | 0.7999 |
| 1.0 | 1.932 | 0.7381 | 0.7381 | 0.5119 | 0.5615 | 0.7999 | 0.7961 |
| 2.0 | 1.276 | 0.7143 | 0.7143 | 0.4881 | 0.5407 | 0.8016 | 0.7964 |
| 5.0 | 0.825 | 0.6905 | 0.6905 | 0.4643 | 0.5517 | 0.8027 | 0.7993 |
| 9.0 | 0.642 | 0.6905 | 0.6905 | 0.4643 | 0.5796 | 0.8050 | 0.8004 |
| 15.0 | 0.519 | 0.6667 | 0.6667 | 0.4524 | 0.6143 | 0.8050 | 0.7977 |
| 50.0 | 0.318 | **0.6667** | 0.6667 | 0.4524 | 0.7227 | 0.8005 | 0.7922 |
| (no DP) | - | **0.5714** | 0.5714 | 0.4048 | 0.7957 | 0.7761 | 0.7648 |

Before the fix the same grid gave recall 0.5238 at eps 0.5, 1.0 **and** 9.0 -- bit
identical. It is now monotone non-increasing across the whole grid with a spread of
**0.119 = 5 held-out positives = 5x the 1/42 = 0.024 granularity of the test set**,
in the same direction and of the same order as Experiment 1 (published single-seed
FedAvg 0.4524 -> 0.2857, spread 0.167; published 5-seed FedProx mean 0.6524 ->
0.4524, spread 0.200). The predicted-positive rate tracks recall monotonically
across the whole grid (0.5357 -> 0.4524), which is the signature of a moving
operating point rather than threshold jitter; BCE is U-shaped with its minimum at
eps 2 and climbs monotonically from there to the no-DP endpoint.

The `plain` row continues the same trajectory on recall, predicted-positive rate
and BCE -- it is the eps -> infinity endpoint of the curve rather than a
disconnected arm, which is the strongest evidence the axis is now real. AUC is the
one column it does not continue (0.7761 against 0.8005 at eps 50); the multi-seed
section below explains why.

### Seed robustness, and the privacy-cost narrative stated honestly

Seed 42 alone is not enough to characterise an 84-row test set, so the epsilon
sweep was repeated on seeds 42-45 (stroke, FedProx, `dp` stack, plus the `plain`
baseline). Mean over the four seeds:

| eps | recall | pred-pos rate | BCE | AUC | AP |
|---|---|---|---|---|---|
| 0.5 | 0.8690 +- 0.049 | 0.6815 | 0.6128 | 0.7893 | 0.7795 |
| 1.0 | 0.8512 +- 0.066 | 0.6577 | 0.5884 | 0.7949 | 0.7811 |
| 2.0 | 0.8333 +- 0.073 | 0.6220 | 0.5776 | 0.7978 | 0.7818 |
| 5.0 | 0.8095 +- 0.077 | 0.5893 | 0.5845 | 0.8000 | 0.7828 |
| 9.0 | 0.7857 +- 0.069 | 0.5655 | 0.6011 | 0.8017 | 0.7869 |
| 15.0 | 0.7798 +- 0.078 | 0.5625 | 0.6223 | 0.8029 | 0.7874 |
| 50.0 | 0.7738 +- 0.080 | 0.5506 | 0.6876 | 0.8053 | 0.7913 |
| `plain` (no DP) | 0.7262 +- 0.104 | 0.5030 | 0.6537 | 0.7895 | 0.7655 |

Per-seed recall across the seven grid points:

| seed | 0.5 | 1.0 | 2.0 | 5.0 | 9.0 | 15 | 50 | spread | monotone? | `plain` |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7857 | 0.7381 | 0.7143 | 0.6905 | 0.6905 | 0.6667 | 0.6667 | 0.119 | yes | 0.5714 |
| 43 | 0.8810 | 0.8810 | 0.8333 | 0.8095 | 0.7619 | 0.7619 | 0.7381 | 0.143 | yes | 0.6905 |
| 44 | 0.9048 | 0.9048 | 0.9048 | 0.9048 | 0.8810 | 0.8810 | 0.8810 | 0.024 | yes | 0.8333 |
| 45 | 0.9048 | 0.8810 | 0.8810 | 0.8333 | 0.8095 | 0.8095 | 0.8095 | 0.095 | yes | 0.8095 |

**Monotone non-increasing on every seed**, mean spread 0.095 (4 held-out positives),
best seed 0.143. Seed 44 is the weak one at 0.024, which is a fair reminder that
this test set has 42 positives; the multi-seed campaign is what the frontier plot
should be built on.

Now the narrative, stated as the measurements actually support it -- there are two
different answers depending on which metric you ask:

**Within the DP family, the privacy-utility trade-off is in the textbook
direction on the threshold-free metrics.** Mean AUC rises monotonically with the
budget, 0.7893 (eps 0.5) -> 0.8053 (eps 50); mean AP likewise, 0.7795 -> 0.7913.
Tighter privacy costs discriminative quality, exactly as expected. Nothing is
inverted here.

**On recall -- stroke's headline metric -- the ordering is reversed, and that is an
operating-point effect, not a discrimination effect.** Recall rises as epsilon
falls (0.7738 -> 0.8690) and so does the predicted-positive rate (0.5506 ->
0.6815) in lockstep, while AUC moves the other way. The mechanism is the one
Experiment 1 documented for the same dataset: clients train on SMOTE-*balanced*
data, so the model's discriminative ability barely depends on epsilon; what
epsilon changes is where the model sits on its ROC curve at the fixed 0.5
threshold. More noise -> the run stays nearer the balanced-training prior ->
more positives predicted -> higher recall and lower precision.

**The `plain` arm is the natural eps -> infinity endpoint of that same trajectory,
and it does not beat DP.** It has the lowest recall (0.7262), the lowest
predicted-positive rate (0.5030 -- and note that is essentially perfect
calibration against a 50/50 test set) and the lowest AP (0.7655) of any arm, and
its AUC (0.7895) is back down at the eps-0.5 level rather than continuing up to
0.8053. So the answer to "does DP cost utility?" is:

* **yes**, monotonically, on AUC/AP *within* the DP grid;
* **no** relative to the non-private baseline at moderate epsilon -- DP at
  eps 2-50 beats `plain` on AUC, AP, recall and predicted-positive rate alike.

That second half is a real finding, not a residual bug, and the mechanism is Fact
A: per-client SMOTETomek leaves each client a separable local problem built on
14x-112x duplicated minority points, so training harder means fitting the
interpolation artefacts harder. Without noise the run goes furthest down that
path and comes out slightly past the AUC peak. Experiment 1 saw a stronger version
of the same thing (its published 5-seed AUC *falls* 0.7872 -> 0.7494 as epsilon
rises, because it does 3.3x more local work and so overfits harder at every
budget). This trial sits earlier on the same curve, which is why its AUC ordering
is the milder, opposite-signed one.

On loan -- IID, not resampled, not separable -- DP costs utility in the ordinary
direction on every metric (accuracy 0.9614 -> 0.9403, BCE 0.1054 -> 0.2022,
AUC 0.9939 -> 0.9798), so the pipeline is perfectly capable of showing a normal
privacy cost. The stroke behaviour is a property of the SMOTETomek recipe, not of
the DP implementation.

### Lattice equality (`he` == `plain`, `dp_he` == `dp_spe` == `dp`)

Preserved, and on stroke it is now *better* than before. Encryption still enters
only at aggregation time; nothing in the training path branches on the stack.

| | before (10 rounds) | after (20 rounds) |
|---|---|---|
| stroke, all discrete metrics (recall / accuracy / precision / f1 / pred-pos rate) | identical | **identical, at every one of the 20 rounds** |
| stroke `he` vs `plain`, worst metric gap | 2.8e-2 (fedavg AUC) | **1.3e-6 (BCE only)** |
| stroke `dp_he` / `dp_spe` vs `dp`, worst metric gap | 3.0e-6 | 3.0e-7 |
| loan `dp_he` / `dp_spe` vs `dp`, discrete metrics | identical | identical |
| loan `he` vs `plain`, discrete metrics | 1 test row of 854 | 1-5 test rows of 854 |

The residual loan jitter is **not** introduced by this fix and does not grow with
rounds. Tracing both arms round by round, `he` and `plain` first differ at round
**6** -- inside the old 10-round schedule -- and the gap then oscillates in a bounded
0-5.7e-3 band with no trend (round 10: 1.17e-3; round 20: 1.17e-3). It is chaotic
amplification of the ~2e-7 CKKS rounding error through 20 rounds of a
sign-like Adam update, plus the fact that CKKS encryption is itself randomised, so
the HE arms are only reproducible to within `config.CKKS_TOLERANCE`. Stroke is
immune because its balanced test set has 84 rows, coarser than the perturbation.

### Runtime (4 cores, default torch threads)

Before-figures are `timing.total_sec` from `results/arms_BROKEN_reference.json`
(fedprox rows); after-figures are wall-clock over a whole `run_stack` call.

| arm | before (10 rounds) | after (20 rounds) |
|---|---|---|
| stroke `plain` | 11.4 s | 15.8 s |
| stroke `he` | 8.1 s | 15-16 s |
| stroke `dp` (heaviest) | 23.2 s | **43.9 s** |
| stroke `dp_he` / `dp_spe` | 18-19 s | 41-43 s |
| loan `dp_he` (heaviest) | 8.4 s | 18.4 s |
| `run_smoke.py` | - | 13.8 s, 7/7 PASS |
| `verify_claims.py` | - | config cross-check OK |

Well inside the 90 s per-stroke-run budget. The DP arms roughly double (1.9-2.1x),
so `run_frontier.py` (21 points) and `run_multiseed.py` (5 x 21) will take about
twice as long as they would have. That is the price of having a utility axis at
all: at 10 rounds those 21 frontier points would have returned the same recall.

---

## What still needs regenerating

`results/arms.json`, `arms_table_*.md`, `frontier.json`, `multiseed.json` and
`seed_*.json` are all pre-fix and must be re-run. The pre-fix arm matrix is
preserved as `results/arms_BROKEN_reference.json`.

`results/lossless.json` and `results/sensitivity.json` were produced by
`run_losslessness.py` / `run_sensitivity.py`, which build their client updates
with `config.LR[dataset]` and record `config.ROUNDS` in their metadata. Neither
script's logic was touched and neither depends on `AGG_WEIGHTED` (both use
`weighted=True` consistently on both sides of their own comparison), but the
update magnitudes they encrypt now come from a 1e-4 learning rate, so
`update_scale` and hence `rel_to_noise_floor` will shift when they are re-run.
The losslessness verdict itself (CKKS deviation ~1e-7, orders of magnitude below
the DP noise floor) is scale-free and does not move -- `run_smoke.py` check 4
still measures 3.7e-8 and `run_stack` still reports 5.96e-08 to 7.15e-08 per round
against a 1.49e-08 float32 floor.

`verify_claims.py` cross-checks a verbatim copy of the config constants; its
`ROUNDS` copy was updated from 10 to 20 to match. Its one standing FAIL
(CLAIM 5b, the "below-random Jaccard at eps=0.5" in `sensitivity.json`) is a
pre-existing 3-repeat sampling artefact in `run_sensitivity.py` and is unrelated
to anything here.

---

# ADDENDUM -- the five defects from the independent crypto verification

`VERIFICATION.md` (independent agent, re-derived from raw TenSEAL / Opacus /
numpy) confirmed losslessness, weighted HE, the payload ratio and the
`dp_noise_std` formula, and raised five defects. All five are fixed below.

Constraint under which this was done: the full campaign was running in four
background processes at the time, so nothing they depend on was touched --
`obscura/orchestrator.py`, `fedprox.py`, `aggregate.py`, `data.py`, `models.py`,
`dp.py`, `metrics.py` and every config constant those runs read are unchanged.
Every new quantity is an ADDED key; no existing key changed meaning, so results
produced before and after remain comparable.

Files changed: `obscura/he.py`, `obscura/sensitivity.py`,
`obscura/config.py` (`SENS_REPEATS` only), `run_losslessness.py`,
`run_sensitivity.py`, `run_smoke.py`.

## D1 -- `fedavg_he_timed` ignored its own `slots` argument

`encrypt_vector` chunked at `config.CKKS_SLOTS` while the aggregation loop
chunked at `slots`, so for any non-default slot count the loop indexed past the
end of the ciphertext list. Fixed by giving `encrypt_vector` a `slots` parameter
(defaulting to `config.CKKS_SLOTS`, so every existing caller is unaffected) and
threading it through from `fedavg_he_timed`.

| call | before | after |
|---|---|---|
| `fedavg_he(..., slots=1024)` | dev 2.9e-08 (correct) | unchanged |
| `fedavg_he_timed(..., slots=1024)` | **IndexError** | dev 3.85e-08, 4 chunks, agrees with `fedavg_he` |

Regression added to `run_smoke.py` check 4: it now aggregates the 3137-parameter
model at `slots=1024`, asserts the deviation is within `CKKS_TOLERANCE`, asserts
`fedavg_he` and `fedavg_he_timed` agree, and asserts `n_chunks == 4`. Still 7/7.

## D2 -- SPE payload accounting understated the wire cost

Added `wire_payload_ratio = (ciphertext_bytes + 4 * n_plaintext_params) /
(4 * n_params)` to `fedavg_he_timed`'s stats. `payload_ratio` is untouched and
still means "ciphertext cost against a full-plaintext baseline"; both are now
defined in he.py's module docstring.

| p (n_params = 3137) | `payload_ratio` | `wire_payload_ratio` | verifier's predicted true ratio |
|---|---|---|---|
| 0.25 | 18.757 | **19.505** | 19.507 |
| 1.00 | 18.758 | **18.758** | 18.758 |

At `n_encrypted == 0` the new key is 1.0, not 0.0 -- the client sends the whole
model in the clear, which is exactly 1.0x the plaintext baseline.

**Propagation, and where it stops.** `aggregate.aggregate_states` passes the HE
stats dict through wholesale, so `wire_payload_ratio` appears automatically in
its `he` and `spe` stats with no edit to `aggregate.py` (verified: `he` 18.756,
`spe` 19.506). It does **not** appear in two places, both of which would have
required editing a file the running campaign depends on:

* `aggregate.aggregate_states(mode="plaintext")`, which builds its own literal
  stats dict -- the key is simply absent there;
* `run_stack`'s `crypto` block, which copies a hard-coded list of keys in
  `orchestrator.py`.

Both are one-line additions once the campaign is finished; flagging rather than
doing them, as instructed. `run_losslessness.py` records both ratios on every
record in the meantime, so the number is available in `results/lossless.json`.

## D3 -- duplicate `(epsilon, 1.0, "he")` records

The full-HE aggregation does not depend on p but sat inside the p loop, so with
`LOSSLESS_FRACTIONS = [0.25, 1.0]` it emitted two indistinguishable records per
epsilon. Hoisted out of the loop.

| | before | after |
|---|---|---|
| records | 16 | 12 |
| distinct `(eps, p, mode)` keys | 12 | **12** |
| duplicated keys | 4 (one per epsilon) | **none** |

Per epsilon the file now holds exactly `(eps, 1.0, "he")`, `(eps, 0.25, "spe")`,
`(eps, 1.0, "spe")`.

## D4 -- `rel_to_noise_floor` is dimensionally inconsistent

It divides a weight-space deviation by a per-step gradient-space noise std.
`rel_to_noise_floor` is kept for continuity; `rel_to_update_scale =
max_abs_dev / update_scale` is added and is what the verdict table now leads
with. Measured over all 12 records of the regenerated `lossless.json`:

| normalisation | range | spread |
|---|---|---|
| `rel_to_update_scale` (new, same space) | 4.12e-07 .. 5.19e-07 | **1.26x** |
| `rel_to_noise_floor` (legacy, mixed spaces) | 3.57e-07 .. 4.11e-06 | 11.52x |

Per-epsilon HE rows, showing the numerator barely moves while only the legacy
denominator does:

| eps | max abs dev | update scale | `rel_to_update_scale` | `rel_to_noise_floor` |
|---|---|---|---|---|
| 0.5 | 4.28e-08 | 8.422e-02 | 5.08e-07 | 4.25e-07 |
| 2.0 | 4.37e-08 | 8.432e-02 | 5.19e-07 | 1.10e-06 |
| 9.0 | 4.27e-08 | 8.445e-02 | 5.05e-07 | 2.13e-06 |
| 50.0 | 4.08e-08 | 8.468e-02 | 4.82e-07 | 4.11e-06 |

The defensible headline is therefore: **CKKS aggregation error is 4.1e-07 to
5.2e-07 of the magnitude of the update being aggregated -- about seven orders of
magnitude below the signal -- at every epsilon tested.** The verifier measured
5.6e-07..7.6e-07 on the pre-fix lr=1e-3 data; the shift is the smaller update
scale at lr=1e-4, and the stability is the point.

## D5 -- the sensitivity experiment was underpowered and its estimator unstable

**(a) `SENS_REPEATS` 3 -> 64.** The whole Part-1 sweep costs ~40 s because the
DP scores for a given (epsilon, draw) do not depend on p and the clean scores
depend on nothing, so both are now computed once and reused across fractions --
bit-identical to recomputing them. Every cell is reported as mean +- s.d. with a
standard error, never a point estimate.

**The "below random" finding is refuted, reproducibly.** At eps=0.5, p=0.25:

| | shipped (3 draws) | now (64 draws) | analytic baseline p/(2-p) |
|---|---|---|---|
| Jaccard | 0.044 | **0.371 +- 0.041 (s.e.)** | 0.1429 |

0.371 sits **5.5 standard errors ABOVE** the baseline. The old 0.044 was three
draws of a statistic whose per-draw s.d. is 0.33. `run_sensitivity.py`'s printed
conclusion now says explicitly that no anti-correlation exists.

**(b) New primary metric: P(the DP mask selects the same TENSORS as the clean
mask).** Jaccard over scalars mostly measures `np.argpartition`: every scalar
inherits its tensor's single score, so the boundary tensor contributes a block of
exactly-equal scores, and at p=0.25, 655 of 784 selected scalars (84%) are picked
arbitrarily from inside `net.2.weight`. Added
`sensitivity.tensor_selection()` and `sensitivity.tensor_selection_agreement()`,
which rank tensors by score, fill greedily, and compare the resulting tensor
sets -- a pure function of the ranking, invariant to any tie-break. Recorded as
`same_tensor_selection` on every record and reported first.

Mean +- s.d. over 64 draws (primary):

| eps | P(same tensors) | s.e. | Jaccard over scalars (secondary, p=0.25) | s.e. |
|---|---|---|---|---|
| 0.5 | **0.094** | 0.036 | 0.371 | 0.041 |
| 2.0 | **0.266** | 0.055 | 0.508 | 0.046 |
| 9.0 | **0.422** | 0.062 | 0.705 | 0.037 |
| 50.0 | **0.734** | 0.055 | 0.891 | 0.023 |

Monotone in epsilon and close to the verifier's independent 0.02 / 0.20 / 0.42 /
0.73. The claim the trial can now make is clean and tie-break free: **at eps=0.5
the DP-scored mask picks the same set of tensors as the clean mask only 9% of the
time; at eps=50, 73%.**

One property worth stating so nobody reads it as a bug: `same_tensor_selection`
is **identical across p=0.1, 0.25 and 0.5** on this model. `net.2.weight` holds
2048 of 3137 parameters, so the greedy fill reaches k inside that same tensor at
all three fractions and the selected tensor set does not depend on p. That is a
true statement about a 6-tensor model, not a copy-paste error; on a deeper model
the columns would separate.

**(c) Analytic random baseline recorded.** `random_jaccard_analytic = p/(2-p)` is
now a field on every record (0.0526 / 0.1429 / 0.3333 for p = 0.1 / 0.25 / 0.5)
alongside the measured random control, so the baseline is checkable without
recomputation.

**Bonus correction the verifier flagged.** The clean per-tensor score span was
written up as "8.2e-04 to 2.0e-02, a 24x range". The actual span at the seed-42
init on client 0's 256-row batch is **5.13e-04 to 2.67e-02, a 52x range**;
`run_sensitivity.py` now prints the full score table and the span it measured, so
the number cannot drift from the configuration again. The qualitative point --
clean scores span decades, DP scores collapse onto the `E|N(0, sigma*C/B)|` floor
-- is strengthened. The driver's printed conclusion also now records the
mechanism the write-up missed: the per-tensor sampling s.d. of `mean|noise|`
falls as `1/sqrt(numel)`, so saturation systematically promotes SMALL tensors --
a bias, not a coin flip.

## Regenerated, and unchanged

`results/lossless.json` and `results/sensitivity.json` were re-run sequentially
(nice'd, single-threaded, ~40 s and ~4 min) so both reflect the fixes and the
post-fix learning rate. `results/overhead.json` was NOT re-run -- `run_overhead.py`
is one of the drivers held back while the campaign occupies the box, and its
`payload_ratio` figures are unaffected by these changes in any case; it should be
re-run afterwards to pick up `wire_payload_ratio`.

Utility across mask modes is still exactly identical (accuracy 0.7262, recall
0.6905, F1 0.7160, AUC 0.8050 for clean / dp / random alike), which is the SPE
losslessness property and is unaffected by any of this. `run_smoke.py` passes
7/7 with the two new regression assertions.
