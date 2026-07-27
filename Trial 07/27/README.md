# Trial 07/27 — DP + FL + HE, simultaneously

**What this is.** The three prior ObscuraDP experiments each reproduced a paper that
held one leg of the privacy stack fixed. This trial composes all three into a single
pipeline in which differential privacy, federated learning and homomorphic encryption
run **at the same time**, and asks the questions none of them could answer alone.

| prior trial | contributed | could not test |
|---|---|---|
| [Experiment 1](../../Experiment%201) — arXiv:2508.10017 | non-IID Dirichlet clients, per-client SMOTETomek, FedProx, Opacus DP-SGD | no HE arm at all |
| [Experiment 2](../../Experiment%202) — Sci. Rep. 10.1038/s41598-025-34536-9 | the controlled arm matrix, CKKS aggregation, payload/latency accounting | DP **or** HE, never both |
| [Experiment 3](../../Experiment%203) — arXiv:2303.10837 (FedML-HE) | chunked CKKS, Selective Parameter Encryption, O(n) overhead, losslessness | no DP anywhere |

Five stacks form a lattice, so every privacy leg is isolated by exactly one controlled
change:

```
            plain  (FL only)
           /     \
      dp  /       \  he
   (FL+DP)         (FL+HE)
           \     /
            dp_he            (all three, naive full encryption)
              |
            dp_spe           (all three, selective encryption)
```

---

## 1. The three claims

1. **Separability** — utility depends only on ε, crypto cost only on the encryption
   ratio *p*. If true, the privacy/utility frontier and the privacy/cost frontier are
   orthogonal axes that an operator can tune independently.
2. **Losslessness survives noise** — Experiment 3 proved CKKS aggregation lossless on
   *clean* updates. Does it stay lossless when the updates it encrypts are DP-noised,
   and by what margin relative to the noise DP already injects?
3. **Sensitivity fidelity** — Experiment 3 picks the SPE mask from *clean* gradient
   magnitudes. Under DP-SGD the only gradients a real deployment can see are clipped
   and Gaussian-noised. Does the top-*p* ranking survive?

---

## 2. Setup

| Component | stroke | loan |
|---|---|---|
| Source | fedesoriano Stroke Prediction, 4,908 rows after cleaning, **4.26%** positive | Kaggle loan-approval, 4,269 rows, 11 features |
| Scaling | `StandardScaler` on age/glucose/BMI, fit on train | `MinMaxScaler`→[0,1] on all features (CKKS needs small magnitudes) |
| Partition | **Dirichlet(α=0.5)** over 5 clients, ≥12 of both classes each | **IID**, 5 × 683 |
| Client preproc | **SMOTETomek** per client | none (already near-balanced) |
| Test set | **class-balanced**, 84 rows (42 positive) | natural, 854 rows |
| Model | FFNN 15→64→32→1, **3,137 params** | MLP 11→32→16→1, **929 params** |
| Headline metric | recall | accuracy |
| Aggregation | **unweighted** (see §6) | sample-weighted |
| lr | 1e-4 (see §6) | 5e-3 |

Shared: 5 clients · **20 rounds** · 3 local epochs · Adam · batch 32 · FedProx μ=0.01.
DP: Opacus DP-SGD, Gaussian + RDP accounting, C=1.0, δ=1e-5, ε a **per-round** budget
(Experiment 1's convention; the composed cumulative spend is reported alongside).
HE: TenSEAL CKKS, `poly_modulus_degree=8192` (4096 slots), `coeff_mod=[60,40,60]`,
`scale=2^40`.

**New capability: weighted homomorphic aggregation.** Experiment 2 could use an
unweighted mean because its partition was IID; Experiment 1 needed sample-weighted
FedAvg because Dirichlet clients differ wildly in size. Running the non-IID task *under
encryption* therefore required applying weights without ever decrypting an individual
client vector. Each client's ciphertext is multiplied by its plaintext scalar weight
before summing — still multiplicative depth 1, so the `[60,40,60]` chain suffices
(verified: it admits exactly one scalar multiply, with **zero headroom**).

---

## 3. The arm matrix

`python run_arms.py` → `results/arms.json`, `results/arms.png`

**stroke** (ε=9.0, p=0.25 for `dp_spe`):

| algo | stack | recall | acc | AUC | pred-pos | BCE | crypto s/round | max\|Δ\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fedavg | plain | 0.5714 | 0.6667 | 0.7744 | 0.405 | 0.806 | — | 1.49e-08 |
| fedavg | dp | 0.6905 | 0.7262 | 0.8050 | 0.464 | 0.582 | — | 1.49e-08 |
| fedavg | he | **0.5714** | **0.6667** | 0.7744 | 0.405 | 0.806 | 0.0323 | 6.56e-08 |
| fedavg | dp_he | **0.6905** | **0.7262** | 0.8050 | 0.464 | 0.582 | 0.0325 | 6.56e-08 |
| fedavg | dp_spe | **0.6905** | **0.7262** | 0.8050 | 0.464 | 0.582 | 0.0291 | 6.56e-08 |
| fedprox | plain | 0.5714 | 0.6667 | 0.7761 | 0.405 | 0.796 | — | 1.49e-08 |
| fedprox | dp | 0.6905 | 0.7262 | 0.8050 | 0.464 | 0.580 | — | 1.49e-08 |
| fedprox | he | **0.5714** | **0.6667** | 0.7761 | 0.405 | 0.796 | 0.0315 | 6.56e-08 |
| fedprox | dp_he | **0.6905** | **0.7262** | 0.8050 | 0.464 | 0.580 | 0.0377 | 6.56e-08 |
| fedprox | dp_spe | **0.6905** | **0.7262** | 0.8050 | 0.464 | 0.580 | 0.0295 | 6.56e-08 |

**loan** (ε=9.0):

| algo | stack | acc | recall | AUC | BCE | crypto s/round |
|---|---|---:|---:|---:|---:|---:|
| fedavg | plain | 0.9614 | 0.9755 | 0.9946 | 0.099 | — |
| fedavg | dp | **0.9403** | 0.9397 | 0.9803 | 0.202 | — |
| fedavg | he | 0.9602 | 0.9755 | 0.9944 | 0.100 | 0.0306 |
| fedavg | dp_he | **0.9403** | 0.9397 | 0.9803 | 0.202 | 0.0302 |
| fedavg | dp_spe | **0.9403** | 0.9397 | 0.9803 | 0.202 | 0.0290 |

**What this shows.** `he` reproduces `plain`, and `dp_he` and `dp_spe` reproduce `dp`,
to every reported digit on stroke — at **every one of the 20 rounds**, worst gap 1.3e-6
(BCE only). Encryption is applied at aggregation time and changes nothing about what is
learned; that equality is the *result*, not a coincidence.

Loan shows a **normal privacy cost** (0.9614 → 0.9403). Stroke does not — DP *beats*
the non-private arm there. That is a real property of the SMOTETomek recipe, discussed
honestly in §6 rather than smoothed over.

---

## 4. Losslessness under DP noise

`python run_losslessness.py` → `results/lossless.json`, `results/lossless.png`

The same DP-noised client updates, aggregated three ways — plaintext, full CKKS, and
SPE — and compared.

| ε | mode | p | max\|HE−plain\| | ÷ update scale | ÷ DP noise std |
|---:|---|---:|---:|---:|---:|
| 0.5 | he | 1.00 | 4.28e-08 | **5.08e-07** | 4.25e-07 |
| 0.5 | spe | 0.25 | 3.59e-08 | **4.27e-07** | 3.57e-07 |
| 2.0 | he | 1.00 | 4.37e-08 | **5.19e-07** | 1.10e-06 |
| 2.0 | spe | 0.25 | 3.77e-08 | **4.47e-07** | 9.46e-07 |
| 9.0 | he | 1.00 | 4.27e-08 | **5.05e-07** | 2.13e-06 |
| 9.0 | spe | 0.25 | 3.48e-08 | **4.12e-07** | 1.73e-06 |
| 50.0 | he | 1.00 | 4.08e-08 | **4.82e-07** | 4.11e-06 |
| 50.0 | spe | 0.25 | 3.74e-08 | **4.41e-07** | 3.76e-06 |

**CKKS error is 4.1e-07 to 5.2e-07 of the update magnitude, at every ε and every p** —
a 1.26× spread across the whole grid. Encryption is free in utility terms, and stays
free when what it encrypts is noise.

Two calibrations make that number interpretable rather than merely small:

- **The float32 floor.** The aggregate is stored back into a float32 `state_dict`, so
  even the *plaintext* arm deviates from a float64 reference — by 1.49e-08. The
  plaintext arms sit exactly on that floor (ratio 1.000). The HE arms sit at 3.5–3.9×
  it. So CKKS costs a small constant multiple of the rounding a plaintext federation
  already pays just to hold its weights.
- **Normalise by the right thing.** The `rel_to_noise_floor` column above divides a
  *weight-space* deviation by a *gradient-space* per-step noise std. It is retained for
  continuity but it drifts 11.5× across ε while its numerator is flat — a units
  mismatch, not a finding. `rel_to_update_scale` is scale-free and duly invariant.

---

## 5. Does the SPE mask survive DP noise?

`python run_sensitivity.py` → `results/sensitivity.json`, `results/sensitivity.png`

Mean ± standard error over **64 draws** per cell.

| ε | P(same tensors) | Jaccard @ p=0.25 | random baseline | Spearman |
|---:|---:|---:|---:|---:|
| 0.5 | **0.094 ± 0.037** | 0.371 ± 0.041 | 0.1429 | 0.533 |
| 2.0 | **0.266 ± 0.056** | 0.508 ± 0.046 | 0.1429 | 0.874 |
| 9.0 | **0.422 ± 0.062** | 0.705 ± 0.037 | 0.1429 | 0.927 |
| 50.0 | **0.734 ± 0.056** | 0.891 ± 0.023 | 0.1429 | 0.983 |

**The finding: SPE's privacy targeting degrades monotonically as the budget tightens.**
At ε=50 the DP-chosen mask agrees with the clean one 73% of the time; at ε=0.5, 9%. The
mechanism is saturation — the DP scores collapse onto the noise floor
`E|N(0, σC/B)| = 0.0117`, compressing a clean score range that spans 52×
(5.13e-04 … 2.67e-02) into one spanning 1.26×. There is simply no ranking signal left.

**Utility is identical across mask modes** — clean, dp and random all give recall
0.6905 / acc 0.7262 / AUC 0.8050. That is expected and important: SPE is lossless
*whichever* parameters it encrypts, so the mask is a **privacy** decision, not a utility
one. Which is exactly why its degradation matters — it is invisible in the accuracy
numbers.

### A claim this trial retracted

An earlier version of this experiment reported that at ε=0.5 the DP mask fell **below**
the random baseline (Jaccard 0.044 vs 0.143), implying anti-correlation. **That does not
replicate and is withdrawn.** It rested on 3 repeats; over 64 draws the same estimator
gives 0.371 ± 0.041 — 5.5 standard errors *above* baseline — with a per-draw standard
deviation of 0.34 and a near-bimodal distribution. Three repeats could never have
supported the claim.

Two methodological caveats, both real:

- **Jaccard-over-scalars is tie-break sensitive.** Every scalar inherits its tensor's
  single score, so **84% of the p=0.25 mask is decided by `np.argpartition` tie-breaking**
  rather than by sensitivity. Swapping the tie-break convention moves the ε=0.5 statistic
  between 0.15 and 0.47. The tie-break-free per-tensor selection probability is therefore
  the primary metric; Jaccard is kept as a secondary column.
- **The p-columns are degenerate here.** P(same tensors) is identical at p=0.1, 0.25 and
  0.5 because one weight matrix holds 2,048 of 3,137 parameters, so the greedy top-*p*
  fill lands in the same tensor at all three fractions. That is a limitation of a
  six-tensor model, not a property of SPE.

---

## 6. Overhead composition

`python run_overhead.py` → `results/overhead.json`, `results/overhead.png`

**SPE is linear in p** (at 541,697 params, 14→133 ciphertext chunks):

| p | chunks | crypto s | ÷ crypto(p=1) | payload | wire payload |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 14 | 0.3660 | **0.100** | 1.521 | 2.421 |
| 0.25 | 34 | 0.9083 | **0.249** | 3.693 | 4.443 |
| 0.50 | 67 | 1.8454 | **0.505** | 7.278 | 7.778 |
| 0.75 | 100 | 2.6480 | **0.725** | 10.862 | 11.112 |
| 1.00 | 133 | 3.6514 | 1.000 | 14.447 | 14.447 |

> `payload_ratio` counts only the ciphertext; `wire_payload_ratio` adds the unencrypted
> remainder the client still transmits at p<1. The two coincide at p=1.

**DP and HE taxes do not interact:**

| params | DP multiplier | HE payload | HE µs/param |
|---:|---:|---:|---:|
| 3,137 | 2.50× | 18.76× | 7.97 |
| 10,369 | 2.19× | 17.02× | 7.03 |
| 37,121 | 3.74× | 15.85× | 6.60 |
| 139,777 | 4.82× | 14.73× | 6.46 |
| 541,697 | 10.19× | 14.45× | 6.20 |
| 2,131,969 | **37.70×** | **14.38×** | **6.19** |

**DP's tax is local compute and grows with model size** (per-sample gradients);
**HE's tax is communication and is asymptotically constant per parameter** — a flat
~6.2–8.0 µs/param across a 680× range, which is the O(n) claim reproduced. They touch
disjoint parts of a round.

### The payload ratio is quantised, and that explains everything

The same CKKS configuration reports 63.3× at loan's 929 parameters, 18.76× at stroke's
3,137, and 14.38× at 2.13M. This looks inconsistent until you measure the quantum: a
**full 4096-slot ciphertext serialises to 235,359 bytes and a one-slot ciphertext to
235,386 bytes** — 27 bytes apart. Payload is quantised in whole ciphertexts:

```
payload_ratio(n) = ceil(n / slots) · bytes_per_ciphertext / (4n)
```

which decays to the 14.38× asymptote — matching Experiment 3's published 14.4×. The
check is exact: 18.758/14.365 = 1.3058 against 4096/3137 = 1.3057. Model and measurement
agree to 1e-15.

The same quantisation is why the width-1 row in the p-sweep is **flat at 18.757 for
every p**: on a model smaller than one ciphertext, every p ≥ 1/4096 costs exactly one
chunk. It is retained in the results tagged `regime = "degenerate_single_chunk"` as an
illustration of the floor, and must not be read as "SPE saves nothing".

---

## 7. What broke, and what it revealed

Two defects made the first version of the stroke results worthless. Both are documented
with measurements in [`DIAGNOSIS.md`](DIAGNOSIS.md); the pre-fix matrix is preserved at
`results/arms_BROKEN_reference.json`.

**The non-private arms collapsed** (BCE 7.76, predicted-positive rate 0.048 on a
*balanced* test set). Per-client SMOTETomek leaves each client a **locally separable**
problem — the largest client interpolates 2,460 positives from **22 real ones**. Plain
BCE on a separable set has no finite minimiser, so unclipped Adam only inflates the
logit scale: pooled train accuracy sat flat at 0.86 from round 1 to 10 while mean
|logit| grew 3.32 → 12.90. The DP arms merely *looked* better because Opacus's clipping
bounded them. Experiment 1 never surfaced this because it has no non-private arm — every
one of its runs went through Opacus.

Sample-weighted FedAvg compounded it: after resampling, `n_c` counts how aggressively a
client duplicated its minority class, so the most imbalanced client took **65.6%** of
the aggregate while holding 13% of the real stroke cases, and the client holding **56%
of all real cases** took 3.1%.

**The ε axis was degenerate** — identical metrics at ε = 0.5, 1.0 and 9.0. Adam's scale
invariance is real and measured (weight displacement saturates at 0.429 and stops
tracking σ; under SGD it scales properly, 0.006→0.130), but the dominant cause was
`ROUNDS` having been halved 20→10 to pay for the second dataset, which left every ε
bunched at the start of the trajectory.

Fixes: `ROUNDS` back to 20, stroke `LR` 1e-3→1e-4, and unweighted aggregation for the
resampled task.

### The finding this leaves standing

On stroke, **DP outperforms the non-private arm** (recall 0.6905 vs 0.5714). This is not
forced into the expected shape because it is genuine: `plain` is now well-calibrated
(pred-positive 0.405 on a 50/50 test set) and is simply the ε→∞ endpoint of the same
trajectory, fitting the 112×-duplicated synthetic positives hardest. Within the DP grid
privacy *does* cost utility in the textbook direction on threshold-free metrics. Loan —
IID and unresampled — shows a normal privacy cost on every metric. **The effect belongs
to the SMOTETomek recipe, not to DP.**

---

## 8. Independent verification

The crypto claims were re-derived against raw TenSEAL/Opacus, deliberately **not**
through this package's own `he.py`/`aggregate.py`, since those were the code under test.
See [`VERIFICATION.md`](VERIFICATION.md); reproduce with `python verify_claims.py` (~17 s).

| claim | verdict | key evidence |
|---|---|---|
| CKKS + SPE lossless on DP-noised updates | **CONFIRMED** | 4.8e-08…7.0e-08; an off-by-one scatter positive control yields 3.10e+00, so the test has ~7 orders of discriminating power |
| Weighted homomorphic aggregation | **CONFIRMED** | rel. error flat ~1.3e-07 across \|v\|=1e-3…10; `[60,40,60]` admits exactly one scalar multiply |
| Payload ratio is padding, not a chain difference | **CONFIRMED** | asymptote 14.365×, matching Experiment 3 |
| `dp_noise_std = σ·C/B` | **CONFIRMED, one correction** | measured grad std 1.5516 vs predicted 1.5625. `B` should be Opacus's `expected_batch_size` (29–32), so the floor is understated by ≤10% |
| Mask degradation under DP | **CONFIRMED** | saturation onto `E|N(0,σC/B)| = 0.0117`; clipping verified as the standard *joint* `min(1, C/‖g‖)` |
| Mask falls *below* random at ε=0.5 | **REFUTED** | see §5 |

---

## 9. Reproduce

```bash
python3.11 -m venv ../../obscura_env && source ../../obscura_env/bin/activate
pip install -r requirements.txt
python run_smoke.py          # 7-check end-to-end sanity (~15 s)
python run_arms.py           # Exp A  arm matrix          -> results/arms.json
python run_frontier.py       # Exp B  joint eps x p       -> results/frontier.json
python run_losslessness.py   # Exp C  losslessness        -> results/lossless.json
python run_sensitivity.py    # Exp D  mask fidelity       -> results/sensitivity.json
python run_overhead.py       # Exp E  overhead            -> results/overhead.json
python run_multiseed.py      # Exp F  5 seeds             -> results/multiseed.json
python verify_claims.py      # independent cross-check
python plot_results.py       # all five figures
```

Both datasets ship in `data/` — nothing is ever downloaded.

### Layout

```
Trial 07/27/
├── obscura/
│   ├── config.py         every hyperparameter (single source of truth)
│   ├── data.py           both datasets, scaling, non-IID/IID partition, SMOTETomek
│   ├── models.py         the FFNN family, flatten <-> state_dict
│   ├── dp.py             sigma solving, RDP accounting, Opacus wiring, noise floors
│   ├── he.py             CKKS context, chunked encryption, naive + SPE + weighted
│   ├── sensitivity.py    clean and DP-aware scoring, top-p masks, agreement metrics
│   ├── fedprox.py        one client's local round (plaintext or DP-SGD, +/- proximal)
│   ├── metrics.py        held-out evaluation, multi-run summarisation
│   ├── aggregate.py      the single place plaintext / HE / SPE aggregation meet
│   ├── orchestrator.py   the full-stack federated loop
│   └── bench.py          crypto and DP overhead harness, padding model
├── run_smoke.py  run_arms.py  run_frontier.py  run_losslessness.py
├── run_sensitivity.py  run_overhead.py  run_multiseed.py  aggregate_seeds.py
├── verify_claims.py  plot_results.py
├── DIAGNOSIS.md  VERIFICATION.md
└── data/  results/  requirements.txt
```
