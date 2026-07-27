# Independent verification of Trial 07/27's crypto-side claims

Auditor: independent verification agent. Date: 2026-07-27.
Reproduce with `/home/user/ObscuraDP/obscura_env/bin/python verify_claims.py` from the
trial root (~14 s, deterministic apart from SEAL's encryption RNG).

## Validated against — check this first

This document is only as current as the data it was run against. `verify_claims.py`
prints the same fingerprint on every run; if it disagrees with the block below, this
document has gone stale and must be re-run before it is quoted.

| results file | sha256 (12) | records |
|---|---|---|
| `results/sensitivity.json` | `ee0de5c6cc54` | 771 (768 agreement + 3 utility) |
| `results/lossless.json` | `0995df961b17` | 12 |
| `results/overhead.json` | `38761d5ce23d` | 37 |
| `results/arms.json` | `fc4d7783d0ce` | 20 |

Config at time of validation (all 18 constants cross-checked verbatim against
`obscura/config.py`): `SENS_REPEATS = 64`, `LR = {stroke: 1e-4, loan: 5e-3}`,
`ROUNDS = 20`, `LOCAL_EPOCHS = 3`, `BATCH_SIZE = 32`, `MAX_GRAD_NORM = 1.0`,
CKKS `[60, 40, 60] @ poly_modulus_degree 8192`, `SPE_DEFAULT = 0.25`.

**What changed since the previous verification.** `SENS_REPEATS` went 3 → 64 and
`results/sensitivity.json` was regenerated (this is the defect the previous round of
this audit identified); `LR["stroke"]` went 1e-3 → 1e-4, which moved every trained
weight magnitude and therefore every deviation that is normalised by one; and the
previous round's extra findings E1, E2 and E4 were all fixed in the package. The
harness itself was rebuilt: it now separates status from verdict, validates against the
64-repeat data, checks the tie-break-free primary metric rather than only the secondary
Jaccard, and adds Claim 6.

**Method.** Every reference number below was re-derived from raw TenSEAL, raw Opacus,
raw numpy and plain `torch.nn`. `obscura.he`, `obscura.aggregate` and
`obscura.sensitivity` were never used to *produce* a reference — only as the *subject*
of a comparison. `obscura.data` was used only to obtain the same client partition the
experiment used (it is not under test). No long driver was re-run; `results/*.json` is
read as authoritative. Nothing outside `verify_claims.py` and this file was modified.

**Scope.** Crypto and DP-accounting only. Training utility, recall, the ε-vs-accuracy
frontier and the multi-seed campaign were deliberately not examined.

---

## How to read the verdicts

Two orthogonal things are reported, and conflating them is what made the previous
version of this harness unreadable.

* **STATUS** — did the check *execute* and reach a confident conclusion?
  `OK` the measurement ran and its controls behaved; `ERROR` it could not be completed.
* **VERDICT** — is the *claim* true?
  `CONFIRMED` · `CONFIRMED-WITH-CORRECTION` · `REFUTED`.

**A `REFUTED` claim with `STATUS = OK` is a success for this harness.** The check did
its job and established the claim is false. `verify_claims.py` therefore exits nonzero
only when (a) a check failed to execute, or (b) a measured verdict disagrees with the
verdict recorded in the table below. It never exits nonzero merely because a claim
turned out to be false. The harness parses the table below on every run and compares it
against what it just measured, so a write-up that drifts away from the data fails
loudly.

<!-- VERDICT-TABLE -->

| # | Claim | Status | Verdict |
|---|---|---|---|
| 0 | Harness config cross-check | OK | CONFIRMED |
| 1 | CKKS/SPE losslessness on DP-noised updates | OK | CONFIRMED |
| 2 | Weighted homomorphic aggregation | OK | CONFIRMED |
| 3 | Ciphertext/plaintext payload ratio | OK | CONFIRMED |
| 4 | DP accounting and `dp_noise_std` | OK | CONFIRMED-WITH-CORRECTION |
| 5a | DP-aware sensitivity: the reported numbers reproduce | OK | CONFIRMED-WITH-CORRECTION |
| 5b | "Mask agreement falls BELOW the random baseline at ε=0.5" | OK | REFUTED |
| 6a | The float32 floor is a correctly-computed calibration | OK | CONFIRMED |
| 6b | "The HE arms sit at 3.5–3.9× the float32 storage floor" | OK | REFUTED |

<!-- /VERDICT-TABLE -->

Both refutations are *recorded, expected* refutations: 5b is the claim the trial has
already retracted, and re-measurement confirms the retraction was right; 6b is a stale
number in the README that this round identified and priced. Exit code 0.

---

## CLAIM 1 — CKKS aggregation is lossless, and stays lossless on DP-noised updates

**STATUS: OK. VERDICT: CONFIRMED.**

| quantity | reported (`lossless.json`, 12 records) | independently measured |
|---|---|---|
| `max_abs_dev` over ε × p | 3.476e-08 … 4.375e-08 | 3.89e-08 … 4.25e-08 |
| `rel_to_update_scale` | **4.116e-07 … 5.188e-07** | 4.62e-07 … 5.05e-07 |
| `rel_to_noise_floor` | 3.568e-07 … 4.109e-06 (11.5× spread) | — |

**The reference is built on the right vectors, and that is checkable.** The seed-42
stroke init has `mean|w| = 0.084163`; the reported `update_scale` averages `0.084417`
across all 12 records — agreement −0.30%. At `lr = 1e-4` the weights barely leave the
init, so the object being encrypted is essentially the init itself. The independent
CKKS reference is therefore run on `w0 + N(0, lr·√(steps)·σC/B)` per client rather than
on a guessed magnitude, and it lands inside the reported band at every ε.

A falsifiable ordering both my reference and the shipped file satisfy: SPE at p=0.25
encrypts fewer coordinates than full HE, so its max-over-coordinates deviation must be
the smaller one. It is, in both.

**Internal consistency.** `rel_to_update_scale == max_abs_dev / update_scale` on 12/12
records; `rel_to_noise_floor == max_abs_dev / noise_std` on 12/12.

### Is the scale-free normalisation still the defensible headline? Yes — and now measured

Sweeping the magnitude of the encrypted vectors over the operating band
(`mean|ref|` from 0.055 to 55, a ~1000× sweep, everything else fixed):

| `mean|ref|` | `max_abs_dev` | `÷ mean|ref|` | `÷ (σC/B)` |
|---|---|---|---|
| 0.00053 | 1.9e-08 | 3.5e-05 | 1.9e-07 |
| 0.00550 | 1.5e-08 | 2.8e-06 | 1.5e-07 |
| 0.05524 | 3.7e-08 | **6.8e-07** | 3.7e-07 |
| 0.54691 | 3.4e-07 | **6.3e-07** | 3.4e-06 |
| 5.53205 | 3.5e-06 | **6.3e-07** | 3.4e-05 |
| 55.1436 | 3.5e-05 | **6.4e-07** | 3.5e-04 |

Over that band the deviation itself moves ~940×, while `dev / update_scale` moves
**1.08×**. A fixed absolute error would have moved the ratio by the full 1000×; it did
not. This is a real invariance, not an artifact of a narrow sweep.

**Caveat, also measured:** below `mean|ref| ≈ 0.01` an additive CKKS floor (~1.9e-08)
takes over and the ratio degrades to 3.5e-05. "Scale-free" holds over the operating
range — the trial's `update_scale` of 0.0844 sits comfortably inside it — not
asymptotically. Say "scale-free over the operating range", not "scale-free".

By contrast `rel_to_noise_floor` divides a **weight-space** deviation by a **per-step
gradient-space** noise std, which is why it wanders 11.5× across ε while its numerator
is flat. `rel_to_update_scale` is the right headline and the write-up has adopted it.

### SPE scatter-back — specifically tested for mis-indexing

A 5000-coordinate case built so any misalignment is loud: 1250 True positions drawn by
random permutation (longest contiguous True run = 6), client vectors whose neighbouring
coordinates differ by ~0.6 on a `|v| ∈ [0, 1.005]` range.

* independent SPE reference vs plaintext weighted mean: `2.5e-07`;
* **positive control** — shifting the encrypted coordinates by one slot on scatter-back:
  `3.10e+00`, i.e. ~7 orders of magnitude of detection power. It did not fire;
* `obscura.he.fedavg_he` on the same input: `2.5e-07`; `fedavg_he_timed`: `2.5e-07`;
* mask-polarity probe: the **plaintext half** matches to `4.4e-16` (exact float64) while
  the **encrypted half** matches to `2.5e-07` (CKKS precision). An inverted or shifted
  mask would swap or smear that signature. It did not;
* all 5000 coordinates finite, none dropped, both entry points agree.

---

## CLAIM 2 — Weighted homomorphic aggregation is correct

**STATUS: OK. VERDICT: CONFIRMED.**

Weights ∝ the real stroke client sizes `[484, 234, 1120, 4920, 738]` →
`[0.0646, 0.0312, 0.1494, 0.6564, 0.0985]`. Deliberately unequal: the weighted answer
differs from the unweighted mean by 0.20 at `|v| ~ 0.1`, i.e. **7 orders of magnitude
larger than the HE error**, so the test genuinely distinguishes weighted from unweighted
rather than merely confirming that both are close to something.

| `|v|` scale | max abs dev | max rel dev | \|weighted − unweighted\| |
|---|---|---|---|
| 1e-03 | 1.6e-08 | 7.6e-06 | 2.0e-03 |
| 1e-02 | 1.9e-08 | 8.2e-07 | 2.0e-02 |
| 1e-01 | 3.9e-08 | 1.4e-07 | 2.0e-01 |
| 1e+00 | 3.2e-07 | 1.3e-07 | 2.5e+00 |
| 1e+01 | 3.1e-06 | 1.3e-07 | 1.98e+01 |

Relative error is flat at ~1.3e-07 across four decades — the signature of a correct
depth-1 CKKS computation with no silent precision collapse. `obscura.he`'s weighted
full-HE path matches the independent plaintext weighted mean to `3.5e-08`.

**Depth budget, directly probed.** With `[60, 40, 60]`: one scalar multiply on a fresh
ciphertext **works**; two **raise `ValueError: scale out of bounds`**. So `cᵢ · wᵢ` then
`Σ` is exactly depth 1 and fits — with **zero headroom left**. Worth stating in the
write-up: there is no room for a second homomorphic operation without moving to
`[60, 40, 40, 60]`, the ~20× chain Experiment 2 used.

---

## CLAIM 3 — The reported payload ratio

**STATUS: OK. VERDICT: CONFIRMED. The 18.0–18.8× is 100% zero-padding.**

A TenSEAL CKKS ciphertext under `poly_modulus_degree=8192`, `coeff_mod=[60,40,60]`
serialises to **≈ 235 360 B regardless of how many of its 4096 slots are filled**
(measured at 1, 512, 3137 and 4096 filled slots). Analytic model:
`ratio(n) = ceil(n / 4096) · 235 360 B / (4n)`.

| n_params | chunks | my prediction | reported | abs diff |
|---|---|---|---|---|
| 3 137 | 1 | 18.757 | 18.757 | 0.0001 |
| 10 369 | 3 | 17.024 | 17.024 | 0.0002 |
| 37 121 | 10 | 15.851 | 15.851 | 0.0003 |
| 139 777 | 35 | 14.734 | 14.733 | 0.0002 |
| 541 697 | 133 | 14.447 | 14.447 | 0.0001 |
| 2 131 969 | 521 | 14.379 | 14.379 | 0.0002 |

And decisively: `18.757 / 14.365 = 1.3058` against `4096 / 3137 = 1.3057`. The padding
factor accounts for the discrepancy to four significant figures.

**Asymptotic ratio measured: 14.365×** on a slot-aligned 81 920-element vector — exactly
Experiment 3's constant. Cross-check on the chain: an uncompressed fresh ciphertext under
`[60,40,60]` is `2 polys × 8192 coeffs × 2 data primes × 8 B = 262 144 B` (the trailing
60-bit prime is SEAL's special key-switching prime, not part of a fresh ciphertext),
giving a raw ratio of exactly 16.0; serialised is 0.898× that (SEAL's ZSTD), landing at
14.365. Both routes confirm the chain is `[60,40,60]`, not a longer one.

**The correction this harness asked for has been adopted.** `overhead.json` now carries
`wire_payload_ratio`, and it re-derives independently as
`(ciphertext_bytes + 4·(n_params − n_encrypted)) / plaintext_bytes` on **24/24** records.
At `n = 541 697`: p=0.10 ciphertext-only 1.521 → true wire 2.421; p=0.25 3.693 → 4.443;
p=0.50 7.278 → 7.778; p=0.75 10.862 → 11.112.

The flatness across p at `n = 3137` remains an artifact of a 3137-parameter model needing
one chunk at any p ≥ 0.1, and the driver already labels that sweep
`regime: "degenerate_single_chunk"`. Keep saying so, because as printed it reads as
"SPE does not reduce payload", which is false for any model above ~4k parameters.

---

## CLAIM 4 — The DP accounting

**STATUS: OK. VERDICT: CONFIRMED WITH CORRECTION.**

### σ → ε round trip (raw Opacus `get_noise_multiplier` + `RDPAccountant`, n_local = 484)

| target ε | σ | ε_round recovered | err | ε cumulative over **20** rounds |
|---|---|---|---|---|
| 0.5 | 3.750000 | 0.4976 | −0.48% | 2.404 |
| 2.0 | 1.406250 | 1.9980 | −0.10% | 8.681 |
| 9.0 | 0.686035 | 8.9957 | −0.05% | 35.715 |
| 50.0 | 0.329285 | 49.9958 | −0.01% | 271.623 |

Round trip is correct (residual is `get_noise_multiplier`'s binary-search granularity).
Composition is monotone and sub-linear: ε(1 round) = 9.00 → ε(20 rounds) = 35.71, far
below 20 × 9.00. `obscura.dp` reproduces σ, ε_round and ε_cumulative bit-for-bit.

### Is `dp_noise_std = σ·C/B` the right expression? — now tested where the two candidates *differ*

Opacus adds noise of std `σ·C` to the **summed** clipped gradient, then divides by
`expected_batch_size`. So the noise on the **averaged** gradient has std
`σ·C / expected_batch_size`, and `σ·C/B` is the correct *form*.

The previous version of this check ran at `n_data = 1024`, where `1024/32 = 32` exactly
and `expected_batch_size == BATCH_SIZE` — so it could not distinguish the two candidate
denominators at all. It is now run at **n_data = 484**, the real client-0 size, where
Opacus uses `expected_batch_size = 30`:

| n_data | Opacus `expected_batch_size` | measured `std(p.grad)` | `σC/32` | `σC/expected` | `σC` (no 1/B) |
|---|---|---|---|---|---|
| 484 | 30 | **1.6661** | 1.5625 (ratio 1.066) | **1.6667 (ratio 1.000)** | 50.0 (ratio 0.033) |
| 1024 | 32 (degenerate) | 1.5597 | 1.5625 (0.998) | 1.5625 (0.998) | 50.0 (0.031) |

At the discriminating size the measurement picks `expected_batch_size` over the nominal
batch size decisively, and excludes the "no 1/B" alternative by 30×.

### The correction

`dp_noise_std` hard-codes `B = config.BATCH_SIZE = 32`. Opacus uses
`expected_batch_size = int(len(dataset) / len(loader))`:

| n_local | len(loader) | Opacus `expected_batch_size` | `dp_noise_std` understates the floor by |
|---|---|---|---|
| 484 | 16 | 30 | +6.7% |
| 234 | 8 | 29 | +10.3% |
| 1120 | 35 | 32 | 0.0% |
| 4920 | 154 | 31 | +3.2% |
| 738 | 24 | 30 | +6.7% |

Up to 10%, mean ≈ 5%. Immaterial to a "1e-7 of the signal" headline, but the write-up
should say `B` is Opacus's `expected_batch_size`, not the nominal batch size.

A related, separately reportable mismatch: `get_sigma` accounts with
`sample_rate = min(1, batch_size / n_local)` (= 32/484 = 0.0661 for client 0), whereas
Opacus's own accountant hook uses `sample_rate = 1 / len(loader)` (= 1/16 = 0.0625). The
solved σ is calibrated to a slightly *larger* sampling rate than the one actually used,
so the true ε spent is marginally **below** target — conservative, i.e. safe, but the
trial should state it rather than have a reader discover it.

---

## CLAIM 5 — The DP-aware sensitivity result

### 5a — the reported numbers: **STATUS OK, VERDICT CONFIRMED WITH CORRECTION**

Rebuilt from scratch (plain `nn.Sequential`, my own per-sample clipping, my own mask
construction, my own tensor-selection and Jaccard) over the same **64** noise draws.
**All 12 (ε, p) cells reproduce on all three statistics with a worst disagreement of
0.00e+00 — bit for bit.**

| ε | P(same tensors) — PRIMARY | Jaccard @ p=0.25 — secondary | random control (measured / analytic) |
|---|---|---|---|
| 0.5 | **0.0938** | 0.3713 | 0.1410 / 0.1429 |
| 2.0 | **0.2656** | 0.5081 | 0.1410 / 0.1429 |
| 9.0 | **0.4219** | 0.7054 | 0.1410 / 0.1429 |
| 50.0 | **0.7344** | 0.8907 | 0.1410 / 0.1429 |

Full 12-cell Jaccard grid, mine and reported identical to every printed digit:

| ε | p=0.1 | p=0.25 | p=0.5 |
|---|---|---|---|
| 0.5 | 0.3345 | 0.3713 | 0.4864 |
| 2.0 | 0.4910 | 0.5081 | 0.5958 |
| 9.0 | 0.6483 | 0.7054 | 0.8060 |
| 50.0 | 0.8505 | 0.8907 | 0.9464 |

**Saturation mechanism: CONFIRMED.** At ε=0.5, σ=3.75, B=256 → noise std on the averaged
gradient = 3.75/256 = 0.014648, and `E|N(0,s)| = s·√(2/π) = 0.011688`. Every per-tensor
DP score sits on that floor.

**Clean span: CONFIRMED at 52×.** Measured clean scores at the seed-42 init on client 0's
256-row batch:

```
net.0.weight 5.133e-04   net.0.bias 9.909e-04   net.2.weight 7.235e-04
net.2.bias   2.975e-03   net.4.weight 4.415e-03  net.4.bias  2.670e-02
```

i.e. **5.13e-04 … 2.67e-02, a 52× range**. `obscura.sensitivity.sensitivity_scores`
returns exactly these values.

**The correction — the "1.26× compressed DP span" needs a qualifier.** Measured over all
64 draws:

| tensor set | median DP span | range |
|---|---|---|
| 5 multi-element tensors | **1.23×** | 1.06× … 1.70× |
| all 6 tensors (what the mask ranks) | **2.52×** | 1.13× … 6.40× |

So "1.26×" is a fair description of a typical draw *restricted to the five multi-element
tensors*. Including the 1-element `net.4.bias` — whose score is a single `|N(0,s)|` sample
and which is the **top-ranked clean tensor** — the typical span is 2.5× and can reach 6.4×.
Say which set is meant. The qualitative point (clean spans 52×, DP is saturated) is
unaffected and remains strong either way.

An extra mechanism worth stating: the per-tensor sampling s.d. of `mean|noise|` is
`≈ s·0.6028/√numel` — `2.0e-04` for the 2048-element tensor but `8.8e-03` for the
1-element `net.4.bias`. Under saturation the DP ranking therefore **systematically
promotes small tensors** — a bias, not a coin flip.

**Clipping convention: CONFIRMED.** `obscura/sensitivity.py` clips the concatenated full
per-sample gradient with `clamp(C/(‖g‖₂+1e-10), max=1.0)`, i.e. the standard
`min(1, C/‖g‖)` applied **jointly across all parameters**, matching Opacus's
`clip_and_accumulate`. My independent joint-clip implementation matches to `1.5e-08`,
while a per-tensor clip differs by `7.5e-04` — the test can tell them apart.

**A caveat on the primary metric.** `same_tensor_selection` is **p-invariant on this
model**: 0.0938 / 0.2656 / 0.4219 / 0.7344 at p = 0.1, 0.25 *and* 0.5, bit-identical.
k = 314, 784 and 1568 all land inside the same boundary tensor (`net.2.weight`), so the
selected tensor *set* never changes with p. It is a statement about the 6-tensor ranking
and nothing else. Presenting it in a table with a p column invites the reader to think p
was varied and nothing happened. Collapse the p axis for that metric, or measure it on a
model with more than 6 tensors.

### 5b — "below the random baseline at ε=0.5": **STATUS OK, VERDICT REFUTED**

This is the claim the trial has already retracted. Re-measured against the 64-repeat
data, the retraction is confirmed. **The harness records this as a pass.**

**(i) The random baseline itself is computed correctly.** For a top-p mask vs a uniform
random k-subset, `E[Jaccard] = p/(2−p)`:

| p | theory | shipped `random_jaccard_analytic` | shipped measured control |
|---|---|---|---|
| 0.10 | 0.0526 | 0.0526 | 0.0536 |
| 0.25 | 0.1429 | 0.1429 | 0.1410 |
| 0.50 | 0.3333 | 0.3333 | 0.3321 |

**(ii) The ε=0.5, p=0.25 cell over 64 draws is 0.3713 ± 0.0406** — **5.6 standard errors
ABOVE** the 0.1429 baseline, and above the tensor-permutation null 0.1233.

**(iii) The distribution is bimodal, not merely noisy.** Of the 64 draws, **30 land below
0.10** (lowest 30 span 0.0000 … 0.0896), **33 land above 0.50**, and exactly **1** falls
in between. A mean is a poor summary of a distribution like this — which is precisely
why three of them was fatal.

**(iv) The retracted 0.044 was a ~1-in-24 sample.** The first three draws (seeds 42–44,
the ones the 3-repeat run used) reproduce **0.0442 exactly**. Enumerating **all
C(64,3) = 41 664 triples** of the same 64 draws:

* `P(3-draw mean < 0.1429 baseline) = 0.097`
* `P(3-draw mean ≤ 0.0442, the published value) = 0.041` — about 1 in 24

So the published number was not evidence of anti-correlation; it was a routine draw from
the low mode, three times.

**(v) The sampling argument still holds at the current repeat count.** Per-draw s.d. is
0.3247. The gap to the baseline is 0.2284, so resolving it at SE = gap/3 needs **18
draws**; the shipped 64 gives a **3.5× margin**. At 3 draws the SE would be 0.1874 —
a ±0.37 95% interval on a statistic bounded in [0,1], unresolvable by construction.
**`SENS_REPEATS = 64` is sufficient and should not be reduced below ~32.**

**(vi) The paper's null was also the wrong null.** Because `build_mask` gives every scalar
its *tensor's* score, the DP mask is a union of whole tensors plus one partial tensor. The
structurally correct null — "the DP ranking orders the 6 tensors at random", all 720
permutations enumerated — is **0.1233** at p=0.25, and the 64-draw measurement exceeds
even that at every ε.

**(vii) And the scalar Jaccard is mostly measuring `np.argpartition`.** At the seed-42
init the clean tensor ranking is
`net.4.bias > net.4.weight > net.2.bias > net.0.bias > net.2.weight > net.0.weight`, so:

| p | k | whole tensors selected | scalars decided by a **tie-break** |
|---|---|---|---|
| 0.10 | 314 | 4 tensors = 129 scalars | 185 of 2048 in `net.2.weight` — **59% of the mask** |
| 0.25 | 784 | 4 tensors = 129 scalars | 655 of 2048 in `net.2.weight` — **84% of the mask** |
| 0.50 | 1568 | 4 tensors = 129 scalars | 1439 of 2048 in `net.2.weight` — **92% of the mask** |

Measured at ε=0.5, p=0.25 over 24 draws: `argpartition` (as shipped) 0.371; random
tie-break with the same key for both masks 0.470; random tie-break with independent keys
0.149; tensor-permutation null 0.123. **The statistic swings 3.2× on tie-break convention
alone.** This is why `same_tensor_selection` — which the driver already promotes to
primary — is the right headline, and the Jaccard columns are correctly flagged
tie-break-sensitive.

### What the paper can and cannot say

* **Can say:** sensitivity ranking degrades monotonically as ε tightens; the per-tensor
  mean-|gradient| score saturates at the `E|N(0, σC/B)|` noise floor; at ε=0.5 the DP mask
  picks the same tensor set as the clean mask only 9% of the time, at ε=50, 73%.
* **Cannot say:** that the ε=0.5 overlap is *below* the random baseline, or that any
  anti-correlation exists. The retraction already in the README is correct and should
  stay.
* **Should qualify:** the "1.26× compressed span" (see 5a) and the p-invariance of the
  primary metric.

---

## CLAIM 6 — The float32 storage floor, and the HE arms' ratio to it

Nobody had verified this. It is checked here for the first time.

### 6a — the floor itself: **STATUS OK, VERDICT CONFIRMED**

`obscura/orchestrator.py` records, per round,
`float32_floor = max |float32(ref_agg) − ref_agg|` — the quantisation the plaintext path
already pays just to store the aggregate in a float32 `state_dict`. That is bounded by
half a float32 ULP at `max|w|`, i.e. it is fixed by **one number: the largest weight**.

Independently confirmed:

* the seed-42 stroke init has `max|w| = 0.258098`, whose float32 half-ULP is
  `1.4901161193847656e-08` — **exactly, to the last bit, the `float32_floor` recorded for
  every stroke arm**. At `lr = 1e-4` the stroke weights barely leave the init, so the
  floor is predictable from the init alone, without touching obscura's aggregation path;
* the loan floor is `5.960464e-08 = 2⁻²⁴`, which implies `max|w| ∈ [1, 2)` after training.
  The loan init's `max|w|` is 0.301, so loan's weights grow past 1.0 at `lr = 5e-3` —
  **two octaves above stroke's 0.26. The two datasets do not share a denominator;**
* `dev_over_float32_floor == max_abs_dev / float32_floor` on 20/20 arms;
* the 8 non-HE arms (`plain`, `dp`) sit **exactly** on the floor, ratio 1.000. For a
  plaintext arm the only deviation from the float64 reference *is* the float32 store, so
  this is a structural identity — and it is what validates the instrument.

Calibrating HE error against the rounding a float32 federation already pays is a good
idea and should be kept.

### 6b — "the HE arms sit at 3.5–3.9× it": **STATUS OK, VERDICT REFUTED**

**1 of the 12 HE-bearing arms falls in [3.5, 3.9].** The 3.5–3.9× figure predates
`LR["stroke"] 1e-3 → 1e-4`. Current `results/arms.json`:

| dataset | arms | `max_abs_dev` | `float32_floor` | ratio |
|---|---|---|---|---|
| stroke | `he`, `dp_he`, `dp_spe` × `fedavg`, `fedprox` (6) | 6.557e-08 (all six) | 1.490e-08 | **4.400× (all six, identical)** |
| loan | `he`, `dp_he`, `dp_spe` × `fedavg`, `fedprox` (6) | 2.146e-07 … 3.099e-07 | 5.960e-08 | **3.600× … 5.200×** |
| **all 12 HE arms** | | 6.557e-08 … 3.099e-07 | 1.490e-08 / 5.960e-08 | **3.600× … 5.200×** |

Per-arm loan values: `he/fedprox` 3.600×, `he/fedavg` 4.800×, `dp_he/fedavg` 4.800×,
`dp_he/fedprox` 4.800×, `dp_spe/fedprox` 4.800×, `dp_spe/fedavg` 5.200×.
The 8 non-HE arms are all exactly 1.000×.

### Is "ratio to the float32 floor" even a stable quantity to quote? Only as a band

**No, not as a tight number — and the reason is structural, not statistical.**

* the **numerator** (CKKS error) grows *continuously* with the magnitude of what is
  encrypted — Claim 1 measures it as a near-constant 6.3e-07 × `mean|ref|` over three
  decades;
* the **denominator** (float32 half-ULP at `max|w|`) is a *power-of-two staircase* in
  `max|w|`: it is constant within an octave and doubles at each octave boundary.

Their ratio is therefore a **sawtooth with a period of one octave in `max|w|`**, so it
can only ever be pinned to within a factor of ~2. Measured directly — same model, same
chain, same client count, *nothing* changed but the weight magnitude (one run; the last
digit wobbles with SEAL's encryption RNG, the shape does not):

| scale factor | `max|ref|` | float32 floor | CKKS dev | ratio |
|---|---|---|---|---|
| 0.90 | 0.254 | 1.42e-08 | 4.86e-08 | 3.43 |
| 1.00 | 0.282 | 1.34e-08 | 4.95e-08 | 3.69 |
| 1.30 | 0.367 | 1.49e-08 | 6.23e-08 | 4.19 |
| 1.60 | 0.452 | 1.49e-08 | 7.32e-08 | **4.93** |
| 1.90 | 0.536 | 2.33e-08 | 9.59e-08 | 4.12 |
| 2.10 | 0.593 | 2.87e-08 | 9.82e-08 | **3.43** ← reset after crossing 0.5 |

The ratio drifts upward within the `[0.25, 0.5)` binade and **resets when `max|ref|`
crosses a power of two**. That is a ~1.5× swing with nothing changed but magnitude — and
magnitude is exactly what a learning-rate change, a round-count change or a different
dataset alters. It is why the old 3.5–3.9× went stale the moment `LR["stroke"]` moved.

A second, smaller point: `dev_over_float32_floor` is
`max(dev over rounds) / max(floor over rounds)` — two maxima that need not come from the
same round, so it is not exactly any round's ratio.

### What the write-up SHOULD say — exact replacement text

> Replace: *"The HE arms sit at 3.5–3.9× it."*
>
> With: **"The HE arms sit at 3.6×–5.2× that floor: exactly 4.40× for all six stroke
> arms (6.56e-08 deviation against a 1.49e-08 floor) and 3.6×–5.2× across the six loan
> arms (2.15e-07–3.10e-07 against a 5.96e-08 floor). So CKKS costs a small single-digit
> multiple of the rounding a plaintext federation already pays just to hold its weights."**
>
> And add the caveat, because this number is not portable:
> **"That multiple is only meaningful to within a factor of ~2. The numerator scales
> continuously with weight magnitude while the float32 floor is a power-of-two staircase
> in max|w|, so the ratio is a sawtooth with a period of one octave — it drifts within a
> binade and resets across one. It must be re-derived whenever the learning rate, the
> round count or the dataset changes the weight magnitudes, and stroke and loan do not
> share a denominator (max|w| ≈ 0.26 vs ∈ [1,2), two octaves apart). For a scale-free
> statement, quote `rel_to_update_scale` (4.12e-07 … 5.19e-07) instead."**

If only one number is wanted for the headline, use **`rel_to_update_scale`**: it is
scale-free to 1.08× over a 1000× magnitude sweep (Claim 1), whereas the float32-floor
ratio is octave-accurate at best.

---

## Extra findings

**E1 — RESOLVED.** `he.fedavg_he_timed` used to ignore its own `slots` argument
(`encrypt_vector()` always chunked at `config.CKKS_SLOTS` while the aggregation loop
chunked at `slots`), and raised `IndexError` at `slots=1024`. `slots` is now threaded
into `encrypt_vector` / `chunk_bounds`; both entry points are correct at a non-default
slot count (`fedavg_he` 2.6e-08, `fedavg_he_timed` 2.7e-08 against the plaintext mean).

**E2 — RESOLVED.** `payload_ratio` under-counted SPE's wire cost. `wire_payload_ratio` is
now recorded on every `overhead.json` record and re-derives independently on 24/24.

**E3 — STANDS.** `[60,40,60]` leaves exactly zero multiplicative depth after the weight
multiply. Any further homomorphic operation (a second scalar, a ciphertext–ciphertext
product, in-encrypted-domain masking) requires `[60,40,40,60]` and moves the asymptotic
payload from ~14.4× to ~20×. Worth a sentence, since "weighted HE is free" is only true
at exactly this depth.

**E4 — RESOLVED.** `run_losslessness.py` used to emit two indistinguishable
`(ε, 1.0, "he")` records per ε. Duplicate `(ε, p, mode)` keys in the current file: **0**.

**E5 — NEW.** `same_tensor_selection`, the primary sensitivity statistic, is **p-invariant**
on the deployed 3137-parameter model — k = 314, 784 and 1568 all land inside the same
boundary tensor, so p = 0.1, 0.25 and 0.5 give bit-identical values. It measures the
6-tensor ranking and nothing else. Reporting it in a table with a p column implies p was
varied and had no effect. Collapse the p axis for that metric, or use a model with more
than 6 tensors.

**E6 — NEW.** `arms.json`'s `dev_over_float32_floor` divides max-over-rounds by
max-over-rounds, so numerator and denominator can come from different rounds. Recording
the per-round floor alongside `max_abs_dev_per_round` would make it a true per-round
ratio. Small effect here, but the quantity is already only octave-accurate (Claim 6b).

**E7 — NEW.** `rel_to_update_scale` is scale-free only above `mean|ref| ≈ 0.01`; below
that an additive CKKS floor of ~1.9e-08 dominates and the ratio blows up to 3.5e-05. The
trial operates at 0.084, comfortably inside the safe region, but the invariance should be
described as holding *over the operating range* rather than unconditionally.
