# Independent verification of Trial 07/27's crypto-side claims

Auditor: independent verification agent. Date: 2026-07-27.
Reproduce with `/home/user/ObscuraDP/obscura_env/bin/python verify_claims.py` from the
trial root (~17 s, deterministic).

**Method.** Every reference number below was re-derived from raw TenSEAL, raw Opacus,
raw numpy and plain `torch.nn`. `obscura.he`, `obscura.aggregate` and
`obscura.sensitivity` were never used to *produce* a reference — only as the *subject*
of a comparison. `obscura.data` was used only to obtain the same client partition the
experiment used (it is not under test). Nothing outside `verify_claims.py` and this
file was modified.

**Scope.** Crypto and DP-accounting only. Training utility, recall, the ε-vs-accuracy
frontier and the non-private arms were deliberately not examined.

---

## Verdict summary

| Claim | Verdict |
|---|---|
| 1 — CKKS/SPE lossless on DP-noised updates | **CONFIRMED** |
| 2 — Weighted homomorphic aggregation | **CONFIRMED** |
| 3 — 18.0–18.8× payload ratio | **CONFIRMED** (explanation exactly as hypothesised: zero-padding) |
| 4 — DP accounting / `dp_noise_std` | **CONFIRMED WITH CORRECTION** (formula right; `B` is Opacus's `expected_batch_size`, 29–32, not 32; and the *denominator choice* in the headline is not defensible) |
| 5a — DP-aware sensitivity, reported numbers | **CONFIRMED** (reproduced to 4 decimal places) |
| 5b — "below the random baseline at ε=0.5" | **REFUTED** |

---

## CLAIM 1 — CKKS aggregation is lossless, and stays lossless on DP-noised updates

**VERDICT: CONFIRMED.**

| quantity | reported | independently measured |
|---|---|---|
| `max abs dev` over ε×p | 4.90e-08 … 6.90e-08 | 4.91e-08 … 6.65e-08 |
| ratio to DP noise std | 4.87e-07 … 6.94e-06 | 6.10e-07 … 5.83e-06 |

Re-derived with raw `ts.ckks_vector` on five client vectors of the same magnitude the
experiment reported (`update_scale ≈ 0.087`), plus i.i.d. `N(0, σC/B)` DP noise at each
of the four σ values in `results/lossless.json`. Every value lands in the reported band.
The error is CKKS encoding error and is essentially independent of ε — the 14× spread
in the *ratio* comes entirely from the denominator moving, not the numerator.

### SPE scatter-back — specifically tested for mis-indexing

Constructed a 5000-coordinate case designed so that any misalignment is loud:

* mask: 1250 True positions drawn by random permutation — **longest contiguous True run
  is 6**, i.e. definitively non-contiguous;
* client vectors with `mean |v[i] - v[i+1]| = 0.625` on a `|v| ∈ [0, 1.005]` range, so
  neighbouring coordinates are nothing like each other.

Results:

* independent SPE reference vs plaintext weighted mean: `2.57e-07`;
* **positive control** — shifting the encrypted coordinates by exactly one slot on
  scatter-back gives `3.10e+00`. The test therefore has ~7 orders of magnitude of power
  to detect an off-by-one; it did not fire;
* `obscura.he.fedavg_he` on the same input: `2.54e-07`;
  `obscura.he.fedavg_he_timed`: `2.52e-07`;
* mask-polarity probe: the **plaintext half** of the output matches the numpy weighted
  mean to `4.4e-16` (exact float64) while the **encrypted half** matches to `2.5e-07`
  (CKKS precision). An inverted or shifted mask would swap or smear that signature. It
  did not;
* all 5000 coordinates finite, none dropped, `fedavg_he` and `fedavg_he_timed` agree.

The scatter-back is correct.

### Correction the write-up should make

`rel_to_noise_floor` divides a deviation measured on **aggregated model weights** by a
**per-step gradient noise standard deviation**. Those are different spaces, and it is
why the published ratio wanders over 14× (4.9e-07 → 6.9e-06) with ε while the numerator
barely moves. Dividing by the scale of the thing actually being encrypted
(`mean|w| ≈ 0.089`) gives a stable **5.59e-07 … 7.60e-07** across all 16 records.

> Replace "HE error is 5e-7 to 7e-6 times the DP noise floor" with
> "HE aggregation error is 5.6e-07 to 7.6e-07 of the magnitude of the update being
> aggregated, i.e. ~7 orders of magnitude below the signal, at every ε tested."

That is the same headline, defensible, and *more* stable.

---

## CLAIM 2 — Weighted homomorphic aggregation is correct

**VERDICT: CONFIRMED.**

Weights ∝ the real stroke client sizes `[484, 234, 1120, 4920, 738]` →
`[0.0646, 0.0312, 0.1494, 0.6564, 0.0985]`. Deliberately unequal: the weighted answer
differs from the unweighted mean by 0.20 at `|v| ~ 0.1`, i.e. **7 orders of magnitude
larger than the HE error**, so the test genuinely distinguishes weighted from unweighted.

| `|v|` scale | max abs dev | max rel dev | \|weighted − unweighted\| |
|---|---|---|---|
| 1e-03 | 1.3e-08 | 6.3e-06 | 2.0e-03 |
| 1e-02 | 1.9e-08 | 8.1e-07 | 2.0e-02 |
| 1e-01 | 3.9e-08 | 1.4e-07 | 2.0e-01 |
| 1e+00 | 3.2e-07 | 1.3e-07 | 2.5e+00 |
| 1e+01 | 3.2e-06 | 1.3e-07 | 1.98e+01 |

Relative error is flat at ~1.3e-07 across four decades of magnitude — the signature of a
correct depth-1 CKKS computation, with no silent precision collapse. `obscura.he`'s
weighted full-HE path matches the independent plaintext weighted mean to `3.6e-08`.

**Depth budget, directly probed.** With `[60, 40, 60]`:

* one scalar multiply on a fresh ciphertext: **works**;
* two scalar multiplies: **raises `ValueError: scale out of bounds`**.

So `cᵢ · wᵢ` then `Σ` is exactly depth 1 and fits — but with **zero headroom left**. The
docstring's claim that the chain "still suffices" is true and the reason is right (the
weight multiply costs exactly what the old `1/N` multiply cost). Worth saying explicitly
in the write-up: there is no room for a second homomorphic operation without moving to
`[60, 40, 40, 60]`, which is the ~20× chain Experiment 2 used.

By construction, only the aggregate ciphertext is ever `.decrypt()`ed — no individual
client vector is decrypted anywhere in `he.py`.

---

## CLAIM 3 — The reported payload ratio

**VERDICT: CONFIRMED. The 18.0–18.8× is 100% zero-padding. No modulus-chain difference,
no measurement error.**

Measured, directly: a TenSEAL CKKS ciphertext under `poly_modulus_degree=8192`,
`coeff_mod=[60,40,60]` serialises to **≈ 235 360 B regardless of how many of its 4096
slots are filled** (1 slot → 235 324 B; 4096 slots → 235 377 B).

Analytic model: `ratio(n) = ceil(n / 4096) · 235 360 B / (4n)`.

| n_params | chunks | predicted | reported (`overhead.json`) | abs diff |
|---|---|---|---|---|
| 3 137 | 1 | 18.758 | 18.757 | 0.0006 |
| 10 369 | 3 | 17.025 | 17.023 | 0.0017 |
| 37 121 | 10 | 15.852 | 15.851 | 0.0006 |
| 139 777 | 35 | 14.734 | 14.733 | 0.0010 |
| 541 697 | 133 | 14.448 | 14.447 | 0.0009 |
| 2 131 969 | 521 | 14.380 | 14.379 | 0.0010 |

Every reported value is reproduced to < 0.002. And decisively:

```
18.758 / 14.365 = 1.3058      4096 / 3137 = 1.3057
```

The padding factor accounts for the discrepancy to four significant figures.

**Asymptotic ratio measured: 14.365×** on a slot-aligned 81 920-element vector. Exactly
Experiment 3's constant 14.4× — and Experiment 3's own README says "4096 reals pack into
one ~235 KB ciphertext", the same ciphertext I measured. Same chain, same everything.

Cross-check on the chain itself: an uncompressed fresh ciphertext under `[60,40,60]` is
`2 polys × 8192 coeffs × 2 data primes × 8 B = 262 144 B` (the trailing 60-bit prime is
SEAL's special key-switching prime and is not part of a fresh ciphertext), giving a raw
ratio of exactly 16.0. Serialised is 0.898× that, i.e. SEAL's ZSTD compression, which
lands the ratio at 14.365. Both numbers confirm the chain is `[60,40,60]`, not a longer
one.

### Correction the write-up should make

`payload_ratio` for SPE rows divides the ciphertext bytes of the **encrypted subset** by
the plaintext bytes of the **whole model**. It omits the unencrypted parameters the
client still has to transmit. True wire ratio at `n=3137`:

| p | reported | true wire ratio |
|---|---|---|
| 0.10 | 18.758 | 19.658 |
| 0.25 | 18.757 | 19.507 |
| 0.50 | 18.757 | 19.257 |
| 0.75 | 18.758 | 19.008 |
| 1.00 | 18.758 | 18.758 |

Either report the true wire ratio, or label the current metric "ciphertext payload
relative to a full-plaintext baseline". At `n=3137` the flatness across p is an artifact
of a 3137-parameter model needing one chunk at any p ≥ 0.1 — say so, because as printed
it reads as "SPE does not reduce payload", which is false for any model above ~4k
parameters (`crypto_vs_size` shows 3.6× at p=0.25 for 2.1M params).

---

## CLAIM 4 — The DP accounting

**VERDICT: CONFIRMED WITH CORRECTION.**

### σ → ε round trip (raw Opacus `get_noise_multiplier` + `RDPAccountant`, n_local = 484)

| target ε | σ | ε_round recovered | err | ε cumulative over 10 rounds |
|---|---|---|---|---|
| 0.5 | 3.750000 | 0.4976 | −0.48% | 1.653 |
| 2.0 | 1.406250 | 1.9980 | −0.10% | 5.935 |
| 9.0 | 0.686035 | 8.9957 | −0.05% | 24.240 |
| 50.0 | 0.329285 | 49.9958 | −0.01% | 165.987 |

Round trip is correct (residual is `get_noise_multiplier`'s binary-search granularity).
Composition is monotone and sub-linear: ε(1 round)=9.00 → ε(10 rounds)=24.24, well below
10 × 9.00, which is the expected RDP behaviour. `obscura.dp` reproduces all three
quantities bit-for-bit.

### Is `dp_noise_std = σ·C/B` the right expression? — checked against Opacus's source *and* empirically

From `opacus/optimizers/optimizer.py`:

```python
def add_noise(self):
    noise = _generate_noise(std=self.noise_multiplier * self.max_grad_norm, ...)
    p.grad = (p.summed_grad + noise).view_as(p)

def scale_grad(self):
    if self.loss_reduction == "mean":
        p.grad /= self.expected_batch_size * self.accumulated_iterations
```

Noise of std `σ·C` is added to the **summed** clipped gradient, then divided by
`expected_batch_size`. So the noise on the **averaged** gradient has std
`σ·C / expected_batch_size`. **`σ·C/B` is the correct expression** — it is *not* off by
a factor of B.

Empirically confirmed: with σ=50, C=1, a real `PrivacyEngine`-wrapped 3137-parameter
model, `std(p.grad)` measured over 6 steps = **1.5516**, against `σC/B = 1.5625`
(ratio 0.993). The "no 1/B" alternative would predict 50.0 (ratio 0.031) and is
decisively excluded.

### The correction

`dp_noise_std` hard-codes `B = config.BATCH_SIZE = 32`. Opacus uses
`expected_batch_size = int(len(dataset) / len(loader))`, which for these clients is:

| n_local | len(loader) | Opacus `expected_batch_size` | error in `dp_noise_std` |
|---|---|---|---|
| 484 | 16 | 30 | +6.7% |
| 234 | 8 | 29 | +10.3% |
| 1120 | 35 | 32 | 0.0% |
| 4920 | 154 | 31 | +3.2% |
| 738 | 24 | 30 | +6.7% |

`dp_noise_std` therefore **understates the true noise floor by up to 10%** (mean ≈ 5%).
This is immaterial to a "1e-6 ×" headline, but the write-up should say `B` is Opacus's
`expected_batch_size`, not the nominal batch size.

A related, separately reportable mismatch: `get_sigma` accounts with
`sample_rate = min(1, batch_size / n_local)` (= 32/484 = 0.0661 for client 0), whereas
Opacus's own accountant hook uses `sample_rate = 1 / len(loader)` (= 1/16 = 0.0625).
The solved σ is therefore calibrated to a slightly *larger* sampling rate than the one
actually used, so the true ε spent is marginally **below** target — conservative, i.e.
safe, but the trial should state it rather than have a reader discover it.

### The bigger issue with the headline

See Claim 1: even with the correct `B`, comparing a weight-space HE deviation to a
gradient-space per-step noise std is the wrong ratio to publish. Use the update-scale
normalisation.

---

## CLAIM 5 — The DP-aware sensitivity result

### 5a — the reported numbers: **CONFIRMED, exactly**

Rebuilt from scratch (plain `nn.Sequential`, my own per-sample clipping, my own mask
construction, my own Jaccard) — mean over the same 3 repeats:

| ε | p=0.1 | p=0.25 | p=0.5 |
|---|---|---|---|
| 0.5 | 0.1223 / **0.1223** | 0.0442 / **0.0442** | 0.1039 / **0.1039** |
| 2.0 | 0.4380 / **0.4380** | 0.3704 / **0.3704** | 0.4020 / **0.4020** |
| 9.0 | 0.8124 / **0.8124** | 0.8629 / **0.8629** | 0.9327 / **0.9327** |
| 50.0 | 1.0000 / **1.0000** | 1.0000 / **1.0000** | 1.0000 / **1.0000** |

(mine / reported — identical to 4 decimals in all 12 cells.)

**Saturation mechanism: CONFIRMED.** At ε=0.5, σ=3.75, B=256 → noise std on the averaged
gradient = 3.75/256 = 0.014648, and `E|N(0,s)| = s·√(2/π) = 0.011688`. Every per-tensor
DP score sits on that floor. Measured DP spans for the three shipped repeats, restricted
to the five multi-element tensors: **1.07× / 1.22× / 1.44×**. So the write-up's "1.09e-2
to 1.37e-2, a 1.26× range" is a fair description of a typical draw.

**But the quoted clean span is wrong.** The measured clean scores at the seed-42 init on
client 0's 256-row batch are:

```
net.0.weight 5.133e-04   net.0.bias 9.909e-04   net.2.weight 7.235e-04
net.2.bias   2.975e-03   net.4.weight 4.415e-03  net.4.bias  2.670e-02
```

i.e. span **5.13e-04 … 2.67e-02, a 52× range**, not "8.2e-04 to 2.0e-02, a 24× range".
`obscura.sensitivity.sensitivity_scores` returns exactly these values, so this is not a
disagreement between my code and theirs — the number in the write-up does not correspond
to the configuration that produced `results/sensitivity.json`. **Fix the quoted clean
range to 5.1e-04 … 2.7e-02 (52×).** (The qualitative point — clean is a wide range, DP is
saturated — is *strengthened*, not weakened.)

An extra mechanism the write-up misses: the per-tensor sampling s.d. of `mean|noise|` is
`≈ s·0.6028/√numel`, i.e. `2.0e-04` for the 2048-element tensor but `8.8e-03` for the
1-element `net.4.bias`. Under saturation the DP ranking therefore **systematically
promotes small tensors** — a bias, not a coin flip. That is worth stating.

**5a(iii) clipping convention: CONFIRMED.** `obscura/sensitivity.py` clips the
concatenated full per-sample gradient with `clamp(C/(‖g‖₂+1e-10), max=1.0)`, i.e. the
standard `min(1, C/‖g‖)` applied **jointly across all parameters**. Numerically: my
independent joint-clip implementation matches to `1.5e-08`, while a per-tensor clip
differs by `7.5e-04` — so the test can tell them apart, and obscura is on the joint side.
This matches Opacus's `clip_and_accumulate`, which also norms across all params jointly.

### 5b — "below the random baseline at ε=0.5": **REFUTED**

**(i) The random baseline itself is computed correctly.** For a top-p mask vs a uniform
random k-subset, `E[Jaccard] = p/(2−p)`:

| p | theory | reported |
|---|---|---|
| 0.10 | 0.0526 | 0.0520 |
| 0.25 | 0.1429 | 0.1415 |
| 0.50 | 0.3333 | 0.3356 |

All three match. The baseline is right.

**(ii) The sub-random result is not real.** Two independent reasons, both measured.

**Reason 1 — three repeats cannot resolve this statistic.** The per-draw Jaccard at
ε=0.5, p=0.25 is near-bimodal with s.d. 0.34. Running the *identical* estimator, the
*identical* tie-break, on 64 noise draws instead of 3:

| ε | p | shipped (3 rep) | 64 rep ± SE | frac of draws > 0.5 | tensor-permutation null |
|---|---|---|---|---|---|
| 0.5 | 0.10 | 0.1223 | **0.3345 ± 0.031** | 0.14 | 0.1032 |
| 0.5 | 0.25 | 0.0442 | **0.3713 ± 0.041** | 0.52 | 0.1234 |
| 0.5 | 0.50 | 0.1039 | **0.4864 ± 0.046** | 0.53 | 0.3599 |
| 2.0 | 0.25 | 0.3704 | 0.5081 ± 0.046 | 0.64 | 0.1234 |
| 9.0 | 0.25 | 0.8629 | 0.7054 ± 0.037 | 0.89 | 0.1234 |
| 50.0 | 0.25 | 1.0000 | 0.8907 ± 0.023 | 1.00 | 0.1234 |

**0.044 → 0.371 ± 0.041.** The published value is a three-sample fluke: seeds 42/43/44
happened to draw the low mode three times. The 64-draw estimate sits *above* the
p/(2−p) = 0.143 baseline by more than 5 standard errors. **There is no anti-correlation.**
Note that the endpoint 1.00 at ε=50 is also a small-sample artifact — the honest value
is 0.89 ± 0.02.

**Reason 2 — the paper's null is the wrong null anyway.** Because
`build_mask` gives every scalar its *tensor's* score, the DP mask is not an arbitrary
subset of scalars: it is a union of whole tensors plus one partial tensor. The correct
null is "the DP ranking orders the 6 tensors at random", which I computed by enumerating
all 720 permutations: **0.1032 / 0.1234 / 0.3599** for p = 0.1 / 0.25 / 0.5. Even that
lower null is exceeded by the 64-draw measurement at every ε.

**(iii) And the metric is mostly measuring `np.argpartition`.** This is the most
important structural finding. At the seed-42 init, the clean tensor ranking is
`net.4.bias > net.4.weight > net.2.bias > net.0.bias > net.2.weight > net.0.weight`, so:

| p | k | whole tensors selected | scalars decided by a **tie-break** among equal scores |
|---|---|---|---|
| 0.10 | 314 | 4 tensors = 129 scalars | 185 of 2048 in `net.2.weight` — **59% of the mask** |
| 0.25 | 784 | 4 tensors = 129 scalars | 655 of 2048 in `net.2.weight` — **84% of the mask** |
| 0.50 | 1568 | 4 tensors = 129 scalars | 1439 of 2048 in `net.2.weight` — **92% of the mask** |

At the headline operating point p=0.25, **84% of the mask is an arbitrary subset of one
tensor**, picked by whatever order `np.argpartition`'s introselect happens to produce.
Whether the two masks agree there is determined by the whole score array's values, not by
sensitivity. Directly measured at ε=0.5, p=0.25 over 24 draws:

* `np.argpartition` tie-break (as shipped): mean 0.371, sd 0.345
* random tie-break, same key for both masks: mean 0.470, sd 0.422
* random tie-break, independent keys: mean 0.149, sd 0.101
* tensor-permutation null: 0.124

The statistic swings by 3× depending purely on the tie-break convention. It survives none
of these perturbations as a "sub-random" result.

### The metric the write-up should report instead

Make the tie-break deterministic (stable, lowest-index-first) so the Jaccard becomes a
pure function of the 6-tensor ranking, and/or report the ranking-agreement probability
directly. Both are stable and both still show the monotone degradation the trial wants:

| ε | Jaccard, index tie-break, p=0.25, 64 draws | P(DP ranking selects the same tensors as clean) |
|---|---|---|
| 0.5 | 0.488 | **0.02** |
| 2.0 | 0.610 | **0.20** |
| 9.0 | 0.828 | **0.42** |
| 50.0 | 0.960 | **0.73** |

`P(same tensor selection)` is the honest headline: **at ε=0.5 the DP-scored mask picks
the same set of tensors as the clean mask only 2% of the time; at ε=50, 73%.** That is a
strong, clean, tie-break-free version of exactly the finding the trial wants to claim —
and it does not require the (false) "worse than random" wording.

### What the paper can and cannot say

* **Can say:** sensitivity ranking degrades monotonically as ε tightens; the per-tensor
  mean-|gradient| score saturates at the `E|N(0, σC/B)|` noise floor; at ε=0.5 the mask
  is essentially uninformative.
* **Cannot say:** that the ε=0.5 overlap is *below* the random baseline, or that any
  anti-correlation exists. Delete that sentence.
* **Must fix:** `SENS_REPEATS = 3` is far too few for a statistic with s.d. 0.34. Use
  ≥ 50. And fix the quoted clean-score range.

---

## Extra findings (not asked for)

**E1 — latent correctness bug in `he.fedavg_he_timed`.** It ignores its own `slots`
argument: `encrypt_vector()` always chunks at `config.CKKS_SLOTS`, while the aggregation
loop chunks at `slots`. Whenever `slots != 4096` the two disagree.

```
fedavg_he(clients, mask, ctx, slots=1024)        -> dev 2.9e-08 (correct)
fedavg_he_timed(clients, mask, ctx, slots=1024)  -> IndexError: list index out of range
```

Latent only — every current caller uses the default — but `fedavg_he` and
`fedavg_he_timed` are documented as the same computation and are not.
Fix: thread `slots` into `encrypt_vector` / `chunk_bounds`. (I did not edit `he.py`;
that file belongs to another agent.)

**E2 — `payload_ratio` under-counts SPE's wire cost.** See Claim 3.

**E3 — zero remaining multiplicative depth.** `[60,40,60]` supports exactly the one
scalar multiply the weighted aggregation needs. Any further homomorphic operation
(a second scalar, a ciphertext–ciphertext product, in-encrypted-domain masking) requires
`[60,40,40,60]` and moves the asymptotic payload from ~14.4× to ~20×. Worth a sentence in
the write-up, since "weighted HE is free" is only true at exactly this depth.

**E4 — `run_losslessness.py` label collision.** With `LOSSLESS_FRACTIONS = [0.25, 1.0]`
the loop emits a `mode="he"` record with `spe_fraction: 1.0` on *both* iterations, so
`results/lossless.json` contains two indistinguishable `(ε, 1.0, "he")` records per ε and
only one `(ε, 0.25, "spe")`. Harmless for the min/max ranges quoted, but any per-p
grouping of that file will silently double-count.
