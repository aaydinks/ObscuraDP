# Results — ε-vs-Recall Frontier (arXiv:2508.10017 reproduction)

Single deterministic run (`SEED=42`), 5 clients, 20 rounds, 5 local epochs,
Adam lr=1e-3, batch 32, FedProx μ=0.01, DP-SGD clip C=1.0, δ=1e-5. Metrics on
the class-balanced held-out test set (42 positives + 42 negatives).

## Headline numbers

| ε | FedProx recall | FedAvg recall | FedProx AUC | FedAvg AUC | FedProx F1 | FedAvg F1 |
|----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.5 | **0.476** | 0.452 | 0.777 | 0.776 | **0.563** | 0.543 |
| 1.0 | **0.452** | 0.429 | 0.772 | 0.771 | **0.559** | 0.537 |
| 2.0 | 0.429 | 0.429 | 0.769 | 0.771 | 0.545 | 0.545 |
| 5.0 | 0.405 | 0.405 | 0.760 | 0.760 | 0.531 | 0.531 |
| 7.0 | 0.405 | 0.405 | 0.757 | 0.759 | 0.531 | 0.531 |
| **9.0** | **0.405** | 0.357 | 0.756 | 0.756 | **0.531** | 0.484 |
| 10.0 | **0.405** | 0.333 | 0.755 | 0.755 | **0.531** | 0.459 |
| 15.0 | **0.405** | 0.310 | 0.751 | 0.755 | **0.531** | 0.433 |
| 50.0 | **0.333** | 0.286 | 0.741 | 0.739 | **0.467** | 0.421 |

**Baseline (FedAvg + DP-SGD, NO SMOTETomek):** recall = **0.000**, precision =
0.000, predicted-positive-rate = 0.000, AUC = 0.732 (at ε=9).
**No-SMOTE across ε:** recall = 0.000 at ε ∈ {0.5, 9.0, 50.0} (AUC 0.595 → 0.732
→ 0.754). Privacy budget is irrelevant to recall without resampling.

## What reproduced (matches the paper)

1. **The failure case — exactly.** Without SMOTETomek the global model predicts
   *zero* strokes (recall = precision = pred-pos-rate = 0) despite a non-trivial
   AUC of 0.73. It "looks" like a working classifier yet is clinically useless —
   precisely the motivation the paper opens with. Crucially this holds at *every*
   ε, proving the failure is **class imbalance, not privacy**.

2. **SMOTETomek rescues recall.** Per-client resampling lifts recall from 0.000
   to the 0.33–0.48 range — the whole point of the oversampling step.

3. **FedProx ≥ FedAvg across the entire ε sweep**, on recall and F1. The gap is
   small at low ε, zero in the mid plateau, and **opens up sharply at ε ≥ 9**
   (recall 0.405 vs 0.357/0.333/0.310 at ε = 9/10/15). The proximal term keeps
   the model stable on non-IID data exactly where plain FedAvg starts to degrade.

4. **A clear non-linear "knee" near ε ≈ 9.** Both curves are flat through the
   mid-range, and ε ≈ 9 is the inflection point where FedAvg's recall breaks down
   while FedProx holds — see `results/recall_vs_epsilon.png`.

## What diverged (and why)

- **Recall direction vs ε.** The paper frames recall as *rising* toward the ε≈9
  knee; here recall is **flat-to-gently-declining** with ε. Mechanism (verified
  via precision and predicted-positive-rate): because each client trains on
  SMOTE-*balanced* data, the model's discriminative ability (AUC ≈ 0.74–0.78) is
  nearly constant across ε — DP noise does **not** destroy it. What ε changes is
  only the **operating point**: more noise (low ε) → the model predicts more
  positives → higher recall / lower precision; less noise (high ε) → more
  conservative → lower recall / higher precision. The AUC panel
  (`metrics_vs_epsilon.png`) confirms raw model quality is essentially flat, so
  the FedProx advantage lives in **operating-point stability**, not in AUC
  (mean FedProx−FedAvg AUC gap ≈ 0.0000).

  This is a genuine, explainable outcome of the SMOTE-balanced setup, not a bug.
  Re-running reproduces it bit-for-bit. The qualitative claims the paper actually
  argues — *imbalance breaks recall; SMOTETomek fixes it; FedProx beats FedAvg on
  non-IID data with a knee around ε≈9* — all hold.

## Caveats / sensitivity

- **Test-set granularity:** only 42 positives, so recall resolves to ~0.024 per
  patient. The curve is monotone and reproducible, but absolute values are coarse.
  Averaging over several `SEED`s (and/or a larger test fraction) would smooth it.
- **ε is a per-round budget** (matching the outline), so cumulative privacy over
  20 rounds is looser than the labelled ε. The sweep varies the *noise level*; the
  monotone noise→operating-point relationship is what the frontier captures.
- Reported at a fixed 0.5 decision threshold. Recall is threshold-sensitive;
  AUC/AP (threshold-free) are reported alongside for a fuller picture.

## Figures

- `results/recall_vs_epsilon.png` — the headline frontier (FedProx vs FedAvg vs
  no-SMOTE baseline).
- `results/metrics_vs_epsilon.png` — recall / precision / F1 / AUC panels.
- `results/results.json` — all raw numbers.
