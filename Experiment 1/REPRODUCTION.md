# Reproducing arXiv:2508.10017 — SMOTETomek + FedProx under DP-SGD

Reproduction of the **ε-vs-recall frontier** on non-IID clinical (stroke) data,
showing (1) the non-linear privacy/utility tradeoff with a "knee" around ε≈9.0,
(2) FedProx's advantage over FedAvg, and (3) the SMOTETomek failure baseline.

## Quick start

```bash
python3.11 -m venv fedprox_env && source fedprox_env/bin/activate
pip install -r requirements.txt
# dataset already in data/healthcare-dataset-stroke-data.csv (fedesoriano stroke set)
python run_sweep.py        # baseline + full sweep -> results/results.json  (~10-15 min, CPU)
python plot_results.py     # -> results/recall_vs_epsilon.png, metrics_vs_epsilon.png
```

## Project layout

| File | Role |
|------|------|
| `stroke_fl/config.py` | All hyperparameters (paper Table 3) and the ε grid |
| `stroke_fl/data.py`   | Clean → encode → scale → stratified 80/20 → balanced test → Dirichlet non-IID partition |
| `stroke_fl/model.py`  | FFNN: input→64→32→1 (sigmoid), BCELoss |
| `stroke_fl/train.py`  | SMOTETomek, DP-SGD local training, FedProx proximal term, FedAvg aggregation, metrics |
| `run_sweep.py`        | Baseline failure case + FedProx/FedAvg ε sweep |
| `plot_results.py`     | Frontier + per-metric figures |

## Methodology (as implemented)

- **Data.** fedesoriano stroke set (5110 rows, 4.87% positive). Drop 201 BMI-null
  rows and the single `gender=Other` row → 4908 rows. One-hot encode nominal
  categoricals; `StandardScaler` on `age, avg_glucose_level, bmi` **fit on train
  only**. Stratified 80/20 split; the held-out test set is then **class-balanced**
  (all positives + equal random negatives) so precision/accuracy are meaningful.
- **Non-IID partition.** Per-class **Dirichlet(α=0.5)** across 5 clients (the
  outline explicitly endorses α=0.5), with a **minimum-positives floor** per
  client so SMOTETomek (needs ≥6 minority samples) is feasible everywhere.
- **Client preprocessing.** `SMOTETomek(random_state=42)` applied **independently
  per client** (never globally — that would leak across clients).
- **Local training.** DP-SGD via **Opacus**, Adam lr=1e-3, batch 32, 5 local
  epochs, gradient-clipping C=1.0. FedProx proximal coefficient μ=0.01.
- **FL.** 5 clients, 20 rounds, FedAvg sample-weighted aggregation.
- **ε grid.** [0.5, 1.0, 2.0, 5.0, 7.0, 9.0, 10.0, 15.0, 50.0]; recall is the
  headline metric (false negatives are the dangerous clinical failure).

## Deviations from the literal outline (and why)

1. **No Flower (`flwr`).** The federation is simulated in-process. For a 5-client
   local run, Flower's client/server orchestration changes none of the
   aggregation math but adds fragility (notably its Opacus integration). The
   frontier — the paper's actual claim — is unaffected. Swappable later if needed.

2. **Proximal term injected *after* Opacus's private step.** Under DP-SGD the
   optimizer overwrites `p.grad` with the clipped+noised per-sample gradient, so a
   proximal term merely *added to the loss* (as the outline's snippet shows) is
   **silently discarded** — FedProx would collapse into FedAvg. We instead run
   Opacus's `pre_step()` (clip+noise), add `μ·(w − w_global)` to `p.grad`, then let
   Adam apply the sum. The proximal term is data-independent (it depends only on
   the public global weights), so adding it post-clipping does **not** weaken the
   DP guarantee.

3. **ε is a per-round privacy budget.** Like the outline's per-round
   `make_private_with_epsilon`, each round calibrates noise to hit `target_epsilon`
   over the 5 local epochs. The cumulative budget across 20 rounds is therefore
   looser than the labelled ε; what the experiment varies is the *noise knob*, and
   the monotonic noise→recall relationship that produces the frontier is preserved.
   (We precompute σ with `get_noise_multiplier` and reuse it — identical calibration,
   faster than re-running the accountant search every round.)

4. **SMOTETomek computed once per client, not literally re-run every round.** It is
   deterministic (`random_state=42`) on fixed local data, so every round would
   produce identical resampled data. Caching it is a pure speed optimization.

5. **Dataset from a public GitHub mirror** (Kaggle requires auth); byte-identical
   schema and class balance to fedesoriano's original.

## Reproducibility

Single global initialization shared across all configs; `torch`/`numpy` seeded
from `SEED=42`; Opacus in default (seedable) mode. Re-running `run_sweep.py`
reproduces `results/results.json` exactly on the same machine.
