"""Central configuration for Trial 07/27 -- the unified DP + FL + HE stack.

This trial composes the three prior ObscuraDP reproductions into ONE pipeline:

  * Experiment 1 (arXiv:2508.10017)  -> non-IID Dirichlet partitioning,
    per-client SMOTETomek rebalancing, FedProx proximal term, Opacus DP-SGD on
    imbalanced clinical data, and the epsilon-vs-recall frontier.
  * Experiment 2 (Sci. Rep. 10.1038/s41598-025-34536-9) -> the arm-matrix
    methodology (one controlled variable: only the privacy mechanism changes),
    MinMax->[0,1] scaling for CKKS-friendly magnitudes, and TenSEAL CKKS
    homomorphic aggregation with payload / latency accounting.
  * Experiment 3 (arXiv:2303.10837, FedML-HE) -> chunked CKKS aggregation,
    Selective Parameter Encryption (SPE) driven by gradient-magnitude
    sensitivity, O(n) overhead scaling, and the exact losslessness check.

The new question none of the three could answer on its own: when DP, FL and HE
run SIMULTANEOUSLY, do their costs compose additively, and does SPE stay
lossless and sensitivity-faithful when the updates it encrypts are DP-noised?

Every hyperparameter lives here so all six drivers share one source of truth.
"""

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
STROKE_PATH = "data/healthcare-dataset-stroke-data.csv"
LOAN_PATH = "data/loan_approval_dataset.csv"

SEED = 42
SEEDS = [42, 43, 44, 45, 46]          # multi-seed campaign (Experiment F)

# --------------------------------------------------------------------------
# Federated learning (shared by every arm -- only the privacy stack varies)
# --------------------------------------------------------------------------
NUM_CLIENTS = 5
ROUNDS = 20                            # Exp 1's round count; see the note below
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
MU = 0.01                              # FedProx proximal coefficient (mu=0 -> FedAvg)

# ROUNDS was briefly halved to 10 to pay for the second dataset and the five
# stacks. That was a false economy: epsilon acts on utility only through how far
# the noisy trajectory gets, so at 10 rounds every epsilon in EPSILON_GRID landed
# on the same operating point (recall 0.5238 at eps = 0.5, 1.0 and 9.0 alike) and
# the frontier had no utility axis at all. Restoring Exp 1's 20 rounds restores a
# monotone recall-vs-epsilon curve. LOCAL_EPOCHS stays at 3 rather than Exp 1's 5
# because per-round sigma is solved for LOCAL_EPOCHS: more local epochs buy the
# same epsilon at a higher sigma and measurably COMPRESS the epsilon spread
# (0.119 recall spread at 3 local epochs vs 0.071 at 5), besides costing time.

# Per-dataset learning rates, applied identically across every arm of their
# dataset so the privacy mechanism remains the single controlled variable.
# Exp 2 needed 5e-3 on loan so DP-SGD could escape the 62% majority-class basin.
#
# Stroke deviates from Exp 1's 1e-3 down to 1e-4, and this is the one deviation
# that matters. Exp 1 had no non-private arm: every one of its runs went through
# Opacus, whose per-sample clipping bounded the local update. This trial's
# `plain` and `he` arms do not, and per-client SMOTETomek leaves each client a
# LOCALLY SEPARABLE problem (the largest client interpolates 2460 positives from
# just 22 real ones), so plain BCE has no finite minimiser there and Adam simply
# inflates the logit scale for ever. At 1e-3 that produced |logit| growing 3.3 ->
# 12.9 over 10 rounds, BCE 7.76 and a 4.8% predicted-positive rate on a balanced
# test set. 1e-4 keeps every arm inside the calibrated regime (plain BCE 0.81)
# without changing the optimiser, the schedule or anything else.
LR = {"stroke": 1e-4, "loan": 5e-3}

# FedAvg aggregation weights. Standard FedAvg weights client c by n_c because
# n_c counts that client's independent samples. After per-client SMOTETomek that
# is no longer true: n_c counts how aggressively the client had to duplicate its
# minority class, so the MOST imbalanced client wins the most weight. On the
# stroke partition client 3 holds 22 of the 167 real stroke cases (13%) yet its
# 4920 resampled rows carry 66% of the aggregate, while client 1 -- which holds
# 93 real cases, 56% of all of them -- carries 3%. Weighting the resampled
# datasets uniformly gives every client's real evidence a voice and is what makes
# the non-private arm sane. Loan is not resampled, so it keeps sample weighting.
AGG_WEIGHTED = {"stroke": False, "loan": True}

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# stroke: 15 -> 64 -> 32 -> 1  (Exp 1's FFNN, ~3.3k params)
# loan:   11 -> 32 -> 16 -> 1  (Exp 2's MLP,  929 params)
HIDDEN = {"stroke": (64, 32), "loan": (32, 16)}

# Width multipliers for the overhead-vs-model-size curve (Experiment E). The
# stroke FFNN is widened by these factors to sweep ~3.3k -> ~1M parameters and
# expose the O(n) crypto scaling on the SAME architecture family.
WIDTH_SCALES = [1, 2, 4, 8, 16, 32]

# --------------------------------------------------------------------------
# Differential privacy (Opacus DP-SGD, Gaussian mechanism + RDP accounting)
# --------------------------------------------------------------------------
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0                    # per-sample clipping threshold C
# Per-round epsilon budget (Exp 1 convention: sigma solved for LOCAL_EPOCHS at
# the client's sampling rate, reused every round). `cumulative_epsilon()` in
# dp.py reports the composed spend over all ROUNDS alongside it.
EPSILON_GRID = [0.5, 1.0, 2.0, 5.0, 9.0, 15.0, 50.0]
EPSILON_DEFAULT = 9.0                  # the Exp-1 "knee", used for single-eps runs

# --------------------------------------------------------------------------
# Homomorphic encryption (TenSEAL CKKS)
# --------------------------------------------------------------------------
# poly_modulus_degree 8192 -> 4096 packed reals/ciphertext. The [60,40,60]
# chain (depth 1 -- enough for the single 1/N average) is Exp 3's setting and
# lands the payload at ~14x plaintext; Exp 2 used [60,40,40,60] (~20x). We keep
# Exp 3's chain as the default because this trial's headline is overhead.
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40
CKKS_SLOTS = CKKS_POLY_MODULUS_DEGREE // 2      # 4096

# --------------------------------------------------------------------------
# Selective Parameter Encryption
# --------------------------------------------------------------------------
# p = fraction of parameters aggregated under CKKS (the rest in plaintext).
# p=0.0 -> no encryption, p=1.0 -> naive full HE.
SPE_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SPE_DEFAULT = 0.25                     # Exp 3's headline operating point

# How the top-p mask is selected. "clean" scores gradients without DP (Exp 3's
# recipe); "dp" scores them through the same clip+noise DP-SGD applies, which is
# what a real DP deployment would have to do; "random" is the control.
MASK_MODES = ["clean", "dp", "random"]
MASK_MODE_DEFAULT = "clean"

# --------------------------------------------------------------------------
# Non-IID partitioning (stroke) / IID partitioning (loan)
# --------------------------------------------------------------------------
DIRICHLET_ALPHA = 0.5                  # label-skew concentration; lower = more non-IID
MIN_POS_PER_CLIENT = 12                # floor so per-client SMOTETomek is feasible
TEST_FRACTION = 0.20

# --------------------------------------------------------------------------
# Experiment grids
# --------------------------------------------------------------------------
# A -- arm matrix: every (stack, algorithm) pair on both datasets.
ARM_STACKS = ["plain", "dp", "he", "dp_he", "dp_spe"]
ARM_ALGOS = ["fedavg", "fedprox"]
ARM_DATASETS = ["stroke", "loan"]

# B -- joint frontier: does utility depend only on eps and cost only on p?
FRONTIER_EPSILONS = EPSILON_GRID
FRONTIER_FRACTIONS = [0.0, 0.25, 1.0]
FRONTIER_ALGO = "fedprox"
FRONTIER_DATASET = "stroke"

# C -- losslessness under DP noise: aggregate the SAME noised updates three ways.
LOSSLESS_EPSILONS = [0.5, 2.0, 9.0, 50.0]
LOSSLESS_FRACTIONS = [0.25, 1.0]

# D -- DP-aware sensitivity: mask agreement + utility under each mask mode.
SENS_EPSILONS = [0.5, 2.0, 9.0, 50.0]
SENS_FRACTIONS = [0.1, 0.25, 0.5]
SENS_REPEATS = 3                       # noise draws per (eps, p) for mask stability

# E -- overhead composition: crypto vs p and vs model size, plus DP's compute tax.
OVERHEAD_FRACTIONS = SPE_FRACTIONS
OVERHEAD_REPEATS = 3                   # timing repeats; medians are reported

# --------------------------------------------------------------------------
# Numerical tolerances (the exactness assertions in run_smoke.py)
# --------------------------------------------------------------------------
CKKS_TOLERANCE = 1e-4                  # max |HE - plaintext| on a model update
