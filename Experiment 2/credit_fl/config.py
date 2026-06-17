"""Central configuration for the credit-risk three-way FL comparison.

Reproduction of *Scientific Reports* 10.1038/s41598-025-34536-9 -- comparing
Standard FedAvg vs. DP-FL (Opacus DP-SGD) vs. HE-FL (TenSEAL CKKS) on the
Kaggle Loan-Approval-Prediction task, quantifying the accuracy cost of DP and
the accuracy-neutrality of HE.
"""

DATA_PATH = "data/loan_approval_dataset.csv"
RESULTS_DIR = "results"

SEED = 42

# --- FL configuration (paper: 5 clients, 5 rounds, 5 local epochs, IID) ---
NUM_CLIENTS = 5
ROUNDS = 5
LOCAL_EPOCHS = 5
# Shared across ALL arms (Standard / DP / HE) -- only the privacy mechanism
# varies, never the optimiser. 5e-3 lets DP-SGD escape the majority-class basin
# within 25 local epochs; at 1e-3 the DP arms collapse to the 62% base rate while
# the noise-free arms are unaffected, which would overstate DP's cost.
LR = 5e-3
BATCH_SIZE = 64

# --- Model: two-layer MLP (input -> 32 -> 16 -> 1) ---
HIDDEN1 = 32
HIDDEN2 = 16

# --- Differential privacy (Opacus / DP-SGD, Gaussian mechanism + RDP) ---
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0            # per-sample gradient clipping threshold C
# Cumulative epsilon targets the paper derives over the 5 rounds. The
# best-balance point reported in the paper is eps=8.65 (~87-90% accuracy).
DP_EPSILONS = [14.13, 8.65, 5.74]

# --- Split / partition ---
TEST_FRACTION = 0.20          # global 80/20 train/test split
# IID partition: each client gets a random ~1/NUM_CLIENTS share of the train set.
# Intentional -- isolates the privacy mechanism from non-IID heterogeneity.

# --- Homomorphic encryption (TenSEAL CKKS) ---
# poly_modulus_degree 8192 -> 4096 plaintext slots, enough for the ~929-param
# flattened model in a single ciphertext. The 60/40/40/60 coeff-mod chain and a
# 2**40 global scale are the standard CKKS settings for one multiplicative depth
# (the single scalar 1/N average), balancing precision against noise growth.
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40
