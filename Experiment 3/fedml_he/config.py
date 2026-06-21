"""Central configuration for the FedML-HE overhead reproduction.

Reproduction of arXiv:2303.10837 (FedML-HE) -- the HE overhead benchmarks:
  * the ~15x naive-HE penalty and its O(n) scaling in model size, and
  * the Selective Parameter Encryption (SPE) reduction (overhead ~ p).
"""

RESULTS_DIR = "results"
DATA_DIR = "data"

SEED = 42

# --- Federated setup (common to all benchmarks) ---
NUM_CLIENTS = 5               # IID clients; the focus is HE overhead, not non-IID
LOCAL_EPOCHS = 1              # paper cares about per-round overhead, not convergence

# --- CKKS (TenSEAL) ----------------------------------------------------------
# poly_modulus_degree 8192 -> 4096 plaintext slots per ciphertext, so a model
# with millions of parameters is encrypted as ceil(n/4096) chunked ciphertexts.
# The [60,40,60] modulus chain (depth 1, enough for the single 1/N average) with
# a 2**40 scale lands the per-parameter payload at ~14x plaintext -- reproducing
# the paper's "~15x" communication blow-up. A longer chain (e.g. [60,40,40,60])
# pushes it to ~20x; a shorter one loses precision.
CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40
CKKS_SLOTS = CKKS_POLY_MODULUS_DEGREE // 2     # 4096 packed reals per ciphertext

# --- Models for the scaling curve (Experiment A/B) ---
# Ordered by parameter count; spans ~0.1M -> ~86M to expose the O(n) line.
# ViT-B/16 (~86M) is the heavy end (~1.5 min to encrypt) -- toggled by RUN_VIT.
OVERHEAD_MODELS = ["SmallCNN", "LSTM", "ResNet-18", "ResNet-34", "ResNet-50"]
RUN_VIT = True                # append "ViT-B/16" to reach BERT-scale parameter counts
SPE_MODELS = ["LSTM", "ResNet-18", "ResNet-50"]   # subset for the overhead-vs-p sweep

# --- Selective Parameter Encryption sweep (Experiment B) ---
SPE_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

# --- Accuracy / losslessness verification (Experiment C) ---
# ResNet-18 on CIFAR-10 under plaintext / full-HE / SPE(p=0.25) FedAvg.
# HE/SPE aggregation is numerically identical to plaintext (CKKS rounding ~1e-6),
# so losslessness is EXACT and independent of training length; the defaults run a
# reduced-but-faithful schedule that finishes in minutes. Raise ACC_ROUNDS /
# ACC_TRAIN_SUBSET (or set it to None for the full 50k set) for the paper's 100
# rounds if a GPU is available.
ACC_MODEL = "ResNet-18"
ACC_ROUNDS = 6
ACC_LOCAL_EPOCHS = 1
ACC_BATCH_SIZE = 64
ACC_LR = 0.01
ACC_SPE_FRACTION = 0.25
ACC_TRAIN_SUBSET = 5000       # images/round across all clients (None = full 50k)
ACC_TEST_SUBSET = 2000        # test images for the accuracy estimate (None = 10k)
