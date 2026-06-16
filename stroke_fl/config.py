"""Central configuration for the SMOTETomek + FedProx stroke experiment.

Reproduction of arXiv:2508.10017 -- the epsilon-vs-recall frontier on
non-IID clinical (stroke) data, comparing FedProx vs FedAvg under DP-SGD.
"""

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
RESULTS_DIR = "results"

SEED = 42

# --- FL configuration (paper Table 3) ---
NUM_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
MU = 0.01                      # FedProx proximal coefficient

# --- Model ---
HIDDEN1 = 64
HIDDEN2 = 32

# --- Differential privacy (Opacus / DP-SGD) ---
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0            # clipping threshold C

# --- epsilon sweep (fine-grained around the ~9.0 "sweet spot") ---
EPSILON_GRID = [0.5, 1.0, 2.0, 5.0, 7.0, 9.0, 10.0, 15.0, 50.0]

# --- Non-IID partitioning ---
DIRICHLET_ALPHA = 0.5         # label-skew concentration; lower = more non-IID
MIN_POS_PER_CLIENT = 12       # floor so SMOTETomek (needs >=6 minority) is feasible

# --- Test set ---
TEST_FRACTION = 0.20          # 80/20 split before partitioning
