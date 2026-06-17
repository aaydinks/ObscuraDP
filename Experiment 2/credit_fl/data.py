"""Data loading, cleaning, encoding, splitting and IID partitioning.

Pipeline (mirrors the paper's outline):
  1. Load CSV, strip whitespace from headers/values, drop `loan_id`.
  2. Handle nulls (this dataset has none after stripping, but we coerce + drop
     defensively so the pipeline is robust to the raw Kaggle download).
  3. Label-encode the binary categoricals (education, self_employed) and the
     target (loan_status: Approved=1 / Rejected=0).
  4. MinMaxScaler on ALL features into [0,1] -- required by CKKS, which operates
     on real numbers where large magnitudes amplify ciphertext rounding noise.
     The scaler is fit on TRAIN only (no leakage).
  5. Global stratified 80/20 split, then an IID partition of the train set into
     NUM_CLIENTS roughly-equal random shards.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import config

TARGET = "loan_status"
DROP_COLS = ["loan_id"]
BINARY_CAT = {
    "education": {"Graduate": 1, "Not Graduate": 0},
    "self_employed": {"Yes": 1, "No": 0},
}
LABEL_MAP = {"Approved": 1, "Rejected": 0}


def load_clean_encode():
    """Return (X, y) as float32 / int64 numpy arrays plus feature names."""
    df = pd.read_csv(config.DATA_PATH, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=DROP_COLS)

    # Strip whitespace from object/string columns (raw CSV pads " Graduate").
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = df[c].astype(str).str.strip()

    # Label-encode the binary categoricals and the target.
    for col, mapping in BINARY_CAT.items():
        df[col] = df[col].map(mapping)
    y = df[TARGET].map(LABEL_MAP).astype(np.int64).values
    df = df.drop(columns=[TARGET])

    # Drop any rows that still carry nulls (defensive; dataset is clean).
    mask = ~df.isnull().any(axis=1) & ~pd.isnull(y)
    df = df[mask].reset_index(drop=True)
    y = y[mask.values]

    feature_names = list(df.columns)
    X = df.astype(np.float32).values
    return X, y, feature_names


def split_and_scale():
    """Stratified 80/20 split + MinMax-to-[0,1]. Returns (X_tr,y_tr,X_te,y_te,info)."""
    X, y, feature_names = load_clean_encode()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_FRACTION, stratify=y,
        random_state=config.SEED,
    )

    # MinMax all features into [0,1], fit on train only.
    scaler = MinMaxScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    info = {
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "train_pos_rate": float(y_tr.mean()),
        "test_pos_rate": float(y_te.mean()),
    }
    return X_tr, y_tr, X_te, y_te, info


def iid_partition(n_train, num_clients, seed):
    """Shuffle train indices and split into `num_clients` near-equal IID shards."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return [np.array(p) for p in parts], {"sizes": [len(p) for p in parts]}
