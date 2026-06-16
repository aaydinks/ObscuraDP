"""Data loading, cleaning, encoding, splitting and non-IID partitioning.

Pipeline (mirrors the paper's outline):
  1. Load CSV, drop `id`.
  2. Drop rows with missing BMI; drop the single ambiguous gender='Other' row.
  3. One-hot encode nominal categoricals; standardize continuous features
     (scaler fit on TRAIN only -> no leakage).
  4. Stratified 80/20 split, then build a CLASS-BALANCED held-out test set
     so accuracy/precision are not dominated by the majority class.
  5. Partition the training set across NUM_CLIENTS with per-class
     Dirichlet(alpha) skew + a minimum-positives floor.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config

CONT_COLS = ["age", "avg_glucose_level", "bmi"]
CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def load_clean_encode():
    """Return (X, y) as float32 / int64 numpy arrays plus feature names."""
    df = pd.read_csv(config.DATA_PATH, na_values=["N/A", "Unknown"])
    df = df.drop(columns=["id"])

    # `smoking_status` has a legitimate 'Unknown' category we want to KEEP as a
    # category (not treat as NaN). Re-read smoking separately to be safe.
    raw = pd.read_csv(config.DATA_PATH)
    df["smoking_status"] = raw["smoking_status"]

    # Drop rows with missing BMI (the only column with true nulls).
    df = df.dropna(subset=["bmi"]).reset_index(drop=True)
    # Drop the single gender='Other' row (1 sample) for clean encoding.
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    y = df["stroke"].astype(np.int64).values
    df = df.drop(columns=["stroke"])

    # One-hot encode nominal categoricals (hypertension/heart_disease already 0/1).
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    feature_names = list(df.columns)
    X = df.astype(np.float32).values
    return X, y, feature_names


def split_and_scale():
    """Stratified 80/20 split, standardize continuous cols, build a balanced
    test set. Returns (X_train, y_train, X_test_bal, y_test_bal, info)."""
    X, y, feature_names = load_clean_encode()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_FRACTION, stratify=y,
        random_state=config.SEED,
    )

    # StandardScaler on continuous columns only, fit on train.
    cont_idx = [feature_names.index(c) for c in CONT_COLS]
    scaler = StandardScaler().fit(X_tr[:, cont_idx])
    X_tr = X_tr.copy(); X_te = X_te.copy()
    X_tr[:, cont_idx] = scaler.transform(X_tr[:, cont_idx]).astype(np.float32)
    X_te[:, cont_idx] = scaler.transform(X_te[:, cont_idx]).astype(np.float32)

    # Class-balanced held-out test set: all positives + equal random negatives.
    rng = np.random.RandomState(config.SEED)
    pos = np.where(y_te == 1)[0]
    neg = np.where(y_te == 0)[0]
    n = min(len(pos), len(neg))
    neg_sel = rng.choice(neg, size=n, replace=False)
    sel = np.concatenate([pos, neg_sel])
    rng.shuffle(sel)
    X_test_bal, y_test_bal = X_te[sel], y_te[sel]

    info = {
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "n_train": len(y_tr),
        "train_pos": int(y_tr.sum()),
        "n_test_balanced": len(y_test_bal),
        "test_pos": int(y_test_bal.sum()),
        # also keep the full (natural-distribution) test set for reference
        "X_test_full": X_te, "y_test_full": y_te,
    }
    return X_tr, y_tr, X_test_bal, y_test_bal, info


def dirichlet_partition(y_train, num_clients, alpha, min_pos, seed):
    """Per-class Dirichlet partition of indices across clients.

    Draw an independent Dirichlet(alpha) proportion vector per class and split
    that class's samples accordingly -> genuine non-IID (clients differ in size
    AND in stroke prevalence). Re-draw until every client has >= min_pos samples
    of BOTH classes, so SMOTETomek (which oversamples whichever class is the
    local minority) is always feasible. NOTE: the per-class floor must cover both
    classes -- with severe label skew a small client can end up *mostly positive*,
    making negatives the local minority.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    attempt = 0
    while True:
        attempt += 1
        client_idx = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(y_train == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet([alpha] * num_clients)
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(idx_c, cuts)):
                client_idx[cid].extend(chunk.tolist())
        # per-client count of EACH class
        per_class = [[int((y_train[ix] == c).sum()) for c in classes]
                     for ix in client_idx]
        min_any_class = min(min(pc) for pc in per_class)
        if min_any_class >= min_pos:
            for ix in client_idx:
                rng.shuffle(ix)
            return [np.array(ix) for ix in client_idx], {
                "attempts": attempt,
                "sizes": [len(ix) for ix in client_idx],
                "pos_counts": [int(y_train[ix].sum()) for ix in client_idx],
                "neg_counts": [int((y_train[ix] == 0).sum()) for ix in client_idx],
            }
        if attempt > 5000:
            raise RuntimeError("Could not satisfy per-class floor; lower min_pos.")
