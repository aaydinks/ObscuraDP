"""Data loading, cleaning, encoding, splitting and partitioning.

Unifies two datasets from prior experiments:
- Stroke: non-IID Dirichlet partitioning, per-client SMOTETomek, class-balanced test set,
  StandardScaler on continuous features only.
- Loan: IID partitioning, no resampling, natural test set, MinMaxScaler on all features
  into [0,1] (required by CKKS).

Both pipelines accept seed arguments to support multi-seed campaigns. The stroke recipe
preserves Experiment 1's per-class Dirichlet floor + redraw loop; the loan recipe matches
Experiment 2's MinMax-to-[0,1] scaling for CKKS compatibility.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from . import config


# ============================================================================
# STROKE DATASET (Experiment 1 recipe)
# ============================================================================

STROKE_CONT_COLS = ["age", "avg_glucose_level", "bmi"]
STROKE_CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def load_stroke():
    """Load and clean stroke dataset: drop id, missing BMI, gender='Other';
    one-hot encode categoricals; return (X, y, feature_names)."""
    df = pd.read_csv(config.STROKE_PATH, na_values=["N/A", "Unknown"])
    df = df.drop(columns=["id"])

    # Preserve 'Unknown' as a smoking_status category; re-read separately.
    raw = pd.read_csv(config.STROKE_PATH)
    df["smoking_status"] = raw["smoking_status"]

    # Drop rows with missing BMI and ambiguous gender='Other'.
    df = df.dropna(subset=["bmi"]).reset_index(drop=True)
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    y = df["stroke"].astype(np.int64).values
    df = df.drop(columns=["stroke"])

    # One-hot encode nominal categoricals; hypertension/heart_disease already 0/1.
    df = pd.get_dummies(df, columns=STROKE_CAT_COLS, drop_first=True)
    feature_names = list(df.columns)
    X = df.astype(np.float32).values
    return X, y, feature_names


# ============================================================================
# LOAN DATASET (Experiment 2 recipe)
# ============================================================================

LOAN_TARGET = "loan_status"
LOAN_DROP_COLS = ["loan_id"]
LOAN_BINARY_CAT = {
    "education": {"Graduate": 1, "Not Graduate": 0},
    "self_employed": {"Yes": 1, "No": 0},
}
LOAN_LABEL_MAP = {"Approved": 1, "Rejected": 0}


def load_loan():
    """Load and clean loan dataset: strip whitespace, drop loan_id,
    label-encode binary categoricals; return (X, y, feature_names)."""
    df = pd.read_csv(config.LOAN_PATH, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=LOAN_DROP_COLS)

    # Strip whitespace from all object/string columns.
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            df[c] = df[c].astype(str).str.strip()

    # Label-encode binary categoricals and target.
    for col, mapping in LOAN_BINARY_CAT.items():
        df[col] = df[col].map(mapping)
    y = df[LOAN_TARGET].map(LOAN_LABEL_MAP).astype(np.int64).values
    df = df.drop(columns=[LOAN_TARGET])

    # Drop any rows with remaining nulls (defensive).
    mask = ~df.isnull().any(axis=1) & ~pd.isnull(y)
    df = df[mask].reset_index(drop=True)
    y = y[mask.values]

    feature_names = list(df.columns)
    X = df.astype(np.float32).values
    return X, y, feature_names


# ============================================================================
# PUBLIC API: LOAD & DISPATCH
# ============================================================================

def load_dataset(name):
    """Load and clean a dataset: name in {"stroke", "loan"}.
    Returns (X float32, y int64, feature_names)."""
    if name == "stroke":
        return load_stroke()
    elif name == "loan":
        return load_loan()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def split_and_scale(name, seed=None):
    """Stratified 80/20 split + scaler fit on train only.

    Stroke: StandardScaler on continuous cols only, class-balanced test set.
    Loan: MinMaxScaler on ALL features into [0,1], natural test set.

    Returns (X_train, y_train, X_test, y_test, info).
    info keys: n_features, feature_names, n_train, n_test, train_pos_rate,
    test_pos_rate, scaler_kind, balanced_test."""
    if seed is None:
        seed = config.SEED

    if name == "stroke":
        return _split_and_scale_stroke(seed)
    elif name == "loan":
        return _split_and_scale_loan(seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _split_and_scale_stroke(seed):
    """Stroke-specific split and scaling."""
    X, y, feature_names = load_stroke()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_FRACTION, stratify=y,
        random_state=seed,
    )

    # StandardScaler on continuous columns only, fit on train.
    cont_idx = [feature_names.index(c) for c in STROKE_CONT_COLS]
    scaler = StandardScaler().fit(X_tr[:, cont_idx])
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    X_tr[:, cont_idx] = scaler.transform(X_tr[:, cont_idx]).astype(np.float32)
    X_te[:, cont_idx] = scaler.transform(X_te[:, cont_idx]).astype(np.float32)

    # Class-balanced held-out test set: all positives + equal random negatives.
    rng = np.random.RandomState(seed)
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
        "n_test": len(y_test_bal),
        "train_pos_rate": float(y_tr.mean()),
        "test_pos_rate": float(y_test_bal.mean()),
        "scaler_kind": "standard",
        "balanced_test": True,
    }
    return X_tr, y_tr, X_test_bal, y_test_bal, info


def _split_and_scale_loan(seed):
    """Loan-specific split and scaling."""
    X, y, feature_names = load_loan()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_FRACTION, stratify=y,
        random_state=seed,
    )

    # MinMaxScaler on ALL features into [0,1], fit on train only.
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
        "scaler_kind": "minmax",
        "balanced_test": False,
    }
    return X_tr, y_tr, X_te, y_te, info


# ============================================================================
# PARTITIONING
# ============================================================================

def partition(name, y_train, num_clients=None, seed=None):
    """Partition training indices across clients.

    Stroke -> Dirichlet(alpha) with per-class minimum positives floor.
    Loan -> IID random shards.

    Returns (parts list[np.ndarray], info)."""
    if num_clients is None:
        num_clients = config.NUM_CLIENTS
    if seed is None:
        seed = config.SEED

    if name == "stroke":
        return dirichlet_partition(
            y_train, num_clients,
            alpha=config.DIRICHLET_ALPHA,
            min_pos=config.MIN_POS_PER_CLIENT,
            seed=seed
        )
    elif name == "loan":
        return iid_partition(len(y_train), num_clients, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def dirichlet_partition(y_train, num_clients, alpha, min_pos, seed):
    """Per-class Dirichlet partition with per-class minimum floor.

    Draws independent Dirichlet(alpha) per class and splits accordingly -> non-IID.
    Redraws until every client has >= min_pos samples of BOTH classes, because
    severe label skew can make negatives the local minority on a small client."""
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
                "kind": "dirichlet",
                "attempts": attempt,
                "sizes": [len(ix) for ix in client_idx],
                "pos_counts": [int(y_train[ix].sum()) for ix in client_idx],
                "neg_counts": [int((y_train[ix] == 0).sum()) for ix in client_idx],
            }
        if attempt > 5000:
            raise RuntimeError("Could not satisfy per-class floor; lower min_pos.")


def iid_partition(n_train, num_clients, seed):
    """Shuffle train indices and split into num_clients near-equal IID shards."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return [np.array(p) for p in parts], {
        "kind": "iid",
        "sizes": [len(p) for p in parts],
    }


# ============================================================================
# RESAMPLING
# ============================================================================

def smote_resample(X, y, seed=None):
    """Apply SMOTETomek to a client's local data (oversample + clean).

    k_neighbors is clamped to n_minority - 1 so resampling never fails
    on a small/skewed client."""
    if seed is None:
        seed = config.SEED

    n_min = int(np.bincount(y).min())
    k = max(1, min(5, n_min - 1))
    smt = SMOTETomek(
        random_state=seed,
        smote=SMOTE(random_state=seed, k_neighbors=k),
    )
    Xr, yr = smt.fit_resample(X, y)
    return Xr.astype(np.float32), yr.astype(np.int64)


# ============================================================================
# CLIENT DATA ASSEMBLY
# ============================================================================

def build_client_data(name, X_train, y_train, parts, use_smote=None, seed=None):
    """Build per-client (Xc, yc) tuples from partition indices.

    use_smote=None -> True for stroke, False for loan.
    Returns list of (Xc float32, yc int64) tuples."""
    if seed is None:
        seed = config.SEED

    if use_smote is None:
        use_smote = (name == "stroke")

    client_data = []
    for idx in parts:
        Xc, yc = X_train[idx], y_train[idx]
        if use_smote:
            Xc, yc = smote_resample(Xc, yc, seed=seed)
        client_data.append((Xc, yc))

    return client_data
