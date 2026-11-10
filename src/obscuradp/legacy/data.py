"""Data preparation exactly as in Trial 07/27 (comparison only, see the package docstring).

Known defects that are reproduced here on purpose and must not be copied into new code:
scalers are fit on the pooled training rows of all participants (finding G12), the stroke
test set is rebalanced to 50/50 (G25), and SMOTETomek builds synthetic rows before private
training (G3).

Needs pandas, scikit-learn and imbalanced-learn (the ``legacy`` extra).
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.partition import IndexArray, dirichlet_partition, iid_partition

Array = np.ndarray[Any, Any]

STROKE_CONT_COLS = ["age", "avg_glucose_level", "bmi"]
STROKE_CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

LOAN_TARGET = "loan_status"
LOAN_DROP_COLS = ["loan_id"]
LOAN_BINARY_CAT = {
    "education": {"Graduate": 1, "Not Graduate": 0},
    "self_employed": {"Yes": 1, "No": 0},
}
LOAN_LABEL_MAP = {"Approved": 1, "Rejected": 0}


class Prepared(TypedDict):
    """Everything one run needs: the split, the participants' rows and their description."""

    X_tr: Array
    y_tr: Array
    X_te: Array
    y_te: Array
    info: dict[str, Any]
    parts: list[IndexArray]
    part_info: dict[str, Any]
    client_data: list[tuple[Array, Array]]
    in_features: int


def load_stroke(cfg: LegacyTrialConfig) -> tuple[Array, Array, list[str]]:
    """Load and clean the stroke table. Returns (features, labels, feature names)."""
    import pandas as pd

    df = pd.read_csv(cfg.stroke_path, na_values=["N/A", "Unknown"])
    df = df.drop(columns=["id"])

    # Preserve 'Unknown' as a smoking_status category; re-read separately.
    raw = pd.read_csv(cfg.stroke_path)
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


def load_loan(cfg: LegacyTrialConfig) -> tuple[Array, Array, list[str]]:
    """Load and clean the loan table. Returns (features, labels, feature names)."""
    import pandas as pd

    df = pd.read_csv(cfg.loan_path, skipinitialspace=True)
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


def load_dataset(cfg: LegacyTrialConfig, name: str) -> tuple[Array, Array, list[str]]:
    """Load ``"stroke"`` or ``"loan"``."""
    if name == "stroke":
        return load_stroke(cfg)
    if name == "loan":
        return load_loan(cfg)
    raise ValueError(f"Unknown dataset: {name}")


def split_and_scale(
    cfg: LegacyTrialConfig, name: str, seed: int | None = None
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    """Stratified 80/20 split, then a scaler fit on the pooled training rows."""
    if seed is None:
        seed = cfg.seed
    if name == "stroke":
        return _split_and_scale_stroke(cfg, seed)
    if name == "loan":
        return _split_and_scale_loan(cfg, seed)
    raise ValueError(f"Unknown dataset: {name}")


def _split_and_scale_stroke(
    cfg: LegacyTrialConfig, seed: int
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names = load_stroke(cfg)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg.test_fraction,
        stratify=y,
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
    rng = np.random.RandomState(seed)  # RandomState on purpose: same draws as the research code
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


def _split_and_scale_loan(
    cfg: LegacyTrialConfig, seed: int
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X, y, feature_names = load_loan(cfg)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg.test_fraction,
        stratify=y,
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


def partition(
    cfg: LegacyTrialConfig,
    name: str,
    y_train: Array,
    num_clients: int | None = None,
    seed: int | None = None,
) -> tuple[list[IndexArray], dict[str, Any]]:
    """Split training rows among participants: uneven class mixes for stroke, even for loan.

    The description is returned in the research code's dictionary layout.
    """
    if num_clients is None:
        num_clients = cfg.num_clients
    if seed is None:
        seed = cfg.seed

    if name == "stroke":
        parts, info = dirichlet_partition(
            y_train,
            num_clients,
            alpha=cfg.dirichlet_alpha,
            seed=seed,
            min_per_class=cfg.min_pos_per_client,
        )
        return parts, {
            "kind": "dirichlet",
            "attempts": info.attempts,
            "sizes": list(info.sizes),
            "pos_counts": [int(y_train[ix].sum()) for ix in parts],
            "neg_counts": [int((y_train[ix] == 0).sum()) for ix in parts],
        }
    if name == "loan":
        parts, info = iid_partition(len(y_train), num_clients, seed)
        return parts, {"kind": "iid", "sizes": list(info.sizes)}
    raise ValueError(f"Unknown dataset: {name}")


def smote_resample(
    cfg: LegacyTrialConfig, X: Array, y: Array, seed: int | None = None
) -> tuple[Array, Array]:
    """Apply SMOTETomek to one participant's rows (see finding G3: this voids the guarantee)."""
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE

    if seed is None:
        seed = cfg.seed

    n_min = int(np.bincount(y).min())
    k = max(1, min(5, n_min - 1))
    smt = SMOTETomek(
        random_state=seed,
        smote=SMOTE(random_state=seed, k_neighbors=k),
    )
    Xr, yr = smt.fit_resample(X, y)
    return Xr.astype(np.float32), yr.astype(np.int64)


def build_client_data(
    cfg: LegacyTrialConfig,
    name: str,
    X_train: Array,
    y_train: Array,
    parts: list[IndexArray],
    use_smote: bool | None = None,
    seed: int | None = None,
) -> list[tuple[Array, Array]]:
    """Build each participant's (features, labels). Resampling is on by default for stroke."""
    if seed is None:
        seed = cfg.seed
    if use_smote is None:
        use_smote = name == "stroke"

    client_data = []
    for idx in parts:
        Xc, yc = X_train[idx], y_train[idx]
        if use_smote:
            Xc, yc = smote_resample(cfg, Xc, yc, seed=seed)
        client_data.append((Xc, yc))
    return client_data


def prepare(
    cfg: LegacyTrialConfig, dataset: str, seed: int | None = None, use_smote: bool | None = None
) -> Prepared:
    """Split, scale, partition and build participants' data once, for every arm to share."""
    if seed is None:
        seed = cfg.seed
    X_tr, y_tr, X_te, y_te, info = split_and_scale(cfg, dataset, seed=seed)
    in_features = info["n_features"]
    parts, part_info = partition(cfg, dataset, y_tr, num_clients=cfg.num_clients, seed=seed)
    client_data = build_client_data(cfg, dataset, X_tr, y_tr, parts, use_smote=use_smote, seed=seed)
    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_te": X_te,
        "y_te": y_te,
        "info": info,
        "parts": parts,
        "part_info": part_info,
        "client_data": client_data,
        "in_features": in_features,
    }
