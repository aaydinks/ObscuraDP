"""The two research tables, prepared the way the library asks for.

Nothing here is computed from the rows themselves: every range and category list below is a
declaration. The ranges are round numbers a data owner could state from what each column
means (an age, a credit score with a defined scale, a body-mass index). They were written
by someone who had seen the column ranges of these two public datasets, which is stated
in ``docs/GATE_PHASE2.md``; a real deployment declares them before looking at any data.

The files are read from the frozen research folder and are never shipped in the package.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from obscuradp.partition import dirichlet_partition, iid_partition
from obscuradp.seeding import derive_seed
from obscuradp.tables import (
    CategoricalFeature,
    ClientData,
    NumericFeature,
    TableSchema,
    make_client_data,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Trial 07" / "27" / "data"
FILES = {
    "stroke": DATA_DIR / "healthcare-dataset-stroke-data.csv",
    "loan": DATA_DIR / "loan_approval_dataset.csv",
}
NUM_CLIENTS = 5

STROKE_SCHEMA = TableSchema(
    (
        NumericFeature("age", 0.0, 100.0),  # years
        NumericFeature("hypertension", 0.0, 1.0),  # a yes/no flag stored as 0 or 1
        NumericFeature("heart_disease", 0.0, 1.0),
        NumericFeature("avg_glucose_level", 40.0, 300.0),  # mg/dL
        NumericFeature("bmi", 10.0, 60.0),  # kg per square metre
        CategoricalFeature("gender", ("Male", "Female")),
        CategoricalFeature("ever_married", ("Yes", "No")),
        CategoricalFeature(
            "work_type", ("Private", "Self-employed", "Govt_job", "children", "Never_worked")
        ),
        CategoricalFeature("Residence_type", ("Urban", "Rural")),
        CategoricalFeature(
            "smoking_status", ("never smoked", "formerly smoked", "smokes", "Unknown")
        ),
    )
)

LOAN_SCHEMA = TableSchema(
    (
        NumericFeature("no_of_dependents", 0.0, 5.0),
        NumericFeature("income_annum", 0.0, 10_000_000.0),
        NumericFeature("loan_amount", 0.0, 40_000_000.0),
        NumericFeature("loan_term", 2.0, 20.0),  # years
        NumericFeature("cibil_score", 300.0, 900.0),  # the score's defined scale
        NumericFeature("residential_assets_value", -100_000.0, 30_000_000.0),
        NumericFeature("commercial_assets_value", 0.0, 20_000_000.0),
        NumericFeature("luxury_assets_value", 0.0, 40_000_000.0),
        NumericFeature("bank_asset_value", 0.0, 15_000_000.0),
        CategoricalFeature("education", ("Graduate", "Not Graduate")),
        CategoricalFeature("self_employed", ("Yes", "No")),
    )
)

SCHEMAS = {"stroke": STROKE_SCHEMA, "loan": LOAN_SCHEMA}

# Share of positive rows expected in the population, from a public source, for class weights.
# Stroke: 2.9% of US adults report ever having had a stroke (CDC, MMWR 73(20), May 2024,
# doi:10.15585/mmwr.mm7320a1). Loan: no public figure is used; the classes are not rare.
PUBLIC_PRIOR: dict[str, float | None] = {"stroke": 0.029, "loan": None}

# Hidden layers of the research networks.
HIDDEN = {"stroke": (64, 32), "loan": (32, 16)}


def load_table(name: str) -> tuple[dict[str, list[str]], np.ndarray]:
    """Read one table as text columns plus 0/1 labels, with the research code's row filter.

    Stroke rows without a body-mass index and the single row with gender "Other" are
    dropped, as in the research code, so old and new runs use the same rows. Whether a row
    is kept depends on that row alone.
    """
    if name not in FILES:
        raise ValueError(f"unknown dataset {name!r}")
    with FILES[name].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
    if name == "stroke":
        rows = [row for row in rows if row["bmi"] != "N/A" and row["gender"] != "Other"]
        labels = np.array([int(row["stroke"]) for row in rows], dtype=np.int64)
    else:
        labels = np.array([int(row["loan_status"] == "Approved") for row in rows], dtype=np.int64)
    columns = {
        feature.name: [row[feature.name] for row in rows] for feature in SCHEMAS[name].features
    }
    return columns, labels


def split_among_clients(
    name: str, features: np.ndarray, labels: np.ndarray, seed: int
) -> list[ClientData]:
    """Give the rows to five participants as the research code did, then cut each three ways.

    Stroke: uneven class mixes (Dirichlet, concentration 0.5, at least 12 positive rows
    each). Loan: an even random split. Every participant then keeps 70% of its own rows
    for training, 15% for validation and 15% for testing.
    """
    if name == "stroke":
        parts, _ = dirichlet_partition(labels, NUM_CLIENTS, alpha=0.5, seed=seed, min_per_class=12)
    else:
        parts, _ = iid_partition(len(labels), NUM_CLIENTS, seed)
    return [
        make_client_data(
            f"site-{index}",
            features[part],
            labels[part],
            validation=0.15,
            test=0.15,
            seed=derive_seed(seed, "data", index),
        )
        for index, part in enumerate(parts)
    ]


def make_clients(name: str, seed: int) -> list[ClientData]:
    """The real table ``name`` encoded from its declared schema and split for ``seed``."""
    columns, labels = load_table(name)
    return split_among_clients(name, SCHEMAS[name].encode(columns), labels, seed)


def balanced_rows(clients: list[ClientData], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """All positive test rows plus as many random negative ones, as the research code scored.

    Kept beside the natural test rows only so old and new numbers can be compared.
    """
    features = np.concatenate([client.test_x for client in clients])
    labels = np.concatenate([client.test_y for client in clients])
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    count = min(positive.size, negative.size)
    rng = np.random.default_rng(derive_seed(seed, "evaluation"))
    chosen = np.concatenate(
        [rng.choice(positive, count, replace=False), rng.choice(negative, count, replace=False)]
    )
    return features[chosen], labels[chosen]
