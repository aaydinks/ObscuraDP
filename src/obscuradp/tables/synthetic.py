"""A made-up health table, split among hospitals that differ from each other.

Every row is drawn from random numbers and a hand-written rule. **No row describes a real
person**, and the rule is not medical knowledge: it only gives the table a shape that is
typical for the kind of data this library is meant for, so that examples, set-up checks
and benchmarks can run without anybody's records.

What the table looks like by default: about 20,000 adults in 10 hospitals of very
different sizes, about 6% of them with the outcome, and hospitals that differ in the age
and habits of their patients and in how common the outcome is. The same seed always gives
the same table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from obscuradp.seeding import derive_seed, numpy_generator
from obscuradp.tables.schema import CategoricalFeature, NumericFeature, TableSchema
from obscuradp.tables.split import ClientData, make_client_data

SYNTHETIC_HEALTH_SCHEMA = TableSchema(
    (
        NumericFeature("age", 18.0, 95.0),  # years
        NumericFeature("bmi", 15.0, 50.0),  # kg per square metre
        NumericFeature("systolic_pressure", 80.0, 200.0),  # mmHg
        NumericFeature("glucose", 60.0, 300.0),  # mg/dL
        NumericFeature("diabetes", 0.0, 1.0),  # a yes/no flag stored as 0 or 1
        NumericFeature("earlier_event", 0.0, 1.0),
        CategoricalFeature("sex", ("female", "male")),
        CategoricalFeature("smoking", ("never", "former", "current")),
    )
)
# The share of the outcome in the whole population that the rule is tuned to. It is a
# property of this made-up population and may be used as a public prior for class weights.
SYNTHETIC_HEALTH_PRIOR = 0.06


@dataclass(frozen=True)
class SyntheticTable:
    """The made-up table before it is encoded.

    Attributes:
        columns: The raw columns by name, as a hospital's own system might hold them.
        outcome: 1 where the outcome occurred, else 0.
        site: The number of the hospital each row belongs to.
    """

    columns: dict[str, npt.NDArray[np.generic]]
    outcome: npt.NDArray[np.int64]
    site: npt.NDArray[np.int64]


def _site_sizes(generator: np.random.Generator, sites: int, rows: int) -> npt.NDArray[np.int64]:
    """Uneven sizes that add up to ``rows``; no hospital has fewer than 2% of the rows."""
    floor = max(1, rows // 50)
    if floor * sites > rows:
        raise ValueError("too few rows for this many sites")
    shares = generator.dirichlet(np.full(sites, 1.5))
    sizes = floor + np.floor(shares * (rows - floor * sites)).astype(np.int64)
    sizes[np.argmax(sizes)] += rows - int(sizes.sum())
    return np.asarray(sizes, dtype=np.int64)


def synthetic_health_rows(
    seed: int,
    *,
    sites: int = 10,
    rows: int = 20_000,
    positive_share: float = SYNTHETIC_HEALTH_PRIOR,
) -> SyntheticTable:
    """Draw the made-up table.

    Args:
        seed: The same seed gives the same table.
        sites: Number of hospitals.
        rows: Number of rows over all hospitals.
        positive_share: About this share of all rows has the outcome.
    """
    if sites < 1:
        raise ValueError("sites must be at least 1")
    if rows < sites:
        raise ValueError("need at least one row per site")
    if not 0.0 < positive_share < 1.0:
        raise ValueError("positive_share must be in (0, 1)")
    generator = numpy_generator(seed, "data")
    sizes = _site_sizes(generator, sites, rows)
    site = np.repeat(np.arange(sites, dtype=np.int64), sizes)
    # What makes one hospital unlike another: who comes there, and how often things go wrong.
    older = generator.normal(0.0, 6.0, sites)[site]
    smokers = generator.normal(0.0, 0.5, sites)[site]
    riskier = generator.normal(0.0, 0.35, sites)[site]

    age = np.clip(generator.normal(55.0 + older, 16.0), 18.0, 95.0)
    male = generator.random(rows) < 0.48
    bmi = np.clip(generator.normal(27.0 + 0.02 * (age - 55.0), 5.0), 15.0, 50.0)
    pressure = np.clip(
        generator.normal(118.0 + 0.45 * (age - 55.0) + 0.9 * (bmi - 27.0), 14.0), 80.0, 200.0
    )
    diabetes = generator.random(rows) < 1.0 / (
        1.0 + np.exp(-(-2.6 + 0.03 * (age - 55.0) + 0.09 * (bmi - 27.0)))
    )
    glucose = np.clip(
        generator.normal(98.0 + 55.0 * diabetes + 0.5 * (bmi - 27.0), 18.0), 60.0, 300.0
    )
    pull = generator.normal(smokers, 1.0)
    smoking = np.where(pull > 0.9, "current", np.where(pull > 0.1, "former", "never"))
    earlier = generator.random(rows) < 1.0 / (1.0 + np.exp(-(-3.4 + 0.05 * (age - 55.0))))

    score = (
        0.055 * (age - 55.0)
        + 0.022 * (pressure - 118.0)
        + 0.035 * (bmi - 27.0)
        + 0.7 * diabetes
        + 1.3 * earlier
        + 0.25 * male
        + 0.6 * (smoking == "current")
        + 0.2 * (smoking == "former")
        + riskier
    )
    # Shift the scores so that the expected share of the outcome is the one asked for.
    low, high = -20.0, 20.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if float(np.mean(1.0 / (1.0 + np.exp(-(score + middle))))) < positive_share:
            low = middle
        else:
            high = middle
    outcome = generator.random(rows) < 1.0 / (1.0 + np.exp(-(score + low)))
    return SyntheticTable(
        columns={
            "age": np.round(age).astype(np.int64),
            "bmi": np.round(bmi, 1),
            "systolic_pressure": np.round(pressure).astype(np.int64),
            "glucose": np.round(glucose).astype(np.int64),
            "diabetes": diabetes.astype(np.int64),
            "earlier_event": earlier.astype(np.int64),
            "sex": np.where(male, "male", "female"),
            "smoking": smoking,
        },
        outcome=outcome.astype(np.int64),
        site=site,
    )


def synthetic_health(
    seed: int,
    *,
    sites: int = 10,
    rows: int = 20_000,
    positive_share: float = SYNTHETIC_HEALTH_PRIOR,
    validation: float = 0.15,
    test: float = 0.15,
    split_seed: int | None = None,
) -> list[ClientData]:
    """The made-up table, encoded and cut into training, validation and test rows per hospital.

    The encoding is ``SYNTHETIC_HEALTH_SCHEMA``, which declares every range and category
    in advance, so preparing the table costs no privacy budget.

    ``split_seed`` cuts the same table (the one of ``seed``) in another way, for repeated
    runs on one table. By default the cut follows ``seed``.
    """
    table = synthetic_health_rows(seed, sites=sites, rows=rows, positive_share=positive_share)
    encoded = SYNTHETIC_HEALTH_SCHEMA.encode(table.columns)
    clients = []
    for index in range(sites):
        own = table.site == index
        clients.append(
            make_client_data(
                f"site-{index}",
                encoded[own],
                table.outcome[own],
                validation=validation,
                test=test,
                seed=derive_seed(seed if split_seed is None else split_seed, "data", index),
            )
        )
    return clients
