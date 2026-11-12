"""The made-up health table: the same from the same seed, shaped as described, no real row."""

import numpy as np
import pytest

import obscuradp as odp
from obscuradp.models import TabularMLP
from obscuradp.tables import (
    SYNTHETIC_HEALTH_PRIOR,
    SYNTHETIC_HEALTH_SCHEMA,
    synthetic_health,
    synthetic_health_rows,
)


def test_the_table_has_the_described_shape() -> None:
    table = synthetic_health_rows(seed=1)
    assert table.outcome.shape == table.site.shape == (20_000,)
    assert set(table.columns) == {feature.name for feature in SYNTHETIC_HEALTH_SCHEMA.features}
    assert float(table.outcome.mean()) == pytest.approx(SYNTHETIC_HEALTH_PRIOR, abs=0.01)
    sizes = np.bincount(table.site, minlength=10)
    assert sizes.sum() == 20_000
    assert sizes.min() >= 400  # no hospital below 2% of the rows
    assert sizes.max() > 2 * sizes.min()  # and they are far from equal
    # Hospitals differ in who comes there and in how common the outcome is.
    shares = [float(table.outcome[table.site == site].mean()) for site in range(10)]
    ages = [float(table.columns["age"][table.site == site].mean()) for site in range(10)]
    assert max(shares) > 1.5 * min(shares)
    assert max(ages) - min(ages) > 5.0
    # Every value is inside the range or list the schema declares.
    for feature in SYNTHETIC_HEALTH_SCHEMA.features:
        values = table.columns[feature.name]
        if hasattr(feature, "categories"):
            assert set(values.tolist()) <= set(feature.categories)
        else:
            assert feature.lower <= values.min()
            assert values.max() <= feature.upper


def test_the_same_seed_gives_the_same_table_and_another_seed_another() -> None:
    first, again, other = (synthetic_health_rows(seed=seed) for seed in (7, 7, 8))
    assert np.array_equal(first.outcome, again.outcome)
    assert all(np.array_equal(first.columns[name], again.columns[name]) for name in first.columns)
    assert not np.array_equal(first.outcome, other.outcome)
    assert not np.array_equal(np.bincount(first.site), np.bincount(other.site))


def test_size_and_mix_can_be_chosen() -> None:
    small = synthetic_health_rows(seed=2, sites=3, rows=3_000, positive_share=0.2)
    assert small.outcome.size == 3_000
    assert set(small.site.tolist()) == {0, 1, 2}
    assert float(small.outcome.mean()) == pytest.approx(0.2, abs=0.03)
    with pytest.raises(ValueError, match="sites"):
        synthetic_health_rows(seed=2, sites=0)
    with pytest.raises(ValueError, match="one row per site"):
        synthetic_health_rows(seed=2, sites=10, rows=5)
    with pytest.raises(ValueError, match="positive_share"):
        synthetic_health_rows(seed=2, positive_share=1.0)
    with pytest.raises(ValueError, match="too few rows"):
        synthetic_health_rows(seed=2, sites=100, rows=120)


def test_the_encoded_table_is_ready_for_a_simulation() -> None:
    clients = synthetic_health(seed=3, sites=4, rows=4_000)
    assert [client.name for client in clients] == [f"site-{index}" for index in range(4)]
    # The same table can be cut in another way: same rows per hospital, other training rows.
    recut = synthetic_health(seed=3, sites=4, rows=4_000, split_seed=9)
    assert [c.n_train for c in recut] == [c.n_train for c in clients]
    assert not np.array_equal(recut[0].train_x, clients[0].train_x)
    assert np.array_equal(
        np.sort(np.concatenate([recut[0].train_x, recut[0].validation_x, recut[0].test_x]), 0),
        np.sort(
            np.concatenate([clients[0].train_x, clients[0].validation_x, clients[0].test_x]), 0
        ),
    )
    assert sum(c.n_train + c.validation_y.size + c.test_y.size for c in clients) == 4_000
    for client in clients:
        assert client.train_x.shape[1] == SYNTHETIC_HEALTH_SCHEMA.width == 11
        assert client.train_x.dtype == np.float32
        assert float(np.abs(client.train_x).max()) <= 1.0
        assert 0 < client.train_y.sum() < client.n_train  # both outcomes in every hospital
    # The outcome can be learned from the columns, which is what makes the table useful.
    result = odp.Simulation(
        model_fn=lambda: TabularMLP(SYNTHETIC_HEALTH_SCHEMA.width, (40, 20)),
        clients=clients,
        plan=odp.Plan(dp=None, imbalance=odp.imbalance.ClassWeighted(SYNTHETIC_HEALTH_PRIOR)),
        policy=None,
        rounds=15,
        local_epochs=2.0,
        learning_rate=0.1,
        evaluate_every=0,
    ).run(seed=3)
    assert result.test["auc"] > 0.7
    # The network the benchmark plan names for this table: about 1,300 numbers.
    size = sum(value.numel() for value in TabularMLP(11, (40, 20)).parameters())
    assert 1_250 < size < 1_350
