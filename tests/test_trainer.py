"""One participant's private round: ledger first, exact sampling, fixed divisor."""

import pytest
import torch
from torch import nn

from obscuradp.accounting import SubsampledGaussianEvent, calibrate_noise_multiplier, total_steps
from obscuradp.client import RoundStats, train_private_round
from obscuradp.ledger import BudgetExceeded, PrivacyLedger
from obscuradp.mechanisms import clipped_sum, per_record_gradients
from obscuradp.models import TabularMLP
from obscuradp.policy import PrivacyPolicy
from obscuradp.state import get_state

DELTA = 1e-5


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _table(n: int = 40, d: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator(seed)
    x = torch.randn(n, d, generator=generator)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).float()
    return x, y


def _model(d: int = 4, seed: int = 0) -> TabularMLP:
    torch.manual_seed(seed)
    return TabularMLP(d, (6,))


def _ledger(epsilon: float = 1e6) -> PrivacyLedger:
    return PrivacyLedger(PrivacyPolicy(epsilon, DELTA, accountant="rdp"))


def _round(model: nn.Module, x: torch.Tensor, y: torch.Tensor, **overrides: object) -> RoundStats:
    arguments: dict[str, object] = {
        "loss_fn": nn.BCELoss(),
        "optimizer": torch.optim.SGD(model.parameters(), lr=0.1),
        "ledger": _ledger(),
        "scope": "client-0",
        "sampling_rate": 0.25,
        "steps": 5,
        "clip_norm": 1.0,
        "noise_multiplier": 1.0,
        "sampling_generator": _generator(1),
        "noise_generator": _generator(2),
        **overrides,
    }
    return train_private_round(model, x, y, **arguments)  # type: ignore[arg-type]


def test_the_round_is_charged_before_any_data_is_used() -> None:
    x, y = _table()
    model = _model()
    before = get_state(model)
    sampling, noise = _generator(1), _generator(2)
    untouched = (sampling.get_state().clone(), noise.get_state().clone())
    ledger = PrivacyLedger(PrivacyPolicy(0.5, DELTA))
    with pytest.raises(BudgetExceeded):
        _round(
            model,
            x,
            y,
            ledger=ledger,
            noise_multiplier=0.8,
            steps=50,
            sampling_generator=sampling,
            noise_generator=noise,
        )
    assert ledger.scopes == ()
    assert all(torch.equal(before[name], value) for name, value in get_state(model).items())
    # Not a single random number was drawn: the refusal came first.
    assert torch.equal(sampling.get_state(), untouched[0])
    assert torch.equal(noise.get_state(), untouched[1])


def test_a_round_records_exactly_what_it_did() -> None:
    x, y = _table()
    model = _model()
    ledger = _ledger()
    stats = _round(model, x, y, ledger=ledger, sampling_rate=0.3, steps=7, noise_multiplier=1.3)
    assert stats.steps == 7
    assert 0 <= stats.empty_batches <= 7
    assert ledger.events("client-0") == (SubsampledGaussianEvent(1.3, 0.3, 7),)
    _round(model, x, y, ledger=ledger, sampling_rate=0.3, steps=7, noise_multiplier=1.3)
    assert ledger.events("client-0") == (SubsampledGaussianEvent(1.3, 0.3, 14),)


def test_every_step_matches_the_formula_and_uses_poisson_batches() -> None:
    """Replay the round from its seeds: batches, clipped sums, noise and the fixed divisor."""
    x, y = _table(n=30)
    model = _model()
    shadow = _model()
    rate, clip, sigma, lr = 0.2, 0.4, 1.7, 0.05
    seen: list[tuple[int, list[int], dict[str, torch.Tensor]]] = []
    stats = _round(
        model,
        x,
        y,
        optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        sampling_rate=rate,
        steps=6,
        clip_norm=clip,
        noise_multiplier=sigma,
        on_step=lambda step, indices, noisy: seen.append((step, indices.tolist(), noisy)),
    )
    assert [step for step, _, _ in seen] == list(range(6))
    sampling, noise = _generator(1), _generator(2)
    sizes = []
    for _, indices, noisy in seen:
        expected = torch.nonzero(torch.rand(30, generator=sampling, dtype=torch.float64) < rate)
        assert indices == expected.flatten().tolist()
        sizes.append(len(indices))
        total = clipped_sum(
            per_record_gradients(shadow, nn.BCELoss(), x[indices], y[indices]), clip
        )
        with torch.no_grad():
            for name, parameter in shadow.named_parameters():
                draw = torch.normal(0.0, sigma * clip, size=parameter.shape, generator=noise)
                gradient = (total[name] + draw) / (rate * 30)  # never the actual batch size
                assert torch.allclose(noisy[name], gradient, atol=1e-6), name
                parameter -= lr * gradient
    assert len(set(sizes)) > 1  # batch sizes vary, as Poisson sampling requires
    assert stats.records_drawn == sum(sizes)
    assert stats.empty_batches == sizes.count(0)
    for name, value in get_state(model).items():
        assert torch.allclose(value, get_state(shadow)[name], atol=1e-6), name


def test_empty_batches_still_take_a_noisy_step() -> None:
    """Finding G27: skipping an empty batch would reveal that it was empty."""
    x, y = _table(n=3)
    model = _model()
    before = get_state(model)
    observed = []
    stats = _round(
        model,
        x,
        y,
        sampling_rate=1e-9,
        steps=4,
        on_step=lambda step, indices, noisy: observed.append(indices.numel()),
    )
    assert observed == [0, 0, 0, 0]
    assert stats == RoundStats(steps=4, empty_batches=4, records_drawn=0)
    assert any(not torch.equal(before[name], value) for name, value in get_state(model).items())


def test_the_declared_size_sets_the_divisor() -> None:
    x, y = _table(n=20)
    scales = {}
    for declared in (None, 80):
        model = _model()
        grabbed: list[dict[str, torch.Tensor]] = []
        _round(
            model,
            x,
            y,
            steps=1,
            sampling_rate=1.0,
            declared_records=declared,
            on_step=lambda step, indices, noisy, sink=grabbed: sink.append(noisy),
        )
        scales[declared] = grabbed[0]["net.0.weight"]
    assert torch.allclose(scales[None], scales[80] * 4.0, atol=1e-6)


def test_same_seeds_same_result_and_other_noise_other_result() -> None:
    x, y = _table()
    results = []
    for noise_seed in (2, 2, 3):
        model = _model()
        _round(model, x, y, noise_generator=_generator(noise_seed))
        results.append(get_state(model)["net.0.weight"])
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])


def test_the_proximal_term_pulls_towards_the_reference_weights() -> None:
    x, y = _table()
    lr, mu = 0.1, 0.5
    plain, pulled = _model(), _model()
    start = get_state(plain)
    reference = {name: value + 1.0 for name, value in start.items()}
    _round(plain, x, y, steps=1, optimizer=torch.optim.SGD(plain.parameters(), lr=lr))
    _round(
        pulled,
        x,
        y,
        steps=1,
        optimizer=torch.optim.SGD(pulled.parameters(), lr=lr),
        proximal_mu=mu,
        reference=reference,
    )
    for name, value in get_state(pulled).items():
        # One step: w - lr * (g + mu * (w - ref)) = plain result + lr * mu, because ref = w + 1.
        assert torch.allclose(value, get_state(plain)[name] + lr * mu, atol=1e-6), name


def test_other_optimizers_work() -> None:
    x, y = _table()
    model = _model()
    before = get_state(model)
    _round(model, x, y, optimizer=torch.optim.Adam(model.parameters(), lr=0.01))
    assert any(not torch.equal(before[name], value) for name, value in get_state(model).items())


def test_private_training_learns_when_the_noise_allows_it() -> None:
    x, y = _table(n=600, seed=3)
    model = _model(seed=1)

    def accuracy() -> float:
        model.eval()
        with torch.no_grad():
            return float(((model(x) > 0.5).float() == y).float().mean())

    before = accuracy()
    _round(
        model,
        x,
        y,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.5),
        sampling_rate=0.2,
        steps=150,
        noise_multiplier=0.8,
    )
    assert accuracy() > 0.9
    assert accuracy() > before + 0.2


def test_a_whole_run_stays_inside_a_total_budget() -> None:
    """Calibrate once for the whole run; every round fits; one extra round is refused."""
    x, y = _table(n=64)
    rounds, rate = 4, 0.25
    steps = total_steps(rounds, 1, rate)  # 4 rounds of 4 steps
    assert steps == 16
    sigma = calibrate_noise_multiplier(1.0, DELTA, rate, steps).noise_multiplier
    ledger = PrivacyLedger(PrivacyPolicy(1.0, DELTA))
    model = _model()
    for _ in range(rounds):
        _round(
            model,
            x,
            y,
            ledger=ledger,
            sampling_rate=rate,
            steps=steps // rounds,
            noise_multiplier=sigma,
        )
    assert 0.99 < ledger.spent("client-0") <= 1.0
    with pytest.raises(BudgetExceeded):
        _round(
            model,
            x,
            y,
            ledger=ledger,
            sampling_rate=rate,
            steps=steps // rounds,
            noise_multiplier=sigma,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"steps": -1}, "steps"),
        ({"proximal_mu": -0.1}, "proximal_mu"),
        ({"proximal_mu": 0.1}, "reference weights are required"),
        ({"declared_records": 0}, "declared_records"),
        ({"sampling_rate": 0.0}, "sampling_rate"),
        ({"noise_multiplier": 0.0}, "noise_multiplier"),
    ],
)
def test_bad_settings_are_refused_before_anything_is_charged(
    overrides: dict[str, object], message: str
) -> None:
    x, y = _table()
    ledger = _ledger()
    with pytest.raises(ValueError, match=message):
        _round(_model(), x, y, ledger=ledger, **overrides)
    assert ledger.scopes == ()


def test_bad_data_or_models_are_refused_before_anything_is_charged() -> None:
    x, y = _table()
    ledger = _ledger()
    with pytest.raises(ValueError, match="at least one record"):
        _round(_model(), x[:0], y[:0], ledger=ledger)
    with pytest.raises(ValueError, match="same number of records"):
        _round(_model(), x, y[:5], ledger=ledger)
    mixing = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 1), nn.Flatten(0))
    with pytest.raises(ValueError, match="mixes the records"):
        _round(
            mixing,
            x,
            y,
            ledger=ledger,
            loss_fn=nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.SGD(mixing.parameters(), lr=0.1),
        )
    assert ledger.scopes == ()


def test_a_round_of_zero_steps_does_nothing() -> None:
    x, y = _table()
    model = _model()
    before = get_state(model)
    ledger = _ledger()
    assert _round(model, x, y, ledger=ledger, steps=0) == RoundStats(0, 0, 0)
    assert ledger.scopes == ()
    assert all(torch.equal(before[name], value) for name, value in get_state(model).items())
