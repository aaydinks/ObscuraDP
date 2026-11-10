"""Poisson sampling and the noise source."""

import math

import pytest
import torch

from obscuradp.mechanisms import entropy_generator, gaussian_noise, poisson_sample


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_every_record_is_drawn_with_the_stated_probability() -> None:
    n, rate, steps = 200, 0.15, 4000
    generator = _generator(0)
    counts = torch.zeros(n)
    sizes = []
    for _ in range(steps):
        chosen = poisson_sample(n, rate, generator)
        counts[chosen] += 1
        sizes.append(chosen.numel())
    # Each record's count is Binomial(steps, rate): mean 600, standard deviation 22.6.
    spread = math.sqrt(steps * rate * (1 - rate))
    assert float((counts - steps * rate).abs().max()) < 4.5 * spread
    # The batch size is Binomial(n, rate): mean 30, variance 25.5.
    size = torch.tensor(sizes, dtype=torch.float64)
    assert float(size.mean()) == pytest.approx(n * rate, abs=0.4)
    assert float(size.var()) == pytest.approx(n * rate * (1 - rate), rel=0.1)
    assert min(sizes) < n * rate < max(sizes)  # the size really varies


def test_draws_are_independent_between_steps() -> None:
    generator = _generator(1)
    first = torch.zeros(20000, dtype=torch.bool)
    second = torch.zeros(20000, dtype=torch.bool)
    first[poisson_sample(20000, 0.5, generator)] = True
    second[poisson_sample(20000, 0.5, generator)] = True
    both = float((first & second).double().mean())
    assert both == pytest.approx(0.25, abs=0.015)  # 0.5 if the draws were repeated


def test_empty_batches_happen_as_often_as_they_should() -> None:
    n, rate, steps = 5, 0.2, 5000
    generator = _generator(2)
    empty = sum(poisson_sample(n, rate, generator).numel() == 0 for _ in range(steps))
    expected = (1 - rate) ** n  # 0.328
    assert empty / steps == pytest.approx(expected, abs=0.025)


def test_edges_and_repeatability() -> None:
    assert poisson_sample(7, 1.0, _generator(3)).tolist() == list(range(7))
    assert poisson_sample(0, 0.5, _generator(3)).numel() == 0
    a = poisson_sample(50, 0.3, _generator(4))
    b = poisson_sample(50, 0.3, _generator(4))
    assert a.tolist() == b.tolist()
    assert a.tolist() == sorted(a.tolist())
    assert a.dtype == torch.int64
    for rate in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="sampling_rate"):
            poisson_sample(5, rate, _generator(0))
    with pytest.raises(ValueError, match="n_records"):
        poisson_sample(-1, 0.5, _generator(0))


def test_noise_has_the_stated_mean_and_spread() -> None:
    count, spread = 200_000, 2.5
    noise = gaussian_noise((count,), spread, _generator(5), dtype=torch.float64)
    assert noise.shape == (count,)
    # The mean of 200,000 draws has standard deviation 0.0056.
    assert abs(float(noise.mean())) < 4.5 * spread / math.sqrt(count)
    # The sample standard deviation is within 0.16% (one sigma) of the truth.
    assert float(noise.std()) == pytest.approx(spread, rel=0.008)
    # Gaussian shape: 68.27% within one standard deviation, 0.27% beyond three.
    assert float((noise.abs() < spread).double().mean()) == pytest.approx(0.6827, abs=0.005)
    assert float((noise.abs() > 3 * spread).double().mean()) == pytest.approx(0.0027, abs=0.0006)


def test_noise_edges() -> None:
    assert gaussian_noise((2, 3), 1.0, _generator(6)).dtype == torch.float32
    assert gaussian_noise((2, 3), 1.0, _generator(6)).shape == (2, 3)
    assert gaussian_noise((), 1.0, _generator(6)).shape == ()
    assert torch.equal(gaussian_noise((4,), 0.0, _generator(6)), torch.zeros(4))
    assert torch.equal(
        gaussian_noise((4,), 1.0, _generator(7)), gaussian_noise((4,), 1.0, _generator(7))
    )
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="standard_deviation"):
            gaussian_noise((4,), bad, _generator(6))


def test_noise_for_real_data_is_not_predictable() -> None:
    first = gaussian_noise((8,), 1.0, entropy_generator())
    second = gaussian_noise((8,), 1.0, entropy_generator())
    assert not torch.equal(first, second)
