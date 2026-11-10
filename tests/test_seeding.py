"""Seeds for the separate random streams of a run."""

import itertools

import pytest
import torch

from obscuradp.seeding import PURPOSES, derive_seed, numpy_generator, torch_generator


def test_same_inputs_give_the_same_seed() -> None:
    assert derive_seed(42, "train", 3, 7) == derive_seed(42, "train", 3, 7)
    assert 0 <= derive_seed(42, "train", 3, 7) < 2**63


def test_no_collisions_where_the_old_formula_collides() -> None:
    """The research code used seed*1000 + round*10 + participant: (0, 10) equals (1, 0)."""
    assert 42 * 1000 + 0 * 10 + 10 == 42 * 1000 + 1 * 10 + 0
    assert derive_seed(42, "train", 0, 10) != derive_seed(42, "train", 1, 0)


def test_all_streams_of_a_large_run_are_distinct() -> None:
    seeds = {
        derive_seed(root, purpose, rnd, client)
        for root in (0, 1, 42)
        for purpose in PURPOSES
        for rnd, client in itertools.product(range(30), range(40))
    }
    assert len(seeds) == 3 * len(PURPOSES) * 30 * 40


def test_purpose_and_path_length_matter() -> None:
    assert derive_seed(1, "train", 0) != derive_seed(1, "noise", 0)
    assert derive_seed(1, "train", 0) != derive_seed(1, "train", 0, 0)
    assert derive_seed(1, "train") != derive_seed(2, "train")


def test_generators_are_repeatable_and_independent() -> None:
    a = numpy_generator(5, "partition", 1).random(4)
    b = numpy_generator(5, "partition", 1).random(4)
    c = numpy_generator(5, "partition", 2).random(4)
    assert (a == b).all()
    assert not (a == c).all()
    x = torch.randn(4, generator=torch_generator(5, "noise", 0, 0))
    y = torch.randn(4, generator=torch_generator(5, "noise", 0, 0))
    z = torch.randn(4, generator=torch_generator(5, "noise", 0, 1))
    assert torch.equal(x, y)
    assert not torch.equal(x, z)


def test_using_a_generator_leaves_the_global_stream_alone() -> None:
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    torch.randn(100, generator=torch_generator(9, "init"))
    assert torch.equal(torch.rand(3), expected)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((-1, "train"), "negative"),
        ((1, "banana"), "unknown purpose"),
        ((1, "train", -2), "negative"),
    ],
)
def test_bad_inputs_are_refused(args: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        derive_seed(*args)
