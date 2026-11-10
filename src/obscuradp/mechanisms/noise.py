"""The noise that protects a release.

Two rules:

- Noise for real data must not be predictable. ``entropy_generator`` seeds the random stream
  from the operating system. A fixed seed is for simulations and tests only, because anyone
  who knows the seed can subtract the noise.
- This module draws ordinary floating-point Gaussian noise. Floating-point noise has known
  weaknesses against an attacker who can inspect the exact low-order bits of a released
  value (finding G16 in ``docs/AUDIT.md``). The integer noise path planned for a later step
  closes that gap; until then the library is labelled as research software.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence

import torch


def entropy_generator() -> torch.Generator:
    """A random stream seeded from the operating system, for noise that protects real data."""
    generator = torch.Generator()
    generator.manual_seed(secrets.randbits(63))
    return generator


def gaussian_noise(
    shape: Sequence[int],
    standard_deviation: float,
    generator: torch.Generator,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Independent Gaussian noise with mean 0 and the given standard deviation."""
    if not standard_deviation >= 0.0:
        raise ValueError("standard_deviation must not be negative")
    if standard_deviation == 0.0:
        return torch.zeros(tuple(shape), dtype=dtype)
    return torch.normal(
        mean=0.0, std=standard_deviation, size=tuple(shape), generator=generator, dtype=dtype
    )
