"""Seeds for the separate random streams of one run.

A run needs many independent random streams: one per participant per round for training,
others for model initialization, masks and data splits. Deriving them by arithmetic on the
run's seed (for example ``seed * 1000 + round * 10 + participant``) makes different streams
collide as soon as there are ten participants. Here every stream gets its own seed derived
with ``numpy.random.SeedSequence``, which is designed for exactly this.

These seeds make simulations repeatable. They are not a source of secret randomness: noise
that protects real data must come from the operating system, never from a seed someone chose.
"""

from __future__ import annotations

import numpy as np
import torch

# Fixed numbers for each purpose, so adding a purpose never changes the seeds of the others.
PURPOSES = {
    "init": 0,
    "partition": 1,
    "train": 2,
    "sampling": 3,
    "noise": 4,
    "mask": 5,
    "evaluation": 6,
    "data": 7,
    "quantize": 8,
    "canary": 9,
}


def derive_seed(root: int, purpose: str, *path: int) -> int:
    """Return a 63-bit seed for one stream, determined by ``root``, ``purpose`` and ``path``.

    ``path`` identifies the stream within its purpose, for example ``(round, participant)``.
    Different ``(purpose, path)`` pairs give statistically independent seeds.
    """
    if root < 0:
        raise ValueError("root seed must not be negative")
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; use one of {sorted(PURPOSES)}")
    if any(part < 0 for part in path):
        raise ValueError("path entries must not be negative")
    sequence = np.random.SeedSequence(entropy=root, spawn_key=(PURPOSES[purpose], *path))
    return int(sequence.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def numpy_generator(root: int, purpose: str, *path: int) -> np.random.Generator:
    """A numpy random generator for one stream."""
    return np.random.default_rng(derive_seed(root, purpose, *path))


def torch_generator(root: int, purpose: str, *path: int) -> torch.Generator:
    """A CPU torch random generator for one stream."""
    generator = torch.Generator()
    generator.manual_seed(derive_seed(root, purpose, *path))
    return generator
