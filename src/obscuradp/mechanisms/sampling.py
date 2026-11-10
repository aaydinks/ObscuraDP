"""Choosing which records take part in one training step.

The privacy accountants assume **Poisson sampling**: for every step, every record is
included independently with the same probability $q$. The batch size is then random (it
can be zero), and that randomness is part of the guarantee. Shuffling the data and cutting
it into fixed-size batches is a different procedure with a different (weaker, and harder to
state) guarantee, so the private trainer never does it.
"""

from __future__ import annotations

import torch


def poisson_sample(
    n_records: int, sampling_rate: float, generator: torch.Generator
) -> torch.Tensor:
    """Indices of the records drawn for one step, each included with probability $q$.

    Args:
        n_records: Number of records the participant holds.
        sampling_rate: Inclusion probability $q$, in (0, 1].
        generator: The random stream of this participant's sampling.

    Returns:
        A sorted 1-D tensor of indices. It may be empty.
    """
    if n_records < 0:
        raise ValueError("n_records must not be negative")
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1]")
    # float64 draws lie in [0, 1), so a rate of exactly 1 always includes every record.
    draws = torch.rand(n_records, generator=generator, dtype=torch.float64)
    return torch.nonzero(draws < sampling_rate).flatten()
