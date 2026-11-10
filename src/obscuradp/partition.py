"""Splitting one dataset's rows among simulated participants.

These functions are for simulations and benchmarks. In a real deployment each participant
already holds its own rows and nothing is split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.int_]


@dataclass(frozen=True)
class PartitionInfo:
    """What a split looks like: its kind, each participant's size, and class counts if known."""

    kind: str
    sizes: tuple[int, ...]
    attempts: int = 1
    class_counts: dict[int, tuple[int, ...]] = field(default_factory=dict)


def iid_partition(
    n_rows: int, num_clients: int, seed: int
) -> tuple[list[IndexArray], PartitionInfo]:
    """Shuffle row indices and cut them into ``num_clients`` near-equal parts."""
    _check_clients(num_clients)
    if n_rows < num_clients:
        raise ValueError("need at least one row per participant")
    rng = np.random.RandomState(seed)  # RandomState on purpose: same draws as the research code
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    parts = [np.array(part) for part in np.array_split(indices, num_clients)]
    return parts, PartitionInfo(kind="iid", sizes=tuple(len(p) for p in parts))


def dirichlet_partition(
    labels: npt.ArrayLike,
    num_clients: int,
    alpha: float,
    seed: int,
    *,
    min_per_class: int = 0,
    max_attempts: int = 5000,
) -> tuple[list[IndexArray], PartitionInfo]:
    """Split rows so that participants see different class mixes.

    For each class, the share each participant receives is drawn from a Dirichlet
    distribution with concentration ``alpha``. A small ``alpha`` gives very uneven mixes; a
    large one approaches an even split. The draw is repeated until every participant holds
    at least ``min_per_class`` rows of every class.

    Args:
        labels: Class label of each row.
        num_clients: Number of participants.
        alpha: Concentration of the Dirichlet distribution. Must be positive.
        seed: Seed for the split.
        min_per_class: Smallest number of rows of each class a participant may end up with.
        max_attempts: How many draws to try before giving up.

    Raises:
        RuntimeError: If no draw within ``max_attempts`` satisfies ``min_per_class``.
    """
    _check_clients(num_clients)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    y = np.asarray(labels)
    rng = np.random.RandomState(seed)  # RandomState on purpose: same draws as the research code
    classes = np.unique(y)
    attempt = 0
    while True:
        attempt += 1
        client_rows: list[list[int]] = [[] for _ in range(num_clients)]
        for label in classes:
            rows = np.where(y == label)[0]
            rng.shuffle(rows)
            shares = rng.dirichlet([alpha] * num_clients)
            cuts = (np.cumsum(shares) * len(rows)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(rows, cuts)):
                client_rows[client].extend(chunk.tolist())
        per_class = [[int((y[owned] == label).sum()) for label in classes] for owned in client_rows]
        if min(min(row_counts) for row_counts in per_class) >= min_per_class:
            for owned in client_rows:
                rng.shuffle(owned)
            parts = [np.array(owned) for owned in client_rows]
            counts = {
                int(label): tuple(per_class[c][k] for c in range(num_clients))
                for k, label in enumerate(classes)
            }
            info = PartitionInfo(
                kind="dirichlet",
                sizes=tuple(len(p) for p in parts),
                attempts=attempt,
                class_counts=counts,
            )
            return parts, info
        if attempt > max_attempts:
            raise RuntimeError(
                "could not give every participant enough rows of every class; "
                "lower min_per_class or raise alpha"
            )


def _check_clients(num_clients: int) -> None:
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
