"""Each participant's own training, validation and test rows.

Settings must be chosen on rows that are neither trained on nor used for the final score.
Without a validation part, people tune on the test rows and report numbers that are too
good. So every participant's table is cut three ways, on that participant's machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.int_]
FloatTable = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ClientData:
    """One participant's encoded rows, already cut into the three parts."""

    name: str
    train_x: FloatTable
    train_y: FloatTable
    validation_x: FloatTable
    validation_y: FloatTable
    test_x: FloatTable
    test_y: FloatTable

    def __post_init__(self) -> None:
        for part in ("train", "validation", "test"):
            x, y = getattr(self, f"{part}_x"), getattr(self, f"{part}_y")
            if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{self.name}: {part} features must be a table with one label per row"
                )
        if self.train_x.shape[0] == 0:
            raise ValueError(f"{self.name}: at least one training row is needed")
        widths = {self.train_x.shape[1], self.validation_x.shape[1], self.test_x.shape[1]}
        if len(widths) != 1:
            raise ValueError(f"{self.name}: all parts must have the same columns")

    @property
    def n_train(self) -> int:
        """Number of training rows. The protocol treats this number as public."""
        return int(self.train_x.shape[0])


def split_indices(
    n_rows: int,
    *,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int,
    labels: npt.ArrayLike | None = None,
) -> tuple[IndexArray, IndexArray, IndexArray]:
    """Cut row numbers ``0 .. n_rows - 1`` into training, validation and test parts.

    With ``labels``, each class is cut separately, so a rare class appears in every part as
    far as its size allows. Every row lands in exactly one part.

    Returns:
        ``(train, validation, test)``, each sorted.
    """
    if n_rows < 1:
        raise ValueError("need at least one row")
    if validation < 0 or test < 0 or validation + test >= 1:
        raise ValueError("validation and test shares must be non-negative and sum to less than 1")
    rng = np.random.default_rng(seed)
    if labels is None:
        groups = [np.arange(n_rows)]
    else:
        y = np.asarray(labels).ravel()
        if y.shape[0] != n_rows:
            raise ValueError("need exactly one label per row")
        groups = [np.flatnonzero(y == value) for value in np.unique(y)]
    parts: tuple[list[IndexArray], list[IndexArray], list[IndexArray]] = ([], [], [])
    for group in groups:
        shuffled = rng.permutation(group)
        n_validation = round(validation * shuffled.size)
        n_test = round(test * shuffled.size)
        # Training keeps at least one row of every class that has any.
        while n_validation + n_test >= shuffled.size and n_validation + n_test > 0:
            if n_validation >= n_test:
                n_validation -= 1
            else:
                n_test -= 1
        parts[1].append(shuffled[:n_validation])
        parts[2].append(shuffled[n_validation : n_validation + n_test])
        parts[0].append(shuffled[n_validation + n_test :])
    train, validation_rows, test_rows = (np.sort(np.concatenate(part)) for part in parts)
    return train, validation_rows, test_rows


def make_client_data(
    name: str,
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    validation: float = 0.15,
    test: float = 0.15,
    seed: int,
    stratify: bool = True,
) -> ClientData:
    """Cut one participant's encoded table into its three parts."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).ravel()
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be a table with one label per row")
    train, validation_rows, test_rows = split_indices(
        x.shape[0], validation=validation, test=test, seed=seed, labels=y if stratify else None
    )
    return ClientData(
        name=name,
        train_x=x[train],
        train_y=y[train],
        validation_x=x[validation_rows],
        validation_y=y[validation_rows],
        test_x=x[test_rows],
        test_y=y[test_rows],
    )
