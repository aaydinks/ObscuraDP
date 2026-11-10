"""Made-up tables with the public shape of the real ones, for choosing settings.

Settings such as the learning rate must not be chosen by looking at results on the private
tables. These stand-ins copy only facts that are public about each task: the number of rows
and encoded columns, roughly how common the positive class is, and how the rows are spread
over participants. The rows themselves are random, with a made-up relation between columns
and outcome.
"""

from __future__ import annotations

import numpy as np

from benchmarks.phase2.tables import SCHEMAS, split_among_clients
from obscuradp.tables import ClientData

# name: (rows, share of positive rows, how strongly the columns predict the outcome)
# The strengths make the best possible AUC about 0.83 (stroke-shaped) and 0.96 (loan-shaped),
# in line with what is commonly reported for these two public tasks.
SHAPES = {"stroke": (4900, 0.043, 3.5), "loan": (4270, 0.62, 12.0)}


def make_standin(name: str, seed: int) -> list[ClientData]:
    """A random table shaped like ``name``, split among participants like the real one."""
    rows, positive_share, strength = SHAPES[name]
    width = SCHEMAS[name].width
    rng = np.random.default_rng(1_000_003 * seed + len(name))
    features = rng.uniform(-1.0, 1.0, size=(rows, width)).astype(np.float32)
    # The last columns stand for 0/1 category indicators.
    indicators = width // 2
    features[:, -indicators:] = (features[:, -indicators:] > 0.2).astype(np.float32)
    weights = rng.normal(size=width)
    weights[rng.random(width) < 0.4] = 0.0  # some columns carry no signal
    score = strength * (features @ weights) / np.sqrt(max(1.0, np.count_nonzero(weights)))
    score = score + rng.logistic(size=rows)
    labels = (score > np.quantile(score, 1.0 - positive_share)).astype(np.int64)
    return split_among_clients(name, features, labels, seed)
