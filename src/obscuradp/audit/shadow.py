"""A second opinion on "was this record used?": the shadow-model attack.

The attacker here is an outsider who gets the final model. A plain look at how well the
model fits a record says little, because some records are simply easy and others hard
whether or not they were used. The shadow-model attack removes that: the pipeline is run
many times, each time on a random half of the rows, so that every row is in the training
rows of about half of the runs. For a run under attack, each row's fit is then compared
with how the **same row** fared in the other runs when it was in and when it was out
(the likelihood-ratio attack of Carlini, Chien, Nasr, Song, Terzis and Tramer, "Membership
Inference Attacks From First Principles", 2022).

This is the strongest known attack of its kind for an outsider, and it is expensive: it
needs the whole pipeline many times over. It is a measuring tool for the people who run a
federation, not something a real outsider could do without the participants' rows.

The functions here work on plain tables of numbers, one row of the table per run and one
column per record, so they can be tested without training anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from obscuradp.audit.membership import (
    Canaries,
    SimulationBuilder,
    choose_canaries,
    leave_out,
)
from obscuradp.sim.simulation import Simulation, SimulationResult
from obscuradp.state import State, set_state
from obscuradp.tables.split import ClientData

ATTACK_SHADOW = "shadow models"

BoolTable = npt.NDArray[np.bool_]
FloatTable = npt.NDArray[np.float64]
FloatVector = npt.NDArray[np.float64]

# A model's confidence is kept this far from 0 and 1, so its log-odds stay finite.
_EDGE = 1e-7
# With fewer runs than this on one side, a record is compared with all records instead.
_MIN_RUNS_PER_SIDE = 2


def confidence(
    model: nn.Module,
    state: State,
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    from_logits: bool,
) -> FloatVector:
    """How sure the model is of each record's true label, as log-odds.

    For a record with label $y$ and model probability $p$ of that label this is
    $\\log\\frac{p}{1 - p}$. On this scale the values of one record over many training
    runs are close to bell-shaped, which is what the attack counts on.

    Args:
        model: A model of the right shape; its weights are replaced by ``state``.
        state: The weights to judge.
        features: The records, one per row.
        labels: Their 0/1 labels.
        from_logits: Whether the model returns raw scores (true) or probabilities.
    """
    set_state(model, state)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(torch.as_tensor(np.asarray(features, dtype=np.float32))).double()
    finally:
        model.train(was_training)
    if from_logits:
        log_odds = output
    else:
        probability = output.clamp(_EDGE, 1.0 - _EDGE)
        log_odds = torch.log(probability) - torch.log1p(-probability)
    positive = torch.as_tensor(np.asarray(labels, dtype=np.float64)) > 0.5
    limit = float(np.log((1.0 - _EDGE) / _EDGE))
    signed = torch.where(positive, log_odds, -log_odds).clamp(-limit, limit)
    result: FloatVector = signed.numpy()
    return result


@dataclass(frozen=True)
class HalfRun:
    """One run of a pipeline on a random half of the rows.

    Attributes:
        kept: One coin per training row of the full tables (participant after participant):
            true if the row was in this run's training rows.
        labels: The rows' labels, in the same order.
        confidence: The final model's confidence in each row's true label (log-odds).
        result: The run's own result, with its privacy report.
    """

    kept: npt.NDArray[np.bool_]
    labels: FloatVector
    confidence: FloatVector
    result: SimulationResult


def all_rows(clients: Sequence[ClientData], seed: int) -> Canaries:
    """Every training row as a test record, each with its own fair coin."""
    return choose_canaries(clients, seed, fraction=1.0, cap=10**9)


def _from_logits(simulation: Simulation, model: nn.Module) -> bool:
    if simulation.outputs is not None:
        return simulation.outputs == "logit"
    return not bool(getattr(model, "probability_output", True))


def random_half_run(build: SimulationBuilder, clients: Sequence[ClientData], seed: int) -> HalfRun:
    """Train once on a random half of every participant's rows and judge every row.

    Each row is kept or left out by a fair coin drawn from ``seed``. The participants'
    public sizes stay their full sizes, as in ``obscuradp.audit.membership``. Call this
    with the same participants and different seeds to get the runs the attack needs.

    Args:
        build: ``build(clients, declared_rows)`` returns the simulation to attack.
        clients: The participants with all of their rows.
        seed: Seed of the coins and of the run.
    """
    coins = all_rows(clients, seed)
    reduced, declared = leave_out(clients, coins)
    simulation = build(reduced, declared)
    if simulation.declared_rows != declared:
        raise ValueError("build must pass declared_rows on to the Simulation")
    # Model building draws from the global random stream, so it happens before the run.
    judge = simulation.model_fn()
    result = simulation.run(seed)
    features = np.concatenate([client.train_x for client in clients])
    labels = np.concatenate([client.train_y for client in clients]).astype(np.float64)
    return HalfRun(
        kept=coins.coins(),
        labels=labels,
        confidence=confidence(
            judge, result.state, features, labels, from_logits=_from_logits(simulation, judge)
        ),
        result=result,
    )


def likelihood_ratio_scores(
    target: npt.ArrayLike, shadow_confidence: npt.ArrayLike, shadow_kept: npt.ArrayLike
) -> FloatVector:
    """Score every record of one run: how much more its fit looks like "in" than like "out".

    For each record the other runs give two groups of values, those where the record was in
    and those where it was out. Each group is summed up by its mean; the spread is taken
    from all records together (with a few dozen runs a spread per record would be too
    shaky). The score is the log of the ratio of the two bell curves at the value seen in
    the run under attack. A record that was in too few or out of too few of the other runs
    is compared with the means over all records.

    Args:
        target: The confidence of every record in the run under attack.
        shadow_confidence: The same for the other runs, one row per run.
        shadow_kept: For the other runs, whether each record was in (same shape).

    Returns:
        One score per record; higher means "more likely in".
    """
    seen = np.asarray(target, dtype=np.float64)
    values = np.asarray(shadow_confidence, dtype=np.float64)
    inside = np.asarray(shadow_kept, dtype=bool)
    if values.ndim != 2 or values.shape != inside.shape or values.shape[1] != seen.shape[0]:
        raise ValueError("need one row per shadow run and one column per record")
    if seen.ndim != 1:
        raise ValueError("target must hold one value per record")
    means = []
    spreads = []
    for side in (inside, ~inside):
        count = side.sum(axis=0)
        total = np.where(side, values, 0.0).sum(axis=0)
        enough = count >= _MIN_RUNS_PER_SIDE
        overall = float(values[side].mean()) if side.any() else 0.0
        mean = np.where(enough, total / np.maximum(count, 1), overall)
        # One spread for all records: the scatter of each record around its own mean.
        residual = np.where(side & enough, values - mean, 0.0)
        freedom = int((count[enough] - 1).sum())
        spread = float(np.sqrt((residual**2).sum() / freedom)) if freedom > 0 else 1.0
        means.append(mean)
        spreads.append(max(spread, 1e-6))
    (mean_in, mean_out), (spread_in, spread_out) = means, spreads
    scores: FloatVector = (
        -0.5 * ((seen - mean_in) / spread_in) ** 2
        - np.log(spread_in)
        + 0.5 * ((seen - mean_out) / spread_out) ** 2
        + np.log(spread_out)
    )
    return scores


def attack_each_run(
    confidence_by_run: npt.ArrayLike,
    kept_by_run: npt.ArrayLike,
    *,
    targets: Sequence[int] | None = None,
    shadows: Sequence[int] | None = None,
) -> dict[int, FloatVector]:
    """Attack some runs with the others as their shadows.

    By default every run is attacked in turn with all the others as shadows. To get runs
    whose guesses may be pooled into one lower limit on the budget, name ``targets`` and
    ``shadows`` that do not overlap: a target's guesses then depend on other targets
    through nothing at all.

    Args:
        confidence_by_run: One row per run, one column per record.
        kept_by_run: Whether each record was in each run (same shape).
        targets: The runs to attack (default: all).
        shadows: The runs to learn from (default: for each target, all the others).

    Returns:
        For each attacked run, one score per record.
    """
    values = np.asarray(confidence_by_run, dtype=np.float64)
    inside = np.asarray(kept_by_run, dtype=bool)
    if values.ndim != 2 or values.shape != inside.shape:
        raise ValueError("need one row per run and one column per record, twice")
    runs = values.shape[0]
    attacked = list(range(runs)) if targets is None else [int(t) for t in targets]
    scores = {}
    for target in attacked:
        others = (
            [run for run in range(runs) if run != target]
            if shadows is None
            else [int(run) for run in shadows]
        )
        if target in others:
            raise ValueError("a run cannot be its own shadow")
        if len(others) < 2 * _MIN_RUNS_PER_SIDE:
            raise ValueError("too few shadow runs")
        scores[target] = likelihood_ratio_scores(values[target], values[others], inside[others])
    return scores
