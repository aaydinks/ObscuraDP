"""One run of the benchmark: a dataset, a way of protecting it and its settings in, a record out.

Five datasets. Three tables in which the budget protects single rows (a made-up health
table with ten hospitals, and the stroke and loan tables of the earlier phases with five
participants), and two public datasets with a natural split into many participants, in
which the budget protects a whole participant (handwriting by 3,400 writers, text by 644
speaking roles).

What is fixed here is what a run **is**. Which runs are made, with which seeds, and how
their settings are chosen is fixed in ``benchmarks/PREREGISTRATION.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

import obscuradp as odp
from benchmarks.phase2.tables import HIDDEN, PUBLIC_PRIOR, make_clients
from benchmarks.phase5 import datasets
from obscuradp.models import CharGRU, SmallCNN, TabularMLP
from obscuradp.plan import SmallFederationWarning
from obscuradp.results import make_result
from obscuradp.tables import (
    SYNTHETIC_HEALTH_PRIOR,
    ClientData,
    synthetic_health,
)
from obscuradp.tasks import ManyClasses, Task, YesNo

DELTA = 1e-5
BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0)
HEALTH_TABLE_SEED = 2026  # the one made-up health table every run uses; runs differ in its cut
FULL_BATCH = 10**9
BASELINE = "none"
# Ways of protecting single rows (tables).
ROW_ARMS = {
    "local": "the default plan: every participant adds all of its own noise",
    "shared": "shared noise, ordinary (Gaussian), one assumed colluder",
    "shared_whole": "shared noise, whole numbers (Skellam), one assumed colluder",
}
# The way of protecting whole participants (handwriting, text).
PARTICIPANT_ARM = "participant"
COLLUDING_SHARE = 0.1  # of the participants are assumed to work with the server


@dataclass(frozen=True)
class Dataset:
    """What the benchmark needs to know about a dataset.

    Attributes:
        unit: What a budget protects here: ``"record"`` or ``"client"``.
        task: What the model predicts.
        models: The models that are run, by name.
        per_round: For the many-participant datasets, how many take part per round on
            average. ``None`` where everybody takes part in every round.
        measure: The headline measure, a key of the run's test scores.
        chance: What that measure is for a model that has learned nothing.
        prior: A public share of the positive class for class weights, if one is used.
    """

    unit: str
    task: Task
    models: dict[str, Callable[[int], nn.Module]]
    per_round: int | None
    measure: str
    chance: float
    prior: float | None = None


def _table_models(hidden: tuple[int, ...]) -> dict[str, Callable[[int], nn.Module]]:
    return {
        "network": lambda width: TabularMLP(width, hidden),
        "logistic": lambda width: TabularMLP(width, ()),
    }


DATASETS: dict[str, Dataset] = {
    "health": Dataset(
        "record", YesNo(), _table_models((40, 20)), None, "auc", 0.5, SYNTHETIC_HEALTH_PRIOR
    ),
    "stroke": Dataset(
        "record", YesNo(), _table_models(HIDDEN["stroke"]), None, "auc", 0.5, PUBLIC_PRIOR["stroke"]
    ),
    "loan": Dataset(
        "record", YesNo(), _table_models(HIDDEN["loan"]), None, "auc", 0.5, PUBLIC_PRIOR["loan"]
    ),
    "femnist": Dataset(
        "client",
        ManyClasses(datasets.FEMNIST_CLASSES),
        {"cnn": lambda width: SmallCNN(datasets.FEMNIST_CLASSES, datasets.FEMNIST_SIDE)},
        200,
        "accuracy",
        # Always answering the most common class is right about this often.
        0.057,
    ),
    "shakespeare": Dataset(
        "client",
        ManyClasses(datasets.TEXT_VOCABULARY),
        {"gru": lambda width: CharGRU(datasets.TEXT_VOCABULARY)},
        50,
        "accuracy",
        # Always answering the most common symbol (a blank) is right about this often.
        0.18,
    ),
}


# (rounds, batch size, local passes per round) of every arm, where this plan fixes them.
# Stroke and loan keep the settings of Phases 2 and 3 (``benchmarks/phase3/settings.json``).
PROTOCOL: dict[str, dict[str, tuple[int, int, float]]] = {
    "health": {
        BASELINE: (30, 32, 1.0),
        "local": (30, 32, 1.0),
        "shared": (100, 640, 1.0),
        "shared_whole": (100, FULL_BATCH, 1.0),
    },
    "femnist": {BASELINE: (100, 20, 1.0), PARTICIPANT_ARM: (100, 20, 1.0)},
    "shakespeare": {BASELINE: (100, 20, 1.0), PARTICIPANT_ARM: (100, 20, 1.0)},
}
# Where the search for a learning rate and a clipping norm starts (the pilot runs with these).
STARTING_POINT = {"health": (0.1, 1.0), "femnist": (0.1, 1.0), "shakespeare": (1.0, 1.0)}


def arms_of(dataset: str) -> tuple[str, ...]:
    """The ways of protecting ``dataset`` that the benchmark runs."""
    return (PARTICIPANT_ARM,) if DATASETS[dataset].unit == "client" else tuple(ROW_ARMS)


def load(dataset: str, seed: int) -> list[ClientData]:
    """The participants of ``dataset`` for a run with ``seed``.

    Tables are cut anew for every seed (which rows are training, validation and test
    rows; for stroke and loan also which rows a participant holds, as in the earlier
    phases). The many-participant datasets have one fixed cut.
    """
    if dataset == "health":
        return synthetic_health(HEALTH_TABLE_SEED, split_seed=seed)
    if dataset in ("stroke", "loan"):
        return make_clients(dataset, seed)
    if dataset == "femnist":
        return datasets.load_femnist()
    if dataset == "shakespeare":
        return datasets.load_shakespeare()
    raise ValueError(f"unknown dataset {dataset!r}")


@dataclass(frozen=True)
class Setting:
    """What distinguishes one run from another."""

    dataset: str
    model: str
    arm: str  # "none", a key of ROW_ARMS, or "participant"
    epsilon: float | None  # total budget for the whole run; None for no protection
    seed: int
    learning_rate: float
    clip_norm: float
    rounds: int
    batch_size: int
    local_epochs: float

    @property
    def label(self) -> str:
        budget = "none" if self.epsilon is None else f"{self.epsilon:g}"
        return f"{self.dataset}_{self.model}_{self.arm}_eps-{budget}_seed-{self.seed}"


def build_simulation(
    setting: Setting, clients: list[ClientData], *, real_secure_sum: bool = False
) -> odp.Simulation:
    """The simulation for ``setting``.

    For the many-participant datasets the secure sum is simulated unless ``real_secure_sum``
    asks for the real one (the small confirming case of the written plan). The tables
    always run the real one.
    """
    data = DATASETS[setting.dataset]
    width = int(clients[0].train_x.shape[1])
    arm, clip = setting.arm, setting.clip_norm
    imbalance = (
        odp.imbalance.NoRebalancing()
        if data.prior is None
        else odp.imbalance.ClassWeighted(data.prior)
    )
    common: dict[str, Any] = {
        "model_fn": lambda: data.models[setting.model](width),
        "clients": clients,
        "rounds": setting.rounds,
        "learning_rate": setting.learning_rate,
        "optimizer": "sgd",
        "evaluate_every": 0,
        "task": data.task,
    }
    if data.unit == "client":
        rate = min(1.0, (data.per_round or len(clients)) / len(clients))
        if arm == BASELINE:
            return odp.Simulation(
                plan=odp.Plan(dp=None),
                policy=None,
                local_epochs=setting.local_epochs,
                batch_size=setting.batch_size,
                participation=rate,
                **common,
            )
        if arm != PARTICIPANT_ARM:
            raise ValueError(
                f"{setting.dataset} is run without protection or with {PARTICIPANT_ARM!r}"
            )
        assert setting.epsilon is not None
        plan = odp.Plan(
            dp=odp.ClientLevelDP(
                sampling_rate=rate,
                clip_norm=clip,
                max_colluding=round(COLLUDING_SHARE * len(clients)),
            ),
            secure_sum=odp.crypto.PairwiseMasking(),
        )
        with warnings.catch_warnings():
            # The text dataset has fewer than a thousand roles; showing what that costs
            # is its purpose.
            warnings.simplefilter("ignore", SmallFederationWarning)
            return odp.Simulation(
                plan=plan,
                policy=odp.PrivacyPolicy(setting.epsilon, DELTA, unit="client"),
                local_epochs=setting.local_epochs,
                batch_size=setting.batch_size,
                # Thousands of participants: the secure sum is simulated (see the written plan).
                crypto_in_loop=real_secure_sum,
                **common,
            )
    policy = None if setting.epsilon is None else odp.PrivacyPolicy(setting.epsilon, DELTA)
    if arm == BASELINE:
        plan = odp.Plan(dp=None, imbalance=imbalance)
    elif arm == "local":
        plan = odp.Plan(dp=odp.LocalDPSGD(clip_norm=clip), imbalance=imbalance)
    elif arm in ("shared", "shared_whole"):
        noise = "skellam" if arm == "shared_whole" else "gaussian"
        plan = odp.Plan(
            dp=odp.DistributedDPSGD(max_colluding=1, clip_norm=clip, noise=noise),
            imbalance=imbalance,
            secure_sum=odp.crypto.PairwiseMasking(),
        )
        return odp.Simulation(plan=plan, policy=policy, batch_size=setting.batch_size, **common)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return odp.Simulation(
        plan=plan,
        policy=policy,
        local_epochs=setting.local_epochs,
        batch_size=setting.batch_size,
        **common,
    )


def run_setting(
    setting: Setting,
    clients: list[ClientData],
    *,
    keep_test: bool = True,
    real_secure_sum: bool = False,
) -> dict[str, Any]:
    """Run one setting and return its result record.

    ``keep_test=False`` leaves the scores on the test rows out of the record. The pilot
    and the search use it: nothing may be chosen, or even seen, on test rows before the
    reported runs.
    """
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # read-only arrays mapped from the cache
        simulation = build_simulation(setting, clients, real_secure_sum=real_secure_sum)
        result = simulation.run(seed=setting.seed)
    base = result.to_record()
    data = {**base["data"], "label": setting.label}
    if not keep_test:
        data["test"] = None
        data["test_at_chosen_threshold"] = None
    config = {**base["config"], "benchmark": asdict(setting)}
    if len(clients) > 100:
        # Thousands of names say nothing; their number and their rows do.
        config["clients"] = {
            "participants": len(clients),
            "training_rows": int(sum(client.n_train for client in clients)),
        }
    return make_result(kind="phase5_run", config=config, seeds=[setting.seed], data=data)
