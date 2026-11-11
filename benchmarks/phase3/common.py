"""One run of the Phase 3 comparison: an arm and its settings in, a result record out.

An arm is one way of protecting a run. All arms are run on the same tables, splits and
seeds and at the same total budgets, so that they differ in the means of protection only.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import torch

import obscuradp as odp
from benchmarks.phase2.common import BATCH_SIZE, DELTA, LOCAL_EPOCHS, ROUNDS
from benchmarks.phase2.tables import HIDDEN, PUBLIC_PRIOR
from obscuradp.models import TabularMLP
from obscuradp.plan import SmallFederationWarning
from obscuradp.results import make_result
from obscuradp.tables import ClientData

BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0)
FEW_BUDGETS = (1.0, 8.0)
SEEDS = {"stroke": tuple(range(42, 52)), "loan": tuple(range(42, 47))}
MODELS = ("network", "logistic")
COLLUDING = 1
# Every row in every round: any batch size above the number of rows does that.
FULL_BATCH = 10**9

# name: (what it is, the budgets it is run at)
ARMS: dict[str, tuple[str, tuple[float, ...]]] = {
    "local": ("every participant adds all of its own noise (Phase 2 protocol)", BUDGETS),
    "local_tuned": ("the same, with rounds and local passes chosen like the other arms", BUDGETS),
    "shared": ("shared noise, ordinary (Gaussian), one assumed colluder", BUDGETS),
    "shared_whole": ("shared noise, whole numbers (Skellam), one assumed colluder", BUDGETS),
    "server": ("a trusted server adds the noise (upper limit)", BUDGETS),
    "local_masked": ("local noise through the secure sum, 12-bit values", FEW_BUDGETS),
    "local_masked8": ("local noise through the secure sum, 8-bit values", FEW_BUDGETS),
    "participant": ("whole participants protected, shared noise", FEW_BUDGETS),
}
BASELINE = "none"
ONE_STEP = ("shared", "shared_whole", "server")


@dataclass(frozen=True)
class Setting:
    """What distinguishes one run from another."""

    dataset: str
    model: str  # "network" (the research network) or "logistic" (no hidden layer)
    arm: str  # a key of ARMS, or "none" for no protection
    epsilon: float | None  # total budget for the whole run; None for no protection
    seed: int
    learning_rate: float
    clip_norm: float
    rounds: int = ROUNDS
    batch_size: int = BATCH_SIZE
    local_epochs: float = LOCAL_EPOCHS

    @property
    def label(self) -> str:
        budget = "none" if self.epsilon is None else f"{self.epsilon:g}"
        return f"{self.dataset}_{self.model}_{self.arm}_eps-{budget}_seed-{self.seed}"


def _model(setting: Setting, width: int) -> TabularMLP:
    hidden = HIDDEN[setting.dataset] if setting.model == "network" else ()
    return TabularMLP(width, hidden)


def build_simulation(setting: Setting, clients: list[ClientData]) -> odp.Simulation:
    """The simulation for ``setting``.

    The local arms and the baseline keep the research protocol of Phase 2 (20 rounds, 3
    local passes, batch 32) and its settings, so their results can be compared with the
    stored Phase 2 runs number for number. The one-step arms have their own number of
    rounds and batch size, because a round means something else there.
    """
    width = clients[0].train_x.shape[1]
    prior = PUBLIC_PRIOR[setting.dataset]
    imbalance = (
        odp.imbalance.NoRebalancing() if prior is None else odp.imbalance.ClassWeighted(prior)
    )
    arm, clip = setting.arm, setting.clip_norm
    masking = odp.crypto.PairwiseMasking()
    unit = "client" if arm == "participant" else "record"
    policy = (
        None if setting.epsilon is None else odp.PrivacyPolicy(setting.epsilon, DELTA, unit=unit)
    )
    common: dict[str, Any] = {
        "model_fn": lambda: _model(setting, width),
        "clients": clients,
        "policy": policy,
        "rounds": setting.rounds,
        "learning_rate": setting.learning_rate,
        "optimizer": "sgd",
        "weighted_average": True,
        "evaluate_every": 0,
        "choose_threshold": True,
    }
    if arm in ONE_STEP:
        if arm == "server":
            plan = odp.Plan(dp=odp.CentralDPSGD(clip_norm=clip), imbalance=imbalance)
        else:
            noise = "skellam" if arm == "shared_whole" else "gaussian"
            plan = odp.Plan(
                dp=odp.DistributedDPSGD(max_colluding=COLLUDING, clip_norm=clip, noise=noise),
                imbalance=imbalance,
                secure_sum=masking,
            )
        return odp.Simulation(
            plan=plan, batch_size=setting.batch_size, allow_unsafe=arm == "server", **common
        )
    if arm == "participant":
        plan = odp.Plan(
            dp=odp.ClientLevelDP(sampling_rate=1.0, clip_norm=clip, max_colluding=COLLUDING),
            imbalance=imbalance,
            secure_sum=masking,
        )
    elif arm == BASELINE:
        plan = odp.Plan(dp=None, imbalance=imbalance)
    else:
        bits = {"local": None, "local_tuned": None, "local_masked": 12, "local_masked8": 8}[arm]
        plan = odp.Plan(
            dp=odp.LocalDPSGD(clip_norm=clip),
            imbalance=imbalance,
            secure_sum=None
            if bits is None
            else odp.crypto.PairwiseMasking(value_bits=bits, value_range=1.0),
        )
    with warnings.catch_warnings():
        # Five participants are far too few for this arm; showing that is its purpose.
        warnings.simplefilter("ignore", SmallFederationWarning)
        return odp.Simulation(
            plan=plan, local_epochs=setting.local_epochs, batch_size=setting.batch_size, **common
        )


def run_setting(setting: Setting, clients: list[ClientData]) -> dict[str, Any]:
    """Run one setting and return its result record."""
    torch.set_num_threads(1)
    result = build_simulation(setting, clients).run(seed=setting.seed)
    base = result.to_record()
    communication = base["data"]["communication"]
    data = {
        **base["data"],
        # What one participant uploads over the whole run, key setup included.
        "upload_bytes_per_run": communication["upload_bytes_per_round"] * setting.rounds
        + communication["setup_bytes"],
        "test_rows": int(sum(client.test_y.size for client in clients)),
        "test_positive_rows": int(sum(client.test_y.sum() for client in clients)),
        "label": setting.label,
    }
    config = {**base["config"], "benchmark": asdict(setting)}
    return make_result(kind="phase3_run", config=config, seeds=[setting.seed], data=data)
