"""One run of the Phase 2 benchmark: settings in, a result record out."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

import obscuradp as odp
from benchmarks.phase2.tables import HIDDEN, PUBLIC_PRIOR, balanced_rows
from obscuradp.metrics import binary_metrics
from obscuradp.models import TabularMLP
from obscuradp.results import make_result
from obscuradp.state import set_state
from obscuradp.tables import ClientData

DELTA = 1e-5
ROUNDS = 20
LOCAL_EPOCHS = 3.0
BATCH_SIZE = 32
BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0)
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("network", "logistic")
VALUE_BITS = 12


@dataclass(frozen=True)
class Setting:
    """What distinguishes one run from another."""

    dataset: str
    model: str  # "network" (the research network) or "logistic" (no hidden layer)
    epsilon: float | None  # total budget for the whole run; None for no protection
    secure_sum: bool
    learning_rate: float
    clip_norm: float
    seed: int
    rounds: int = ROUNDS

    @property
    def label(self) -> str:
        budget = "none" if self.epsilon is None else f"{self.epsilon:g}"
        masked = "masked" if self.secure_sum else "clear"
        return f"{self.dataset}_{self.model}_eps-{budget}_{masked}_seed-{self.seed}"


def build_simulation(setting: Setting, clients: list[ClientData]) -> odp.Simulation:
    """The simulation for ``setting``. The research protocol (20 rounds, 3 local passes,
    batch 32) is kept; only what the audit found wrong is changed."""
    width = clients[0].train_x.shape[1]
    hidden = HIDDEN[setting.dataset] if setting.model == "network" else ()
    prior = PUBLIC_PRIOR[setting.dataset]
    imbalance = (
        odp.imbalance.NoRebalancing() if prior is None else odp.imbalance.ClassWeighted(prior)
    )
    private = setting.epsilon is not None
    plan = odp.Plan(
        dp=odp.LocalDPSGD(clip_norm=setting.clip_norm) if private else None,
        imbalance=imbalance,
        secure_sum=odp.crypto.PairwiseMasking(value_bits=VALUE_BITS, value_range=1.0)
        if setting.secure_sum
        else None,
    )
    return odp.Simulation(
        model_fn=lambda: TabularMLP(width, hidden),
        clients=clients,
        plan=plan,
        policy=odp.PrivacyPolicy(setting.epsilon, DELTA) if setting.epsilon is not None else None,
        rounds=setting.rounds,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=setting.learning_rate,
        optimizer="sgd",
        weighted_average=True,
        evaluate_every=5,
        choose_threshold=True,
    )


def run_setting(setting: Setting, clients: list[ClientData]) -> dict[str, Any]:
    """Run one setting and return its result record, with the balanced-set scores added."""
    torch.set_num_threads(1)
    simulation = build_simulation(setting, clients)
    result = simulation.run(seed=setting.seed)
    base = result.to_record()

    width = clients[0].train_x.shape[1]
    hidden = HIDDEN[setting.dataset] if setting.model == "network" else ()
    model = TabularMLP(width, hidden)
    set_state(model, result.state)
    model.eval()
    features, labels = balanced_rows(clients, setting.seed)
    with torch.no_grad():
        scores = model(torch.as_tensor(features)).numpy()
    data = {
        **base["data"],
        "test_balanced": binary_metrics(scores, labels),
        "test_balanced_at_chosen_threshold": binary_metrics(
            scores, labels, threshold=result.threshold
        ),
        "test_rows": int(sum(client.test_y.size for client in clients)),
        "test_positive_rows": int(sum(client.test_y.sum() for client in clients)),
        "balanced_rows": int(labels.size),
        "label": setting.label,
    }
    # The benchmark's own settings are part of the fingerprinted configuration.
    config = {**base["config"], "benchmark": asdict(setting)}
    return make_result(kind="phase2_run", config=config, seeds=[setting.seed], data=data)


def mean_and_interval(values: list[float]) -> dict[str, float]:
    """Mean, sample standard deviation and a 95% interval (Student's t) of a few runs."""
    from scipy import stats

    data = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if data.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n": 0,
        }
    mean = float(data.mean())
    if data.size == 1:
        return {
            "mean": mean,
            "std": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n": 1,
        }
    std = float(data.std(ddof=1))
    half = float(stats.t.ppf(0.975, data.size - 1)) * std / np.sqrt(data.size)
    return {"mean": mean, "std": std, "low": mean - half, "high": mean + half, "n": int(data.size)}
