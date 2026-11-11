"""One audited run of the Phase 4 attack tests: an arm and its settings in, counts out.

Every run trains once on a real table in which about a quarter of each participant's
training rows (at most 250) are test records, each kept or left out by a fair coin, and is
then attacked (``obscuradp.audit.membership``). What is stored of a run are counts and
summary figures; no single record's score is written down.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

import obscuradp as odp
from benchmarks.phase2.common import BATCH_SIZE, LOCAL_EPOCHS, ROUNDS
from benchmarks.phase3.common import Setting, build_simulation
from obscuradp.audit.membership import AuditRun, audit_run, summarize_run
from obscuradp.mechanisms import dpsgd
from obscuradp.tables import ClientData

DATASETS = ("stroke", "loan")
MODELS = ("network", "logistic")
BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0)
FEW_BUDGETS = (1.0, 8.0)
# Seeds of the pilot (never used for a reported number) and of the reported runs. Both
# differ from every seed of Phases 2 and 3 (42 to 51).
PILOT_SEEDS = tuple(range(90, 95))
SEEDS = tuple(range(100, 130))

BASELINE = "none"
SILENT = "local_silent"
# name: (what it is, the budgets it is run at, the models it is run with)
ARMS: dict[str, tuple[str, tuple[float | None, ...], tuple[str, ...]]] = {
    BASELINE: ("no protection (Phase 2 protocol)", (None,), MODELS),
    "local": ("the default plan: every participant adds all of its own noise", BUDGETS, MODELS),
    SILENT: (
        "planted defect: the default plan with its noise switched off, still stating a budget of 1",
        (1.0,),
        MODELS,
    ),
    "shared": (
        "shared noise, ordinary (Gaussian), one assumed colluder",
        FEW_BUDGETS,
        ("logistic",),
    ),
    "shared_whole": (
        "shared noise, whole numbers (Skellam), one assumed colluder",
        FEW_BUDGETS,
        ("logistic",),
    ),
}
# The arm whose settings an arm runs with.
SETTINGS_OF = {SILENT: "local"}


@contextlib.contextmanager
def noise_switched_off() -> Iterator[None]:
    """Run the private training step without its noise, while the ledger counts it as added.

    This is the defect an audit exists to catch. It is planted here, in the benchmark, and
    nowhere in the library.
    """
    honest = dpsgd.gaussian_noise

    def silent(shape: Any, standard_deviation: float, generator: Any, **kwargs: Any) -> Any:
        return honest(shape, 0.0, generator, **kwargs)

    dpsgd.gaussian_noise = silent  # type: ignore[assignment]
    try:
        yield
    finally:
        dpsgd.gaussian_noise = honest


def audit_setting(setting: Setting, clients: Sequence[ClientData]) -> AuditRun:
    """Audit one run. ``setting.arm`` may be any arm of ``ARMS``."""
    torch.set_num_threads(1)
    real = replace(setting, arm=SETTINGS_OF.get(setting.arm, setting.arm))

    def build(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        simulation = build_simulation(real, list(rows))
        # The audit needs no validation scores and no chosen cut-off.
        return replace(simulation, declared_rows=dict(declared), choose_threshold=False)

    if setting.arm == SILENT:
        with noise_switched_off():
            return audit_run(build, clients, setting.seed)
    return audit_run(build, clients, setting.seed)


def run_setting(setting: Setting, clients: Sequence[ClientData]) -> dict[str, Any]:
    """Audit one run and return what is stored of it."""
    run = audit_setting(setting, clients)
    summary = summarize_run(run).to_dict()
    return {
        **summary,
        "attacker": run.attacker,
        "colluders": list(run.colluders),
        "test_auc": run.result.test.get("auc"),
        "seconds": round(run.result.seconds, 2),
        "rounds": setting.rounds,
        # Kept in full for one seed of every cell, so that it can be recomputed.
        "privacy_report": None if run.result.report is None else run.result.report.to_dict(),
    }


__all__ = [
    "ARMS",
    "BASELINE",
    "BATCH_SIZE",
    "BUDGETS",
    "DATASETS",
    "LOCAL_EPOCHS",
    "MODELS",
    "PILOT_SEEDS",
    "ROUNDS",
    "SEEDS",
    "SETTINGS_OF",
    "SILENT",
    "audit_setting",
    "noise_switched_off",
    "run_setting",
]
