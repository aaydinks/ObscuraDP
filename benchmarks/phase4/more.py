"""Three further attacks on the default plan (step S16), and one group of rows looked at apart.

    python -m benchmarks.phase4.more            # everything (resumes: finished parts are kept)
    python -m benchmarks.phase4.more --quick    # two rounds, few runs, to check the pipeline

What is run, always on the real tables with the settings of Phases 2 and 3:

- **Shadow models.** For each cell (table, model, and no protection or the default plan at
  a budget of 1 or 8) the pipeline is run 32 times, each time on a random half of every
  participant's rows. Every run is then attacked with the others as its shadows
  (``obscuradp.audit.shadow``). For a lower limit on the budget, runs 0 to 15 are attacked
  with runs 16 to 31 as shadows only, so that their guesses may be pooled.
- **A hidden column.** The same runs give models with known used and unused rows. For
  each, an attacker who lacks one yes/no column of a record tries to guess it with the
  model, and is compared with an attacker who has no model (``obscuradp.audit.attribute``).
- **Reading a record out of an update** (``obscuradp.audit.inversion``): one record alone
  in an update, as each plan shows it to its attacker.
- **The rare rows apart.** On the stroke table the whole-run audit of step S15 is repeated
  with every row of the rare class as a test record, beside equally many rows of the
  common class, and read by class.

What is stored are counts and summary figures, never a single record's score.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.phase2.tables import SCHEMAS, make_clients
from benchmarks.phase3.common import Setting
from benchmarks.phase4.common import DATASETS, MODELS, audit_setting, builder
from benchmarks.phase4.run import settings_for
from obscuradp.audit import attribute, inversion, shadow
from obscuradp.audit.membership import (
    ATTACK_FINAL_MODEL,
    ATTACK_UPDATES,
    count_guesses,
    measured_advantage,
)
from obscuradp.mechanisms.dpsgd import clipped_sum, per_record_gradients, private_gradient
from obscuradp.results import make_result
from obscuradp.seeding import numpy_generator, torch_generator
from obscuradp.tables import ClientData

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "benchmarks" / "results" / "phase4" / "more"

SPLIT_SEED = 42  # one split of each table for all shadow runs (the first seed of Phase 2)
SHADOW_SEEDS = tuple(range(200, 232))
# For a pooled lower limit: these runs are attacked, the others are only learned from.
POOLED_TARGETS = tuple(range(16))
GUESS_SHARE = 0.05  # as fixed by the pilot of step S15, before any run of this step
RARE_SEEDS = tuple(range(300, 330))
# (arm, total budget): no protection, and the default plan at two budgets.
PROTECTION: tuple[tuple[str, float | None], ...] = (("none", None), ("local", 1.0), ("local", 8.0))
HIDDEN_COLUMNS = {
    "stroke": ("hypertension", "heart_disease", "ever_married"),
    "loan": ("self_employed", "education"),
}
INVERSION_RECORDS = 200
INVERSION_BATCH = 32


def _setting(
    dataset: str, model: str, arm: str, epsilon: float | None, seed: int, quick: bool
) -> Setting:
    for setting in settings_for(dataset, seed, quick=quick):
        if (setting.model, setting.arm, setting.epsilon) == (model, arm, epsilon):
            return setting
    raise ValueError("no such cell")


def cell_name(dataset: str, model: str, arm: str, epsilon: float | None) -> str:
    return f"{dataset}_{model}_{arm}_eps-{'none' if epsilon is None else f'{epsilon:g}'}"


def _write(
    path: Path, kind: str, config: dict[str, Any], seeds: list[int], data: dict[str, Any]
) -> None:
    record = make_result(kind=kind, config=config, seeds=seeds, data=data)
    scratch = path.with_suffix(".part")
    scratch.write_text(
        json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    )
    scratch.replace(path)


# ---------------------------------------------------------------------- shadow models


def _hidden_column_figures(
    dataset: str, clients: list[ClientData], setting: Setting, state: Any, kept: np.ndarray
) -> dict[str, dict[str, float]]:
    """How well a hidden column is guessed with this run's model, for used and unused rows."""
    schema = SCHEMAS[dataset]
    features = np.concatenate([client.train_x for client in clients])
    labels = np.concatenate([client.train_y for client in clients])
    known_x = np.concatenate([client.validation_x for client in clients])
    simulation = builder(setting)(clients, {client.name: client.n_train for client in clients})
    judge = simulation.model_fn()
    from_logits = not bool(getattr(judge, "probability_output", True))
    figures = {}
    for name in HIDDEN_COLUMNS[dataset]:
        hidden = attribute.attribute_from_schema(schema, name)
        truth = hidden.read(features)
        likelihood = attribute.outcome_log_likelihood(
            judge, state, features, labels, hidden, from_logits=from_logits
        )
        # What the attacker believes before looking at the model: learned without outcomes
        # from rows of the same population that it holds itself.
        prior = attribute.fit_imputer(hidden, known_x).log_odds(features)
        with_model = attribute.infer_with_model(likelihood, prior)
        model_alone = attribute.infer_with_model(likelihood)
        count = len(hidden.values)
        figures[name] = {
            "used": attribute.balanced_accuracy(truth[kept], with_model[kept], count),
            "unused": attribute.balanced_accuracy(truth[~kept], with_model[~kept], count),
            "model_alone_used": attribute.balanced_accuracy(truth[kept], model_alone[kept], count),
            "model_alone_unused": attribute.balanced_accuracy(
                truth[~kept], model_alone[~kept], count
            ),
        }
    return figures


def _without_model(dataset: str, clients: list[ClientData]) -> dict[str, dict[str, float]]:
    """The attacker without any model: the hidden column guessed from the others and the outcome."""
    features = np.concatenate([client.train_x for client in clients])
    labels = np.concatenate([client.train_y for client in clients])
    known_x = np.concatenate([client.validation_x for client in clients])
    known_y = np.concatenate([client.validation_y for client in clients])
    figures = {}
    for name in HIDDEN_COLUMNS[dataset]:
        hidden = attribute.attribute_from_schema(SCHEMAS[dataset], name)
        truth = hidden.read(features)
        guess = attribute.fit_imputer(hidden, known_x, known_y).guess(features, labels)
        blind = attribute.fit_imputer(hidden, known_x).guess(features)
        share = float(np.mean(truth == 1))
        figures[name] = {
            "with_outcome": attribute.balanced_accuracy(truth, guess, len(hidden.values)),
            "without_outcome": attribute.balanced_accuracy(truth, blind, len(hidden.values)),
            "share_of_second_value": share,
            "second_value": hidden.labels[1],
        }
    return figures


def _advantages(scores: np.ndarray, kept: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rare = labels > 0.5
    return {
        "all": measured_advantage(scores, kept),
        "label_1": measured_advantage(scores[rare], kept[rare]),
        "label_0": measured_advantage(scores[~rare], kept[~rare]),
    }


def shadow_cell(task: tuple[str, str, str, float | None, bool, str, str]) -> str:
    """All shadow runs of one cell, their attacks, and the hidden-column guesses."""
    dataset, model, arm, epsilon, quick, directory, cache = task
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    name = cell_name(dataset, model, arm, epsilon)
    path = Path(directory) / f"shadow_{name}.json"
    if path.exists():
        return name
    seeds = SHADOW_SEEDS[:8] if quick else SHADOW_SEEDS
    targets = POOLED_TARGETS[:4] if quick else POOLED_TARGETS
    clients = make_clients(dataset, SPLIT_SEED)
    setting = _setting(dataset, model, arm, epsilon, SPLIT_SEED, quick)
    build = builder(setting)
    confidence, kept, runs = [], [], []
    labels = np.concatenate([client.train_y for client in clients]).astype(np.float64)
    for seed in seeds:
        saved = Path(cache) / f"{name}_{'quick_' if quick else ''}seed-{seed}.npz"
        if saved.exists():
            with np.load(saved, allow_pickle=False) as data:
                confidence.append(data["confidence"])
                kept.append(data["kept"])
                runs.append(json.loads(str(data["figures"])))
            continue
        half = shadow.random_half_run(build, clients, seed)
        report = half.result.report
        figures = {
            "seed": seed,
            "epsilon": None if report is None else report.epsilon,
            "delta": None if report is None else report.policy.delta,
            "ceiling_mechanism": None if report is None else report.advantage_ceiling,
            "seconds": round(half.result.seconds, 2),
            "hidden_columns": _hidden_column_figures(
                dataset, clients, setting, half.result.state, half.kept
            ),
            # The plain look at the fit, for comparison: one cut-off for all records.
            "plain_advantage": _advantages(half.confidence, half.kept, labels),
        }
        np.savez(saved, confidence=half.confidence, kept=half.kept, figures=json.dumps(figures))
        confidence.append(half.confidence)
        kept.append(half.kept)
        runs.append(figures)
    values, inside = np.stack(confidence), np.stack(kept)
    # Every run attacked in turn, with all the others as its shadows.
    for run, scores in shadow.attack_each_run(values, inside).items():
        runs[run]["shadow_advantage"] = _advantages(scores, inside[run], labels)
    # The first runs attacked with only the last ones as shadows: guesses that may be pooled.
    others = [run for run in range(len(seeds)) if run not in targets]
    pooled = shadow.attack_each_run(values, inside, targets=targets, shadows=others)
    for run, scores in pooled.items():
        runs[run]["pooled"] = {
            "canaries": int(scores.size),
            "guesses": list(count_guesses(scores, inside[run], GUESS_SHARE)),
            "advantage": measured_advantage(scores, inside[run]),
        }
    _write(
        path,
        "phase4_shadow_cell",
        {"dataset": dataset, "model": model, "arm": arm, "epsilon": epsilon, "quick": quick},
        list(seeds),
        {
            "runs": runs,
            "rows": int(labels.size),
            "rows_label_1": int((labels > 0.5).sum()),
            "without_model": _without_model(dataset, clients),
        },
    )
    return name


# ---------------------------------------------------------------------- the rare rows apart


def _rare_and_as_many_common(clients: list[ClientData], seed: int) -> dict[str, np.ndarray]:
    """Per participant: every row of the rare class and equally many rows of the other."""
    eligible = {}
    for index, client in enumerate(clients):
        rare = np.flatnonzero(client.train_y > 0.5)
        common = np.flatnonzero(client.train_y <= 0.5)
        count = min(rare.size, common.size)
        if count == 0:
            continue
        picked = numpy_generator(seed, "canary", 1000 + index).choice(common, count, replace=False)
        eligible[client.name] = np.concatenate([rare, picked])
    return eligible


def rare_seed(task: tuple[int, bool, str]) -> str:
    """The stroke audit with the rare rows as test records, for one seed."""
    seed, quick, directory = task
    warnings.simplefilter("ignore")
    path = Path(directory) / f"rare_stroke_seed-{seed}.json"
    if path.exists():
        return path.stem
    clients = make_clients("stroke", seed)
    eligible = _rare_and_as_many_common(clients, seed)
    cells = {}
    for model in MODELS:
        for epsilon in (1.0, 8.0):
            run = audit_setting(
                _setting("stroke", model, "local", epsilon, seed, quick), clients, eligible
            )
            rare = run.labels > 0.5
            report = run.result.report
            assert report is not None
            cells[cell_name("stroke", model, "local", epsilon)] = {
                "canaries": int(run.kept.size),
                "canaries_label_1": int(rare.sum()),
                "epsilon": report.epsilon,
                "ceiling_mechanism": report.advantage_ceiling,
                "advantage": {
                    attack: _advantages(run.scores[attack], run.kept, run.labels)
                    for attack in (ATTACK_UPDATES, ATTACK_FINAL_MODEL)
                },
            }
    _write(
        path, "phase4_rare_seed", {"dataset": "stroke", "quick": quick}, [seed], {"cells": cells}
    )
    return path.stem


# ---------------------------------------------------------------------- reading a record back


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy(output, target, reduction="none")


def inversion_part(directory: str, quick: bool) -> str:
    """One record alone in an update, as each plan shows it, and what can be read back."""
    path = Path(directory) / "inversion.json"
    if path.exists():
        return path.stem
    warnings.simplefilter("ignore")
    torch.set_num_threads(1)
    cells = {}
    count = 20 if quick else INVERSION_RECORDS
    for dataset in DATASETS:
        clients = make_clients(dataset, SPLIT_SEED)
        largest = max(clients, key=lambda client: client.n_train)
        features = torch.as_tensor(largest.train_x)
        targets = torch.as_tensor(largest.train_y)
        mean_row = features.double().mean(dim=0).numpy()
        for model_name in MODELS:
            sigmas = {}
            for epsilon in (1.0, 8.0):
                setting = _setting(dataset, model_name, "local", epsilon, SPLIT_SEED, quick)
                simulation = builder(setting)(clients, {c.name: c.n_train for c in clients})
                sigmas[epsilon] = simulation.calibrations()[largest.name].noise_multiplier
                clip = setting.clip_norm
            torch.manual_seed(0)
            model = simulation.model_fn()
            noise = torch_generator(0, "noise", 0)
            order = numpy_generator(0, "sampling", 0).permutation(largest.n_train)
            errors: dict[str, list[float]] = {
                "alone, no protection": [],
                "alone, default plan, budget 8": [],
                "alone, default plan, budget 1": [],
                f"one of {INVERSION_BATCH}, no protection": [],
                "no update at all (the average record)": [],
            }
            for position in order[:count]:
                x, y = features[position : position + 1], targets[position : position + 1]
                truth = x[0].numpy()
                alone = clipped_sum(per_record_gradients(model, _loss, x, y), 1e9)
                errors["alone, no protection"].append(
                    inversion.reconstruction_error(truth, inversion.reconstruct_input(alone, model))
                )
                for epsilon in (8.0, 1.0):
                    noisy = private_gradient(
                        model,
                        _loss,
                        x,
                        y,
                        clip_norm=clip,
                        noise_multiplier=sigmas[epsilon],
                        expected_batch_size=1.0,
                        generator=noise,
                    )
                    errors[f"alone, default plan, budget {epsilon:g}"].append(
                        inversion.reconstruction_error(
                            truth, inversion.reconstruct_input(noisy, model)
                        )
                    )
                others = numpy_generator(0, "sampling", 1, int(position)).choice(
                    largest.n_train, INVERSION_BATCH - 1, replace=False
                )
                rows = np.concatenate([[position], others])
                batch = clipped_sum(
                    per_record_gradients(model, _loss, features[rows], targets[rows]), 1e9
                )
                errors[f"one of {INVERSION_BATCH}, no protection"].append(
                    inversion.reconstruction_error(truth, inversion.reconstruct_input(batch, model))
                )
                errors["no update at all (the average record)"].append(
                    inversion.reconstruction_error(truth, mean_row)
                )
            cells[f"{dataset}_{model_name}"] = {
                "records": count,
                "noise_multiplier": {f"{eps:g}": sigma for eps, sigma in sigmas.items()},
                "median_error": {key: float(np.median(value)) for key, value in errors.items()},
                "share_within_10_percent": {
                    key: float(np.mean(np.asarray(value) < 0.1)) for key, value in errors.items()
                },
            }
    _write(path, "phase4_inversion", {"quick": quick}, [SPLIT_SEED], {"cells": cells})
    return path.stem


# ---------------------------------------------------------------------- everything


def _dispatch(task: tuple[str, Any]) -> str:
    kind, arguments = task
    if kind == "shadow":
        return shadow_cell(arguments)
    if kind == "rare":
        return rare_seed(arguments)
    return inversion_part(*arguments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="two rounds and few runs only")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument(
        "--cache",
        default=str(Path(tempfile.gettempdir()) / "obscuradp-phase4-more"),
        help="where unfinished cells keep their runs (outside the repository)",
    )
    args = parser.parse_args()
    directory = RESULTS / "quick" if args.quick else RESULTS
    directory.mkdir(parents=True, exist_ok=True)
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Any]] = [("inversion", (str(directory), args.quick))]
    # The longest cells first.
    for model in MODELS:
        for dataset in DATASETS:
            for arm, epsilon in reversed(PROTECTION):
                if args.quick and epsilon == 8.0:
                    continue
                tasks.append(
                    (
                        "shadow",
                        (dataset, model, arm, epsilon, args.quick, str(directory), args.cache),
                    )
                )
    for seed in RARE_SEEDS[:1] if args.quick else RARE_SEEDS:
        tasks.append(("rare", (seed, args.quick, str(directory))))
    with multiprocessing.Pool(args.processes) as pool:
        for name in pool.imap_unordered(_dispatch, tasks, chunksize=1):
            print(f"finished {name}", flush=True)


if __name__ == "__main__":
    main()
