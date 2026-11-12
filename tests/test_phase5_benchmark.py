"""The Phase 5 benchmark scripts: checked downloads, dataset readers, the runner.

The scripts live in ``benchmarks/`` and are not part of the installed package, so these
tests only run from a full checkout of the repository. No test here uses the network.
"""

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "benchmarks" / "phase5" / "fetch.py").is_file():
    pytest.skip("the benchmark scripts are not part of this installation", allow_module_level=True)

sys.path.insert(0, str(ROOT))

from benchmarks.phase5 import datasets, fetch  # noqa: E402

CONTENT = b"a public file, " * 1000


def _source(content: bytes = CONTENT, name: str = "file.bin") -> fetch.Source:
    return fetch.Source(
        name=name,
        url="https://example.org/file.bin",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        origin="made up for a test",
    )


def test_a_file_is_downloaded_once_checked_and_kept(tmp_path: Path) -> None:
    calls = []

    def opener(url: str) -> io.BytesIO:
        calls.append(url)
        return io.BytesIO(CONTENT)

    path = fetch.fetch(_source(), tmp_path, opener=opener)
    assert path == tmp_path / "file.bin"
    assert path.read_bytes() == CONTENT
    assert fetch.fingerprint(path) == _source().sha256
    # The second time it comes from the cache.
    assert fetch.fetch(_source(), tmp_path, opener=opener) == path
    assert calls == ["https://example.org/file.bin"]
    assert not list(tmp_path.glob("*.part"))


def test_a_file_with_the_wrong_fingerprint_is_never_kept(tmp_path: Path) -> None:
    def tampered(url: str) -> io.BytesIO:
        return io.BytesIO(CONTENT + b"x")

    with pytest.raises(fetch.FingerprintMismatch, match="expected sha256"):
        fetch.fetch(_source(), tmp_path, opener=tampered)
    assert list(tmp_path.iterdir()) == []
    # A damaged file in the cache is replaced, not trusted.
    (tmp_path / "file.bin").write_bytes(b"damaged")
    path = fetch.fetch(_source(), tmp_path, opener=lambda url: io.BytesIO(CONTENT))
    assert path.read_bytes() == CONTENT


def test_only_https_addresses_are_downloaded_and_the_cache_can_be_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = fetch.Source("f", "http://example.org/f", "0" * 64, 1, "made up")
    with pytest.raises(ValueError, match="only https"):
        fetch.fetch(plain, tmp_path)
    monkeypatch.setenv(fetch.CACHE_VARIABLE, str(tmp_path / "elsewhere"))
    assert fetch.cache_directory() == tmp_path / "elsewhere"
    monkeypatch.delenv(fetch.CACHE_VARIABLE)
    assert fetch.cache_directory().name == "obscuradp-benchmarks"
    # The cache is never inside the repository.
    assert ROOT not in fetch.cache_directory().parents


def test_the_recorded_sources_say_where_they_come_from() -> None:
    for source in (datasets.FEMNIST, datasets.SHAKESPEARE):
        assert source.url.startswith("https://storage.googleapis.com/tff-datasets-public/")
        assert len(source.sha256) == 64
        assert source.size > 1_000_000
        assert "public domain" in source.origin


def test_texts_become_symbol_numbers_and_reading_windows() -> None:
    assert len(datasets.SYMBOLS) + 1 == datasets.TEXT_VOCABULARY
    assert len(set(datasets.SYMBOLS)) == len(datasets.SYMBOLS)
    encoded = datasets.encode_text("To be\n~")
    assert encoded.dtype == np.float32
    assert encoded[-1] == 0  # a symbol that is not on the list
    assert all(1 <= value < datasets.TEXT_VOCABULARY for value in encoded[:-1])
    assert datasets.SYMBOLS[int(encoded[0]) - 1] == "T"
    # 100 symbols: windows start at 0, 4, 8, 12, 16 and each is followed by its next symbol.
    symbols = np.arange(100, dtype=np.float32)
    read, following = datasets.windows(symbols)
    assert read.shape == (5, datasets.TEXT_WINDOW)
    assert following.tolist() == [80.0, 84.0, 88.0, 92.0, 96.0]
    assert read[1, 0] == 4.0
    assert read[1, -1] == 83.0
    # Too short a text gives no window; a long one is capped.
    assert datasets.windows(np.zeros(80, dtype=np.float32))[0].shape == (0, 80)
    assert datasets.windows(np.zeros(81, dtype=np.float32))[0].shape == (1, 80)
    assert datasets.windows(np.zeros(50_000, dtype=np.float32))[1].shape == (datasets.TEXT_CAP,)


def _archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:bz2") as handle:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            handle.addfile(info, io.BytesIO(content))


def test_a_made_up_archive_is_read_into_participants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    generator = np.random.default_rng(0)
    contents = {}
    counts = {"train": (30, 1, 12), "test": (4, 3, 2)}  # the second writer is nearly empty
    for part, sizes in counts.items():
        scratch = tmp_path / f"{part}.h5"
        with h5py.File(scratch, "w") as handle:
            for writer, size in zip(("f0001", "f0002", "f0003"), sizes, strict=True):
                group = handle.create_group(f"examples/{writer}")
                group["pixels"] = generator.random((size, 28, 28), dtype=np.float32)
                group["label"] = generator.integers(0, 62, size).astype(np.int32)
        contents[f"fed_emnist_{part}.h5"] = scratch.read_bytes()
    cache = tmp_path / "cache"
    cache.mkdir()
    _archive(cache / "made-up.tar.bz2", contents)
    made_up = fetch.Source(
        "made-up.tar.bz2",
        "https://example.org/x",
        fetch.fingerprint(cache / "made-up.tar.bz2"),
        (cache / "made-up.tar.bz2").stat().st_size,
        "made up",
    )
    monkeypatch.setattr(datasets, "FEMNIST", made_up)
    monkeypatch.setenv(fetch.CACHE_VARIABLE, str(cache))
    writers = datasets.load_femnist()
    # The writer with a single training row is left out; the others keep a tenth for validation.
    assert [client.name for client in writers] == ["writer-0000", "writer-0002"]
    first = writers[0]
    assert (first.n_train, first.validation_y.size, first.test_y.size) == (27, 3, 4)
    assert first.train_x.shape == (27, 784)
    assert writers[1].n_train == 11
    # Ink is 1 and paper 0: the archive has it the other way round.
    with h5py.File(tmp_path / "train.h5", "r") as handle:
        original = handle["examples/f0001/pixels"][()]
    assert np.allclose(first.train_x[0], 1.0 - original[0].reshape(-1))
    assert set(np.unique(first.train_y)) <= set(range(62))
    # Prepared once: a second call reads the number files and needs no archive.
    (cache / "made-up.tar.bz2").unlink()
    assert len(datasets.load_femnist()) == 2


# --------------------------------------------------------------------------- cells and runner

import json  # noqa: E402

from benchmarks.phase5 import audit, common, pilot, plan, report, run, search  # noqa: E402
from benchmarks.phase5 import main as reported_runs  # noqa: E402

import obscuradp as odp  # noqa: E402
from obscuradp.results import make_result, validate_result  # noqa: E402
from obscuradp.tables import ClientData, make_client_data, synthetic_health  # noqa: E402


def test_every_dataset_says_what_its_budget_protects() -> None:
    assert set(common.DATASETS) == {"health", "stroke", "loan", "femnist", "shakespeare"}
    for name in ("health", "stroke", "loan"):
        data = common.DATASETS[name]
        assert (data.unit, data.measure, data.chance, data.per_round) == (
            "record",
            "auc",
            0.5,
            None,
        )
        assert set(data.models) == {"network", "logistic"}
        assert common.arms_of(name) == ("local", "shared", "shared_whole")
    for name, per_round in (("femnist", 200), ("shakespeare", 50)):
        data = common.DATASETS[name]
        assert (data.unit, data.measure, data.per_round) == ("client", "accuracy", per_round)
        assert common.arms_of(name) == ("participant",)
    # The network named for the made-up health table has about 1,300 numbers.
    network = common.DATASETS["health"].models["network"](11)
    assert 1_250 < sum(value.numel() for value in network.parameters()) < 1_350
    with pytest.raises(ValueError, match="unknown dataset"):
        common.load("census", 0)


def test_the_health_table_is_one_table_cut_anew_for_every_seed() -> None:
    first, second = common.load("health", 1), common.load("health", 2)
    assert [c.n_train for c in first] == [c.n_train for c in second]
    assert len(first) == 10
    assert not np.array_equal(first[0].train_x, second[0].train_x)


@pytest.fixture(scope="module")
def small_health() -> list[ClientData]:
    return synthetic_health(5, sites=3, rows=1_500)


@pytest.mark.parametrize("arm", ["none", "local", "shared", "shared_whole"])
def test_a_table_run_gives_a_checked_record(small_health: list[ClientData], arm: str) -> None:
    epsilon = None if arm == "none" else 8.0
    batch = common.FULL_BATCH if arm == "shared_whole" else 160
    setting = common.Setting("health", "logistic", arm, epsilon, 7, 0.2, 1.0, 2, batch, 1.0)
    record = common.run_setting(setting, small_health)
    validate_result(record)
    assert record["kind"] == "phase5_run"
    assert (
        record["data"]["label"]
        == f"health_logistic_{arm}_eps-{'none' if arm == 'none' else '8'}_seed-7"
    )
    assert record["config"]["benchmark"]["arm"] == arm
    assert "auc" in record["data"]["test"]
    stored = record["data"]["privacy_report"]
    if arm == "none":
        assert stored is None
    else:
        assert odp.verify_report(stored) == []
        assert stored["policy"]["unit"] == "record"
    # The pilot's records leave the test rows out.
    blind = common.run_setting(setting, small_health, keep_test=False)
    assert blind["data"]["test"] is None
    assert blind["data"]["validation"] == record["data"]["validation"]
    with pytest.raises(ValueError, match="unknown arm"):
        common.build_simulation(
            common.Setting(**{**setting.__dict__, "arm": "magic"}), small_health
        )


@pytest.fixture(scope="module")
def crowd() -> list[ClientData]:
    """120 made-up writers of 16 by 16 images are not the real dataset; 784 numbers per row are."""
    generator = np.random.default_rng(0)
    writers = []
    for index in range(120):
        labels = generator.integers(0, 62, 12)
        images = generator.random((12, 784), dtype=np.float32)
        writers.append(
            make_client_data(f"writer-{index:04d}", images, labels, seed=index, stratify=False)
        )
    return writers


def test_a_many_participant_run_protects_whole_participants(crowd: list[ClientData]) -> None:
    protected = common.Setting("femnist", "cnn", "participant", 8.0, 3, 0.1, 1.0, 2, 20, 1.0)
    record = common.run_setting(protected, crowd)
    validate_result(record)
    config, stored = record["config"], record["data"]["privacy_report"]
    assert config["crypto_in_loop"] is False  # the secure sum is simulated at this size
    assert config["task"] == {"classes": 62}
    assert config["clients"] == {
        "participants": 120,
        "training_rows": sum(c.n_train for c in crowd),
    }
    assert stored["policy"]["unit"] == "client"
    assert stored["mode"]["max_colluding"] == 12  # a tenth of the participants
    assert any("simulated" in text for text in stored["caveats"])
    assert set(record["data"]["test"]) == {"accuracy", "loss"}
    plain = common.Setting("femnist", "cnn", "none", None, 3, 0.1, 1.0, 2, 20, 1.0)
    baseline = common.run_setting(plain, crowd)
    assert baseline["data"]["privacy_report"] is None
    # 120 participants are fewer than the 200 a round would draw: everybody takes part.
    assert "participation" not in baseline["config"]
    with pytest.raises(ValueError, match="without protection or with"):
        common.build_simulation(common.Setting(**{**plain.__dict__, "arm": "local"}), crowd)


def test_finished_runs_are_kept(
    small_health: list[ClientData], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "load", lambda dataset, seed: small_health)
    settings = [
        common.Setting("health", "logistic", "none", None, seed, 0.2, 1.0, 2, 32, 1.0)
        for seed in (1, 2)
    ]
    assert run.run_all(settings, tmp_path, processes=1) == 2
    files = sorted(tmp_path.glob("*.json"))
    assert [path.stem for path in files] == [setting.label for setting in settings]
    assert not list(tmp_path.glob("*.part"))
    for path in files:
        validate_result(json.loads(path.read_text()))
    stamps = [path.stat().st_mtime_ns for path in files]
    run.run_all(settings, tmp_path, processes=1)
    assert [path.stat().st_mtime_ns for path in files] == stamps


def test_the_pilot_times_every_arm_and_uses_a_seed_of_its_own() -> None:
    settings, full_rounds = pilot.pilot_settings()
    assert len(settings) == 28
    assert {setting.seed for setting in settings} == {pilot.PILOT_SEED}
    for setting in settings:
        if setting.arm == "none":
            assert setting.rounds == full_rounds[setting.label]  # the unprotected run in full
        else:
            assert setting.rounds == pilot.SHORT_ROUNDS < full_rounds[setting.label]
    # Stroke and loan run with the settings of the earlier phases.
    stroke = pilot.planned("stroke", "logistic", "shared", 1.0)
    assert (stroke.rounds, stroke.batch_size, stroke.learning_rate) == (600, 640, 1.0)
    health = pilot.planned("health", "network", "local", 1.0)
    assert (health.rounds, health.batch_size, health.local_epochs) == (30, 32, 1.0)


# --------------------------------------------------------------------------- the plan and the pilot

PLAN = ROOT / "benchmarks" / "PREREGISTRATION.md"
PILOT = ROOT / "benchmarks" / "results" / "phase5"


def test_the_written_plan_names_what_cannot_be_changed_afterwards() -> None:
    text = PLAN.read_text(encoding="utf-8")
    assert chr(0x2014) not in text
    flowing = " ".join(text.split())  # the same words, whatever the line breaks
    for part in (
        "## 4. Seeds",
        "1000, 1001, 1002, 1003, 1004",
        "## 6. What is expected (written down before the runs)",
        "A difference counts only if that interval does not contain zero.",
        "Table runs are never cut.",
        "crypto_in_loop=False",
        "benchmark-plan-v1",
    ):
        assert part in flowing
    # Later changes go below the line "Amendments" and nowhere else.
    assert plan.plan_is_unchanged()
    # The seeds of the plan are new: no earlier phase used them.
    earlier = set(range(42, 52)) | set(range(90, 95)) | set(range(100, 130))
    earlier |= set(range(200, 232)) | set(range(300, 330))
    assert not earlier & {pilot.PILOT_SEED, 901, 902, 903, *range(1000, 1005), *range(1100, 1130)}


def test_the_pilot_stored_times_and_no_score_on_test_rows() -> None:
    if not (PILOT / "pilot.json").is_file():
        pytest.skip("the pilot's results are not present")
    summary = json.loads((PILOT / "pilot.json").read_text(encoding="utf-8"))
    assert summary["seed"] == pilot.PILOT_SEED
    assert len(summary["rows"]) == 28
    for row in summary["rows"]:
        assert row["seconds"] > 0
        assert (row["validation"] is not None) == (row["arm"] == "none")
        assert row["estimated_seconds_full_run"] >= row["seconds"] - 0.1
    files = sorted((PILOT / "pilot").glob("*.json"))
    assert len(files) == 28
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_result(record)
        assert record["manifest"]["seeds"] == [pilot.PILOT_SEED]
        assert record["data"]["test"] is None
        assert record["data"]["test_at_chosen_threshold"] is None
    # The unprotected runs learn: the pipeline works before anything is measured with it.
    learned = {
        (row["dataset"], row["model"]): row["validation"]
        for row in summary["rows"]
        if row["arm"] == "none"
    }
    assert learned[("femnist", "cnn")] > 0.5
    assert learned[("health", "logistic")] > 0.75


# --------------------------------------------------------------------------- the plan's guard


def test_the_plan_above_its_amendments_cannot_change_unnoticed() -> None:
    text = PLAN.read_text(encoding="utf-8")
    fixed = plan.fixed_part(text)
    assert fixed.endswith("## Amendments\n")
    assert plan.fingerprint(text) == plan.PLAN_SHA256
    # One changed seed in the fixed part is noticed; a new amendment is not a change.
    assert (
        plan.fingerprint(text.replace("1000, 1001, 1002", "1000, 1001, 1005")) != plan.PLAN_SHA256
    )
    assert plan.fingerprint(text + "\n### later\n\nAnother amendment.\n") == plan.PLAN_SHA256
    below = text[len(fixed) :]
    assert "### 2026-11-12, before the first run" in below
    assert plan.PLAN_SHA256 in below
    assert plan.PLAN_COMMIT[:7] in below
    with pytest.raises(ValueError, match="no line"):
        plan.fixed_part("a text without that line")


def test_a_run_counts_only_if_its_commit_comes_after_the_plan() -> None:
    assert plan.comes_after_plan(None) is False
    # Uncommitted changes: the record does not name the code that made it.
    assert plan.comes_after_plan(plan.PLAN_COMMIT + "+dirty") is False
    # A commit this copy of the repository does not know cannot be judged.
    assert plan.comes_after_plan("0" * 40) is None
    # The plan's own commit counts where the history is at hand (not in a shallow copy).
    assert plan.comes_after_plan(plan.PLAN_COMMIT) in (True, None)


# --------------------------------------------------------------------------- the search


def test_the_search_is_the_one_the_plan_describes() -> None:
    settings = search.search_settings()
    assert len(settings) == 241
    assert len({search.file_name(setting) for setting in settings}) == 241
    assert {s.dataset for s in settings} == {"health", "femnist"}
    assert {s.seed for s in settings} == {901, 902, 903}
    assert {s.seed for s in settings if s.model != "logistic"} == {901}  # the larger models
    assert {s.epsilon for s in settings if s.dataset == "health"} == {None, 1.0, 8.0}
    assert {s.epsilon for s in settings if s.dataset == "femnist"} == {None, 8.0}

    def tried(dataset: str, model: str, arm: str) -> tuple[set[float], set[float]]:
        runs = [s for s in settings if (s.dataset, s.model, s.arm) == (dataset, model, arm)]
        return {s.learning_rate for s in runs}, {s.clip_norm for s in runs}

    assert tried("health", "logistic", "local") == ({0.05, 0.1, 0.2, 0.5}, {0.5, 1.0, 2.0})
    assert tried("health", "logistic", "shared_whole") == ({0.3, 1.0, 3.0}, {0.5, 1.0, 2.0})
    assert tried("health", "network", "local") == ({0.05, 0.1, 0.2}, {0.5, 1.0})
    assert tried("health", "network", "shared") == ({0.3, 1.0, 3.0}, {0.5, 1.0})
    assert tried("femnist", "cnn", "participant") == ({0.05, 0.1, 0.2}, {0.5, 1.0})
    assert tried("femnist", "cnn", "none") == ({0.05, 0.1, 0.2}, {search.UNUSED_CLIP})
    # Rounds and batches are the fixed ones, not searched.
    shared = next(s for s in settings if s.arm == "shared")
    assert (shared.rounds, shared.batch_size) == (100, 640)
    # The long runs are listed first.
    assert (settings[0].dataset, settings[0].arm) == ("femnist", "participant")


def test_the_highest_mean_is_taken_and_ties_go_to_the_smaller_values() -> None:
    tried = [
        {"learning_rate": 0.1, "clip_norm": 1.0, "values": [0.8, 0.7]},
        {"learning_rate": 0.2, "clip_norm": 0.5, "values": [0.75, 0.75]},
        {"learning_rate": 0.1, "clip_norm": 0.5, "values": [0.75, 0.75]},
        {"learning_rate": 0.05, "clip_norm": 0.5, "values": [0.99, None]},
        {"learning_rate": 0.01, "clip_norm": 0.5, "values": [float("nan"), 0.99]},
        {"learning_rate": 0.02, "clip_norm": 0.5, "values": [0.6, 0.6]},
    ]
    # Three settings tie at 0.75: the smaller learning rate, then the smaller clipping norm.
    assert search.choose(tried) == tried[2]
    assert search.choose(tried[:2]) == tried[0]
    # A run that did not stay finite makes its setting the worst, whatever its other seeds say.
    assert search.mean_or_worst([0.99, None]) == -np.inf
    assert search.mean_or_worst([]) == -np.inf
    assert search.choose(tried[3:]) == tried[5]
    with pytest.raises(ValueError, match="nothing was tried"):
        search.choose([])


def test_a_search_stores_small_records_without_test_scores(
    small_health: list[ClientData], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "load", lambda dataset, seed: small_health)
    cell = ("health", "logistic", "shared_whole")
    monkeypatch.setattr(search, "GRID", {cell: ((0.3, 1.0), (1.0,), (901, 902))})
    monkeypatch.setattr(search, "ORDER", (cell,))
    monkeypatch.setattr(
        search, "PROTOCOL", {"health": {"shared_whole": (2, common.FULL_BATCH, 1.0)}}
    )
    settings = search.search_settings()
    assert len(settings) == 8  # two budgets, two learning rates, two seeds
    run.run_all(
        settings,
        tmp_path,
        processes=1,
        keep_test=False,
        slimmed=True,
        per_task=1,
        name=search.file_name,
    )
    files = sorted(tmp_path.glob("*.json"))
    assert {path.stem for path in files} == {search.file_name(setting) for setting in settings}
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_result(record)
        assert set(record["data"]) == {
            "label",
            "validation",
            "seconds",
            "epsilon_spent",
            "within_budget",
        }
        assert record["data"]["within_budget"] is True
        assert path.stat().st_size < 4_000
    result = search.collect(tmp_path)
    assert set(result["chosen"]) == {
        "health/logistic/shared_whole/1",
        "health/logistic/shared_whole/8",
    }
    assert result["runs"] == 8
    for key, rows in result["tried"].items():
        assert search.choose(rows)["learning_rate"] == result["chosen"][key]["learning_rate"]
        assert all(len(row["values"]) == 2 for row in rows)
    assert result["chosen"]["health/logistic/shared_whole/8"]["rounds"] == 2
    # A record with scores on test rows stops the collection.
    record = json.loads(files[0].read_text(encoding="utf-8"))
    record["data"]["test"] = {"auc": 0.9}
    files[0].write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="scores on test rows"):
        search.collect(tmp_path)


# --------------------------------------------------------------------------- the reported runs


def _made_up_choice() -> dict[str, dict[str, object]]:
    chosen: dict[str, dict[str, object]] = {}
    for (dataset, model, arm), (rates, clips, _) in search.GRID.items():
        rounds, batch, epochs = common.PROTOCOL[dataset][arm]
        for position, epsilon in enumerate(search.budgets_of(dataset, arm)):
            chosen[search.key_of(dataset, model, arm, epsilon)] = {
                "learning_rate": rates[position],
                "clip_norm": clips[0],
                "rounds": rounds,
                "batch_size": batch,
                "local_epochs": epochs,
            }
    return chosen


def test_the_reported_runs_are_the_ones_the_plan_names(tmp_path: Path) -> None:
    chosen = _made_up_choice()
    tables = reported_runs.table_settings(chosen)
    # 3 tables, 2 models, 5 seeds, each with 1 unprotected run and 3 arms at 5 budgets.
    assert len(tables) == 3 * 2 * 5 * (1 + 3 * 5) == 480
    assert {s.seed for s in tables} == {1000, 1001, 1002, 1003, 1004}
    assert {s.epsilon for s in tables} == {None, 0.5, 1.0, 2.0, 4.0, 8.0}
    handwriting = reported_runs.handwriting_settings(chosen)
    assert len(handwriting) == 21
    protected = [s for s in handwriting if s.arm == "participant"]
    assert sorted((s.epsilon, s.seed) for s in protected if s.epsilon == 1.0) == [(1.0, 1000)]
    assert {s.epsilon for s in protected} == {1.0, 2.0, 4.0, 8.0}
    assert len({(s.learning_rate, s.clip_norm) for s in protected}) == 1  # one choice, every budget
    text = reported_runs.text_settings()
    assert len(text) == 6
    assert {s.seed for s in text} == {1000, 1001, 1002}
    assert {(s.learning_rate, s.clip_norm, s.epsilon) for s in text if s.arm == "participant"} == {
        (1.0, 1.0, 8.0)
    }
    everything = reported_runs.reported_settings(chosen)
    assert len({s.label for s in everything}) == len(everything) == 507

    # The health table: budgets up to 2 use what was chosen at 1, the others what was chosen at 8.
    def rate(epsilon: float) -> float:
        return reported_runs.reported(
            "health", "logistic", "local", epsilon, 1000, chosen
        ).learning_rate

    assert rate(0.5) == rate(1.0) == rate(2.0) == 0.05
    assert rate(4.0) == rate(8.0) == 0.1
    # Stroke and loan keep the settings of the earlier phases, whatever the search says.
    stroke = reported_runs.reported("stroke", "logistic", "shared", 1.0, 1003, chosen)
    assert stroke == common.Setting(
        **{**pilot.planned("stroke", "logistic", "shared", 1.0).__dict__, "seed": 1003}
    )
    # The longest runs come first.
    ordered = reported_runs.by_time(tables)
    assert sorted(s.label for s in ordered) == sorted(s.label for s in tables)
    assert ordered[0].model == "network"
    assert ordered[-1].arm == "none"
    # The small real case: a short protected run.
    small = reported_runs.confirm_setting(1001, chosen)
    assert (small.rounds, small.epsilon, small.seed, small.arm) == (20, 8.0, 1001, "participant")
    assert set(reported_runs.CONFIRM_SEEDS) <= set(reported_runs.SEEDS)
    # No reported run without the search's result, and no unknown stage.
    with pytest.raises(FileNotFoundError, match="run the search first"):
        reported_runs.chosen_settings(tmp_path / "settings.json")
    with pytest.raises(SystemExit, match="unknown stage"):
        reported_runs.main(["everything"])


def test_the_small_real_case_runs_the_real_secure_sum(crowd: list[ClientData]) -> None:
    setting = common.Setting("femnist", "cnn", "participant", 8.0, 3, 0.1, 1.0, 2, 20, 1.0)
    few = crowd[:12]
    real = common.run_setting(setting, few, real_secure_sum=True)
    simulated = common.run_setting(setting, few)
    assert "crypto_in_loop" not in real["config"]  # the real secure sum is the usual case
    assert simulated["config"]["crypto_in_loop"] is False
    assert not any("simulated" in text for text in real["data"]["privacy_report"]["caveats"])
    assert any("simulated" in text for text in simulated["data"]["privacy_report"]["caveats"])
    # The same budget either way: simulating changes how the sum is played, not what is promised.
    assert (
        real["data"]["privacy_report"]["epsilon"] == simulated["data"]["privacy_report"]["epsilon"]
    )
    assert real["data"]["communication"] == pytest.approx(
        simulated["data"]["communication"], rel=1e-9, abs=1.0
    )


# --------------------------------------------------------------------------- the attack tests


def test_test_writers_and_their_coins_follow_the_seed() -> None:
    picked, coins = audit.choose_test_units(120, 40, seed=5)
    again, same = audit.choose_test_units(120, 40, seed=5)
    assert np.array_equal(picked, again)
    assert np.array_equal(coins, same)
    assert len(set(picked.tolist())) == 40
    assert picked.min() >= 0
    assert picked.max() < 120
    assert 8 < coins.sum() < 32  # fair coins
    other, _ = audit.choose_test_units(120, 40, seed=6)
    assert not np.array_equal(picked, other)
    with pytest.raises(ValueError, match="at least two test units"):
        audit.choose_test_units(10, 11, seed=0)


def test_the_attack_on_whole_writers_catches_a_run_without_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every participant is a test unit here. The attack treats what the other participants
    # send as noise; in a small group where everybody takes part in every round that
    # would drown it, in a large one with few drawn per round it is part of the setting.
    generator = np.random.default_rng(1)
    group = [
        make_client_data(
            f"writer-{index:04d}",
            generator.random((12, 784), dtype=np.float32),
            generator.integers(0, 62, 12),
            seed=index,
            stratify=False,
        )
        for index in range(150)
    ]
    setting = common.Setting("femnist", "cnn", "participant", 8.0, 4, 0.1, 1.0, 2, 20, 1.0)
    honest = audit.writer_audit(setting, group, units=150, share=0.2, trials=5)
    assert honest["rounds_seen"] == 2
    assert honest["test_units"] == 150
    assert honest["participants_in_the_run"] == honest["kept"]
    assert 55 < honest["kept"] < 95  # fair coins
    assert (honest["guesses"], honest["guess_share"]) == (30, 0.2)
    assert honest["epsilon"] <= 8.0
    assert honest["epsilon_lower_limit"] <= honest["epsilon"]
    assert 0.0 < honest["ceiling_mechanism"] <= honest["ceiling_general"] < 1.0
    # Watching a run does not change it.
    _, coins = audit.choose_test_units(150, 150, setting.seed)
    unwatched = common.build_simulation(
        setting, [client for client, stays in zip(group, coins, strict=True) if stays]
    ).run(seed=setting.seed)
    assert dict(unwatched.validation) == honest["validation"]

    # A planted defect: the same run claims its budget but adds no noise. The attack then
    # says who was in far better than a coin could.
    def no_noise(shape: tuple[int, ...], *_: object, **__: object) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float64)

    # Patched where the benchmark's Simulation looks the function up (another test
    # re-imports the library, so the module of that name may be a newer copy).
    monkeypatch.setitem(common.odp.Simulation.run.__globals__, "gaussian_noise", no_noise)
    broken = audit.writer_audit(setting, group, units=150, share=0.2, trials=5)
    assert broken["epsilon"] == honest["epsilon"]  # the claim is the same
    assert broken["advantage"] > 0.9
    assert broken["right"] == broken["guesses"]
    assert broken["advantage"] > honest["advantage"]
    monkeypatch.undo()
    with pytest.raises(ValueError, match="only where they are protected"):
        audit.writer_audit(common.Setting(**{**setting.__dict__, "arm": "none"}), group, units=9)


def test_the_health_audit_reads_each_class_apart(small_health: list[ClientData]) -> None:
    setting = common.Setting("health", "logistic", "local", 8.0, 9, 0.2, 1.0, 2, 160, 1.0)
    cell = audit.health_cell(setting, small_health)
    assert cell["epsilon"] <= 8.0
    assert cell["ceiling_mechanism"] <= cell["ceiling_general"]
    assert set(cell["attacks"]) == set(audit.ATTACKS)
    for figures in cell["attacks"].values():
        assert set(figures) == set(audit.GROUPS)
        # Every row of the rare class is a test record, beside equally many of the other class.
        rare = sum(int((client.train_y > 0.5).sum()) for client in small_health)
        assert figures["label_1"]["records"] == rare
        assert figures["label_0"]["records"] == rare
        assert figures["all"]["records"] == 2 * rare
        for group in figures.values():
            assert 0 <= group["right"] <= group["guesses"] <= group["records"]
            assert -1.0 <= group["advantage"] <= 1.0


# --------------------------------------------------------------------------- judging and the report


def test_a_difference_counts_only_if_its_interval_leaves_zero_out() -> None:
    first = {seed: 0.80 + 0.01 * seed for seed in range(5)}
    # Always ahead by about 0.05: the interval is narrow and above zero.
    ahead = {seed: value - 0.05 - 0.001 * (seed % 2) for seed, value in first.items()}
    found = report.paired_difference(first, ahead)
    assert found["n"] == 5
    assert found["mean"] == pytest.approx(0.0504)
    assert found["counts"] is True
    assert 0 < found["low"] < found["mean"] < found["high"]
    # Student's t with four degrees of freedom: 2.776 standard errors to either side.
    spread = np.std([0.05, 0.051, 0.05, 0.051, 0.05], ddof=1)
    assert found["high"] - found["mean"] == pytest.approx(2.7764451 * spread / np.sqrt(5))
    # Ahead on average, but not seed by seed: it does not count.
    mixed = {seed: value - (0.09 if seed % 2 else -0.04) for seed, value in first.items()}
    unsure = report.paired_difference(first, mixed)
    assert unsure["mean"] > 0
    assert unsure["counts"] is False
    # Only seeds that both arms ran are compared, and one seed is no comparison.
    assert report.paired_difference(first, {0: 0.7, 1: 0.7, 9: 0.1})["n"] == 2
    with pytest.raises(ValueError, match="at least two seeds"):
        report.paired_difference(first, {0: 0.7})
    spread_of = report.mean_and_spread([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spread_of == {"mean": 3.0, "sd": pytest.approx(np.sqrt(2.5)), "n": 5}


def _made_up_auc(setting: common.Setting) -> float:
    position = setting.seed - 1000
    if setting.dataset == "loan":
        return 0.96 if setting.arm == "none" else 0.95 + 0.001 * position
    if setting.arm == "none":
        return 0.78
    local = 0.70 + 0.002 * position
    if setting.arm == "local":
        return local
    shared = local + 0.06 + 0.001 * (position % 2)
    return shared if setting.arm == "shared" else shared + 0.004 * (-1) ** position


@pytest.fixture(scope="module")
def made_up_results(
    small_health: list[ClientData],
    crowd: list[ClientData],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A results folder with every planned record, made-up scores and two real privacy reports."""
    folder = tmp_path_factory.mktemp("phase5")
    chosen = _made_up_choice()
    (folder / "settings.json").write_text(
        json.dumps({"chosen": chosen, "runs": 241, "computing_seconds": 7200, "commits": []})
    )
    rows = common.Setting(
        "health", "logistic", "shared_whole", 0.5, 7, 0.3, 1.0, 2, common.FULL_BATCH, 1.0
    )
    whole = common.Setting("femnist", "cnn", "participant", 1.0, 3, 0.1, 1.0, 2, 20, 1.0)
    reports = {
        "record": common.run_setting(rows, small_health)["data"]["privacy_report"],
        "client": common.run_setting(whole, crowd[:12])["data"]["privacy_report"],
    }

    def write(path: Path, kind: str, config: dict[str, object], seed: int, data: object) -> None:
        record = make_result(kind=kind, config=config, seeds=[seed], data=data)  # type: ignore[arg-type]
        record["manifest"]["git_commit"] = plan.PLAN_COMMIT
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    def run_data(setting: common.Setting, score: float) -> dict[str, object]:
        unit = common.DATASETS[setting.dataset].unit
        test = {"auc": score, "ap": 0.3} if unit == "record" else {"accuracy": score, "loss": 1.0}
        return {
            "label": setting.label,
            "test": test,
            "test_at_chosen_threshold": {"recall": 0.5, "precision": 0.25, "f1": 1 / 3},
            "validation": test,
            "history": [{"round": setting.rounds}],
            "communication": {"upload_bytes_per_round": 26.0, "plain_bytes_per_round": 48.0},
            "privacy_report": None if setting.epsilon is None else reports[unit],
            "seconds": 36.0,
        }

    for setting in reported_runs.reported_settings(chosen):
        if common.DATASETS[setting.dataset].unit == "record":
            score = _made_up_auc(setting)
        else:
            score = 0.7 if setting.epsilon is None else 0.05 + 0.01 * setting.epsilon
        write(
            folder / "runs" / f"{setting.label}.json",
            "phase5_run",
            {"benchmark": setting.__dict__},
            setting.seed,
            run_data(setting, score),
        )
    for position, seed in enumerate(reported_runs.CONFIRM_SEEDS):
        setting = reported_runs.confirm_setting(seed, chosen)
        for name, shift in (("real", 0.0), ("simulated", (0.002, -0.002, 0.001)[position])):
            config: dict[str, object] = {"benchmark": setting.__dict__}
            if name == "simulated":
                config["crypto_in_loop"] = False
            write(
                folder / "confirm" / f"{setting.label}_{name}.json",
                "phase5_run",
                config,
                seed,
                run_data(setting, 0.05 + 0.001 * position + shift),
            )
    for model in audit.HEALTH_MODELS:
        for seed in audit.HEALTH_SEEDS:
            cells = {}
            for epsilon in audit.HEALTH_BUDGETS:
                group = {"records": 100, "guesses": 4, "right": 3, "advantage": 0.01 * epsilon}
                cells[f"{epsilon:g}"] = {
                    "epsilon": 0.99 * epsilon,
                    "ceiling_mechanism": 0.1 * epsilon,
                    "ceiling_general": 0.12 * epsilon,
                    "attacks": {
                        attack: {
                            "all": {**group, "records": 200, "guesses": 10, "right": 6},
                            "label_1": {**group, "advantage": 0.02 * epsilon},
                            "label_0": {**group, "advantage": 0.0},
                        }
                        for attack in audit.ATTACKS
                    },
                }
            write(
                folder / "audit" / f"health_{model}_seed-{seed}.json",
                "phase5_audit_health",
                {"model": model},
                seed,
                cells,
            )
    return folder


def test_the_report_judges_made_up_results_by_the_written_rule(made_up_results: Path) -> None:
    summary = report.build_summary(made_up_results)
    assert summary["failed_checks"] == []
    assert (summary["runs_planned"], summary["runs_found"]) == (507, 507)
    assert summary["computing_hours"] == pytest.approx((507 * 36 + 7200) / 3600, abs=0.05)
    assert summary["commits"]["plan_unchanged"] is True
    assert summary["commits"]["all_after_plan"] in (True, None)  # None without the history
    # Cells: the mean of five seeds with the sample standard deviation.
    cell = next(
        row
        for row in summary["cells"]
        if (row["dataset"], row["model"], row["arm"], row["epsilon"])
        == ("health", "network", "local", 1.0)
    )
    assert cell["headline"] == {
        "mean": pytest.approx(0.704),
        "sd": pytest.approx(np.std([0.70, 0.702, 0.704, 0.706, 0.708], ddof=1)),
        "n": 5,
    }
    assert cell["recall"]["mean"] == 0.5
    assert cell["epsilon_spent"] <= 0.5
    # 3 tables, 2 models, 5 budgets: 3 pairs of arms and 3 comparisons with no protection.
    assert len(summary["comparisons"]) == 3 * 2 * 5 * 6
    found = summary["expectations"]
    assert {key: value["of"] for key, value in found.items()} == {"1": 32, "2": 30, "3": 12}
    assert found["1"]["holds"] == 32
    assert found["1"]["confirmed"] is True
    assert found["2"]["confirmed"] is True
    assert found["2"]["largest_gap"] == pytest.approx(0.0, abs=1e-12)
    # Shared noise at a budget of 8 is 0.016 to 0.02 below no protection: within 0.03.
    assert found["3"]["confirmed"] is True
    assert 0.015 < found["3"]["largest_gap"] < 0.021
    # The small real case: the two secure sums differ by less than the seeds scatter.
    assert summary["confirm"]["agree"] is True
    assert summary["confirm"]["difference"]["n"] == 3
    # The attack tests: pooled over 30 runs, each class apart.
    rare = summary["health_audit"][0]["label_1"]
    assert (rare["records"], rare["guesses"], rare["right"]) == (3000, 120, 90)
    assert rare["advantage"]["mean"] == pytest.approx(0.02)
    assert rare["epsilon_lower_limit"] > 0
    assert summary["writer_audit"] is None
    text = report.headline_markdown(summary)
    assert "| 1. Health and stroke" in text
    assert text.count("**confirmed**") == 3
    assert "0.704 ± 0.003" in text
    assert "**counts**" in text
    assert "does not count" in text  # the two shared arms against each other
    assert "The two agree by the plan's rule." in text
    assert text.rstrip().endswith("Not done.")
    assert chr(0x2014) not in text


def test_the_report_names_what_is_wrong_with_stored_results(
    made_up_results: Path, tmp_path: Path
) -> None:
    import shutil

    folder = tmp_path / "copy"
    shutil.copytree(made_up_results, folder)
    chosen = _made_up_choice()
    runs = folder / "runs"

    def change(label: str, edit: object) -> None:
        path = runs / f"{label}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        edit(record)  # type: ignore[operator]
        path.write_text(json.dumps(record), encoding="utf-8")

    # One seed in which local noise is far ahead: expectation 1 no longer holds everywhere.
    behind = reported_runs.reported("stroke", "network", "shared", 2.0, 1002, chosen)
    change(behind.label, lambda record: record["data"]["test"].update(auc=0.2))
    # A run made with uncommitted changes, a run over its budget, a missing run.
    dirty = reported_runs.reported("loan", "logistic", "local", 4.0, 1001, chosen)
    change(dirty.label, lambda record: record["manifest"].update(git_commit="abc+dirty"))
    over = reported_runs.reported("health", "logistic", "local", 0.5, 1000, chosen)
    change(over.label, lambda record: record["data"]["privacy_report"].update(epsilon=0.6))
    missing = reported_runs.reported("femnist", "cnn", "participant", 4.0, 1004, chosen)
    (runs / f"{missing.label}.json").unlink()
    # A handwriting audit that was done.
    writers = {
        "test_units": 400,
        "kept": 199,
        "guesses": 20,
        "right": 13,
        "advantage": 0.04,
        "epsilon": 7.9,
        "epsilon_lower_limit": 0.0,
        "most_the_test_could_show": 1.2,
        "ceiling_mechanism": 0.9,
        "ceiling_general": 0.99,
    }
    record = make_result(kind="phase5_audit_writers", config={}, seeds=[1100], data=writers)
    record["manifest"]["git_commit"] = plan.PLAN_COMMIT
    (folder / "audit" / f"femnist_writers_seed-{audit.WRITER_SEED}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    summary = report.build_summary(folder)
    failed = "\n".join(summary["failed_checks"])
    assert f"{missing.label}.json: missing" in failed
    assert f"{over.label}: over its budget" in failed
    assert "abc+dirty, which does not come after the plan" in failed
    # A changed figure in a stored privacy report is also caught by recomputing the report.
    assert f"{over.label}: the stored privacy report does not recompute" in failed
    assert len(summary["failed_checks"]) == 4
    assert summary["runs_found"] == 506
    found = summary["expectations"]["1"]
    assert (found["holds"], found["of"], found["confirmed"]) == (31, 32, False)
    assert summary["writer_audit"]["right"] == 13
    text = report.headline_markdown(summary)
    assert "**not confirmed**" in text
    assert "holds in 31 of 32 comparisons" in text
    assert "| Guesses made, of them right | 20, 13 |" in text
    # Without the search's result there is nothing to check the runs against.
    assert report.build_summary(tmp_path)["failed_checks"] == [
        "settings.json: missing (the search has not been run)"
    ]
