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

from benchmarks.phase5 import common, pilot, run  # noqa: E402

import obscuradp as odp  # noqa: E402
from obscuradp.results import validate_result  # noqa: E402
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
    # Nothing has been amended yet; later changes go below this line and nowhere else.
    assert text.rstrip().endswith("## Amendments\n\nNone.")
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
