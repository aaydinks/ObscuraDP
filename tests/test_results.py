"""Result records that say where they came from."""

import dataclasses
import json
import math
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import torch

import obscuradp
from obscuradp.results import (
    SCHEMA,
    config_hash,
    environment_manifest,
    git_commit,
    make_result,
    read_result,
    to_jsonable,
    validate_result,
    write_result,
)


def _installed(name: str) -> tuple[str, ...]:
    try:
        return (metadata.version(name),)
    except metadata.PackageNotFoundError:
        return ()


@dataclasses.dataclass(frozen=True)
class _Settings:
    rounds: int = 3
    rates: tuple[float, ...] = (0.1, 0.2)
    where: Path = Path("data") / "x.csv"


def test_to_jsonable_handles_the_usual_suspects() -> None:
    out = to_jsonable(
        {
            "i": np.int64(3),
            "f": np.float32(0.5),
            "b": np.bool_(True),
            "a": np.arange(3),
            "t": torch.tensor([1.5, 2.5]),
            "nan": math.nan,
            "inf": np.float64("inf"),
            "set": frozenset({2}),
            "cfg": _Settings(),
            7: "integer key",
        }
    )
    assert out == {
        "i": 3,
        "f": 0.5,
        "b": True,
        "a": [0, 1, 2],
        "t": [1.5, 2.5],
        "nan": None,
        "inf": None,
        "set": [2],
        "cfg": {"rounds": 3, "rates": [0.1, 0.2], "where": "data/x.csv"},
        "7": "integer key",
    }
    json.dumps(out, allow_nan=False)
    with pytest.raises(TypeError, match="cannot store"):
        to_jsonable({"x": object()})


def test_config_hash_ignores_key_order_but_not_values() -> None:
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert config_hash(_Settings()) == config_hash(_Settings())
    assert config_hash(_Settings()) != config_hash(_Settings(rounds=4))
    assert len(config_hash({})) == 64


def test_manifest_describes_code_and_machine() -> None:
    manifest = environment_manifest()
    assert manifest["packages"]["obscuradp"] == obscuradp.__version__
    assert manifest["packages"]["torch"] == metadata.version("torch")
    assert manifest["packages"]["tenseal"] in (None, *_installed("tenseal"))
    assert manifest["packages"]["numpy"] == np.__version__
    assert manifest["cpu_count"] >= 1
    assert manifest["torch_threads"] >= 1
    assert manifest["python"].count(".") == 2
    assert manifest["created_utc"].endswith("+00:00")


def test_git_commit_outside_a_checkout_is_none(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None


def test_git_commit_inside_a_checkout_marks_local_changes(tmp_path: Path) -> None:
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")

    def run(*args: str) -> None:
        subprocess.run(
            [git, "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    run("init", "-q")
    (tmp_path / "f.txt").write_text("one\n", encoding="utf-8")
    run("add", "f.txt")
    run("commit", "-q", "-m", "first")
    clean = git_commit(tmp_path)
    assert clean is not None
    assert len(clean) == 40
    (tmp_path / "f.txt").write_text("two\n", encoding="utf-8")
    assert git_commit(tmp_path) == f"{clean}+dirty"


def test_round_trip_through_a_file(tmp_path: Path) -> None:
    record = make_result(
        kind="unit_test",
        config=_Settings(),
        seeds=[42, np.int64(43)],
        data={"recall": np.float64(0.5), "curve": np.array([0.1, 0.2])},
        directory=tmp_path,
    )
    assert record["schema"] == SCHEMA
    assert record["manifest"]["seeds"] == [42, 43]
    assert record["manifest"]["git_commit"] is None
    assert record["manifest"]["config_sha256"] == config_hash(_Settings())
    path = tmp_path / "result.json"
    write_result(path, record)
    assert read_result(path) == record
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_tampering_with_the_settings_is_detected(tmp_path: Path) -> None:
    record = make_result(
        kind="unit_test", config={"rounds": 3}, seeds=[1], data={}, directory=tmp_path
    )
    record["config"]["rounds"] = 30
    with pytest.raises(ValueError, match="do not match their fingerprint"):
        validate_result(record)


@pytest.mark.parametrize("missing", ["kind", "manifest", "config", "data"])
def test_incomplete_records_are_refused(missing: str, tmp_path: Path) -> None:
    record = make_result(kind="unit_test", config={}, seeds=[], data={}, directory=tmp_path)
    del record[missing]
    with pytest.raises(ValueError, match="missing"):
        validate_result(record)


def test_wrong_schema_and_empty_kind_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"not a obscuradp\.result/1 record"):
        validate_result({"schema": "something/9"})
    with pytest.raises(ValueError, match="kind"):
        make_result(kind="", config={}, seeds=[], data={})
    record = make_result(kind="k", config={}, seeds=[], data={}, directory=tmp_path)
    del record["manifest"]["seeds"]
    with pytest.raises(ValueError, match="manifest is missing"):
        validate_result(record)
