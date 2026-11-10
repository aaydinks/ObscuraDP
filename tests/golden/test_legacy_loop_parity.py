"""The whole training loop: the copy gives the research code's results.

Each case runs the research code and the copy for 3 rounds of 1 local epoch on the same
seed. Without encryption every number must be equal bit for bit. With encryption the
structure must be equal exactly and the numbers within small stated tolerances, because
encryption randomness cannot be seeded and the decrypted average feeds the next round.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest

from obscuradp.legacy import LegacyTrialConfig
from obscuradp.legacy import data as new_data
from obscuradp.legacy import loop as new_loop

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore")]

ROUNDS, LOCAL_EPOCHS, SEED = 3, 1, 42
TIMING_KEYS = {"encrypt_sec", "aggregate_sec", "decrypt_sec", "crypto_sec"}
EXACT_CRYPTO_KEYS = ("plaintext_bytes", "n_chunks", "n_encrypted", "encrypted_fraction")
SETTING_KEYS = (
    "dataset",
    "stack",
    "algo",
    "seed",
    "epsilon",
    "spe_fraction",
    "mask_mode",
    "rounds",
    "local_epochs",
    "width_scale",
    "n_params",
    "use_smote",
    "mask",
)
ENCRYPTED_METRIC_TOLERANCE = 1e-3
ENCRYPTED_WEIGHT_TOLERANCE = 1e-4


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def prepared(oracle: Any, cfg: LegacyTrialConfig) -> dict[str, tuple[Any, Any]]:
    """Each dataset prepared once by each code (already shown to be identical)."""
    out = {}
    for dataset in ("stroke", "loan"):
        old = oracle.call(oracle.module("orchestrator").prepare, dataset, seed=SEED)
        new = _quiet(new_data.prepare, cfg, dataset, seed=SEED)
        out[dataset] = (old, new)
    return out


def _run_both(
    oracle: Any,
    cfg: LegacyTrialConfig,
    prepared: Any,
    dataset: str,
    stack: str,
    algo: str,
    **extra: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_prep, new_prep = prepared[dataset]
    kwargs = {
        "dataset": dataset,
        "stack": stack,
        "algo": algo,
        "seed": SEED,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "log": None,
        **extra,
    }
    old = oracle.call(oracle.module("orchestrator").run_stack, prepared=old_prep, **kwargs)
    new = _quiet(new_loop.run_stack, cfg, prepared=new_prep, **kwargs)
    return old, new


def _without_timing(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in result.items() if k != "timing"}
    out["crypto"] = {k: v for k, v in result["crypto"].items() if k not in TIMING_KEYS}
    return out


def test_prepared_data_has_the_documented_shapes(prepared: Any) -> None:
    """The first research smoke check: sizes, and every training row is handed out once."""
    expected = {"stroke": (4908, 15), "loan": (4269, 11)}
    for dataset, (rows, features) in expected.items():
        _, new = prepared[dataset]
        assert new["in_features"] == features
        assert new["X_tr"].shape[1] == new["X_te"].shape[1] == features
        full_test_share = 0.2 * rows
        assert len(new["y_tr"]) == rows - round(full_test_share)
        handed_out = sorted(int(i) for part in new["parts"] for i in part)
        assert handed_out == list(range(len(new["y_tr"])))
        assert len(new["client_data"]) == 5


# --------------------------------------------------------------------------- no encryption


@pytest.mark.parametrize("dataset", ["stroke", "loan"])
@pytest.mark.parametrize("stack", ["plain", "dp"])
@pytest.mark.parametrize("algo", ["fedavg", "fedprox"])
def test_unencrypted_stacks_are_identical(
    oracle: Any, cfg: LegacyTrialConfig, prepared: Any, dataset: str, stack: str, algo: str
) -> None:
    old, new = _run_both(oracle, cfg, prepared, dataset, stack, algo)
    assert set(old) == set(new)
    assert set(old["timing"]) == set(new["timing"])
    assert _without_timing(old) == _without_timing(new)

    # Sanity of what was compared: three rounds were recorded and training moved the model.
    assert len(new["curve"]) == ROUNDS
    assert [row["round"] for row in new["curve"]] == [1, 2, 3]
    assert new["n_params"] == (3137 if dataset == "stroke" else 929)
    if stack == "dp":
        assert new["epsilon"] == 9.0
        assert len(new["dp"]["per_client"]["sigma"]) == 5
        # Finding G2: the labelled budget is per round, so the run spends more.
        assert new["dp"]["eps_cumulative"] > new["dp"]["eps_round"]
    else:
        assert new["dp"]["sigma"] is None
    # The plain average differs from its float64 reference only by float32 rounding.
    assert new["lossless"]["max_abs_dev"] == new["lossless"]["float32_floor"]
    assert new["crypto"]["wire_payload_ratio"] == 1.0


def test_other_budgets_and_seeds_are_identical(
    oracle: Any, cfg: LegacyTrialConfig, prepared: Any
) -> None:
    for epsilon in (0.5, 50.0):
        old, new = _run_both(oracle, cfg, prepared, "loan", "dp", "fedprox", epsilon=epsilon)
        assert _without_timing(old) == _without_timing(new)
    # A different seed with data prepared inside the loop (no shared bundle).
    kwargs = {
        "dataset": "loan",
        "stack": "plain",
        "algo": "fedavg",
        "seed": 44,
        "rounds": 2,
        "local_epochs": 1,
        "log": None,
    }
    old = oracle.call(oracle.module("orchestrator").run_stack, **kwargs)
    new = _quiet(new_loop.run_stack, cfg, **kwargs)
    assert _without_timing(old) == _without_timing(new)


def test_wider_model_is_identical(oracle: Any, cfg: LegacyTrialConfig, prepared: Any) -> None:
    old, new = _run_both(oracle, cfg, prepared, "loan", "plain", "fedavg", width_scale=2)
    assert _without_timing(old) == _without_timing(new)
    assert new["width_scale"] == 2


# --------------------------------------------------------------------------- with encryption


def _assert_encrypted_run_matches(old: dict[str, Any], new: dict[str, Any]) -> None:
    assert set(old) == set(new)
    for key in SETTING_KEYS:
        assert old[key] == new[key], key
    # Privacy numbers are computed from sizes and settings, not from the encrypted values.
    assert old["dp"] == new["dp"]
    for key in EXACT_CRYPTO_KEYS:
        assert old["crypto"][key] == new["crypto"][key], key
    assert abs(old["crypto"]["ciphertext_bytes"] - new["crypto"]["ciphertext_bytes"]) <= (
        0.001 * old["crypto"]["ciphertext_bytes"]
    )
    for old_row, new_row in zip(old["curve"], new["curve"], strict=True):
        assert set(old_row) == set(new_row)
        for name, value in old_row.items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(new_row[name])
            else:
                assert new_row[name] == pytest.approx(value, abs=ENCRYPTED_METRIC_TOLERANCE), name
    # The encrypted average stays within the research code's own tolerance of the plain one.
    for result in (old, new):
        assert result["lossless"]["max_abs_dev"] < ENCRYPTED_WEIGHT_TOLERANCE
        assert result["lossless"]["max_abs_dev"] >= result["lossless"]["float32_floor"]


@pytest.mark.parametrize("dataset", ["stroke", "loan"])
@pytest.mark.parametrize("stack", ["he", "dp_he", "dp_spe"])
def test_encrypted_stacks_match_within_tolerance(
    oracle: Any, cfg: LegacyTrialConfig, prepared: Any, dataset: str, stack: str
) -> None:
    old, new = _run_both(oracle, cfg, prepared, dataset, stack, "fedprox")
    _assert_encrypted_run_matches(old, new)
    n_params = 3137 if dataset == "stroke" else 929
    expected_encrypted = round(0.25 * n_params) if stack == "dp_spe" else n_params
    assert new["mask"]["n_encrypted"] == expected_encrypted
    assert new["crypto"]["n_chunks"] == 1.0
    assert new["crypto"]["wire_payload_ratio"] >= new["crypto"]["payload_ratio"]


@pytest.mark.parametrize("mask_mode", ["dp", "random"])
def test_other_mask_modes_match(
    oracle: Any, cfg: LegacyTrialConfig, prepared: Any, mask_mode: str
) -> None:
    old, new = _run_both(
        oracle, cfg, prepared, "stroke", "dp_spe", "fedavg", mask_mode=mask_mode, spe_fraction=0.5
    )
    _assert_encrypted_run_matches(old, new)
    assert new["mask"] == {"mode": mask_mode, "fraction": 0.5, "n_encrypted": round(0.5 * 3137)}


def test_encryption_does_not_change_what_is_learned(cfg: LegacyTrialConfig, prepared: Any) -> None:
    """The research finding the copy must preserve: encrypted arms land on their plain twins."""
    _, new_prep = prepared["stroke"]
    common = {"dataset": "stroke", "algo": "fedprox", "seed": SEED, "rounds": ROUNDS,
              "local_epochs": LOCAL_EPOCHS, "log": None, "prepared": new_prep}  # fmt: skip
    plain = _quiet(new_loop.run_stack, cfg, stack="plain", **common)
    encrypted = _quiet(new_loop.run_stack, cfg, stack="he", **common)
    for name in ("accuracy", "recall", "precision", "f1", "pred_pos_rate"):
        assert encrypted["metrics"][name] == plain["metrics"][name], name
    assert encrypted["metrics"]["loss"] == pytest.approx(
        plain["metrics"]["loss"], abs=ENCRYPTED_METRIC_TOLERANCE
    )


# --------------------------------------------------------------------------- refusals and defects


def test_unknown_names_are_refused_like_the_research_code(
    oracle: Any, cfg: LegacyTrialConfig, prepared: Any
) -> None:
    old_prep, new_prep = prepared["loan"]
    old_run = oracle.module("orchestrator").run_stack
    with pytest.raises(ValueError, match="Unknown algo"):
        old_run(dataset="loan", algo="banana", prepared=old_prep, log=None)
    with pytest.raises(ValueError, match="Unknown algo"):
        new_loop.run_stack(cfg, dataset="loan", algo="banana", prepared=new_prep, log=None)
    with pytest.raises(KeyError):
        old_run(dataset="loan", stack="banana", prepared=old_prep, log=None)
    with pytest.raises(KeyError):
        new_loop.run_stack(cfg, dataset="loan", stack="banana", prepared=new_prep, log=None)


def test_the_documented_defects_are_reproduced(cfg: LegacyTrialConfig, prepared: Any) -> None:
    """The copy must stay wrong in the same ways, or it stops being a reference."""
    _, new_prep = prepared["stroke"]
    lines: list[str] = []
    result = _quiet(
        new_loop.run_stack, cfg, dataset="stroke", stack="dp", algo="fedavg", seed=SEED,
        rounds=2, local_epochs=1, prepared=new_prep, use_smote=False, log=lines.append,
    )  # fmt: skip
    # G18: a prepared bundle silently wins over use_smote, and use_smote is recorded as given.
    assert result["use_smote"] is False
    resampled_sizes = [len(y) for _, y in new_prep["client_data"]]
    assert resampled_sizes != new_prep["part_info"]["sizes"]
    # G3: the noise was calibrated on the resampled sizes, not on the real ones.
    from obscuradp.legacy import dp as new_dp

    expected = [_quiet(new_dp.get_sigma, cfg, 9.0, n, epochs=1) for n in resampled_sizes]
    assert result["dp"]["per_client"]["sigma"] == expected
    # G17: the per-participant seed formula collides once there are ten participants.
    assert SEED * 1000 + 0 * 10 + 10 == SEED * 1000 + 1 * 10 + 0
    assert len(lines) == 2
    assert "recall=" in lines[0]
