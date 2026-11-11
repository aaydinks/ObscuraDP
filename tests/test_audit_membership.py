"""The attack on a whole training run (one-run method) and the auditor's self-test.

What must hold: the test records and their coins are drawn as described and public sizes
stay fixed; auditing a run does not change it; the attack only works from its attacker's
view; a pipeline whose noise is missing is flagged and a correct one is not; and the
summary shows the three numbers side by side with the most the test could have shown.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

import obscuradp as odp
from obscuradp.audit import membership, selftest
from obscuradp.audit.membership import (
    ATTACK_FINAL_MODEL,
    ATTACK_UPDATES,
    ATTACKS,
    GUESS_SHARES,
    RunSummary,
    audit_run,
    best_case_lower_bound,
    choose_canaries,
    count_guesses,
    leave_out,
    measured_advantage,
    summarize,
    summarize_run,
)
from obscuradp.audit.views import MODEL_OBSERVER, SERVER
from obscuradp.cli import main
from obscuradp.client import trainer as trainer_module
from obscuradp.crypto import PairwiseMasking
from obscuradp.mechanisms import dpsgd as dpsgd_module
from obscuradp.models import TabularMLP
from obscuradp.state import flatten_state
from obscuradp.tables import ClientData

DELTA = 1e-5

# Matched by its text, not by its class: another test re-imports the package, after which a
# filter that names the class would look at a different class object than the one raised.
pytestmark = pytest.mark.filterwarnings("ignore:protecting whole participants with only")


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    return selftest.made_up_participants(seed=11)


# --------------------------------------------------------------------------- test records


def test_a_quarter_of_the_rows_become_test_records_up_to_the_cap(
    clients: list[ClientData],
) -> None:
    canaries = choose_canaries(clients, seed=1)
    assert {name: len(rows) for name, rows in canaries.rows.items()} == {
        "site-0": 60,
        "site-1": 60,
        "site-2": 60,
    }
    assert canaries.count == 180
    for name, rows in canaries.rows.items():
        assert len(set(rows.tolist())) == len(rows)  # no row twice
        assert rows.min() >= 0
        assert rows.max() < 240
        assert canaries.kept[name].dtype == bool
    assert choose_canaries(clients, seed=1, cap=25).count == 75
    assert choose_canaries(clients, seed=1, fraction=0.1).count == 72
    assert set(choose_canaries(clients, seed=1, skip=["site-2"]).rows) == {"site-0", "site-1"}
    # The same seed gives the same records and coins, another seed gives others.
    again = choose_canaries(clients, seed=1)
    other = choose_canaries(clients, seed=2)
    assert np.array_equal(canaries.coins(), again.coins())
    assert not np.array_equal(canaries.rows["site-0"], other.rows["site-0"])
    assert not np.array_equal(canaries.coins(), other.coins())


def test_the_coins_are_fair() -> None:
    big = [dataclasses.replace(c) for c in selftest.made_up_participants(seed=0)]
    kept = np.concatenate(
        [choose_canaries(big, seed=seed, fraction=1.0, cap=10**6).coins() for seed in range(20)]
    )
    assert kept.size == 20 * 720
    assert abs(float(kept.mean()) - 0.5) < 0.02


def test_test_record_refusals(clients: list[ClientData]) -> None:
    with pytest.raises(ValueError, match="fraction"):
        choose_canaries(clients, seed=1, fraction=0.0)
    with pytest.raises(ValueError, match="cap"):
        choose_canaries(clients, seed=1, cap=0)


def test_leaving_records_out_keeps_the_public_sizes(clients: list[ClientData]) -> None:
    canaries = choose_canaries(clients, seed=3, skip=["site-1"])
    reduced, declared = leave_out(clients, canaries)
    assert declared == {"site-0": 240, "site-1": 240, "site-2": 240}
    assert reduced[1] is clients[1]
    for index in (0, 2):
        name = clients[index].name
        dropped = int((~canaries.kept[name]).sum())
        assert 10 < dropped < 50
        assert reduced[index].n_train == 240 - dropped
        # Exactly the records whose coin said "out" are gone.
        gone = canaries.rows[name][~canaries.kept[name]]
        stays = np.setdiff1d(np.arange(240), gone)
        assert np.array_equal(reduced[index].train_x, clients[index].train_x[stays])
        assert np.array_equal(reduced[index].train_y, clients[index].train_y[stays])


# --------------------------------------------------------------------------- reading scores


def test_guesses_go_to_the_surest_records_at_both_ends() -> None:
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.4, 0.6, 0.3, 0.7, 0.0])
    kept = np.array([True, False, True, True, False, False, True, False, False, False])
    # All ten: the five highest are guessed "in" (3 right), the five lowest "out" (4 right).
    assert count_guesses(scores, kept, 1.0) == (10, 7)
    # The surest four: 0.9 and 0.8 "in" (both right), 0.0 and 0.1 "out" (both right).
    assert count_guesses(scores, kept, 0.4) == (4, 4)
    # Too few records for a guess at each end.
    assert count_guesses(scores, kept, 0.1) == (0, 0)
    with pytest.raises(ValueError, match="share"):
        count_guesses(scores, kept, 0.0)
    with pytest.raises(ValueError, match="same length"):
        count_guesses(scores, kept[:-1], 1.0)


def test_scores_that_know_nothing_are_right_half_of_the_time() -> None:
    generator = np.random.default_rng(0)
    kept = generator.random(4000) < 0.5
    guesses, right = count_guesses(np.zeros(4000), kept, 1.0)  # all tied
    assert guesses == 4000
    assert abs(right / guesses - 0.5) < 0.03
    assert abs(measured_advantage(generator.normal(size=4000), kept)) < 0.05
    assert measured_advantage(kept.astype(float), kept) == pytest.approx(1.0)
    assert measured_advantage(-kept.astype(float), kept) < 0.05  # wrong way round: no advantage


def test_the_measured_advantage_is_the_gap_between_two_bell_curves() -> None:
    generator = np.random.default_rng(1)
    kept = generator.random(20_000) < 0.5
    scores = generator.normal(size=kept.size) + 1.0 * kept
    # Two unit bell curves one apart: the best cut-off gives 2 * Phi(0.5) - 1 = 0.383.
    assert measured_advantage(scores, kept) == pytest.approx(0.383, abs=0.02)


def test_the_most_the_test_could_show() -> None:
    nothing = best_case_lower_bound(700, 1, 1.0, 0.0, DELTA, trials=20)
    assert nothing == 0.0
    one_run = best_case_lower_bound(700, 1, 1.0, 0.106, DELTA, trials=40)
    pooled = best_case_lower_bound(700, 30, 1.0, 0.106, DELTA, trials=40)
    surest = best_case_lower_bound(700, 30, 0.1, 0.106, DELTA, trials=40)
    # At a budget of 1 (proven advantage limit 10.6%) even a perfect attacker proves little
    # with one run, more with thirty, and most when it only guesses where it is surest.
    assert 0.0 <= one_run < 0.2
    assert one_run < pooled < surest < 1.0
    assert best_case_lower_bound(700, 30, 0.1, 0.5, DELTA, trials=20) > surest
    with pytest.raises(ValueError, match="advantage_ceiling"):
        best_case_lower_bound(700, 1, 1.0, 1.5, DELTA)
    with pytest.raises(ValueError, match="at least one"):
        best_case_lower_bound(0, 1, 1.0, 0.5, DELTA)


def _summary(right: int, guesses: int = 200, canaries: int = 200, **overrides: Any) -> RunSummary:
    by_share = {f"{share:g}": (guesses, right) for share in GUESS_SHARES}
    arguments: dict[str, Any] = {
        "canaries": canaries,
        "guesses": {attack: by_share for attack in ATTACKS},
        "advantage": dict.fromkeys(ATTACKS, 2 * right / guesses - 1),
        "epsilon": 1.0,
        "delta": DELTA,
        "ceiling_mechanism": 0.106,
        **overrides,
    }
    return RunSummary(**arguments)


def test_runs_are_pooled_and_the_three_numbers_stand_side_by_side() -> None:
    runs = [_summary(right) for right in (104, 110, 98, 112, 106)]
    found = summarize(runs, ATTACK_UPDATES, 1.0)
    assert (found.runs, found.canaries, found.guesses, found.correct) == (5, 1000, 1000, 530)
    assert found.epsilon_claimed == 1.0
    assert found.ceiling_any == pytest.approx(0.462, abs=1e-3)
    assert found.ceiling_mechanism == 0.106
    assert found.advantage == pytest.approx(0.06)
    assert found.advantage_low <= found.advantage <= found.advantage_high
    assert found.advantage_low < found.advantage_high
    assert 0.0 <= found.epsilon_lower < 1.0
    assert found.best_case is not None
    assert not found.flagged
    text = found.describe()
    for part in ("stated budget 1", "10.6%", "46.2%", "the most it could show", "not flagged"):
        assert part in text
    assert chr(0x2014) not in text
    assert found.to_dict()["correct"] == 530


def test_guessing_right_too_often_is_flagged() -> None:
    runs = [_summary(right) for right in (190, 186, 194)]
    found = summarize(runs, ATTACK_UPDATES, 1.0)
    assert found.epsilon_lower > 1.0
    assert found.epsilon_lower_median_run > 1.0
    assert found.flagged
    assert "FLAGGED" in found.describe()
    # The pooled limit uses all the guesses and so proves more than a single run does.
    assert found.epsilon_lower > summarize(runs[:1], ATTACK_UPDATES, 1.0).epsilon_lower


def test_a_pipeline_without_a_report_needs_its_claim_spelled_out() -> None:
    bare = _summary(190, epsilon=None, delta=None, ceiling_mechanism=None)
    with pytest.raises(ValueError, match="claimed_epsilon"):
        summarize([bare], ATTACK_UPDATES, 1.0)
    with pytest.raises(ValueError, match="claimed_epsilon"):
        summarize([bare], ATTACK_UPDATES, 1.0, claimed_epsilon=1.0)
    found = summarize([bare], ATTACK_UPDATES, 1.0, claimed_epsilon=1.0, delta=DELTA)
    assert found.flagged
    assert found.ceiling_mechanism is None
    assert found.best_case is None
    assert "none proven" in found.describe()
    with pytest.raises(ValueError, match="no runs"):
        summarize([], ATTACK_UPDATES, 1.0)


def test_a_run_summary_survives_being_written_down() -> None:
    summary = _summary(120)
    assert RunSummary.from_dict(summary.to_dict()) == summary
    bare = _summary(120, epsilon=None, delta=None, ceiling_mechanism=None)
    assert RunSummary.from_dict(bare.to_dict()) == bare


# --------------------------------------------------------------------------- whole runs


def test_auditing_a_run_does_not_change_it(clients: list[ClientData]) -> None:
    audited = audit_run(selftest.correct_pipeline, clients, seed=5)
    reduced, declared = leave_out(clients, choose_canaries(clients, seed=5))
    plain = selftest.correct_pipeline(reduced, declared).run(seed=5)
    assert np.array_equal(flatten_state(audited.result.state), flatten_state(plain.state))
    assert audited.attacker == SERVER
    assert audited.colluders == ()
    assert set(audited.scores) == set(ATTACKS)
    assert audited.kept.shape == (180,)
    for scores in audited.scores.values():
        assert scores.shape == (180,)
        assert np.all(np.isfinite(scores))
    summary = summarize_run(audited)
    assert summary.canaries == 180
    assert summary.epsilon == pytest.approx(1.0, abs=0.01)
    assert summary.ceiling_mechanism == pytest.approx(0.105, abs=0.01)
    assert set(summary.guesses[ATTACK_UPDATES]) == {f"{share:g}" for share in GUESS_SHARES}
    assert summary.guesses[ATTACK_UPDATES]["1"][0] == 180


def test_the_self_test_passes() -> None:
    """The step's exit check: a silent pipeline is flagged, a correct one is not."""
    checks = selftest.self_test()
    assert [check.must_flag for check in checks] == [False, True, False, True]
    assert [check.flagged for check in checks] == [False, True, False, True]
    assert selftest.passed(checks)
    assert all(chr(0x2014) not in check.detail for check in checks)


def test_the_self_test_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["audit", "self-test", "--trials", "2000"]) == 0
    out = capsys.readouterr().out
    assert out.count("[ok]") == 4
    assert "Self-test passed" in out
    # An auditor that flags nothing fails its self-test, and the command says so.
    blind = [
        selftest.SelfTestCheck("a broken pipeline", must_flag=True, flagged=False, detail="-"),
    ]
    monkeypatch.setattr(selftest, "self_test", lambda trials: blind)
    assert main(["audit", "self-test"]) == 1
    out = capsys.readouterr().out
    assert "[FAILED]" in out
    assert "Self-test FAILED" in out


def test_a_private_run_whose_noise_is_missing_is_flagged(
    clients: list[ClientData], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planted defect: the default plan keeps counting its noise but adds none."""
    honest = dpsgd_module.gaussian_noise
    monkeypatch.setattr(
        dpsgd_module,
        "gaussian_noise",
        lambda shape, standard_deviation, generator, **kwargs: honest(
            shape, 0.0, generator, **kwargs
        ),
    )
    runs = [
        summarize_run(audit_run(selftest.correct_pipeline, clients, seed)) for seed in range(5, 10)
    ]
    # The ledger still believes the noise is there.
    assert all(run.epsilon == pytest.approx(1.0, abs=0.01) for run in runs)
    found = summarize(runs, ATTACK_UPDATES, 1.0)
    assert found.flagged
    assert found.advantage > 0.4
    # The attacker measures the noise itself. Had it believed the stated noise, it would
    # have looked away: the correct pipeline, by contrast, stays unflagged (self-test).


def _shared_pipeline(noise: str) -> membership.SimulationBuilder:
    def build(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        return odp.Simulation(
            model_fn=lambda: TabularMLP(selftest.WIDTH, (16,)),
            clients=rows,
            plan=odp.Plan(
                dp=odp.DistributedDPSGD(max_colluding=1, noise=noise),
                secure_sum=PairwiseMasking(),
            ),
            policy=odp.PrivacyPolicy(1.0, DELTA),
            rounds=8,
            batch_size=10**9 if noise == "skellam" else 256,
            learning_rate=0.1,
            evaluate_every=0,
            declared_rows=declared,
        )

    return build


@pytest.mark.parametrize("noise", ["gaussian", "skellam"])
def test_shared_noise_is_attacked_through_the_opened_sum_and_one_colluder(
    clients: list[ClientData], noise: str
) -> None:
    audited = audit_run(_shared_pipeline(noise), clients, seed=6)
    assert audited.attacker == SERVER
    assert audited.colluders == ("site-2",)
    # The colluder knows its own rows, so it holds no test records.
    assert audited.kept.shape == (120,)
    found = summarize([summarize_run(audited)], ATTACK_UPDATES, 1.0)
    assert not found.flagged


def test_shared_noise_that_is_missing_is_flagged(
    clients: list[ClientData], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planted defect: every participant sends its sum without its slice of the noise.

    Shown on the whole-number path, where every row takes part in every round and nothing
    is rounded after the noise: without the noise the sums are exact and the coins can be
    read off. With ordinary noise two things still blur a record when the noise is gone,
    the random draw of rows and the rounding grid (which is sized to the stated noise), so
    one short run of this size does not show the defect there; the single-step check does
    (``test_distributed``).
    """
    monkeypatch.setattr(
        trainer_module,
        "skellam_noise",
        lambda shape, variance, generator: np.zeros(shape, dtype=np.int64),
    )
    run = summarize_run(audit_run(_shared_pipeline("skellam"), clients, seed=6))
    assert run.epsilon == pytest.approx(1.0, abs=0.01)  # the ledger still believes the noise
    found = summarize([run], ATTACK_UPDATES, 1.0)
    assert found.flagged
    assert found.advantage > 0.8


def _central(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
    return odp.Simulation(
        model_fn=lambda: TabularMLP(selftest.WIDTH, (16,)),
        clients=rows,
        plan=odp.Plan(dp=odp.CentralDPSGD()),
        policy=odp.PrivacyPolicy(1.0, DELTA),
        rounds=8,
        batch_size=256,
        learning_rate=0.1,
        evaluate_every=0,
        allow_unsafe=True,
        declared_rows=declared,
    )


def test_a_trusted_server_is_not_the_attacker_of_its_own_plan(
    clients: list[ClientData],
) -> None:
    audited = audit_run(_central, clients, seed=7)
    assert audited.attacker == MODEL_OBSERVER
    assert audited.kept.shape == (180,)
    assert not summarize([summarize_run(audited)], ATTACK_UPDATES, 1.0).flagged
    with pytest.raises(ValueError, match="no colluders"):
        audit_run(_central, clients, seed=7, colluders=["site-0"])


def test_local_noise_through_the_secure_sum_is_attacked_through_the_sum(
    clients: list[ClientData],
) -> None:
    def build(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        simulation = selftest.correct_pipeline(rows, declared)
        return dataclasses.replace(
            simulation, plan=odp.Plan(dp=odp.LocalDPSGD(), secure_sum=PairwiseMasking())
        )

    audited = audit_run(build, clients, seed=8)
    assert audited.attacker == SERVER
    assert np.all(np.isfinite(audited.scores[ATTACK_UPDATES]))
    assert not summarize([summarize_run(audited)], ATTACK_UPDATES, 1.0).flagged
    # The same pipeline seen by someone who only gets the models, and with more path points.
    watched = audit_run(build, clients, seed=8, attacker=MODEL_OBSERVER, path_points=2)
    assert watched.attacker == MODEL_OBSERVER
    assert np.array_equal(watched.kept, audited.kept)


def test_audits_that_cannot_be_run_are_refused(clients: list[ClientData]) -> None:
    def forgetful(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        return dataclasses.replace(selftest.correct_pipeline(rows, declared), declared_rows=None)

    with pytest.raises(ValueError, match="declared_rows"):
        audit_run(forgetful, clients, seed=1)

    def adam(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        return dataclasses.replace(selftest.correct_pipeline(rows, declared), optimizer="adam")

    with pytest.raises(ValueError, match="sgd"):
        audit_run(adam, clients, seed=1)

    def proximal(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        return dataclasses.replace(
            selftest.correct_pipeline(rows, declared),
            plan=odp.Plan(dp=odp.LocalDPSGD(), proximal_mu=0.1),
        )

    with pytest.raises(ValueError, match="proximal"):
        audit_run(proximal, clients, seed=1)

    def whole(rows: Sequence[ClientData], declared: Mapping[str, int]) -> odp.Simulation:
        return odp.Simulation(
            model_fn=lambda: TabularMLP(selftest.WIDTH, (16,)),
            clients=rows,
            plan=odp.Plan(
                dp=odp.ClientLevelDP(sampling_rate=1.0, max_colluding=1),
                secure_sum=PairwiseMasking(),
            ),
            policy=odp.PrivacyPolicy(1.0, DELTA, unit="client"),
            rounds=2,
            declared_rows=declared,
        )

    with pytest.raises(ValueError, match="whole participants"):
        audit_run(whole, clients, seed=1)
    with pytest.raises(ValueError, match="attacker must be"):
        audit_run(selftest.correct_pipeline, clients, seed=1, attacker="a neighbour")
    with pytest.raises(ValueError, match="path_points"):
        audit_run(selftest.correct_pipeline, clients, seed=1, path_points=0)
    tiny = [
        dataclasses.replace(client, train_x=client.train_x[:3], train_y=client.train_y[:3])
        for client in clients
    ]
    with pytest.raises(ValueError, match="no test records"):
        audit_run(selftest.silent_pipeline, tiny, seed=1)


def test_the_outsiders_attack_scores_fit(clients: list[ClientData]) -> None:
    """The final-model attack: a record the model fits better scores higher."""
    audited = audit_run(selftest.silent_pipeline, clients, seed=9)
    scores = audited.scores[ATTACK_FINAL_MODEL]
    assert np.all(scores <= 0.0)  # minus a loss
    assert float(np.std(scores)) > 0.0


@pytest.mark.slow
def test_the_self_test_holds_on_other_tables_too() -> None:
    """Nightly: the auditor tells a silent pipeline from a correct one on ten more tables."""
    for seed in range(1, 11):
        table = selftest.made_up_participants(seed)
        silent = summarize(
            [summarize_run(audit_run(selftest.silent_pipeline, table, seed))],
            ATTACK_UPDATES,
            1.0,
            claimed_epsilon=1.0,
            delta=DELTA,
        )
        correct = summarize(
            [summarize_run(audit_run(selftest.correct_pipeline, table, seed))], ATTACK_UPDATES, 1.0
        )
        assert silent.flagged, seed
        assert not correct.flagged, seed
