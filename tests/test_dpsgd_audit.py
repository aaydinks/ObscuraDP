"""Attack the private trainer and compare what the attack proves with what the ledger says.

The attacker here is the strongest the guarantee allows for: it sees the noisy gradient of
every step, knows every other record, and only has to decide whether one known record (the
"canary") was in the training table. Whatever such an attack can prove is a lower limit on
the true privacy budget. A correct trainer never shows a limit above the ledger's figure.
A trainer that adds half the noise it reports must be caught.
"""

import math

import pytest
import torch
from torch import nn

from obscuradp.accounting import gaussian_epsilon
from obscuradp.audit import epsilon_lower_bound, gaussian_epsilon_lower_bound
from obscuradp.client import train_private_round
from obscuradp.ledger import PrivacyLedger
from obscuradp.mechanisms import clipped_sum, per_record_gradients
from obscuradp.mechanisms import dpsgd as dpsgd_module
from obscuradp.models import TabularMLP
from obscuradp.policy import PrivacyPolicy

DELTA = 1e-5


class _Linear(nn.Module):
    """Loss is linear in the weights, so every record's gradient is the same at every step."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


def _linear_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -output * target


def _full_batch_scores(
    with_canary: bool, trials: int, sigma: float, steps: int, seed: int
) -> tuple[list[float], float]:
    """Run the trainer ``trials`` times; return the attacker's scores and the ledger's budget.

    Every step uses every record (sampling rate 1), so the releases are plain Gaussian noise
    and the best possible attack is known: project each noisy gradient on the canary's
    direction and add the projections up. Its score is Gaussian with standard deviation
    $\\sigma \\sqrt{T}$, shifted by $T$ when the canary is present.
    """
    width, others, clip = 6, 5, 1.0
    data = torch.Generator().manual_seed(0)
    features = torch.randn(others, width, generator=data)
    canary = torch.zeros(1, width)
    canary[0, 0] = 3.0  # longer than the clipping norm, so it is clipped to exactly that
    if with_canary:
        features = torch.cat([features, canary])
    targets = torch.ones(features.shape[0])
    declared = others + 1  # the public size is the same in both worlds
    sampling = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + 1)
    scores = []
    spent = 0.0
    for _ in range(trials):
        model = _Linear(width)
        ledger = PrivacyLedger(PrivacyPolicy(50.0, DELTA))
        seen: list[float] = []
        train_private_round(
            model,
            features,
            targets,
            loss_fn=_linear_loss,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            ledger=ledger,
            scope="audited",
            sampling_rate=1.0,
            steps=steps,
            clip_norm=clip,
            noise_multiplier=sigma,
            sampling_generator=sampling,
            noise_generator=noise,
            declared_records=declared,
            on_step=lambda step, indices, noisy, sink=seen: sink.append(
                -float(noisy["w"][0]) * declared / clip
            ),
        )
        scores.append(sum(seen))
        spent = ledger.spent("audited")
    return scores, spent


STEPS = 2
# Two full-batch steps at this noise level cost a budget of 1 at delta = 1e-5.
SIGMA = 3.73 * math.sqrt(STEPS)
TRIALS = 2000


def test_the_ledger_figure_is_the_exact_gaussian_answer_here() -> None:
    _, spent = _full_batch_scores(True, 1, SIGMA, STEPS, seed=0)
    assert spent == pytest.approx(gaussian_epsilon(SIGMA, DELTA, count=STEPS), rel=1e-3)
    assert spent == pytest.approx(1.0, abs=0.01)


def test_the_strongest_attack_proves_no_more_than_the_ledger_reports() -> None:
    without, spent = _full_batch_scores(False, TRIALS, SIGMA, STEPS, seed=10)
    with_canary, _ = _full_batch_scores(True, TRIALS, SIGMA, STEPS, seed=20)
    proven = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven <= spent
    # The attack is sharp: it proves most of the budget, so the accounting is not loose either.
    estimate = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=1e-9)
    assert estimate == pytest.approx(spent, rel=0.3)


def test_half_the_reported_noise_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: break the noise inside the mechanism and the audit must notice."""
    honest = dpsgd_module.gaussian_noise

    def half_noise(shape, standard_deviation, generator, **kwargs):  # type: ignore[no-untyped-def]
        return honest(shape, standard_deviation / 2.0, generator, **kwargs)

    monkeypatch.setattr(dpsgd_module, "gaussian_noise", half_noise)
    without, spent = _full_batch_scores(False, TRIALS, SIGMA, STEPS, seed=10)
    with_canary, _ = _full_batch_scores(True, TRIALS, SIGMA, STEPS, seed=20)
    assert spent == pytest.approx(1.0, abs=0.01)  # the ledger still believes the full noise
    proven = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven > 1.5 * spent


def test_unclipped_gradients_are_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second mutation: clip to three times the reported norm."""
    honest = dpsgd_module.clipped_sum
    monkeypatch.setattr(
        dpsgd_module,
        "clipped_sum",
        lambda gradients, clip_norm, clip_scale=None: honest(
            gradients, 3.0 * clip_norm, clip_scale
        ),
    )
    without, spent = _full_batch_scores(False, TRIALS, SIGMA, STEPS, seed=10)
    with_canary, _ = _full_batch_scores(True, TRIALS, SIGMA, STEPS, seed=20)
    proven = gaussian_epsilon_lower_bound(without, with_canary, DELTA, confidence=0.999)
    assert proven > 1.5 * spent


def _sampled_scores(with_canary: bool, trials: int, seed: int) -> tuple[list[float], float]:
    """The same attack on a small network with Poisson-sampled batches (no assumptions)."""
    width, others, clip, rate, steps, sigma = 4, 12, 1.0, 0.5, 8, 0.8
    data = torch.Generator().manual_seed(0)
    features = torch.randn(others, width, generator=data)
    targets = (torch.rand(others, generator=data) > 0.5).float()
    canary_x = torch.full((1, width), 4.0)
    canary_y = torch.zeros(1)
    if with_canary:
        features = torch.cat([features, canary_x])
        targets = torch.cat([targets, canary_y])
    sampling = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + 1)
    loss_fn = nn.BCELoss()
    scores = []
    spent = 0.0
    for _ in range(trials):
        torch.manual_seed(0)
        model = TabularMLP(width, (4,))
        ledger = PrivacyLedger(PrivacyPolicy(1e6, DELTA, accountant="rdp"))
        seen: list[float] = []

        def watch(
            step: int,
            indices: torch.Tensor,
            noisy: dict[str, torch.Tensor],
            model: nn.Module = model,
            sink: list[float] = seen,
        ) -> None:
            # The attacker knows the weights at this step, so it knows the canary's gradient.
            direction = clipped_sum(per_record_gradients(model, loss_fn, canary_x, canary_y), clip)
            sink.append(sum(float((noisy[name] * direction[name]).sum()) for name in noisy))

        train_private_round(
            model,
            features,
            targets,
            loss_fn=loss_fn,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
            ledger=ledger,
            scope="audited",
            sampling_rate=rate,
            steps=steps,
            clip_norm=clip,
            noise_multiplier=sigma,
            sampling_generator=sampling,
            noise_generator=noise,
            declared_records=others + 1,
            on_step=watch,
        )
        scores.append(sum(seen))
        spent = PrivacyLedger(PrivacyPolicy(1e6, DELTA)).epsilon_of(ledger.events("audited"))
    return scores, spent


@pytest.mark.slow
def test_white_box_attack_on_sampled_training_stays_below_the_ledger() -> None:
    """Nightly check with Poisson sampling and a real network, using exact error counts."""
    trials = 2000
    without, spent = _sampled_scores(False, trials, seed=100)
    with_canary, _ = _sampled_scores(True, trials, seed=200)
    # Choose the threshold on the first half, count errors on the second half.
    half = trials // 2
    candidates = sorted(without[:half] + with_canary[:half])[:: max(1, half // 50)]
    best_threshold = max(
        candidates,
        key=lambda t: epsilon_lower_bound(
            sum(score > t for score in without[:half]),
            sum(score <= t for score in with_canary[:half]),
            half,
            half,
            DELTA,
        ),
    )
    proven = epsilon_lower_bound(
        sum(score > best_threshold for score in without[half:]),
        sum(score <= best_threshold for score in with_canary[half:]),
        trials - half,
        trials - half,
        DELTA,
        confidence=0.999,
    )
    assert proven <= spent
    assert proven > 0.3  # the attack does prove something, so the comparison has teeth
