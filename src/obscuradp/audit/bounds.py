"""Turning an attack's success into a statement about privacy.

A privacy guarantee limits every attacker. So an attack that is run many times gives a
**lower limit** on the true privacy budget: if the attack tells "record used" from "record
not used" better than a budget $\\varepsilon$ allows, the true budget is larger than
$\\varepsilon$. A correct implementation must never show a lower limit above the budget it
reports. A lower limit far below the reported budget proves nothing; it only means this
attack was weak.

Three estimators are provided:

- ``epsilon_lower_bound`` makes no assumption about the mechanism. It uses the counts of
  wrong guesses in both directions with exact (Clopper-Pearson) confidence limits.
- ``gaussian_epsilon_lower_bound`` assumes the attacker's score is Gaussian with the same
  spread in both worlds, which is true by construction for some white-box tests. It is far
  more sensitive with the same number of trials.
- ``one_run_epsilon_lower_bound`` is for an audit that trains once: many test records are
  each kept or left out by a fair coin, and the attacker guesses for some of them which way
  the coin fell. It is the routine of Steinke, Nasr and Jagielski, "Privacy Auditing with
  One (1) Training Run" (2023), arXiv:2305.08846, Corollary 5.5 and the listing in their
  appendix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats

from obscuradp.accounting.analytic import gaussian_epsilon


def clopper_pearson(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Exact two-sided confidence interval for a probability from ``successes`` of ``trials``."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0 if successes == 0 else float(stats.beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(stats.beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    return lower, upper


def epsilon_lower_bound(
    false_positives: int,
    false_negatives: int,
    trials_without: int,
    trials_with: int,
    delta: float,
    confidence: float = 0.95,
) -> float:
    """Lower limit on $\\varepsilon$ from one attack's error counts, with no assumptions.

    Any $(\\varepsilon, \\delta)$ guarantee forces
    $\\mathrm{FPR} + e^{\\varepsilon}\\,\\mathrm{FNR} \\ge 1 - \\delta$ and the same with the two
    rates exchanged. Upper confidence limits on both rates turn that into a lower limit that
    holds with the stated confidence.

    Args:
        false_positives: Runs without the record where the attack said "used".
        false_negatives: Runs with the record where the attack said "not used".
        trials_without: Number of runs without the record.
        trials_with: Number of runs with the record.
        delta: The $\\delta$ of the guarantee being tested.
        confidence: Joint confidence of the result.

    Returns:
        The lower limit, or 0.0 if the attack shows nothing.
    """
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta must be in [0, 1)")
    each = 1.0 - (1.0 - confidence) / 2.0  # both rates must hold at once
    fpr = clopper_pearson(false_positives, trials_without, each)[1]
    fnr = clopper_pearson(false_negatives, trials_with, each)[1]
    best = 0.0
    for first, second in ((fpr, fnr), (fnr, fpr)):
        if first > 0.0 and 1.0 - delta - second > first:
            best = max(best, math.log((1.0 - delta - second) / first))
    return best


def gaussian_separation_interval(
    scores_without: Sequence[float], scores_with: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """How far apart two Gaussian score distributions are, in units of their spread.

    Returns ``(estimate, lower, upper)`` for $\\mu = (m_1 - m_0) / s$ where $s$ is the pooled
    standard deviation. The interval uses the normal approximation, which is accurate for
    the thousands of trials an audit runs.
    """
    without = np.asarray(scores_without, dtype=np.float64)
    with_record = np.asarray(scores_with, dtype=np.float64)
    if without.size < 2 or with_record.size < 2:
        raise ValueError("at least two scores are needed in each world")
    n0, n1 = without.size, with_record.size
    pooled = math.sqrt(
        ((n0 - 1) * without.var(ddof=1) + (n1 - 1) * with_record.var(ddof=1)) / (n0 + n1 - 2)
    )
    if pooled == 0.0:
        raise ValueError("the scores do not vary")
    estimate = float(with_record.mean() - without.mean()) / pooled
    # Variance of a standardized mean difference (Hedges and Olkin, 1985).
    error = math.sqrt(1.0 / n0 + 1.0 / n1 + estimate**2 / (2.0 * (n0 + n1 - 2)))
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    return estimate, estimate - z * error, estimate + z * error


def gaussian_epsilon_lower_bound(
    scores_without: Sequence[float],
    scores_with: Sequence[float],
    delta: float,
    confidence: float = 0.95,
) -> float:
    """Lower limit on $\\varepsilon$ when the attack's score is Gaussian in both worlds.

    Telling $\\mathcal{N}(0, 1)$ from $\\mathcal{N}(\\mu, 1)$ is exactly as hard as attacking
    Gaussian noise with noise multiplier $1 / \\mu$, whose $\\varepsilon$ at a given $\\delta$
    is known in closed form. The lower confidence limit of $\\mu$ gives the lower limit of
    $\\varepsilon$.
    """
    lower = gaussian_separation_interval(scores_without, scores_with, confidence)[1]
    if lower <= 0.0:
        return 0.0
    return gaussian_epsilon(1.0 / lower, delta)


def one_run_p_value(
    canaries: int, guesses: int, correct: int, epsilon: float, delta: float
) -> float:
    """How surprising ``correct`` right guesses are if the run really is private.

    ``canaries`` test records were each kept in or left out of one training run by a fair
    coin. The attacker made ``guesses`` guesses (it may pass on the other records) and
    ``correct`` of them were right. If the run satisfies $(\\varepsilon, \\delta)$, the number
    of right guesses is hardly larger than that of $r$ coin flips which each come up right
    with probability $e^{\\varepsilon} / (e^{\\varepsilon} + 1)$; $\\delta$ adds a small
    correction. The result is an upper limit on the probability of seeing at least
    ``correct`` right guesses.

    The number of guesses must be fixed before the coins are looked at.

    Args:
        canaries: Number of test records $m$.
        guesses: Number of guesses $r$, at most ``canaries``.
        correct: Number of right guesses $v$, at most ``guesses``.
        epsilon: The $\\varepsilon$ being tested.
        delta: The $\\delta$ being tested.
    """
    if not 0 <= correct <= guesses <= canaries:
        raise ValueError("need 0 <= correct <= guesses <= canaries")
    if epsilon < 0.0:
        raise ValueError("epsilon must not be negative")
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    if correct == 0:
        return 1.0
    right = 1.0 / (1.0 + math.exp(-epsilon))  # how often epsilon alone lets a guess be right
    beta = float(stats.binom.sf(correct - 1, guesses, right))  # P[Binomial(r, right) >= v]
    # alpha = max over i of P[v > Binomial >= v - i] / i, for i = 1 .. v.
    below = stats.binom.pmf(np.arange(correct - 1, -1, -1), guesses, right)
    alpha = float(np.max(np.cumsum(below) / np.arange(1, correct + 1)))
    return min(1.0, beta + alpha * delta * 2.0 * canaries)


def one_run_epsilon_lower_bound(
    canaries: int, guesses: int, correct: int, delta: float, confidence: float = 0.95
) -> float:
    """Lower limit on $\\varepsilon$ from the guesses of an audit that trained once.

    The largest $\\varepsilon$ for which ``one_run_p_value`` stays below ``1 - confidence``:
    with the stated confidence the run is not $(\\varepsilon, \\delta)$-private for any
    smaller value. The limit also holds for the pooled counts of several independent runs
    (each with its own coins): taken together they are one procedure in which a single
    record still only enters one run.

    Returns:
        The lower limit, or 0.0 if the guesses show nothing.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    level = 1.0 - confidence
    if one_run_p_value(canaries, guesses, correct, 0.0, delta) >= level:
        return 0.0
    low, high = 0.0, 1.0  # the p-value is below the level at low, and not below it at high
    while one_run_p_value(canaries, guesses, correct, high, delta) < level:
        low, high = high, high * 2.0
        if high > 64.0:
            # Only reachable when delta is 0 and every guess is right in a huge sample.
            return low
    for _ in range(40):
        middle = (low + high) / 2.0
        if one_run_p_value(canaries, guesses, correct, middle, delta) < level:
            low = middle
        else:
            high = middle
    return low
