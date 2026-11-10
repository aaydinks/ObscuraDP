"""Turning an attack's success into a statement about privacy.

A privacy guarantee limits every attacker. So an attack that is run many times gives a
**lower limit** on the true privacy budget: if the attack tells "record used" from "record
not used" better than a budget $\\varepsilon$ allows, the true budget is larger than
$\\varepsilon$. A correct implementation must never show a lower limit above the budget it
reports. A lower limit far below the reported budget proves nothing; it only means this
attack was weak.

Two estimators are provided:

- ``epsilon_lower_bound`` makes no assumption about the mechanism. It uses the counts of
  wrong guesses in both directions with exact (Clopper-Pearson) confidence limits.
- ``gaussian_epsilon_lower_bound`` assumes the attacker's score is Gaussian with the same
  spread in both worlds, which is true by construction for some white-box tests. It is far
  more sensitive with the same number of trials.
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
