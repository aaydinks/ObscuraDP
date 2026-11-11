"""How much whole-number (Skellam) noise can leak.

Skellam noise is the difference of two independent Poisson counts. It lives on the whole
numbers, and the sum of several Skellam noises is again Skellam noise, with the variances
added up. That makes it the natural noise for a secure sum over whole numbers: every
participant adds a slice and the total is exactly of the same kind.

Its leak is known as a curve of Renyi divergences. For a vector of whole numbers that one
record changes by at most $\\Delta_2$ in length and $\\Delta_1$ in summed absolute size,
noise of variance $\\mu$ per coordinate, and every whole-number order $\\alpha > 1$,

$$\\varepsilon(\\alpha) \\le \\frac{\\alpha \\Delta_2^2}{2\\mu}
+ \\min\\left(\\frac{(2\\alpha - 1)\\Delta_2^2 + 6\\Delta_1}{4\\mu^2},
\\ \\frac{3\\Delta_1}{2\\mu}\\right).$$

This is Corollary 3.6, equation (3.2), of Agarwal, Kairouz and Liu, "The Skellam Mechanism
for Differentially Private Federated Learning" (2021, arXiv:2110.04995). The first term is
the curve of ordinary Gaussian noise of the same variance; the second is the price of
living on whole numbers and shrinks as the noise grows. The result is proven for whole-number
orders only, so only those are used here. The tests compare the formula with an exact
computation from the Skellam probabilities.

Releases add up by adding their curves. The curve is turned into a budget with the
conversion of the ``dp-accounting`` library.
"""

from __future__ import annotations

# Whole-number orders at which the curve is evaluated. Dense where the best order of typical
# runs lies, sparser above.
ORDERS: tuple[int, ...] = (*range(2, 129), *range(132, 257, 4), 320, 384, 448, 512, 768, 1024)


def skellam_rdp(order: int, variance: float, l2_sensitivity: float, l1_sensitivity: float) -> float:
    """The Renyi divergence limit of one Skellam release at a whole-number order.

    Args:
        order: The order $\\alpha$, a whole number above 1.
        variance: Variance $\\mu$ of the noise in every coordinate.
        l2_sensitivity: $\\Delta_2$, the most one record can change the value in length.
        l1_sensitivity: $\\Delta_1$, the most it can change the sum of absolute values.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("order must be a whole number above 1 (the limit is proven for those)")
    if not variance > 0.0:
        raise ValueError("variance must be positive")
    if not l2_sensitivity > 0.0 or not l1_sensitivity > 0.0:
        raise ValueError("sensitivities must be positive")
    square = l2_sensitivity * l2_sensitivity
    gaussian_part = order * square / (2.0 * variance)
    correction = min(
        ((2 * order - 1) * square + 6.0 * l1_sensitivity) / (4.0 * variance * variance),
        3.0 * l1_sensitivity / (2.0 * variance),
    )
    return gaussian_part + correction
