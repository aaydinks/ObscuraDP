"""Checking one accountant against the others.

Any single piece of privacy bookkeeping can be wrong, and a wrong budget fails silently:
training still runs and accuracy even improves. So every budget the library reports is
computed three ways and the three are required to be consistent:

- the PLD lower estimate must not exceed the PLD upper estimate or the RDP limit
  (a lower estimate above an upper limit means one of them is broken);
- the PLD upper and lower estimates must be close (otherwise the grid is too coarse).

``cross_check`` raises ``AccountingMismatch`` when one of these rules is broken, so a run
stops rather than report a number that failed its own check.

One more comparison is a warning, not a rule. For the long, calibrated runs this library
plans (hundreds of steps, total budgets from 0.5 to 8), the RDP limit was measured at 1.08
to 1.24 times the PLD estimate. Far above that band usually means the two accountants were
given different releases. It is not proof, because RDP is known to be much looser for short
runs with strong subsampling (measured: 1.88 times at 60 steps, sampling rate 0.01, noise
multiplier 1). ``CrossCheck.warnings`` reports it; nothing is raised.

A budget above ``MEANINGLESS_EPSILON`` is reported as unbounded by the PLD accountant without
being computed, so there is nothing to compare and no rule applies to it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from obscuradp.accounting.accountants import epsilon_pld, epsilon_rdp
from obscuradp.accounting.events import Event

# Slack for floating-point and grid effects when comparing two estimates.
ORDER_SLACK = 1e-6
# The PLD upper and lower estimates may differ by this share of the budget, plus the absolute slack.
PLD_GAP_RELATIVE = 0.10
PLD_GAP_ABSOLUTE = 0.02
# Above this factor of the PLD upper estimate (plus the absolute slack) the RDP limit is
# unusually loose for a long run, which is worth a warning.
RDP_BAND_RELATIVE = 1.35
RDP_BAND_ABSOLUTE = 0.05


class AccountingMismatch(RuntimeError):
    """Two accountants disagree in a way that cannot happen if both are right."""


@dataclass(frozen=True)
class CrossCheck:
    """One budget computed three ways."""

    delta: float
    pld_upper: float
    pld_lower: float
    rdp: float

    @property
    def epsilon(self) -> float:
        """The budget to report: the PLD upper estimate."""
        return self.pld_upper

    @property
    def unbounded(self) -> bool:
        """True if the budget is too large to be worth computing (reported as infinite)."""
        return math.isinf(self.pld_upper) and math.isinf(self.pld_lower)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def problems(self) -> list[str]:
        """Every consistency rule this result breaks, in plain words. Empty if all hold.

        These cannot happen if all three accountants are right and saw the same releases.
        """
        found: list[str] = []
        if self.unbounded:
            return found
        if self.pld_lower > self.pld_upper + ORDER_SLACK:
            found.append(
                f"the PLD lower estimate ({self.pld_lower:.6g}) is above its upper estimate "
                f"({self.pld_upper:.6g})"
            )
        if self.pld_lower > self.rdp + ORDER_SLACK:
            found.append(
                f"the PLD lower estimate ({self.pld_lower:.6g}) is above the RDP limit "
                f"({self.rdp:.6g})"
            )
        if self.pld_upper - self.pld_lower > PLD_GAP_RELATIVE * self.pld_upper + PLD_GAP_ABSOLUTE:
            found.append(
                f"the PLD upper ({self.pld_upper:.6g}) and lower ({self.pld_lower:.6g}) estimates "
                "are too far apart"
            )
        return found

    def warnings(self) -> list[str]:
        """Observations that are unusual but possible when everything is right."""
        if self.unbounded:
            return ["the budget is too large to mean anything and is reported as unbounded"]
        if self.rdp > RDP_BAND_RELATIVE * self.pld_upper + RDP_BAND_ABSOLUTE:
            return [
                f"the RDP limit ({self.rdp:.6g}) is far above the PLD estimate "
                f"({self.pld_upper:.6g}); this is normal for short runs with strong "
                "subsampling and otherwise worth a second look"
            ]
        return []


def cross_check(events: Sequence[Event], delta: float, *, strict: bool = True) -> CrossCheck:
    """Compute the budget of ``events`` three ways and check that they are consistent.

    Raises ``AccountingMismatch`` on any inconsistency unless ``strict`` is false.
    """
    result = CrossCheck(
        delta=delta,
        pld_upper=epsilon_pld(events, delta, pessimistic=True),
        pld_lower=epsilon_pld(events, delta, pessimistic=False),
        rdp=epsilon_rdp(events, delta),
    )
    if strict:
        problems = result.problems()
        if problems:
            raise AccountingMismatch("privacy accountants disagree: " + "; ".join(problems))
    return result
