"""Attack-based checks of the privacy the library reports.

- ``obscuradp.audit.bounds``: turning an attack's success into a lower limit on the budget.
- ``obscuradp.audit.mechanism``: the strongest allowed attacker against single releases.
- ``obscuradp.audit.membership``: an attack on a whole training run that trains only once.
- ``obscuradp.audit.shadow``: the shadow-model attack, a second opinion for an outsider.
- ``obscuradp.audit.attribute``: guessing a hidden column with the model and without it.
- ``obscuradp.audit.inversion``: reading a record back out of an update.
- ``obscuradp.audit.views``: what each attacker is allowed to see of a round.
- ``obscuradp.audit.selftest``: the auditor shown a correct and a broken pipeline.

The submodules are imported on request; this package itself only loads the arithmetic.
"""

from obscuradp.audit.bounds import (
    clopper_pearson,
    epsilon_lower_bound,
    gaussian_epsilon_lower_bound,
    gaussian_separation_interval,
    one_run_epsilon_lower_bound,
    one_run_p_value,
)

__all__ = [
    "clopper_pearson",
    "epsilon_lower_bound",
    "gaussian_epsilon_lower_bound",
    "gaussian_separation_interval",
    "one_run_epsilon_lower_bound",
    "one_run_p_value",
]
