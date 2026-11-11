"""Obscura DP: federated learning with a total privacy budget and a checkable privacy report.

The pieces most programs need are available here directly::

    import obscuradp as odp

    policy = odp.PrivacyPolicy(epsilon=1.0, delta=1e-5)
    plan = odp.Plan(dp=odp.LocalDPSGD(clip_norm=1.0), imbalance=odp.imbalance.ClassWeighted(0.05))
    result = odp.Simulation(model_fn, clients, plan, policy, rounds=20).run(seed=0)
    print(result.report.summary())

The interface is still growing (see ``docs/ROADMAP.md`` in the repository). Importing the
package never opens a network connection and never reads or sends any data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("obscuradp")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

from obscuradp import crypto, imbalance, tables
from obscuradp.ledger import BudgetExceeded, PrivacyLedger
from obscuradp.plan import CentralDPSGD, DistributedDPSGD, LocalDPSGD, Plan
from obscuradp.policy import PrivacyPolicy
from obscuradp.report import PrivacyReport, verify_report
from obscuradp.sim import Simulation, SimulationResult
from obscuradp.unsafe import UnsafeConfiguration

__all__ = [
    "BudgetExceeded",
    "CentralDPSGD",
    "DistributedDPSGD",
    "LocalDPSGD",
    "Plan",
    "PrivacyLedger",
    "PrivacyPolicy",
    "PrivacyReport",
    "Simulation",
    "SimulationResult",
    "UnsafeConfiguration",
    "__version__",
    "crypto",
    "imbalance",
    "tables",
    "verify_report",
]
