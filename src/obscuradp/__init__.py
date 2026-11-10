"""Obscura DP: federated learning with a total privacy budget and a checkable privacy report.

This is the package skeleton. The public interface is added in later steps of the roadmap
(see ``docs/ROADMAP.md`` in the repository). Importing the package never opens a network
connection and never reads or sends any data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("obscuradp")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

from obscuradp.ledger import BudgetExceeded, PrivacyLedger
from obscuradp.policy import PrivacyPolicy
from obscuradp.report import PrivacyReport, verify_report

__all__ = [
    "BudgetExceeded",
    "PrivacyLedger",
    "PrivacyPolicy",
    "PrivacyReport",
    "__version__",
    "verify_report",
]
