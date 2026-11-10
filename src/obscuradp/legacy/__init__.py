"""A faithful copy of the Trial 07/27 research pipeline, kept only for comparison.

DO NOT USE THIS FOR REAL DATA. The research pipeline has known privacy defects: its privacy
budget is counted per round, it rebalances data in a way that voids the per-record
guarantee, it fits scalers on pooled data, and more. They are listed in ``docs/AUDIT.md``
in the repository.

Why it is here: the library is built from the same building blocks as the research code.
This subpackage wires those blocks together exactly the way the research code did, so
tests can show, number for number, that nothing changed by accident during the move. New
behaviour never goes here.

Nothing in this subpackage is part of the public interface. It needs the ``legacy`` extra
(``pip install "obscuradp[legacy]"``) and the research data files, which are not shipped.
"""

from obscuradp.legacy.config import LegacyTrialConfig

__all__ = ["LegacyTrialConfig"]
