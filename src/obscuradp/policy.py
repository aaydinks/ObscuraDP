"""The privacy promise a run must keep, stated once.

A ``PrivacyPolicy`` is the single dial of the library: a total budget for the whole training
run, and a statement of what the budget protects. Everything else (how much noise each step
needs, when to refuse a release) is derived from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

UNITS = ("record", "client")
ADJACENCIES = ("add_remove",)
ACCOUNTANTS = ("pld", "rdp")

_UNIT_TEXT = {
    "record": "one record (one row of one participant's training table)",
    "client": "one participant (all of that participant's records together)",
}


@dataclass(frozen=True)
class PrivacyPolicy:
    """A total privacy budget and what it protects.

    Args:
        epsilon: Total budget $\\varepsilon$ for the whole run. Smaller is stronger.
        delta: The small failure probability $\\delta$ of the guarantee. It should be well below
            one over the number of records.
        unit: What is protected: ``"record"`` (one row of one participant's table) or
            ``"client"`` (one whole participant).
        adjacency: Which datasets count as neighbours. ``"add_remove"``: one unit is added or
            removed. This is what the accountants assume.
        accountant: ``"pld"`` (tight) or ``"rdp"`` (looser, needs more noise).
        per_scope_epsilon: Optional budgets for individual scopes (for example participants
            who want a stricter budget than the default). Each must be positive.
        max_records_per_person: If one person can own several rows, say how many at most. The
            protection for that person is weaker by this factor, and the report says so.
    """

    epsilon: float
    delta: float
    unit: str = "record"
    adjacency: str = "add_remove"
    accountant: str = "pld"
    per_scope_epsilon: Mapping[str, float] = field(default_factory=dict)
    max_records_per_person: int = 1

    def __post_init__(self) -> None:
        if not self.epsilon > 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}")
        if self.adjacency not in ADJACENCIES:
            raise ValueError(f"adjacency must be one of {ADJACENCIES}")
        if self.accountant not in ACCOUNTANTS:
            raise ValueError(f"accountant must be one of {ACCOUNTANTS}")
        if self.max_records_per_person < 1:
            raise ValueError("max_records_per_person must be at least 1")
        budgets = {str(scope): float(value) for scope, value in self.per_scope_epsilon.items()}
        if any(not value > 0 for value in budgets.values()):
            raise ValueError("every per-scope epsilon must be positive")
        object.__setattr__(self, "per_scope_epsilon", MappingProxyType(budgets))

    def epsilon_for(self, scope: str) -> float:
        """The budget of ``scope``: its own if it has one, otherwise the default."""
        return self.per_scope_epsilon.get(scope, self.epsilon)

    def delta_is_weak_for(self, n_records: int) -> bool:
        """True if $\\delta$ is not clearly below one over the number of records."""
        return n_records > 0 and self.delta >= 1.0 / n_records

    def describe(self) -> str:
        """The promise in plain words."""
        lines = [
            f"Privacy unit: {_UNIT_TEXT[self.unit]}.",
            "Neighbouring datasets differ by adding or removing one such unit; "
            "the sizes participants declare are treated as public.",
            f"Total budget for the whole run: epsilon = {self.epsilon:g}, delta = {self.delta:g}.",
        ]
        if self.per_scope_epsilon:
            parts = ", ".join(
                f"{scope}: {value:g}" for scope, value in sorted(self.per_scope_epsilon.items())
            )
            lines.append(f"Scopes with their own budget: {parts}.")
        if self.max_records_per_person > 1:
            lines.append(
                f"A person may own up to {self.max_records_per_person} records; for such a person "
                f"the protection is weaker by that factor (group privacy)."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "unit": self.unit,
            "adjacency": self.adjacency,
            "accountant": self.accountant,
            "per_scope_epsilon": dict(self.per_scope_epsilon),
            "max_records_per_person": self.max_records_per_person,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrivacyPolicy:
        return cls(
            epsilon=float(data["epsilon"]),
            delta=float(data["delta"]),
            unit=str(data.get("unit", "record")),
            adjacency=str(data.get("adjacency", "add_remove")),
            accountant=str(data.get("accountant", "pld")),
            per_scope_epsilon=dict(data.get("per_scope_epsilon", {})),
            max_records_per_person=int(data.get("max_records_per_person", 1)),
        )
