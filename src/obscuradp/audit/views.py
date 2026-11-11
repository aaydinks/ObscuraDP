"""What one attacker gets to see of a round.

``Simulation.run(observer=...)`` hands out a ``RoundRecord`` with everything that was sent
in a round. No single party sees all of that. An audit that plays an attacker has to cut
the record down first, and this module is the only place where that is done:

- ``server_view``: the server, together with any participants that work with it. Where
  messages go through the secure sum, the server sees the opened sum and nothing else; the
  colluding participants add their own messages to that. Where messages are sent in the
  clear, the server reads all of them.
- ``model_observer_view``: someone who only gets the shared model after every round, which
  is as much as knowing by how much each round moved it. This is the attacker of a plan
  whose server is trusted.

An attack may use a view and anything it knows from elsewhere (the rows of other
participants, the public settings). It may not look at the record itself.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from obscuradp.sim.simulation import RoundRecord
from obscuradp.state import State

SERVER = "server"
MODEL_OBSERVER = "model observer"


@dataclass(frozen=True)
class AttackerView:
    """One round as one attacker sees it.

    Attributes:
        attacker: ``SERVER`` or ``MODEL_OBSERVER``.
        round: The round's number, counted from 0.
        kind: What the messages of this round are (see ``obscuradp.sim.RoundRecord``).
        start: The shared model at the start of the round. Every attacker knows it.
        messages: The single messages this attacker can read, by participant.
        total: What moved the shared model in this round, or ``None`` for a round that
            failed.
        divisor: The public number ``total`` is divided by before it moves the model.
        colluders: The participants that work with this attacker.
    """

    attacker: str
    round: int
    kind: str
    start: State
    messages: Mapping[str, np.ndarray[Any, Any]]
    total: np.ndarray[Any, Any] | None
    divisor: float
    colluders: frozenset[str]

    def honest_total(self) -> np.ndarray[Any, Any] | None:
        """``total`` without the messages of the colluders, who know their own."""
        if self.total is None:
            return None
        rest = self.total.astype(np.float64, copy=True)
        for name in self.colluders:
            if name in self.messages:
                rest -= self.messages[name]
        return rest


def server_view(record: RoundRecord, colluders: Collection[str] = ()) -> AttackerView:
    """The round as the server sees it, together with the participants in ``colluders``.

    Args:
        record: The full record of the round.
        colluders: Names of the participants that share what they know with the server.
    """
    working_together = frozenset(colluders)
    if record.masked:
        visible = {
            name: message for name, message in record.messages.items() if name in working_together
        }
    else:
        visible = dict(record.messages)
    return AttackerView(
        attacker=SERVER,
        round=record.round,
        kind=record.kind,
        start=record.start,
        messages=MappingProxyType(visible),
        total=record.total,
        divisor=record.divisor,
        colluders=working_together,
    )


def model_observer_view(record: RoundRecord) -> AttackerView:
    """The round as someone sees it who only gets the shared model after every round."""
    return AttackerView(
        attacker=MODEL_OBSERVER,
        round=record.round,
        kind=record.kind,
        start=record.start,
        messages=MappingProxyType({}),
        total=record.total,
        divisor=record.divisor,
        colluders=frozenset(),
    )
