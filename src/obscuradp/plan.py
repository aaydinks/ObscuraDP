"""What a run does to protect data: the plan.

A ``PrivacyPolicy`` says how much may leak. A ``Plan`` says by which means the run stays
inside that limit. Keeping them apart means the same policy can be met by different means
(noise added by each participant today; other modes in later versions) without touching
user code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from obscuradp.imbalance import Imbalance, NoRebalancing


@dataclass(frozen=True)
class LocalDPSGD:
    """Each participant trains with private steps and adds all the noise itself.

    What leaves a participant is already protected at record level, whoever receives it:
    the server, the other participants, and anyone who later sees the model.

    Args:
        clip_norm: The clipping norm $C$: the most any single record can move one step.
    """

    clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if not self.clip_norm > 0:
            raise ValueError("clip_norm must be positive")

    def protection(self) -> tuple[str, ...]:
        """Who is protected from whom, in plain words."""
        return (
            "Each participant adds its own noise before anything leaves its machine. Its "
            "training records are protected, at the stated budget, from the server, from the "
            "other participants and from anyone who sees a model.",
            "The updates themselves travel in the clear: the server sees each participant's "
            "noisy update.",
        )


@dataclass(frozen=True)
class Plan:
    """The means a run uses.

    Args:
        dp: How privacy noise is added. ``None`` means no privacy protection at all (a
            baseline for comparison).
        imbalance: How a rare outcome is handled (see ``obscuradp.imbalance``).
        proximal_mu: Strength of the FedProx term that keeps local training close to the
            shared model. 0 gives plain federated averaging.
    """

    dp: LocalDPSGD | None = field(default_factory=LocalDPSGD)
    imbalance: Imbalance = field(default_factory=NoRebalancing)
    proximal_mu: float = 0.0

    def __post_init__(self) -> None:
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu must not be negative")
