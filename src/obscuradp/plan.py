"""What a run does to protect data: the plan.

A ``PrivacyPolicy`` says how much may leak. A ``Plan`` says by which means the run stays
inside that limit. Keeping them apart means the same policy can be met by different means
without touching user code. Three ways of adding the noise exist:

- ``LocalDPSGD``: every participant adds all the noise its records need. Nothing else has
  to be trusted.
- ``DistributedDPSGD``: every participant adds a slice of the noise and the slices add up
  inside a secure sum. Less noise in total, but the guarantee now depends on the secure sum
  and on how many participants pool what they know with the server.
- ``CentralDPSGD``: the server sees clean sums and adds the noise. A baseline for
  comparisons that needs a trusted server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from obscuradp.crypto.pairwise_masking import PairwiseMasking
from obscuradp.imbalance import Imbalance, NoRebalancing
from obscuradp.mechanisms.shared_noise import honest_shares, share_noise_multiplier


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

    def protection(self, *, updates_in_the_clear: bool = True) -> tuple[str, ...]:
        """Who is protected from whom, in plain words."""
        noise = (
            "Each participant adds its own noise before anything leaves its machine. Its "
            "training records are protected, at the stated budget, from the server, from the "
            "other participants and from anyone who sees a model."
        )
        if not updates_in_the_clear:
            return (noise,)
        return (
            noise,
            "The updates themselves travel in the clear: the server sees each participant's "
            "noisy update.",
        )


@dataclass(frozen=True, kw_only=True)
class DistributedDPSGD:
    """The participants share the noise: each adds a slice, a secure sum adds the slices up.

    Every round is one step. Each participant computes one clipped gradient sum at the
    shared model, adds noise with standard deviation $\\sigma C / \\sqrt{N - t - d}$ and
    sends the result through the secure sum. The server only ever sees the total, which
    carries at least the full noise $\\sigma C$ as long as no more than $t$ participants
    give their slices away and no more than $d$ are missing.

    A single message carries only a slice of the noise. This mode is therefore refused
    without a secure sum, and no setting can force it.

    Args:
        max_colluding: $t$, the largest number of participants assumed to pool what they
            know with the server. It has no default: the caller must decide. A participant
            always knows its own slice, so the stated budget holds against the other
            participants only if this is at least 1.
        max_dropouts: $d$, the largest number of participants assumed to be missing from a
            sum. The secure sum of this version cannot open a sum with anyone missing (the
            round is repeated), so a value above 0 only makes every slice larger.
        clip_norm: The clipping norm $C$: the most any single record can move one step.
    """

    max_colluding: int
    max_dropouts: int = 0
    clip_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_colluding", "max_dropouts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative whole number")
        if not self.clip_norm > 0:
            raise ValueError("clip_norm must be positive")

    def honest_shares(self, participants: int) -> int:
        """Slices the guarantee counts on among ``participants``: $N - t - d$."""
        return honest_shares(participants, self.max_colluding, self.max_dropouts)

    def share_noise_multiplier(self, noise_multiplier: float, participants: int) -> float:
        """Noise multiplier of one slice: $\\sigma / \\sqrt{N - t - d}$."""
        return share_noise_multiplier(
            noise_multiplier, participants, self.max_colluding, self.max_dropouts
        )

    def protection(self, participants: int) -> tuple[str, ...]:
        """Who is protected from whom, in plain words."""
        t, d = self.max_colluding, self.max_dropouts
        lines = [
            f"Each of the {participants} participants adds a slice of the noise, and the "
            "slices are added up inside the secure sum. One message alone carries far too "
            "little noise; the protection comes from the sum.",
            "Training records are protected, at the stated budget, from the server and from "
            f"anyone who sees a model, as long as at most {t} of the participants pool what "
            f"they know with the server and at most {d} are missing from a sum. The table in "
            "this report shows the budget for other numbers.",
        ]
        if t == 0:
            lines.append(
                "A participant knows its own slice of the noise. Against a single curious "
                "participant the table row for 1 applies, not the stated budget."
            )
        elif t == 1:
            lines.append(
                "The same budget holds against any one participant, with or without the server."
            )
        else:
            lines.append(
                f"The same budget holds against any group of up to {t} participants, with or "
                "without the server."
            )
        lines += [
            "The server receives every message hidden under masks that only cancel in the "
            "sum. It can read the sum of all messages and no single one, provided it passes "
            "the participants' public keys on unchanged.",
            "If a participant's message is missing, the sum cannot be opened, the model does "
            "not move, and the round still counts against the budget of those who sent one.",
        ]
        return tuple(lines)


@dataclass(frozen=True)
class CentralDPSGD:
    """A trusted server adds the noise. **A baseline for comparisons only.**

    Every round is one step. The participants send their clipped gradient sums without
    noise, and the server adds noise $\\sigma C$ to the total. This is the least noise any
    method needs for a given budget, which makes it the upper limit other modes are compared
    with. The server sees every participant's clean sums, so it must be trusted completely.
    A run in this mode needs ``allow_unsafe=True`` and its report says why.

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
            "The server sees every participant's gradient sums without noise. Records are "
            "not protected from the server.",
            "The server adds the noise before it publishes a model. Records are protected, "
            "at the stated budget, from anyone who sees only the models, including the other "
            "participants.",
        )


@dataclass(frozen=True)
class Plan:
    """The means a run uses.

    Args:
        dp: How privacy noise is added: ``LocalDPSGD``, ``DistributedDPSGD`` or
            ``CentralDPSGD``. ``None`` means no privacy protection at all (a baseline for
            comparison).
        imbalance: How a rare outcome is handled (see ``obscuradp.imbalance``).
        secure_sum: How individual updates are hidden from the server. ``None`` sends them
            in the clear (with ``LocalDPSGD`` they still carry each participant's own
            noise). ``DistributedDPSGD`` requires it and plans the rounding itself, so it
            takes ``PairwiseMasking()`` with its default settings only.
        proximal_mu: Strength of the FedProx term that keeps local training close to the
            shared model. 0 gives plain federated averaging.

    Raises:
        ValueError: For ``DistributedDPSGD`` without a secure sum (a single message would be
            protected by a slice of the noise only; ``allow_unsafe`` cannot override this),
            and for ``CentralDPSGD`` with one.
    """

    dp: LocalDPSGD | DistributedDPSGD | CentralDPSGD | None = field(default_factory=LocalDPSGD)
    imbalance: Imbalance = field(default_factory=NoRebalancing)
    secure_sum: PairwiseMasking | None = None
    proximal_mu: float = 0.0

    def __post_init__(self) -> None:
        if self.proximal_mu < 0:
            raise ValueError("proximal_mu must not be negative")
        if isinstance(self.dp, DistributedDPSGD):
            if self.secure_sum is None:
                raise ValueError(
                    "DistributedDPSGD needs a secure sum: each message carries only a slice "
                    "of the noise and must never be seen on its own. Add "
                    "secure_sum=PairwiseMasking(), or use LocalDPSGD. No setting overrides "
                    "this."
                )
            if self.secure_sum != PairwiseMasking():
                raise ValueError(
                    "DistributedDPSGD plans the range and rounding of its messages itself "
                    "(cutting values off would remove noise that others rely on); use "
                    "PairwiseMasking() with its default settings"
                )
        if isinstance(self.dp, CentralDPSGD) and self.secure_sum is not None:
            raise ValueError(
                "CentralDPSGD sends clean sums to a trusted server and takes no secure sum "
                "in this version"
            )
        if self.single_step and self.proximal_mu > 0:
            raise ValueError("proximal_mu has no effect with one step per round; leave it at 0")

    @property
    def single_step(self) -> bool:
        """True for the modes that take one step per round at the shared model."""
        return isinstance(self.dp, (DistributedDPSGD, CentralDPSGD))
