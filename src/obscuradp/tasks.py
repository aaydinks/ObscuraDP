"""What a model is asked to predict: a yes/no outcome, or one of several classes.

``Simulation`` handles yes/no outcomes unless it is told otherwise (``YesNo``, the
default). ``ManyClasses`` is for outcomes with more than two possible values, such as a
handwritten character or the next letter of a text. It changes three things: labels are
whole numbers from 0 to ``classes - 1``, the model returns one raw score per class, and
the run is scored by accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YesNo:
    """A yes/no outcome: labels 0 or 1, one number per row from the model."""


@dataclass(frozen=True)
class ManyClasses:
    """An outcome with ``classes`` possible values.

    Args:
        classes: The number of possible values. Labels are 0 to ``classes - 1`` and the
            model returns ``classes`` raw scores per row (no final softmax).
    """

    classes: int

    def __post_init__(self) -> None:
        if isinstance(self.classes, bool) or not isinstance(self.classes, int):
            raise ValueError("classes must be a whole number")
        if self.classes < 2:
            raise ValueError("classes must be at least 2")


Task = YesNo | ManyClasses
