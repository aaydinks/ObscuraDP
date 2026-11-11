"""What runs on a participant's own machine."""

from obscuradp.client.trainer import (
    RoundStats,
    clipped_model_change,
    clipped_sum_step,
    flatten_gradients,
    integer_sum_step,
    private_sum_step,
    train_plain_round,
    train_private_round,
)

__all__ = [
    "RoundStats",
    "clipped_model_change",
    "clipped_sum_step",
    "flatten_gradients",
    "integer_sum_step",
    "private_sum_step",
    "train_plain_round",
    "train_private_round",
]
