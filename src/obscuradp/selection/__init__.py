"""Choosing which numbers of a model update get the stronger protection."""

from obscuradp.selection.masks import (
    build_mask,
    mask_agreement,
    random_jaccard_expected,
    random_mask,
    score_rank_correlation,
    tensor_selection,
    tensor_selection_agreement,
)
from obscuradp.selection.policy import (
    mask_from_private_data,
    mask_from_scores,
    private_scores,
    public_mask,
)

__all__ = [
    "build_mask",
    "mask_agreement",
    "mask_from_private_data",
    "mask_from_scores",
    "private_scores",
    "public_mask",
    "random_jaccard_expected",
    "random_mask",
    "score_rank_correlation",
    "tensor_selection",
    "tensor_selection_agreement",
]
