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

__all__ = [
    "build_mask",
    "mask_agreement",
    "random_jaccard_expected",
    "random_mask",
    "score_rank_correlation",
    "tensor_selection",
    "tensor_selection_agreement",
]
