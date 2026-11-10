"""Model construction exactly as in Trial 07/27 (comparison only, see the package docstring)."""

from __future__ import annotations

from obscuradp.legacy.config import LegacyTrialConfig
from obscuradp.models import TabularMLP
from obscuradp.state import State, get_state


def new_model(
    cfg: LegacyTrialConfig, in_features: int, dataset: str = "stroke", width_scale: int = 1
) -> TabularMLP:
    """Build the research network for ``dataset``: two hidden layers and a sigmoid output."""
    hidden = cfg.hidden_for(dataset, width_scale)
    return TabularMLP(in_features, hidden, probability_output=True)


def state_template(
    cfg: LegacyTrialConfig, in_features: int, dataset: str = "stroke", width_scale: int = 1
) -> State:
    """State of a freshly built model, used for its keys and shapes."""
    return get_state(new_model(cfg, in_features, dataset=dataset, width_scale=width_scale))
