"""Feed-forward network for tabular stroke prediction.

input -> 64 -> 32 -> 1 (sigmoid).  Trained with BCELoss.
No BatchNorm (Opacus-incompatible); plain Linear+ReLU is fully DP-SGD friendly.
"""
import torch
import torch.nn as nn

from . import config


class FFNN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, config.HIDDEN1),
            nn.ReLU(),
            nn.Linear(config.HIDDEN1, config.HIDDEN2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def new_model(in_features):
    return FFNN(in_features)


def get_weights(model):
    """Return a CPU state_dict of plain tensors (handles Opacus '_module.' prefix)."""
    sd = model.state_dict()
    clean = {}
    for k, v in sd.items():
        k2 = k.replace("_module.", "")
        clean[k2] = v.detach().cpu().clone()
    return clean


def set_weights(model, state):
    model.load_state_dict(state)
