"""Two-layer MLP for binary loan-approval prediction.

input -> 32 -> 16 -> 1 (sigmoid).  Trained with BCELoss.
No BatchNorm (Opacus-incompatible); plain Linear+ReLU is fully DP-SGD friendly
and keeps the parameter vector flat-and-encryptable for the HE arm.
"""
import numpy as np
import torch
import torch.nn as nn

from . import config


class MLP(nn.Module):
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
    return MLP(in_features)


def get_weights(model):
    """Return a CPU state_dict of plain tensors (handles Opacus '_module.' prefix)."""
    sd = model.state_dict()
    return {k.replace("_module.", ""): v.detach().cpu().clone() for k, v in sd.items()}


def set_weights(model, state):
    model.load_state_dict(state)


# --- flatten / unflatten helpers (used by the HE arm) ---

def flatten_state(state):
    """Concatenate a state_dict into a single 1-D float64 numpy vector."""
    return np.concatenate([state[k].numpy().ravel().astype(np.float64)
                           for k in state]).astype(np.float64)


def unflatten_state(flat, template):
    """Rebuild a state_dict from a flat vector, using `template` for shapes/keys."""
    out, i = {}, 0
    for k, v in template.items():
        n = v.numel()
        chunk = np.asarray(flat[i:i + n], dtype=np.float32).reshape(v.shape)
        out[k] = torch.from_numpy(chunk)
        i += n
    return out
