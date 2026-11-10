"""One private gradient step (DP-SGD).

The step works like this. Compute the gradient of the loss for every sampled record
separately. Shrink any record's gradient whose length exceeds the clipping norm $C$, so no
single record can move the sum by more than $C$. Add the shrunk gradients up, add Gaussian
noise with standard deviation $\\sigma C$ to the sum, and divide by the expected batch size
$q \\cdot n$.

Three details matter for the guarantee and are easy to get wrong:

- **Joint clipping.** The length is taken over all parameters of the model together, not
  layer by layer.
- **Fixed divisor.** The sum is divided by the expected batch size, a number that does not
  depend on which records were drawn. Dividing by the actual batch size would leak it.
- **Empty batches.** If no record was drawn, the step still releases noise. Skipping it
  would reveal that the batch was empty.

Models with layers that mix records (batch normalization) are refused, because then one
record's gradient depends on the others and clipping no longer limits its influence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from obscuradp.mechanisms.noise import gaussian_noise

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Gradients = dict[str, torch.Tensor]

# Gradients are clipped to slightly less than the clipping norm, so that rounding in
# floating-point arithmetic can never leave a record with more influence than was accounted.
CLIP_MARGIN = 1e-6

_MIXING_LAYERS: tuple[type[nn.Module], ...] = (
    nn.modules.batchnorm._BatchNorm,
    nn.modules.batchnorm._LazyNormBase,
)


def check_per_record_model(model: nn.Module) -> None:
    """Refuse models in which one record's output depends on the other records of a batch."""
    for name, module in model.named_modules():
        mixes = isinstance(module, _MIXING_LAYERS) or (
            isinstance(module, nn.modules.instancenorm._InstanceNorm) and module.track_running_stats
        )
        if mixes:
            raise ValueError(
                f"layer {name or '(root)'!r} ({type(module).__name__}) mixes the records of a "
                "batch, so per-record clipping cannot limit one record's influence; use "
                "GroupNorm or LayerNorm instead"
            )


def per_record_gradients(
    model: nn.Module, loss_fn: LossFunction, features: torch.Tensor, targets: torch.Tensor
) -> Gradients:
    """The gradient of the loss for each record separately.

    Args:
        model: The model. Only parameters with ``requires_grad`` are differentiated.
        loss_fn: Called as ``loss_fn(model_output, target)`` on one record at a time (a batch
            of one). Whatever it returns is summed to a single number.
        features: Inputs, one record per row.
        targets: Targets, one per record.

    Returns:
        For every trainable parameter, a tensor whose first dimension runs over the records.
    """
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of records")
    parameters = {
        name: value.detach() for name, value in model.named_parameters() if value.requires_grad
    }
    if features.shape[0] == 0:
        return {name: value.new_zeros((0, *value.shape)) for name, value in parameters.items()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}

    def loss_of_one(
        params: Mapping[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        output = functional_call(model, (dict(params), buffers), (x.unsqueeze(0),))
        result: torch.Tensor = loss_fn(output, y.unsqueeze(0)).sum()
        return result

    gradients: Gradients = vmap(grad(loss_of_one), in_dims=(None, 0, 0), randomness="different")(
        parameters, features, targets
    )
    return gradients


def record_norms(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Length of each record's gradient over all parameters together (64-bit)."""
    squares = [value.double().flatten(1).pow(2).sum(dim=1) for value in gradients.values()]
    if not squares:
        raise ValueError("the model has no trainable parameter")
    return torch.stack(squares).sum(dim=0).sqrt()


def clipped_sum(gradients: Mapping[str, torch.Tensor], clip_norm: float) -> Gradients:
    """Shrink each record's gradient to length at most ``clip_norm`` and add them up.

    A record whose gradient is not a finite number contributes nothing. Whatever the data,
    adding or removing one record changes the result by at most ``clip_norm`` in length.
    """
    if not clip_norm > 0.0:
        raise ValueError("clip_norm must be positive")
    norms = record_norms(gradients)
    finite = torch.isfinite(norms)
    limit = clip_norm * (1.0 - CLIP_MARGIN)
    factors = torch.where(finite, (limit / norms.clamp_min(1e-300)).clamp(max=1.0), 0.0)
    total: Gradients = {}
    for name, value in gradients.items():
        shape = (-1, *([1] * (value.dim() - 1)))
        kept = torch.where(finite.view(shape), value, torch.zeros_like(value))
        total[name] = (kept * factors.to(value.dtype).view(shape)).sum(dim=0)
    return total


def private_gradient(
    model: nn.Module,
    loss_fn: LossFunction,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    clip_norm: float,
    noise_multiplier: float,
    expected_batch_size: float,
    generator: torch.Generator,
) -> Gradients:
    """The noisy average gradient of one step.

    $\\tilde g = \\bigl(\\sum_i \\mathrm{clip}_C(g_i) + z\\bigr) / (q n)$ with
    $z \\sim \\mathcal{N}(0, \\sigma^2 C^2 I)$.

    Args:
        model: The model at its current weights.
        loss_fn: Per-record loss, see ``per_record_gradients``.
        features: The records drawn for this step. May be empty.
        targets: Their targets.
        clip_norm: The clipping norm $C$.
        noise_multiplier: $\\sigma$. Must be positive: a step without noise is not private.
        expected_batch_size: $q \\cdot n$, the same real number at every step.
        generator: The random stream of the noise.
    """
    if not noise_multiplier > 0.0:
        raise ValueError("noise_multiplier must be positive")
    if not expected_batch_size > 0.0:
        raise ValueError("expected_batch_size must be positive")
    total = clipped_sum(per_record_gradients(model, loss_fn, features, targets), clip_norm)
    noisy: Gradients = {}
    for name, value in total.items():
        noise = gaussian_noise(
            value.shape, noise_multiplier * clip_norm, generator, dtype=value.dtype
        )
        noisy[name] = (value + noise.to(value.device)) / expected_batch_size
    return noisy
