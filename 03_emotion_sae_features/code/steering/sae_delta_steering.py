"""Model-agnostic reference hooks for single-feature SAE interventions.

The SAE object must expose ``encode(hidden)`` and ``decode(features)``.
The hook target must be the same residual-stream site on which the SAE was
trained. The code supports Transformer layers that return either a tensor or
a tuple whose first element is the residual tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Intervention = Literal["additive", "multiplicative"]


@dataclass
class SAEDeltaSteering:
    """Callable forward hook that preserves the SAE reconstruction error.

    Additive validation:
        z'_j = z_j + dose * natural_scale

    Multiplicative task intervention:
        z'_j = multiplier * z_j

    Both modes return:
        h' = h + decode(z') - decode(z)
    """

    sae: object
    feature_id: int
    intervention: Intervention
    value: float
    natural_scale: float = 1.0

    def _edit(self, features: torch.Tensor) -> torch.Tensor:
        edited = features.clone()
        if self.intervention == "additive":
            edited[..., self.feature_id] += self.value * self.natural_scale
        elif self.intervention == "multiplicative":
            edited[..., self.feature_id] *= self.value
        else:
            raise ValueError(f"Unknown intervention: {self.intervention}")
        return edited

    def __call__(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        sae_dtype = getattr(getattr(self.sae, "W_enc", None), "dtype", hidden.dtype)
        features = self.sae.encode(hidden.to(dtype=sae_dtype))
        edited = self._edit(features)
        delta = self.sae.decode(edited) - self.sae.decode(features)
        steered = hidden + delta.to(dtype=hidden.dtype, device=hidden.device)
        return (steered,) + output[1:] if isinstance(output, tuple) else steered


def assert_identity(sae, hidden: torch.Tensor, feature_id: int) -> None:
    """Check that multiplicative M=1 produces an exact zero intervention."""
    hook = SAEDeltaSteering(
        sae=sae,
        feature_id=feature_id,
        intervention="multiplicative",
        value=1.0,
    )
    steered = hook(None, None, hidden)
    if not torch.equal(steered, hidden):
        maximum_error = (steered - hidden).abs().max().item()
        raise AssertionError(f"M=1 identity failed; max abs error={maximum_error}")

