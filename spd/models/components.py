from typing import Any, Literal

import einops
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.hooks import HookPoint
from spd.module_utils import init_param_


class Linear(nn.Module):
    """A linear transformation with an optional n_instances dimension."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_instances: int | None = None,
        init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
        init_scale: float = 1.0,
    ):
        super().__init__()
        shape = (n_instances, d_in, d_out) if n_instances is not None else (d_in, d_out)
        self.weight = nn.Parameter(torch.empty(shape))
        init_param_(self.weight, scale=init_scale, init_type=init_type)

        self.hook_pre = HookPoint()  # (batch ... d_in)
        self.hook_post = HookPoint()  # (batch ... d_out)

    def forward(
        self, x: Float[Tensor, "batch ... d_in"], *args: Any, **kwargs: Any
    ) -> Float[Tensor, "batch ... d_out"]:
        x = self.hook_pre(x)
        out = einops.einsum(x, self.weight, "batch ... d_in, ... d_in d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    The weight matrix W is decomposed as W = A @ B, where A and B are learned parameters.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        C: int,
        n_instances: int | None = None,
        init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
        init_scale: float = 1.0,
        m: int | None = None,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.C = C
        self.m = min(d_in, d_out) if m is None else m

        # Initialize A and B matrices
        shape_A = (n_instances, C, d_in, self.m) if n_instances is not None else (C, d_in, self.m)
        shape_B = (n_instances, C, self.m, d_out) if n_instances is not None else (C, self.m, d_out)
        self.A = nn.Parameter(torch.empty(shape_A))
        self.B = nn.Parameter(torch.empty(shape_B))
        self.hook_pre = HookPoint()  # (batch d_in) or (batch n_instances d_in)
        self.hook_component_acts = HookPoint()  # (batch C d_out) or (batch n_instances C d_out)
        self.hook_post = HookPoint()  # (batch d_out) or (batch n_instances d_out)

        init_param_(self.A, scale=init_scale, init_type=init_type)
        init_param_(self.B, scale=init_scale, init_type=init_type)

    @property
    def component_weights(self) -> Float[Tensor, "... C d_in d_out"]:
        """A @ B before summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... C d_in m, ... C m d_out -> ... C d_in d_out")

    @property
    def weight(self) -> Float[Tensor, "... d_in d_out"]:
        """A @ B after summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... C d_in m, ... C m d_out -> ... d_in d_out")

    def forward(
        self,
        x: Float[Tensor, "batch ... d_in"],
        topk_mask: Bool[Tensor, "batch ... C"] | None = None,
    ) -> Float[Tensor, "batch ... d_out"]:
        """Forward pass through A and B matrices which make up the component for this layer.

        Args:
            x: Input tensor
            topk_mask: Boolean tensor indicating which subnetworks to keep
        Returns:
            output: The summed output across all subnetworks
        """
        x = self.hook_pre(x)

        # First multiply by A to get to intermediate dimension m
        inner_acts = einops.einsum(x, self.A, "batch ... d_in, ... C d_in m -> batch ... C m")
        if topk_mask is not None:
            assert topk_mask.shape == inner_acts.shape[:-1]
            inner_acts = einops.einsum(
                inner_acts, topk_mask, "batch ... C m, batch ... C -> batch ... C m"
            )

        # Then multiply by B to get to output dimension
        component_acts = einops.einsum(
            inner_acts, self.B, "batch ... C m, ... C m d_out -> batch ... C d_out"
        )
        self.hook_component_acts(component_acts)

        # Sum over subnetwork dimension
        out = einops.einsum(component_acts, "batch ... C d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out


class TransposedLinear(Linear):
    """Linear layer that uses a transposed weight from another Linear layer.

    We use 'd_in' and 'd_out' to refer to the dimensions of the original Linear layer.
    """

    def __init__(self, original_weight: nn.Parameter):
        # Copy the relevant parts from Linear.__init__. Don't copy operations that will call
        # TransposedLinear.weight.
        nn.Module.__init__(self)
        self.hook_pre = HookPoint()  # (batch ... d_out)
        self.hook_post = HookPoint()  # (batch ... d_in)

        self.register_buffer("original_weight", original_weight, persistent=False)

    @property
    def weight(self) -> Float[Tensor, "... d_out d_in"]:
        return einops.rearrange(self.original_weight, "... d_in d_out -> ... d_out d_in")


class TransposedLinearComponent(LinearComponent):
    """LinearComponent that uses a transposed weight from another LinearComponent.

    We use 'd_in' and 'd_out' to refer to the dimensions of the original LinearComponent.
    """

    def __init__(self, original_A: nn.Parameter, original_B: nn.Parameter):
        # Copy the relevant parts from LinearComponent.__init__. Don't copy operations that will
        # call TransposedLinear.A or TransposedLinear.B.
        nn.Module.__init__(self)
        self.n_instances, self.C, _, self.m = original_A.shape

        self.hook_pre = HookPoint()  # (batch ... d_out)
        self.hook_component_acts = HookPoint()  # (batch ... C d_in)
        self.hook_post = HookPoint()  # (batch ... d_in)

        self.register_buffer("original_A", original_A, persistent=False)
        self.register_buffer("original_B", original_B, persistent=False)

    @property
    def A(self) -> Float[Tensor, "... C d_out m"]:
        # New A is the transpose of the original B
        return einops.rearrange(self.original_B, "... C m d_out -> ... C d_out m")

    @property
    def B(self) -> Float[Tensor, "... C d_in m"]:
        # New B is the transpose of the original A
        return einops.rearrange(self.original_A, "... C d_in m -> ... C m d_in")

    @property
    def component_weights(self) -> Float[Tensor, "... C d_out d_in"]:
        """A @ B before summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... C d_out m, ... C m d_in -> ... C d_out d_in")

    @property
    def weight(self) -> Float[Tensor, "... d_out d_in"]:
        """A @ B after summing over the subnetwork dimension."""
        return einops.einsum(self.A, self.B, "... C d_out m, ... C m d_in -> ... d_out d_in")
