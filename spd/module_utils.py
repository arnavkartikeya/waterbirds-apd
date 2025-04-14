from functools import reduce
from typing import Any, Literal

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def get_nested_module_attr(module: nn.Module, access_string: str) -> Any:
    """Access a specific attribute by its full, path-like name.

    Taken from https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8

    Args:
        module: The module to search through.
        access_string: The full name of the nested attribute to access, with each object separated
            by periods (e.g. "linear1.A").
    """
    names = access_string.split(".")
    try:
        mod = reduce(getattr, names, module)
    except AttributeError as err:
        raise AttributeError(f"{module} does not have nested attribute {access_string}") from err
    return mod


def collect_nested_module_attrs(
    module: nn.Module,
    attr_name: str,
    include_attr_name: bool = True,
) -> dict[str, Tensor]:
    """Collect all attributes matching attr_name from a module and all its submodules.

    Args:
        module: The module to collect attributes from
        attr_name: Name of the attributes to collect from module and all submodules. E.g. "A".
        include_attr_name: If True, the attribute name is included in the key of the dictionary.
            E.g. if attr_name is "A", the key will be "root.A" or "linear1.A".

    Returns:
        Dictionary mapping module names to their attribute values

    Raises:
        - ValueError: If no modules with the specified attribute are found
        - ValueError: If the attribute is not a tensor
    """
    attributes: dict[str, Tensor] = {}

    all_modules = module.named_modules()
    for name, submodule in all_modules:
        if hasattr(submodule, attr_name):
            # For root module, name will be empty string
            submodule_attr = getattr(submodule, attr_name)
            if not isinstance(submodule_attr, Tensor):
                raise ValueError(
                    f"Attribute '{attr_name}' is not a tensor. "
                    f"Available modules: {[name for name, _ in all_modules]}"
                )
            key = name + "." + attr_name if include_attr_name else name
            attributes[key] = submodule_attr

    if not attributes:
        raise ValueError(
            f"No modules found with attribute '{attr_name}'. "
            f"Available modules: {[name for name, _ in all_modules]}"
        )

    return attributes


@torch.inference_mode()
def remove_grad_parallel_to_subnetwork_vecs(
    A: Float[Tensor, "... d_in m"], A_grad: Float[Tensor, "... d_in m"]
) -> None:
    """Modify the gradient by subtracting it's component parallel to the activation.

    I.e. subtract the projection of the gradient vector onto the activation vector.

    This is to stop Adam from changing the norm of A. Note that this will not completely prevent
    Adam from changing the norm due to Adam's (m/(sqrt(v) + eps)) term not preserving the norm
    direction.
    """
    parallel_component = einops.einsum(A_grad, A, "... d_in m, ... d_in m -> ... m")
    A_grad -= einops.einsum(parallel_component, A, "... m, ... d_in m -> ... d_in m")


def init_param_(
    param: torch.Tensor,
    scale: float = 1.0,
    init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
) -> None:
    if init_type == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(param)
        with torch.no_grad():
            param.mul_(scale)
    elif init_type == "xavier_normal":
        torch.nn.init.xavier_normal_(param, gain=scale)
