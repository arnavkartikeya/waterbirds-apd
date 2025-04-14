from torch import Tensor

from spd.hooks import HookedRootModule
from spd.models.components import TransposedLinearComponent
from spd.module_utils import (
    collect_nested_module_attrs,
    get_nested_module_attr,
    remove_grad_parallel_to_subnetwork_vecs,
)


class SPDModel(HookedRootModule):
    def set_subnet_to_zero(self, subnet_idx: int, has_instance_dim: bool) -> dict[str, Tensor]:
        stored_vals = {}
        for attr_name in ["A", "B"]:
            params = collect_nested_module_attrs(self, attr_name)
            for param_name, param in params.items():
                if self.parent_is_transposed_linear(param_name):
                    continue
                if has_instance_dim:
                    stored_vals[param_name] = param.data[:, subnet_idx, :, :].detach().clone()
                    param.data[:, subnet_idx, :, :] = 0.0
                else:
                    stored_vals[param_name] = param.data[subnet_idx, :, :].detach().clone()
                    param.data[subnet_idx, :, :] = 0.0
        return stored_vals

    def restore_subnet(
        self, subnet_idx: int, stored_vals: dict[str, Tensor], has_instance_dim: bool
    ) -> None:
        for name, val in stored_vals.items():
            param = get_nested_module_attr(self, name)
            if has_instance_dim:
                param.data[:, subnet_idx, :, :] = val
            else:
                param.data[subnet_idx, :, :] = val

    def set_As_to_unit_norm(self) -> None:
        """Set all A matrices to unit norm for stability.

        Normalizes over the second last dimension (which is the d_in dimension for A).

        Excludes TransposedLinearComponent matrices.
        """
        params = collect_nested_module_attrs(self, "A")
        for param_name, param in params.items():
            if not self.parent_is_transposed_linear(param_name):
                param.data /= param.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        params = collect_nested_module_attrs(self, "A")
        for param_name, param in params.items():
            if not self.parent_is_transposed_linear(param_name):
                assert param.grad is not None
                remove_grad_parallel_to_subnetwork_vecs(param.data, param.grad)

    def parent_is_transposed_linear(self, param_name: str) -> bool:
        """Check if the parent module of the given parameter is a TransposedLinearComponent.

        We use this to avoid operations on a tensor which is tied to another tensor.
        """
        parent_module_name = ".".join(param_name.split(".")[:-1])
        parent_module = get_nested_module_attr(self, parent_module_name)
        return isinstance(parent_module, TransposedLinearComponent)
