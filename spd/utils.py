import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import einops
import numpy as np
import torch
import yaml
from jaxtyping import Float
from pydantic import BaseModel, PositiveFloat
from pydantic.v1.utils import deep_update
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from spd.hooks import HookedRootModule
from spd.log import logger
from spd.models.base import SPDModel
from spd.module_utils import collect_nested_module_attrs
from spd.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)
Q = TypeVar("Q")

# Avoid seaborn package installation (sns.color_palette("colorblind").as_hex())
COLOR_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def to_root_path(path: str | Path) -> Path:
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def from_root_path(path: str | Path) -> Path:
    """Converts absolute paths to relative ones, relative to the repo root."""
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        # If the path is not relative to REPO_ROOT, return the original path
        return path


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def load_config(config_path_or_obj: Path | str | T, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def replace_pydantic_model(model: BaseModelType, *updates: dict[str, Any]) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


class DatasetGeneratedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that generates batches by calling the dataset's `generate_batch` method."""

    def __init__(
        self,
        dataset: Dataset[Q],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # assert that dataset has a generate_batch method
        assert hasattr(dataset, "generate_batch")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(  # type: ignore
        self,
    ) -> Iterator[Q]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)  # type: ignore


class BatchedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that unpacks the batch in __getitem__.

    This is used for datasets which generate a whole batch in one call to __getitem__.
    """

    def __init__(
        self,
        dataset: Dataset[Q],
        num_workers: int = 0,
    ):
        super().__init__(dataset, num_workers=num_workers)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        for batch, label in super().__iter__():
            yield batch[0], label[0]


def calc_grad_attributions(
    target_out: Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"],
    pre_weight_acts: dict[
        str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]
    ],
    post_weight_acts: dict[
        str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]
    ],
    component_weights: dict[
        str, Float[Tensor, "C d_in d_out"] | Float[Tensor, "n_instances C d_in d_out"]
    ],
    C: int,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the product of the gradient of the target model output w.r.t. the post acts
    and the inner acts (i.e. the output of each subnetwork before being summed).

    Note that we don't use the component_acts collected from the SPD model, because this includes the
    computational graph of the full model. We only want the subnetwork parameters of the current
    layer to be in the computational graph. To do this, we multiply a detached version of the
    pre_weight_acts by the subnet parameters.

    NOTE: This code may be run in between the training forward pass, and the loss.backward() and
    opt.step() calls; it must not mess with the training. The reason the current implementation is
    fine to run anywhere is that we just use autograd rather than backward which does not
    populate the .grad attributes. Unrelatedly, we use retain_graph=True in a bunch of cases
    where we want to later use the `out` variable in e.g. the loss function.

    Args:
        target_out: The output of the target model.
        pre_weight_acts: The activations at the output of each subnetwork before being summed.
        post_weight_acts: The activations at the output of each layer after being summed.
        component_weights: The component weight matrix at each layer.
        k: The number of components.
    Returns:
        The sum of the (squared) attributions from each output dimension.
    """
    # Ensure that all keys are the same after removing the hook suffixes
    post_weight_act_names = [C.removesuffix(".hook_post") for C in post_weight_acts]
    pre_weight_act_names = [C.removesuffix(".hook_pre") for C in pre_weight_acts]
    component_weight_names = list(component_weights.keys())
    assert set(post_weight_act_names) == set(pre_weight_act_names) == set(component_weight_names)

    attr_shape = target_out.shape[:-1] + (C,)  # (batch, C) or (batch, n_instances, C)
    attribution_scores: Float[Tensor, "batch ... C"] = torch.zeros(
        attr_shape, device=target_out.device, dtype=target_out.dtype
    )
    component_acts = {}
    for param_name in pre_weight_act_names:
        component_acts[param_name] = einops.einsum(
            pre_weight_acts[param_name + ".hook_pre"].detach().clone(),
            component_weights[param_name],
            "... d_in, ... C d_in d_out -> ... C d_out",
        )
    out_dim = target_out.shape[-1]
    for feature_idx in range(out_dim):
        feature_attributions: Float[Tensor, "batch ... C"] = torch.zeros(
            attr_shape, device=target_out.device, dtype=target_out.dtype
        )
        grad_post_weight_acts: tuple[Float[Tensor, "batch ... d_out"], ...] = torch.autograd.grad(
            target_out[..., feature_idx].sum(), list(post_weight_acts.values()), retain_graph=True
        )
        for i, param_name in enumerate(post_weight_act_names):
            feature_attributions += einops.einsum(
                grad_post_weight_acts[i],
                component_acts[param_name],
                "... d_out ,... C d_out -> ... C",
            )

        attribution_scores += feature_attributions**2

    return attribution_scores


def collect_subnetwork_attributions(
    spd_model: SPDModel,
    target_model: HookedRootModule,
    device: str,
    n_instances: int | None = None,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """
    Collect subnetwork attributions.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the attributions.

    Args:
        spd_model: The model to collect attributions on.
        target_model: The target model to collect attributions on.
        pre_weight_acts: The activations after the parameter matrix in the target model.
        device: The device to run computations on.
        n_instances: The number of instances in the batch.

    Returns:
        The attribution scores.
    """
    test_batch = torch.eye(spd_model.n_features, device=device)
    if n_instances is not None:
        test_batch = einops.repeat(
            test_batch, "batch n_features -> batch n_instances n_features", n_instances=n_instances
        )
    target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
    target_out, target_cache = target_model.run_with_cache(
        test_batch, names_filter=target_cache_filter
    )
    component_weights = collect_nested_module_attrs(
        spd_model, attr_name="component_weights", include_attr_name=False
    )
    attribution_scores = calc_grad_attributions(
        target_out=target_out,
        component_weights=component_weights,
        pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
        post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
        C=spd_model.C,
    )
    return attribution_scores


@torch.inference_mode()
def calc_ablation_attributions(
    spd_model: SPDModel,
    batch: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    out: Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"],
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the attributions by ablating each subnetwork one at a time."""

    attr_shape = out.shape[:-1] + (spd_model.C,)  # (batch, C) or (batch, n_instances, C)
    has_instance_dim = len(out.shape) == 3
    attributions = torch.zeros(attr_shape, device=out.device, dtype=out.dtype)
    for subnet_idx in range(spd_model.C):
        stored_vals = spd_model.set_subnet_to_zero(subnet_idx, has_instance_dim)
        ablation_out, _, _ = spd_model(batch)
        out_recon = ((out - ablation_out) ** 2).mean(dim=-1)
        attributions[..., subnet_idx] = out_recon
        spd_model.restore_subnet(subnet_idx, stored_vals, has_instance_dim)
    return attributions


def calc_activation_attributions(
    component_acts: dict[
        str, Float[Tensor, "batch C d_out"] | Float[Tensor, "batch n_instances C d_out"]
    ],
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the attributions by taking the L2 norm of the activations in each subnetwork.

    Args:
        component_acts: The activations at the output of each subnetwork before being summed.
    Returns:
        The attributions for each subnetwork.
    """
    first_param = component_acts[next(iter(component_acts.keys()))]
    assert len(first_param.shape) in (3, 4)

    attribution_scores: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"] = (
        torch.zeros(first_param.shape[:-1], device=first_param.device, dtype=first_param.dtype)
    )
    for param_matrix in component_acts.values():
        attribution_scores += param_matrix.pow(2).sum(dim=-1)
    return attribution_scores


def calculate_attributions(
    model: SPDModel,
    batch: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    out: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    target_out: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    pre_weight_acts: dict[
        str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]
    ],
    post_weight_acts: dict[
        str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]
    ],
    component_acts: dict[str, Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]],
    attribution_type: Literal["ablation", "gradient", "activation"],
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    attributions = None
    if attribution_type == "ablation":
        attributions = calc_ablation_attributions(spd_model=model, batch=batch, out=out)
    elif attribution_type == "gradient":
        component_weights = collect_nested_module_attrs(
            model, attr_name="component_weights", include_attr_name=False
        )
        attributions = calc_grad_attributions(
            target_out=target_out,
            pre_weight_acts=pre_weight_acts,
            post_weight_acts=post_weight_acts,
            component_weights=component_weights,
            C=model.C,
        )
    elif attribution_type == "activation":
        attributions = calc_activation_attributions(component_acts=component_acts)
    else:
        raise ValueError(f"Invalid attribution type: {attribution_type}")
    return attributions


def calc_topk_mask(
    attribution_scores: Float[Tensor, "batch ... C"],
    topk: float,
    batch_topk: bool,
) -> Float[Tensor, "batch ... C"]:
    """Calculate the top-k mask.

    Args:
        attribution_scores: The attribution scores to calculate the top-k mask for.
        topk: The number of top-k elements to select. If `batch_topk` is True, this is multiplied
            by the batch size to get the number of top-k elements over the whole batch.
        batch_topk: If True, the top-k mask is calculated over the concatenated batch and k
            dimensions.

    Returns:
        The top-k mask.
    """
    batch_size = attribution_scores.shape[0]
    topk = int(topk * batch_size) if batch_topk else int(topk)

    if batch_topk:
        attribution_scores = einops.rearrange(attribution_scores, "b ... C -> ... (b C)")

    topk_indices = attribution_scores.topk(topk, dim=-1).indices
    topk_mask = torch.zeros_like(attribution_scores, dtype=torch.bool)
    topk_mask.scatter_(dim=-1, index=topk_indices, value=True)

    if batch_topk:
        topk_mask = einops.rearrange(topk_mask, "... (b C) -> b ... C", b=batch_size)

    return topk_mask


class SPDOutputs(NamedTuple):
    target_model_output: (
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"]
    )
    spd_model_output: (
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"]
    )
    spd_topk_model_output: (
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"]
    )
    layer_acts: dict[str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]]
    component_acts: dict[
        str, Float[Tensor, "batch C d_out"] | Float[Tensor, "batch n_instances C d_out"]
    ]
    attribution_scores: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]
    topk_mask: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]


def run_spd_forward_pass(
    spd_model: SPDModel,
    target_model: HookedRootModule,
    input_array: Float[Tensor, "batch n_inputs"],
    attribution_type: Literal["gradient", "ablation", "activation"],
    batch_topk: bool,
    topk: float,
    distil_from_target: bool,
    topk_mask: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"] | None = None,
) -> SPDOutputs:
    # Forward pass on target model
    target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
    target_out, target_cache = target_model.run_with_cache(
        input_array, names_filter=target_cache_filter
    )

    # Do a forward pass with all subnetworks
    spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
    out, spd_cache = spd_model.run_with_cache(input_array, names_filter=spd_cache_filter)

    attribution_scores = calculate_attributions(
        model=spd_model,
        batch=input_array,
        out=out,
        target_out=target_out,
        pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
        post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
        component_acts={k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")},
        attribution_type=attribution_type,
    )

    if topk_mask is None:
        # We always assume the final subnetwork is the one we want to distil
        topk_attrs = attribution_scores[..., :-1] if distil_from_target else attribution_scores

        topk_mask = calc_topk_mask(topk_attrs, topk, batch_topk=batch_topk)
        if distil_from_target:
            # Add back the final subnetwork index to the topk mask and set it to True
            last_subnet_mask = torch.ones(
                (*topk_mask.shape[:-1], 1), dtype=torch.bool, device=attribution_scores.device
            )
            topk_mask = torch.cat((topk_mask, last_subnet_mask), dim=-1)

    topk_spd_out = spd_model(input_array, topk_mask=topk_mask)
    attribution_scores = attribution_scores.cpu().detach()
    return SPDOutputs(
        target_model_output=target_out,
        spd_model_output=out,
        spd_topk_model_output=topk_spd_out,
        layer_acts={k: v for k, v in spd_cache.items() if k.endswith("hook_post")},
        component_acts={k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")},
        attribution_scores=attribution_scores,
        topk_mask=topk_mask,
    )


DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]


class SparseFeatureDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type
        self.value_range = value_range
        self.synced_inputs = synced_inputs

    def __len__(self) -> int:
        return 2**31

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_instances n_features"]
    ) -> Float[Tensor, "batch n_instances n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            # First, get the samples for which there is a non-zero value for any of the indices
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            # Now generate random values in value_range and apply them to the masked elements
            max_val, min_val = self.value_range
            random_values = torch.rand(
                batch.shape[0], self.n_instances, self.n_features, device=self.device
            )
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances n_features"]
    ]:
        # TODO: This is a hack to keep backward compatibility. Probably best to have
        # data_generation_type: Literal["exactly_n_active", "at_least_zero_active"] and
        # data_generation_n: PositiveInt
        number_map = {
            "exactly_one_active": 1,
            "exactly_two_active": 2,
            "exactly_three_active": 3,
            "exactly_four_active": 4,
            "exactly_five_active": 5,
        }
        if self.data_generation_type in number_map:
            n = number_map[self.data_generation_type]
            batch = self._generate_n_feature_active_batch(batch_size, n=n)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
            if self.synced_inputs is not None:
                batch = self.sync_inputs(batch)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_n_feature_active_batch(
        self, batch_size: int, n: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch with exactly n features active per sample and instance.

        Args:
            batch_size: Number of samples in the batch
            n: Number of features to activate per sample and instance
        """
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )

        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)

        # Create indices for all features
        feature_indices = torch.arange(self.n_features, device=self.device)
        # Expand to batch size and n_instances
        feature_indices = feature_indices.expand(batch_size, self.n_instances, self.n_features)

        # For each instance in the batch, randomly permute the features
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)

        # Take first n indices for each instance - guaranteed no duplicates
        active_features = permuted_features[..., :n]

        # Generate random values in value_range for the active features
        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, self.n_instances, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val

        # Place each active feature
        for i in range(n):
            batch.scatter_(
                dim=2, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )

        return batch

    def _masked_batch_generator(
        self, total_batch_size: int
    ) -> Float[Tensor, "total_batch_size n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.

        Args:
            total_batch_size: Number of samples in the batch (either `batch_size` or
                `batch_size * n_instances`)
        """
        min_val, max_val = self.value_range
        batch = (
            torch.rand((total_batch_size, self.n_features), device=self.device)
            * (max_val - min_val)
            + min_val
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

    def _generate_multi_feature_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`."""
        total_batch_size = batch_size * self.n_instances
        batch = self._masked_batch_generator(total_batch_size)
        return einops.rearrange(
            batch,
            "(batch n_instances) n_features -> batch n_instances n_features",
            batch=batch_size,
        )

    def _generate_multi_feature_batch_no_zero_samples(
        self, batch_size: int, buffer_ratio: float
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.

        Ensures that there are no zero samples in the batch.

        Args:
            batch_size: Number of samples in the batch
            buffer_ratio: First generate `buffer_ratio * total_batch_size` samples and count the
                number of samples with all zeros. Then generate another `buffer_ratio *
                n_zeros` samples and fill in the zero samples. Continue until there are no zero
                samples.
        """
        total_batch_size = batch_size * self.n_instances
        buffer_size = int(total_batch_size * buffer_ratio)
        batch = torch.empty(0, device=self.device, dtype=torch.float32)
        n_samples_needed = total_batch_size
        while True:
            buffer = self._masked_batch_generator(buffer_size)
            # Get the indices of the non-zero samples in the buffer
            valid_indices = buffer.sum(dim=-1) != 0
            batch = torch.cat((batch, buffer[valid_indices][:n_samples_needed]))
            if len(batch) == total_batch_size:
                break
            else:
                # We don't have enough valid samples
                n_samples_needed = total_batch_size - len(batch)
                buffer_size = int(n_samples_needed * buffer_ratio)
        return einops.rearrange(
            batch,
            "(batch n_instances) n_features -> batch n_instances n_features",
            batch=batch_size,
        )


def compute_feature_importances(
    batch_size: int,
    n_instances: int,
    n_features: int,
    importance_val: float | None,
    device: str,
) -> Float[Tensor, "batch_size n_instances n_features"]:
    # Defines a tensor where the i^th feature has importance importance^i
    if importance_val is None or importance_val == 1.0:
        importance_tensor = torch.ones(batch_size, n_instances, n_features, device=device)
    else:
        powers = torch.arange(n_features, device=device)
        importances = torch.pow(importance_val, powers)
        # Now make it a tensor of shape (batch_size, n_instances, n_features)
        importance_tensor = einops.repeat(
            importances,
            "n_features -> batch_size n_instances n_features",
            batch_size=batch_size,
            n_instances=n_instances,
        )
    return importance_tensor


def calc_recon_mse(
    output: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    labels: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    recon_loss = (output - labels) ** 2
    if recon_loss.ndim == 3:
        assert has_instance_dim
        recon_loss = einops.reduce(recon_loss, "b i f -> i", "mean")
    elif recon_loss.ndim == 2:
        recon_loss = recon_loss.mean()
    else:
        raise ValueError(f"Expected 2 or 3 dims in recon_loss, got {recon_loss.ndim}")
    return recon_loss


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    elif lr_schedule == "exponential":
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


def replace_deprecated_param_names(
    params: dict[str, Float[Tensor, "..."]], name_map: dict[str, str]
) -> dict[str, Float[Tensor, "..."]]:
    """Replace old parameter names with new parameter names in a dictionary.

    Args:
        params: The dictionary of parameters to fix
        name_map: A dictionary mapping old parameter names to new parameter names
    """
    for k in list(params.keys()):
        for old_name, new_name in name_map.items():
            if old_name in k:
                params[k.replace(old_name, new_name)] = params[k]
                del params[k]
    return params
