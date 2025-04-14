from typing import Any, Literal

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.utils.data import DataLoader

from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDModel
from spd.experiments.tms.models import TMSModel, TMSSPDModel
from spd.hooks import HookedRootModule
from spd.models.base import SPDModel
from spd.run_spd import Config
from spd.utils import (
    DataGenerationType,
    SparseFeatureDataset,
    calc_recon_mse,
    calc_topk_mask,
    calculate_attributions,
    run_spd_forward_pass,
)


def plot_subnetwork_attributions_statistics(
    topk_mask: Float[Tensor, "batch_size n_instances C"],
) -> dict[str, plt.Figure]:
    """Plot vertical bar charts of the number of active subnetworks over the batch for each instance."""
    batch_size = topk_mask.shape[0]
    if topk_mask.ndim == 2:
        n_instances = 1
        topk_mask = einops.repeat(topk_mask, "batch C -> batch n_instances C", n_instances=1)
    else:
        n_instances = topk_mask.shape[1]

    fig, axs = plt.subplots(
        ncols=n_instances, nrows=1, figsize=(5 * n_instances, 5), constrained_layout=True
    )

    axs = np.array([axs]) if n_instances == 1 else np.array(axs)
    for i, ax in enumerate(axs):
        values = topk_mask[:, i].sum(dim=1).cpu().detach().numpy()
        bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
        counts, _ = np.histogram(values, bins=bins)
        bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels([str(b) for b in bins[:-1]])

        # Only add y-label to first subplot
        if i == 0:
            ax.set_ylabel("Count")

        ax.set_xlabel("Number of active subnetworks")
        ax.set_title(f"Instance {i+1}")

        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.suptitle(f"Active subnetworks on current batch (batch_size={batch_size})")
    return {"subnetwork_attributions_statistics": fig}


def plot_subnetwork_correlations(
    dataloader: DataLoader[
        tuple[Float[Tensor, "batch n_inputs"] | Float[Tensor, "batch n_instances? n_inputs"], Any]
    ],
    target_model: HookedRootModule,
    spd_model: SPDModel,
    config: Config,
    device: str,
    n_forward_passes: int = 100,
) -> dict[str, plt.Figure]:
    topk_masks = []
    for batch, _ in dataloader:
        batch = batch.to(device=device)
        assert config.topk is not None

        # Forward pass on target model
        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            batch, names_filter=target_cache_filter
        )

        # Do a forward pass with all subnetworks
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out, spd_cache = spd_model.run_with_cache(batch, names_filter=spd_cache_filter)
        attribution_scores = calculate_attributions(
            model=spd_model,
            batch=batch,
            out=out,
            target_out=target_out,
            pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
            post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
            component_acts={
                k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")
            },
            attribution_type=config.attribution_type,
        )

        # We always assume the final subnetwork is the one we want to distil
        topk_attrs = (
            attribution_scores[..., :-1] if config.distil_from_target else attribution_scores
        )
        if config.exact_topk:
            assert spd_model.n_instances == 1, "exact_topk only works if n_instances = 1"
            topk = (batch != 0).sum() / batch.shape[0]
            topk_mask = calc_topk_mask(topk_attrs, topk, batch_topk=config.batch_topk)
        else:
            topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)

        topk_masks.append(topk_mask)
        if len(topk_masks) > n_forward_passes:
            break
    topk_masks = torch.cat(topk_masks).float()

    if hasattr(spd_model, "n_instances"):
        n_instances = spd_model.n_instances
    else:
        n_instances = 1
        topk_masks = einops.repeat(topk_masks, "batch C -> batch n_instances C", n_instances=1)

    fig, axs = plt.subplots(
        ncols=n_instances, nrows=1, figsize=(5 * n_instances, 5), constrained_layout=True
    )

    axs = np.array([axs]) if n_instances == 1 else np.array(axs)
    im, ax = None, None
    for i, ax in enumerate(axs):
        # Calculate correlation matrix
        corr_matrix = torch.corrcoef(topk_masks[:, i].T).cpu()

        im = ax.matshow(corr_matrix)
        ax.xaxis.set_ticks_position("bottom")
        if corr_matrix.shape[0] * corr_matrix.shape[1] < 200:
            for l in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    ax.text(
                        j,
                        l,
                        f"{corr_matrix[l, j]:.2f}",
                        ha="center",
                        va="center",
                        color="#EE7777",
                        fontsize=8,
                    )
    if (im is not None) and (ax is not None):
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax.set_title("Subnetwork Correlation Matrix")
        ax.set_xlabel("Subnetwork")
        ax.set_ylabel("Subnetwork")
    return {"subnetwork_correlation_matrix": fig}


def collect_sparse_dataset_mse_losses(
    dataset: SparseFeatureDataset,
    target_model: ResidualMLPModel | TMSModel,
    spd_model: TMSSPDModel | ResidualMLPSPDModel,
    batch_size: int,
    device: str,
    topk: float,
    attribution_type: Literal["gradient", "ablation", "activation"],
    batch_topk: bool,
    distil_from_target: bool,
    gen_types: list[DataGenerationType],
) -> dict[str, dict[str, Float[Tensor, ""] | Float[Tensor, " n_instances"]]]:
    """Collect the MSE losses for specific number of active features, as well as for
    'at_least_zero_active'.

    We calculate two baselines:
    - baseline_monosemantic: a baseline loss where the first d_mlp feature indices get mapped to the
        true labels and the final (n_features - d_mlp) features are either 0 (TMS) or the raw inputs
        (ResidualMLP).

    Returns:
        A dictionary keyed by generation type and then by model type (target, spd,
        baseline_monosemantic), with values being MSE losses.
    """
    target_model.to(device)
    spd_model.to(device)
    # Get the entries for the main loss table in the paper
    results = {gen_type: {} for gen_type in gen_types}
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

    for gen_type in gen_types:
        dataset.data_generation_type = gen_type
        batch, labels = dataset.generate_batch(batch_size)

        batch = batch.to(device)
        labels = labels.to(device)

        target_model_output = target_model(batch)

        if gen_type == "at_least_zero_active":
            run_batch_topk = batch_topk
            run_topk = topk
        else:
            run_batch_topk = False
            assert gen_type.startswith("exactly_")
            n_active = word_to_num[gen_type.split("_")[1]]
            run_topk = n_active

        spd_outputs = run_spd_forward_pass(
            spd_model=spd_model,
            target_model=target_model,
            input_array=batch,
            attribution_type=attribution_type,
            batch_topk=run_batch_topk,
            topk=run_topk,
            distil_from_target=distil_from_target,
        )
        # Combine the batch and n_instances dimension for batch, labels, target_model_output,
        # spd_outputs.spd_topk_model_output
        ein_str = "batch n_instances n_features -> (batch n_instances) n_features"
        batch = einops.rearrange(batch, ein_str)
        labels = einops.rearrange(labels, ein_str)
        target_model_output = einops.rearrange(target_model_output, ein_str)
        spd_topk_model_output = einops.rearrange(spd_outputs.spd_topk_model_output, ein_str)

        if gen_type == "at_least_zero_active":
            # Remove all entries where there are no active features
            mask = (batch != 0).any(dim=-1)
            batch = batch[mask]
            labels = labels[mask]
            target_model_output = target_model_output[mask]
            spd_topk_model_output = spd_topk_model_output[mask]

        topk_recon_loss_labels = calc_recon_mse(
            spd_topk_model_output, labels, has_instance_dim=False
        )
        recon_loss = calc_recon_mse(target_model_output, labels, has_instance_dim=False)
        baseline_batch = calc_recon_mse(batch, labels, has_instance_dim=False)

        # Monosemantic baseline
        monosemantic_out = batch.clone()
        # Assumes TMS or ResidualMLP
        if isinstance(target_model, ResidualMLPModel):
            d_mlp = target_model.config.d_mlp * target_model.config.n_layers  # type: ignore
            monosemantic_out[..., :d_mlp] = labels[..., :d_mlp]
        elif isinstance(target_model, TMSModel):
            d_mlp = target_model.config.n_hidden  # type: ignore
            # The first d_mlp features are the true labels (i.e. the batch) and the rest are 0
            monosemantic_out[..., d_mlp:] = 0
        baseline_monosemantic = calc_recon_mse(monosemantic_out, labels, has_instance_dim=False)

        results[gen_type]["target"] = recon_loss
        results[gen_type]["spd"] = topk_recon_loss_labels
        results[gen_type]["baseline_batch"] = baseline_batch
        results[gen_type]["baseline_monosemantic"] = baseline_monosemantic
    return results


def plot_sparse_feature_mse_line_plot(
    results: dict[str, dict[str, float]],
    label_map: list[tuple[str, str, str]],
    log_scale: bool = False,
) -> plt.Figure:
    xtick_label_map = {
        "at_least_zero_active": "Training distribution",
        "exactly_one_active": "Exactly 1 active",
        "exactly_two_active": "Exactly 2 active",
        "exactly_three_active": "Exactly 3 active",
        "exactly_four_active": "Exactly 4 active",
        "exactly_five_active": "Exactly 5 active",
    }
    # Create grouped bar plots for each generation type
    fig, ax = plt.subplots(figsize=(12, 6))

    n_groups = len(results)  # number of generation types
    n_models = len(label_map)  # number of models to compare
    width = 0.8 / n_models  # width of bars

    # Create bars for each model type
    for i, (model_type, label, color) in enumerate(label_map):
        x_positions = np.arange(n_groups) + i * width - (n_models - 1) * width / 2
        heights = [results[gen_type][model_type] for gen_type in results]
        ax.bar(x_positions, heights, width, label=label, color=color)

    # Customize the plot
    ax.set_ylabel("MSE w.r.t true labels")
    ax.set_xticks(np.arange(n_groups))
    xtick_labels = [xtick_label_map[gen_type] for gen_type in results]
    ax.set_xticklabels(xtick_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if log_scale:
        ax.set_yscale("log")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that 0 is the bottom of the y-axis
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_matrix(
    ax: plt.Axes,
    matrix: torch.Tensor,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_format: str = "%.1f",
    norm: plt.Normalize | None = None,
) -> None:
    # Useful to have bigger text for small matrices
    fontsize = 8 if matrix.numel() < 50 else 4
    norm = norm if norm is not None else CenteredNorm()
    im = ax.matshow(matrix.detach().cpu().numpy(), cmap="coolwarm", norm=norm)
    # If less than 500 elements, show the values
    if matrix.numel() < 500:
        for (j, i), label in np.ndenumerate(matrix.detach().cpu().numpy()):
            ax.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=fontsize)
    ax.set_xlabel(xlabel)
    if ylabel != "":
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.1, pad=0.05)
    fig = ax.get_figure()
    assert fig is not None
    fig.colorbar(im, cax=cax, format=tkr.FormatStrFormatter(colorbar_format))
    if ylabel == "Function index":
        n_functions = matrix.shape[0]
        ax.set_yticks(range(n_functions))
        ax.set_yticklabels([f"{L:.0f}" for L in range(1, n_functions + 1)])
