"""Residual Linear decomposition script."""

import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from jaxtyping import Float
from matplotlib.colors import CenteredNorm
from pydantic import PositiveFloat
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_mlp.models import (
    ResidualMLPModel,
    ResidualMLPSPDConfig,
    ResidualMLPSPDModel,
)
from spd.experiments.resid_mlp.plotting import (
    analyze_per_feature_performance,
    plot_individual_feature_response,
    plot_per_feature_performance,
    plot_spd_relu_contribution,
    plot_virtual_weights_target_spd,
    spd_calculate_diag_relu_conns,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import (
    ResidualMLPDataset,
)
from spd.log import logger
from spd.module_utils import collect_nested_module_attrs
from spd.plotting import (
    plot_subnetwork_attributions_statistics,
    plot_subnetwork_correlations,
)
from spd.run_spd import (
    Config,
    ResidualMLPTaskConfig,
    get_common_run_name_suffix,
    optimize,
)
from spd.utils import (
    COLOR_PALETTE,
    DatasetGeneratedDataLoader,
    collect_subnetwork_attributions,
    load_config,
    run_spd_forward_pass,
    set_seed,
)
from spd.wandb_utils import init_wandb

wandb.require("core")


def get_run_name(
    config: Config,
    n_features: int,
    n_layers: int,
    d_resid: int,
    d_mlp: int,
    m: int | None,
    init_scale: float,
) -> str:
    """Generate a run name based on the config."""
    run_suffix = ""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"scale{init_scale}_ft{n_features}_lay{n_layers}_resid{d_resid}_mlp{d_mlp}"
        if m is not None:
            run_suffix += f"_m{m}"
    return config.wandb_run_name_prefix + run_suffix


def calc_n_active_features_per_subnet(
    model: ResidualMLPSPDModel, cutoff: float, device: str
) -> tuple[Float[Tensor, "n_instances C"], Float[Tensor, "n_instances n_features"]]:
    """Calculate the number of active features per subnet (and per instance if n_instances > 1)."""
    n_active_features_per_subnet: Float[Tensor, "n_instances C"] = torch.zeros(
        (model.config.n_instances, model.C), device=device
    )
    active_feature_counts_per_subnet: Float[Tensor, "n_instances n_features"] = torch.zeros(
        (model.config.n_instances, model.config.n_features), device=device
    )
    for c in range(model.C):
        relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = spd_calculate_diag_relu_conns(
            model, device, k_select=c
        )
        # Count the number of features for which each subnet fires beyond the cutoff
        above_cutoff = relu_conns.max(dim=-1).values > cutoff
        n_active_features_per_subnet[:, c] = above_cutoff.sum(dim=-1)
        active_feature_counts_per_subnet[:] += above_cutoff

    return n_active_features_per_subnet, active_feature_counts_per_subnet


def plot_subnetwork_attributions(
    attribution_scores: Float[Tensor, "batch n_instances C"],
    out_dir: Path | None,
    step: int | None,
) -> plt.Figure:
    """Plot subnetwork attributions."""
    # Plot a row with n_instances
    # Each column is a different instance
    n_instances = attribution_scores.shape[1]
    fig, ax = plt.subplots(
        nrows=1, ncols=n_instances, figsize=(5 * n_instances, 5), constrained_layout=True
    )
    axs = np.array([ax]) if n_instances == 1 else np.array(ax)
    im = None
    for i in range(n_instances):
        im = axs[i].matshow(
            attribution_scores[:, i].detach().cpu().numpy(), aspect="auto", cmap="Reds"
        )
        axs[i].set_xlabel("Subnetwork Index")
        axs[i].set_ylabel("Batch Index")
        axs[i].set_title("Subnetwork Attributions")

        # Annotate each cell with the numeric value if less than 200 elements
        if attribution_scores.shape[0] * attribution_scores.shape[-1] < 200:
            for b in range(attribution_scores.shape[0]):
                for j in range(attribution_scores.shape[-1]):
                    axs[i].text(
                        j,
                        b,
                        f"{attribution_scores[b, i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                    )
    plt.colorbar(im)
    if out_dir:
        filename = (
            f"subnetwork_attributions_s{step}.png"
            if step is not None
            else "subnetwork_attributions.png"
        )
        fig.savefig(out_dir / filename, dpi=200)
    return fig


def plot_multiple_component_weights(
    model: ResidualMLPSPDModel,
    out_dir: Path | None,
    step: int | None = None,
) -> plt.Figure:
    """Plot each component weight matrix."""
    all_params = collect_nested_module_attrs(model, "component_weights")
    # Each param (of which there are n_layers): [k, n_features, n_features]
    n_params = len(all_params)
    param_names = list(all_params.keys())
    n_instances = model.config.n_instances
    C = model.C

    # Find global min and max for normalization
    all_values = []
    for param_name in param_names:
        param_values = all_params[param_name].detach().cpu().numpy()
        all_values.append(param_values)
    all_values_concat = np.concatenate([v.flatten() for v in all_values])
    vmax = np.abs(all_values_concat).max()
    norm = CenteredNorm(vcenter=0, halfrange=vmax)

    fig, axs = plt.subplots(
        n_instances * n_params,
        C,
        figsize=(2 * C, n_instances * n_params),
        constrained_layout=False,
    )
    axs = np.array(axs)

    for instance_idx in range(n_instances):
        for param_idx in range(n_params):
            param_name = param_names[param_idx]
            for subnet_idx in range(C):
                col_idx = subnet_idx
                row_idx = instance_idx * n_params + param_idx

                ax = axs[row_idx, col_idx]  # type: ignore
                param = all_params[param_name][instance_idx, subnet_idx].detach().cpu().numpy()
                # If it's a bias with a single dimension, unsqueeze it
                if param.ndim == 1:
                    param = param[:, None]

                # Set aspect ratio based on parameter dimensions
                height, width = param.shape
                aspect = width / height

                im = ax.matshow(param, cmap="RdBu", norm=norm, aspect=aspect)
                ax.set_xticks([])
                ax.set_yticks([])

                if col_idx == 0:
                    ax.set_ylabel(
                        f"Inst.{instance_idx}.{param_name}",
                        rotation=0,
                        ha="right",
                        va="center",
                    )

                if row_idx == ((n_instances * n_params) - 1):
                    ax.set_xlabel(f"Subnet {subnet_idx}", rotation=0, ha="center", va="top")

    # Add colorbar
    fig.colorbar(im, ax=axs.ravel().tolist(), location="right")  # type: ignore

    title_text = "Subnet Parameters"
    if step is not None:
        title_text += f" (Step {step})"
    fig.suptitle(title_text)
    if out_dir:
        fig.savefig(out_dir / f"component_weights_s{step}.png", dpi=200)
    return fig


def plot_subnet_categories(
    model: ResidualMLPSPDModel, device: str, cutoff: float = 4e-2
) -> plt.Figure:
    n_active_features_per_subnet, active_feature_counts_per_subnet = (
        calc_n_active_features_per_subnet(model, cutoff=cutoff, device=device)
    )
    n_dead_subnets = (n_active_features_per_subnet == 0).sum(dim=-1).detach().tolist()
    n_monosemantic_subnets = (n_active_features_per_subnet == 1).sum(dim=-1).detach().tolist()
    n_duosemantic_subnets = (n_active_features_per_subnet == 2).sum(dim=-1).detach().tolist()
    n_polysemantic_subnets = (n_active_features_per_subnet > 2).sum(dim=-1).detach().tolist()
    n_unique_features_represented = (
        (active_feature_counts_per_subnet > 0).sum(dim=-1).detach().tolist()
    )

    n_instances = len(n_dead_subnets)
    fig, ax = plt.subplots(
        nrows=1, ncols=n_instances, figsize=(5 * n_instances, 5), constrained_layout=True
    )
    axs = np.array([ax]) if n_instances == 1 else np.array(ax)
    categories = ["Dead", "Mono", "Duo", "Poly", "Represented"]

    for i in range(n_instances):
        counts = [
            n_dead_subnets[i],
            n_monosemantic_subnets[i],
            n_duosemantic_subnets[i],
            n_polysemantic_subnets[i],
            n_unique_features_represented[i],
        ]
        bars = axs[i].bar(categories, counts)
        axs[i].set_xlabel("Category")
        axs[i].set_ylabel("Count")
        axs[i].set_title(f"Subnet Categories (Instance {i})")

        # Add numbers in the middle of the bars
        for bar, count in zip(bars, counts, strict=False):
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2.0,
                f"{count}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

    return fig


def resid_mlp_plot_results_fn(
    model: ResidualMLPSPDModel,
    target_model: ResidualMLPModel,
    step: int | None,
    out_dir: Path | None,
    device: str,
    config: Config,
    topk_mask: Float[Tensor, "batch_size C"] | None,
    dataloader: DatasetGeneratedDataLoader[
        tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch d_embed"]]
    ]
    | None = None,
    **_,
) -> dict[str, plt.Figure]:
    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    fig_dict = {}

    fig_dict["subnet_categories"] = plot_subnet_categories(model, device)

    ############################################################################################
    # Feature contributions
    ############################################################################################
    fig1, fig2 = plot_spd_relu_contribution(model, target_model, device)
    fig1.suptitle("How much does each ReLU contribute to each feature?")
    fig2.suptitle("How much does each feature route through each ReLU?")
    fig_dict["feature_contributions"] = fig1
    fig_dict["relu_contributions"] = fig2

    fig1, fig2 = plot_spd_relu_contribution(model, target_model, device, k_plot_limit=3)
    fig1.suptitle("How much does each ReLU contribute to each feature?")
    fig2.suptitle("How much does each feature route through each ReLU?")
    fig_dict["cropped_feature_contributions"] = fig1
    fig_dict["cropped_relu_contributions"] = fig2

    ############################################################################################
    # Individual feature responses + per-feature performance
    ############################################################################################
    def spd_model_fn(
        batch: Float[Tensor, "batch n_instances n_features"],
        topk: PositiveFloat | None = config.topk,
        batch_topk: bool = config.batch_topk,
    ) -> Float[Tensor, "batch n_instances n_features"]:
        assert topk is not None
        if config.exact_topk:
            assert model.n_instances == 1, "exact_topk only works if n_instances = 1"
            topk = ((batch != 0).sum() / batch.shape[0]).item()
        return run_spd_forward_pass(
            spd_model=model,
            target_model=target_model,
            input_array=batch,
            attribution_type=config.attribution_type,
            batch_topk=batch_topk,
            topk=topk,
            distil_from_target=config.distil_from_target,
        ).spd_topk_model_output

    def target_model_fn(batch: Float[Tensor, "batch n_instances"]):
        return target_model(batch)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True)
    axes = np.atleast_2d(axes)  # type: ignore
    plot_individual_feature_response(
        model_fn=target_model_fn,
        device=device,
        model_config=model.config,
        ax=axes[0, 0],
    )
    plot_individual_feature_response(
        model_fn=target_model_fn,
        device=device,
        model_config=model.config,
        sweep=True,
        ax=axes[1, 0],
    )
    plot_individual_feature_response(
        model_fn=spd_model_fn,
        device=device,
        model_config=model.config,
        ax=axes[0, 1],
    )
    plot_individual_feature_response(
        model_fn=spd_model_fn,
        device=device,
        model_config=model.config,
        sweep=True,
        ax=axes[1, 1],
    )
    axes[0, 0].set_ylabel(axes[0, 0].get_title())
    axes[1, 0].set_ylabel(axes[1, 0].get_title())
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 0].set_title("Target model")
    axes[0, 1].set_title("SPD model")
    axes[1, 0].set_title("")
    axes[1, 1].set_title("")
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    fig_dict["individual_feature_responses"] = fig

    # Plot per-feature performance when setting topk=1 and batch_topk=False
    fig, ax1 = plt.subplots(figsize=(15, 5))

    losses_target = analyze_per_feature_performance(
        model_fn=target_model_fn,
        model_config=target_model.config,
        device=device,
        batch_size=config.batch_size,
    )
    indices = losses_target.argsort()
    fn_without_batch_topk = lambda batch: spd_model_fn(batch, topk=1, batch_topk=False)  # type: ignore
    losses_spd = analyze_per_feature_performance(
        model_fn=fn_without_batch_topk,
        model_config=model.config,
        device=device,
        batch_size=config.batch_size,
    )

    plot_per_feature_performance(
        losses=losses_spd,
        sorted_indices=indices,
        ax=ax1,
        label="SPD",
        color=COLOR_PALETTE[1],
    )
    plot_per_feature_performance(
        losses=losses_target,
        sorted_indices=indices,
        ax=ax1,
        label="Target",
        color=COLOR_PALETTE[0],
    )
    ax1.legend()

    fig_dict["loss_by_feature_topk_1"] = fig

    # Plot per-feature performance when using batch_topk
    fig, ax2 = plt.subplots(figsize=(15, 5))

    target_losses_batch_topk = analyze_per_feature_performance(
        model_fn=target_model_fn,
        model_config=target_model.config,
        device=device,
        batch_size=config.batch_size,
    )

    spd_losses_batch_topk = analyze_per_feature_performance(
        model_fn=spd_model_fn,
        model_config=model.config,
        device=device,
        batch_size=config.batch_size,
    )

    plot_per_feature_performance(
        losses=spd_losses_batch_topk,
        sorted_indices=indices,
        ax=ax2,
        label="SPD",
        color=COLOR_PALETTE[1],
    )
    plot_per_feature_performance(
        losses=target_losses_batch_topk,
        sorted_indices=indices,
        ax=ax2,
        label="Target",
        color=COLOR_PALETTE[0],
    )
    ax2.legend()
    # Use the same y-axis limits as the topk=1 plot
    ax2.set_ylim(ax1.get_ylim())
    fig_dict["loss_by_feature_batch_topk"] = fig

    ############################################################################################
    # Virtual weights
    ############################################################################################

    fig = plot_virtual_weights_target_spd(target_model, model, device)
    fig_dict["virtual_weights"] = fig

    ############################################################################################
    # Subnetwork attributions
    ############################################################################################
    attribution_scores = collect_subnetwork_attributions(
        spd_model=model,
        target_model=target_model,
        device=device,
        n_instances=model.n_instances,
    )
    fig_dict["subnetwork_attributions"] = plot_subnetwork_attributions(
        attribution_scores, out_dir, step
    )

    if config.topk is not None:
        if dataloader is not None and config.C > 1:
            fig_dict_correlations = plot_subnetwork_correlations(
                dataloader=dataloader,
                target_model=target_model,
                spd_model=model,
                config=config,
                device=device,
            )
            fig_dict.update(fig_dict_correlations)

        assert topk_mask is not None
        fig_dict_attributions = plot_subnetwork_attributions_statistics(topk_mask=topk_mask)
        fig_dict.update(fig_dict_attributions)

    ############################################################################################
    # Subnetwork parameters
    ############################################################################################

    # This can be too big to plot
    n_matrix_params = target_model.config.d_mlp * target_model.config.d_embed
    if n_matrix_params < 1000:
        fig_dict["component_weights"] = plot_multiple_component_weights(
            model=model, out_dir=out_dir, step=step
        )

    # Save plots to files
    if out_dir:
        for k, v in fig_dict.items():
            out_file = out_dir / f"{k}_s{step}.png"
            v.savefig(out_file, dpi=100)
            tqdm.write(f"Saved plot to {out_file}")
    return fig_dict


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    resid_mlp: ResidualMLPModel,
    resid_mlp_train_config_dict: dict[str, Any],
    label_coeffs: Float[Tensor, " n_instances"],
) -> None:
    torch.save(resid_mlp.state_dict(), out_dir / "resid_mlp.pth")

    with open(out_dir / "resid_mlp_train_config.yaml", "w") as f:
        yaml.dump(resid_mlp_train_config_dict, f, indent=2)

    with open(out_dir / "label_coeffs.json", "w") as f:
        json.dump(label_coeffs.detach().cpu().tolist(), f, indent=2)

    if save_to_wandb:
        wandb.save(str(out_dir / "resid_mlp.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "resid_mlp_train_config.yaml"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "label_coeffs.json"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    set_seed(config.seed)
    logger.info(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert isinstance(config.task_config, ResidualMLPTaskConfig)

    target_model, target_model_train_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(
        config,
        n_features=target_model.config.n_features,
        n_layers=target_model.config.n_layers,
        d_resid=target_model.config.d_embed,
        d_mlp=target_model.config.d_mlp,
        m=config.m,
        init_scale=config.task_config.init_scale,
    )
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        resid_mlp=target_model,
        resid_mlp_train_config_dict=target_model_train_config_dict,
        label_coeffs=label_coeffs,
    )

    # Create the SPD model
    model_config = ResidualMLPSPDConfig(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        d_embed=target_model.config.d_embed,
        d_mlp=target_model.config.d_mlp,
        n_layers=target_model.config.n_layers,
        act_fn_name=target_model.config.act_fn_name,
        apply_output_act_fn=target_model.config.apply_output_act_fn,
        in_bias=target_model.config.in_bias,
        out_bias=target_model.config.out_bias,
        init_scale=config.task_config.init_scale,
        C=config.C,
        m=config.m,
    )
    model = ResidualMLPSPDModel(config=model_config).to(device)

    # Use the target_model's embedding matrix and don't train it further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False
    model.W_U.data[:, :] = target_model.W_U.data.detach().clone()
    model.W_U.requires_grad = False

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(target_model.config.n_layers):
        if target_model.config.in_bias:
            model.layers[i].bias1.data[:, :] = target_model.layers[i].bias1.data.detach().clone()
            model.layers[i].bias1.requires_grad = False
        if target_model.config.out_bias:
            model.layers[i].bias2.data[:, :] = target_model.layers[i].bias2.data.detach().clone()
            model.layers[i].bias2.requires_grad = False

    param_names = []
    for i in range(target_model.config.n_layers):
        param_names.append(f"layers.{i}.mlp_in")
        param_names.append(f"layers.{i}.mlp_out")

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = ResidualMLPDataset(
        n_instances=model.config.n_instances,
        n_features=model.config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=synced_inputs,
    )

    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    plot_results_fn = partial(resid_mlp_plot_results_fn, dataloader=dataloader)
    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=out_dir,
        plot_results_fn=plot_results_fn,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
