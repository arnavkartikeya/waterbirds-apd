from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pydantic import PositiveFloat
from torch import Tensor
from tqdm import tqdm

from spd.experiments.resid_mlp.models import (
    ResidualMLPConfig,
    ResidualMLPModel,
    ResidualMLPSPDConfig,
    ResidualMLPSPDModel,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.plotting import plot_matrix
from spd.run_spd import Config
from spd.utils import SPDOutputs, calc_topk_mask, calculate_attributions


def plot_individual_feature_response(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDConfig,
    sweep: bool = False,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    plot_type: Literal["line", "scatter"] = "scatter",
    ax: plt.Axes | None = None,
    cbar: bool = True,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = model_config.n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    inputs = torch.ones(n_features) if not sweep else torch.linspace(-1, 1, n_features)
    batch[torch.arange(n_features), instance_idx, torch.arange(n_features)] = inputs.to(device)
    out = model_fn(batch)

    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    sweep_str = "set to 1" if not sweep else "between -1 and 1"
    title = (
        f"Feature response with one active feature {sweep_str}\n"
        f"n_features={model_config.n_features}, "
        f"d_embed={model_config.d_embed}, "
        f"d_mlp={model_config.d_mlp}"
    )
    ax.set_title(title)
    if subtract_inputs:
        out = out - batch[:, instance_idx, :]
    for f in range(n_features):
        x = torch.arange(n_features)
        y = out[f, :].detach().cpu()
        if plot_type == "line":
            ax.plot(x, y, color=cmap_viridis(f / n_features))
        elif plot_type == "scatter":
            s = torch.ones_like(x)
            alpha = torch.ones_like(x) * 0.33
            s[f] = 20
            alpha[f] = 1
            # Permute order to make zorder random
            order = torch.randperm(n_features)
            ax.scatter(
                x[order],
                y[order],
                color=cmap_viridis(f / n_features),
                marker=".",
                s=s[order],
                alpha=alpha[order].numpy(),  # type: ignore
                # According to the announcement, alpha is allowed to be an iterable since v3.4.0,
                # but the docs & type annotations seem to be wrong. Here's the announcement:
                # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.4.0.html#transparency-alpha-can-be-set-as-an-array-in-collections
            )
        else:
            raise ValueError("Unknown plot_type")
    # Plot labels
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    inputs = batch[torch.arange(n_features), instance_idx, torch.arange(n_features)].detach().cpu()
    targets = label_fn(inputs) if subtract_inputs else inputs + label_fn(inputs)
    baseline = torch.zeros(n_features) if subtract_inputs else inputs
    if plot_type == "line":
        ax.plot(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
        )
        ax.plot(
            torch.arange(n_features),
            baseline,
            color="red",
            linestyle=":",
            label="Baseline (Identity)",
        )
    elif plot_type == "scatter":
        ax.scatter(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
            marker="x",
            s=5,
        )
    else:
        raise ValueError("Unknown plot_type")
    ax.legend()
    if cbar:
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_viridis, norm=plt.Normalize(0, n_features))
        sm.set_array([])
        bar = plt.colorbar(sm, ax=ax, orientation="vertical")
        bar.set_label("Active input feature index")
    ax.set_xlabel("Output index")
    ax.set_ylabel("Output values $x̂_i$")

    ax.set_xticks([0, n_features])
    ax.set_xticklabels(["0", str(n_features)])
    return fig


def plot_single_feature_response(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDConfig,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    feature_idx: int = 15,
    plot_type: Literal["line", "scatter"] = "scatter",
    ax: plt.Axes | None = None,
):
    """Plot the response of the model to a single feature being active.

    If sweep is False then the amplitude of the active feature is 1.
    If sweep is True then the amplitude of the active feature is swept from -1 to 1. This is an
    arbitrary choice (choosing feature 0 to be the one where we test x=-1 etc) made for convenience.
    """
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = 1
    batch_idx = 0
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[batch_idx, instance_idx, feature_idx] = 1
    out = model_fn(batch)

    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    if subtract_inputs:
        out = out - batch[:, instance_idx, :]
    x = torch.arange(n_features)
    y = out[batch_idx, :].detach().cpu()
    inputs = batch[batch_idx, instance_idx, :].detach().cpu()
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    targets = label_fn(inputs) if subtract_inputs else inputs + label_fn(inputs)
    if plot_type == "line":
        ax.plot(x, y, color=cmap_viridis(feature_idx / n_features), label="Target Model")
        ax.plot(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
        )
    elif plot_type == "scatter":
        ax.scatter(
            x,
            y,
            color=cmap_viridis(feature_idx / n_features),
            label="Target Model",
            marker=".",
            s=20,
        )
        ax.scatter(
            torch.arange(n_features),
            targets.cpu().detach(),
            color="red",
            label=r"Label ($x+\mathrm{ReLU}(x)$)",
            marker="x",
            s=5,
        )
    else:
        raise ValueError("Unknown plot_type")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xlabel("Output index")
    ax.set_ylabel(f"Output value $x̂_{{{feature_idx}}}$")
    ax.set_title(f"Output for a single input $x_{{{feature_idx}}}=1$")

    # Only need feature indices 0, feature_idx, n_features.
    ax.set_xticks([0, feature_idx, n_features])
    ax.set_xticklabels(["0", str(feature_idx), str(n_features)])

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_single_relu_curve(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDConfig,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
    feature_idx: int = 15,
    ax: plt.Axes | None = None,
    label: bool = True,
):
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = 1000
    x = torch.linspace(-1, 1, batch_size)
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[:, instance_idx, feature_idx] = x
    out = model_fn(batch)
    out = out[:, instance_idx, :]
    cmap_viridis = plt.get_cmap("viridis")
    fig, ax = plt.subplots(constrained_layout=True) if ax is None else (ax.figure, ax)
    if subtract_inputs:
        out = out - batch[:, instance_idx, :]

    y = out[:, feature_idx].detach().cpu()
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    targets = label_fn(x) if subtract_inputs else x + label_fn(x)
    ax.plot(
        x, y, color=cmap_viridis(feature_idx / n_features), label="Target Model" if label else None
    )
    ax.plot(
        x,
        targets.cpu().detach(),
        color="red",
        label=r"Label ($x+\mathrm{ReLU}(x)$)" if label else None,
        ls="--",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xlabel(f"Input value $x_{{{feature_idx}}}$")
    ax.set_ylabel(f"Output value $x̂_{{{feature_idx}}}$")
    ax.set_title(f"Input-output response for input feature {feature_idx}")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def plot_all_relu_curves(
    model_fn: Callable[[Tensor], Tensor],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDConfig,
    ax: plt.Axes,
    subtract_inputs: bool = True,
    instance_idx: int = 0,
):
    n_features = model_config.n_features
    fig = ax.figure
    for feature_idx in range(n_features):
        plot_single_relu_curve(
            model_fn=model_fn,
            device=device,
            model_config=model_config,
            subtract_inputs=subtract_inputs,
            instance_idx=instance_idx,
            feature_idx=feature_idx,
            ax=ax,
            label=False,
        )
    ax.set_title(f"Input-output response for all {n_features} input features")
    ax.set_xlabel("Input values $x_i$")
    ax.set_ylabel("Output values $x̂_i$")
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def _calculate_snr(
    model: ResidualMLPModel, device: str, input_values: tuple[float, float]
) -> Tensor:
    n_features = model.config.n_features
    n_instances = model.config.n_instances
    batch_size = n_features**2
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    instance_idx = 0
    snr = torch.zeros(n_features, n_features)
    for f1 in range(n_features):
        for f2 in range(n_features):
            idx = f1 * n_features + f2
            batch[idx, instance_idx, f1] = input_values[0]
            batch[idx, instance_idx, f2] = input_values[1]
    out = model(batch)
    out: Float[Tensor, "batch n_features"] = out[:, instance_idx, :]
    for f1 in range(n_features):
        for f2 in range(n_features):
            idx = f1 * n_features + f2
            signal = min(out[idx, f1].abs().item(), out[idx, f2].abs().item())
            noise = out[idx, :].std().item()
            snr[f1, f2] = signal / noise
    return snr


def plot_2d_snr(model: ResidualMLPModel, device: str):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, height_ratios=[1, 10, 10], constrained_layout=True, figsize=(4, 8)
    )  # type: ignore
    # Calculate SNR for (1, 1) and implicitly (1,) too.
    snr = _calculate_snr(model, device, input_values=(1, 1)).cpu().detach()
    # Plot diagonal in top subplot
    diagonal = torch.diag(snr)
    im1 = ax1.imshow(diagonal.unsqueeze(0), aspect="auto", vmin=1, vmax=snr.max())
    ax1.set_yticks([])
    ax1.set_title("SNR for single active features")
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    # Plot main SNR matrix without diagonal
    snr_no_diag = snr.clone()
    snr_no_diag.fill_diagonal_(torch.nan)
    im2 = ax2.imshow(snr_no_diag, aspect="auto", vmin=1, vmax=snr.max())
    ax2.set_title("SNR for pairs of active features set to (1, 1)")
    ax2.set_xlabel("Feature 2 (set to 1)")
    ax2.set_ylabel("Feature 1 (set to 1)")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    # Calculate SNR for (1, -1)
    snr = _calculate_snr(model, device, input_values=(1, -1)).cpu().detach()
    # Plot second SNR matrix without diagonal
    snr_no_diag = snr.clone()
    snr_no_diag.fill_diagonal_(torch.nan)
    im3 = ax3.imshow(snr_no_diag, aspect="auto", vmin=1, vmax=snr.max())
    ax3.set_title("SNR for pairs of active features set to (1, -1)")
    ax3.set_xlabel("Feature 2 (set to -1)")
    ax3.set_ylabel("Feature 1 (set to 1)")
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    return fig


def calculate_virtual_weights(target_model: ResidualMLPModel, device: str) -> dict[str, Tensor]:
    """Currently ignoring interactions between layers. Just flattening (n_layers, d_mlp)"""
    n_instances = target_model.config.n_instances
    n_features = target_model.config.n_features
    d_embed = target_model.config.d_embed
    d_mlp = target_model.config.d_mlp
    has_bias1 = target_model.layers[0].bias1 is not None
    has_bias2 = target_model.layers[0].bias2 is not None
    n_layers = target_model.config.n_layers
    # Get weights
    W_E: Float[Tensor, "n_instances n_features d_embed"] = target_model.W_E
    W_U: Float[Tensor, "n_instances d_embed n_features"] = target_model.W_U
    W_in: Float[Tensor, "n_instances d_embed d_mlp_eff"] = torch.cat(
        [target_model.layers[i].mlp_in.weight.data for i in range(n_layers)], dim=-1
    )
    W_out: Float[Tensor, "n_instances d_mlp_eff d_embed"] = torch.cat(
        [target_model.layers[i].mlp_out.weight.data for i in range(n_layers)],
        dim=-2,
    )
    b_in: Float[Tensor, "n_instances d_mlp_eff"] | None = (
        torch.cat([target_model.layers[i].bias1.data for i in range(n_layers)], dim=-1)
        if has_bias1
        else None
    )
    b_out: Float[Tensor, "n_instances d_embed"] | None = (
        torch.stack([target_model.layers[i].bias2.data for i in range(n_layers)]).sum(dim=0)
        if has_bias2
        else None
    )
    assert W_E.shape == (n_instances, n_features, d_embed)
    assert W_U.shape == (n_instances, d_embed, n_features)
    assert W_in.shape == (n_instances, d_embed, n_layers * d_mlp)
    assert W_out.shape == (n_instances, n_layers * d_mlp, d_embed)
    assert b_in.shape == (n_instances, n_layers * d_mlp) if b_in is not None else True
    assert b_out.shape == (n_instances, d_embed) if b_out is not None else True
    # Calculate connection strengths / virtual weights
    in_conns: Float[Tensor, "n_instances n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_instances n_features d_embed, n_instances d_embed d_mlp -> n_instances n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_instances d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_instances d_mlp d_embed, n_instances n_features d_embed -> n_instances d_mlp n_features",
    )
    diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_instances n_features d_mlp, n_instances d_mlp n_features -> n_instances n_features d_mlp",
    )
    assert in_conns.shape == (n_instances, n_features, n_layers * d_mlp)
    assert out_conns.shape == (n_instances, n_layers * d_mlp, n_features)
    assert diag_relu_conns.shape == (n_instances, n_features, n_layers * d_mlp)
    virtual_weights = {
        "W_E": W_E,
        "W_U": W_U,
        "W_in": W_in,
        "W_out": W_out,
        "in_conns": in_conns,
        "out_conns": out_conns,
        "diag_relu_conns": diag_relu_conns,
    }
    if b_in is not None:
        virtual_weights["b_in"] = b_in
    if b_out is not None:
        virtual_weights["b_out"] = b_out
    return virtual_weights


def relu_contribution_plot(
    ax1: plt.Axes,
    ax2: plt.Axes,
    all_diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"],
    model: ResidualMLPModel | ResidualMLPSPDModel,
    device: str,
    instance_idx: int = 0,
):
    diag_relu_conns: Float[Tensor, "n_features d_mlp"] = (
        all_diag_relu_conns[instance_idx].cpu().detach()
    )
    d_mlp = model.config.d_mlp
    n_layers = model.config.n_layers
    n_features = model.config.n_features

    ax1.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(model.config.n_features):
        ax1.scatter([i] * d_mlp * n_layers, diag_relu_conns[i, :], alpha=0.3, marker=".", c="k")
        ax1.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(d_mlp * n_layers):
            if diag_relu_conns[i, j].item() > 0.1:
                cmap_label = plt.get_cmap("hsv")
                ax1.text(
                    i, diag_relu_conns[i, j].item(), str(j), color=cmap_label(j / d_mlp / n_layers)
                )
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlim(-0.5, model.config.n_features - 0.5)
    ax2.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(d_mlp * n_layers):
        ax2.scatter([i] * n_features, diag_relu_conns[:, i], alpha=0.3, marker=".", c="k")
        ax2.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(n_features):
            if diag_relu_conns[j, i].item() > 0.2:
                cmap_label = plt.get_cmap("hsv")
                ax2.text(i, diag_relu_conns[j, i].item(), str(j), color=cmap_label(j / n_features))
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Features")
    ax2.set_xlabel("ReLUs (consecutively enumerated throughout layers)")
    ax2.set_xlim(-0.5, d_mlp * n_layers - 0.5)


def feature_contribution_plot(
    ax: plt.Axes,
    all_diag_relu_conns: Float[Tensor, "n_features d_mlp"],
    model: ResidualMLPModel | ResidualMLPSPDModel,
    n_features: int,
    pre_labelled_neurons: dict[int, list[int]] | None = None,
    legend: bool = True,
) -> dict[int, list[int]]:
    diag_relu_conns: Float[Tensor, "n_features d_mlp"] = all_diag_relu_conns.cpu().detach()
    d_mlp = model.config.d_mlp
    n_layers = model.config.n_layers

    # Define colors for different layers
    assert n_layers in [1, 2]
    layer_colors = ["grey"] if n_layers == 1 else ["blue", "red"]
    distinct_colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#A65628",  # brown
        "#F781BF",  # pink
        "#1B9E77",  # teal
        "#D95F02",  # dark orange
        "#7570B3",  # slate blue
        "#66A61E",  # lime green
    ]

    # Add legend if there are two layers
    if n_layers == 2 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.legend(loc="upper right")

    labelled_neurons: dict[int, list[int]] = {i: [] for i in range(n_features)}

    ax.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(n_features):
        # Split points by layer and plot separately
        for layer in range(n_layers):
            layer_indices = slice(layer * d_mlp, (layer + 1) * d_mlp)
            ax.scatter(
                [i] * d_mlp,
                diag_relu_conns[i, layer_indices],
                alpha=0.3,
                marker=".",
                c=layer_colors[layer],
            )
        ax.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for j in range(d_mlp * n_layers):
            # Label the neuron if it's in the pre-labelled set or if no pre-labelled set is provided
            # and the neuron has a connection strength greater than 0.1
            if (pre_labelled_neurons is not None and j in pre_labelled_neurons[i]) or (
                pre_labelled_neurons is None and diag_relu_conns[i, j].item() > 0.1
            ):
                color_idx = j % len(distinct_colors)
                # Make the neuron label alternate between left and right (-0.1, 0.1)
                # Add 0.05 or -0.05 to the x coordinate to shift the label left or right
                ax.text(
                    i,
                    diag_relu_conns[i, j].item(),
                    str(j),
                    color=distinct_colors[color_idx],
                    ha="left" if (len(labelled_neurons[i]) + 1) % 2 == 0 else "right",
                )
                labelled_neurons[i].append(j)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_xlabel("Features")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return labelled_neurons


def spd_calculate_virtual_weights(model: ResidualMLPSPDModel, device: str) -> dict[str, Tensor]:
    """Currently ignoring interactions between layers. Just flattening (n_layers, d_mlp)"""
    old_device = next(model.parameters()).device
    model.to(device)
    n_instances = model.config.n_instances
    n_features = model.config.n_features
    d_embed = model.config.d_embed
    d_mlp = model.config.d_mlp
    C = model.config.C
    has_bias1 = model.layers[0].bias1 is not None
    has_bias2 = model.layers[0].bias2 is not None
    n_layers = model.config.n_layers
    # Get weights
    W_E: Float[Tensor, "n_instances n_features d_embed"] = model.W_E
    W_U: Float[Tensor, "n_instances d_embed n_features"] = model.W_U
    W_in: Float[Tensor, "n_instances C d_embed d_mlp_eff"] = torch.cat(
        [model.layers[i].mlp_in.component_weights for i in range(n_layers)], dim=-1
    )
    W_out: Float[Tensor, "n_instances C d_mlp_eff d_embed"] = torch.cat(
        [model.layers[i].mlp_out.component_weights for i in range(n_layers)],
        dim=-2,
    )
    b_in: Float[Tensor, "n_instances C d_mlp_eff"] | None = (
        torch.cat([model.layers[i].bias1 for i in range(n_layers)], dim=-1) if has_bias1 else None
    )
    b_out: Float[Tensor, "n_instances C d_embed"] | None = (
        torch.stack([model.layers[i].bias2 for i in range(n_layers)]).sum(dim=0)
        if has_bias2
        else None
    )
    assert W_E.shape == (n_instances, n_features, d_embed)
    assert W_U.shape == (n_instances, d_embed, n_features)
    assert W_in.shape == (n_instances, C, d_embed, n_layers * d_mlp)
    assert W_out.shape == (n_instances, C, n_layers * d_mlp, d_embed)
    assert b_in.shape == (n_instances, C, n_layers * d_mlp) if b_in is not None else True
    assert b_out.shape == (n_instances, C, d_embed) if b_out is not None else True
    # Calculate connection strengths / virtual weights
    in_conns: Float[Tensor, "n_instances C n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_instances n_features d_embed, n_instances C d_embed d_mlp -> n_instances C n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_instances C d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_instances C d_mlp d_embed, n_instances n_features d_embed -> n_instances C d_mlp n_features",
    )
    diag_relu_conns: Float[Tensor, "n_instances C n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_instances C n_features d_mlp, n_instances C d_mlp n_features -> n_instances C n_features d_mlp",
    )
    assert in_conns.shape == (n_instances, C, n_features, n_layers * d_mlp)
    assert out_conns.shape == (n_instances, C, n_layers * d_mlp, n_features)
    assert diag_relu_conns.shape == (n_instances, C, n_features, n_layers * d_mlp)
    virtual_weights = {
        "W_E": W_E,
        "W_U": W_U,
        "W_in": W_in,
        "W_out": W_out,
        "in_conns": in_conns,
        "out_conns": out_conns,
        "diag_relu_conns": diag_relu_conns,
    }
    if b_in is not None:
        virtual_weights["b_in"] = b_in
    if b_out is not None:
        virtual_weights["b_out"] = b_out

    model.to(old_device)
    return virtual_weights


def spd_calculate_diag_relu_conns(
    model: ResidualMLPSPDModel,
    device: str,
    k_select: int | Literal["sum_before", "sum_nocrossterms", "sum_onlycrossterms"] = 0,
) -> Float[Tensor, "n_instances n_features d_mlp"]:
    virtual_weights = spd_calculate_virtual_weights(model, device)
    if isinstance(k_select, int):
        return virtual_weights["diag_relu_conns"][:, k_select]
    elif k_select == "sum_nocrossterms":
        return virtual_weights["diag_relu_conns"].sum(dim=1)
    else:
        in_conns: Float[Tensor, "n_instances C n_features d_mlp"] = virtual_weights["in_conns"]
        out_conns: Float[Tensor, "n_instances C d_mlp n_features"] = virtual_weights["out_conns"]
        if k_select == "sum_onlycrossterms":
            nocross_diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = (
                virtual_weights["diag_relu_conns"].sum(dim=1)
            )
            all_diag_relu_conns: Float[Tensor, "n_instances k1 k2 n_features d_mlp"] = (
                einops.einsum(
                    in_conns,
                    out_conns,
                    "n_instances k1 n_features d_mlp, n_instance k2 d_mlp n_features -> n_instances k1 k2 n_features d_mlp",
                )
            )
            return all_diag_relu_conns.sum(dim=(-3, -4)) - nocross_diag_relu_conns
        elif k_select == "sum_before":
            sum_diag_relu_conns: Float[Tensor, "n_instances n_features d_mlp"] = einops.einsum(
                in_conns.sum(dim=1),
                out_conns.sum(dim=1),
                "n_instances n_features d_mlp, n_instance d_mlp n_features -> n_instances n_features d_mlp",
            )
            return sum_diag_relu_conns
        else:
            raise ValueError(f"Invalid k_select: {k_select}")


def plot_spd_relu_contribution(
    spd_model: ResidualMLPSPDModel,
    target_model: ResidualMLPModel,
    device: str = "cuda",
    k_plot_limit: int | None = None,
):
    offset = 4
    nrows = (k_plot_limit or spd_model.config.C) + offset
    fig1, axes1 = plt.subplots(nrows, 1, figsize=(20, 3 + 2 * nrows), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)  # type: ignore
    fig2, axes2 = plt.subplots(nrows, 1, figsize=(10, 3 + 2 * nrows), constrained_layout=True)
    axes2 = np.atleast_1d(axes2)  # type: ignore

    virtual_weights = calculate_virtual_weights(target_model, device)
    relu_conns = virtual_weights["diag_relu_conns"]
    relu_contribution_plot(axes1[0], axes2[0], relu_conns, target_model, device)
    axes1[0].set_ylabel("Target model", fontsize=8)
    axes2[0].set_ylabel("Target model", fontsize=8)
    axes1[0].set_xlabel("")
    axes2[0].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_before")
    relu_contribution_plot(axes1[1], axes2[1], relu_conns, spd_model, device)
    axes1[1].set_ylabel("SPD model full sum of all subnets", fontsize=8)
    axes2[1].set_ylabel("SPD model full sum of all subnets", fontsize=8)
    axes1[1].set_xlabel("")
    axes2[1].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_nocrossterms")
    relu_contribution_plot(axes1[2], axes2[2], relu_conns, spd_model, device)
    axes1[2].set_ylabel("SPD model sum without cross terms", fontsize=8)
    axes2[2].set_ylabel("SPD model sum without cross terms", fontsize=8)
    axes1[2].set_xlabel("")
    axes2[2].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_onlycrossterms")
    relu_contribution_plot(axes1[3], axes2[3], relu_conns, spd_model, device)
    axes1[3].set_ylabel("SPD model sum only cross terms", fontsize=8)
    axes2[3].set_ylabel("SPD model sum only cross terms", fontsize=8)
    axes1[3].set_xlabel("")
    axes2[3].set_xlabel("")
    for c in range(k_plot_limit or spd_model.config.C):
        relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select=c)
        relu_contribution_plot(axes1[c + offset], axes2[c + offset], relu_conns, spd_model, device)
        axes1[c + offset].set_ylabel(f"k={c}")
        axes2[c + offset].set_ylabel(f"k={c}")
        if (k_plot_limit or spd_model.config.C) - 1 > c:
            axes1[c + offset].set_xlabel("")
            axes2[c + offset].set_xlabel("")
    return fig1, fig2


def plot_spd_feature_contributions(
    spd_model: ResidualMLPSPDModel,
    target_model: ResidualMLPModel,
    device: str = "cuda",
    k_plot_limit: int | None = None,
) -> plt.Figure:
    instance_idx = 0
    offset = 4
    nrows = (k_plot_limit or spd_model.config.C) + offset
    fig1, axes1 = plt.subplots(nrows, 1, figsize=(20, 3 + 2 * nrows), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)  # type: ignore

    n_features = spd_model.config.n_features

    virtual_weights = calculate_virtual_weights(target_model, device)
    relu_conns = virtual_weights["diag_relu_conns"]
    feature_contribution_plot(
        ax=axes1[0],
        all_diag_relu_conns=relu_conns[instance_idx],
        model=target_model,
        n_features=n_features,
    )
    axes1[0].set_ylabel("Target model", fontsize=8)
    axes1[0].set_xlabel("")
    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_before")
    feature_contribution_plot(
        ax=axes1[1],
        all_diag_relu_conns=relu_conns[instance_idx],
        model=spd_model,
        n_features=n_features,
    )
    axes1[1].set_ylabel("SPD model full sum of all subnets", fontsize=8)
    axes1[1].set_xlabel("")

    # We now use max component instead of sum_nocrossterms now
    # relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_nocrossterms")
    # relu_contribution_plot(axes1[2], axes2[2], relu_conns, spd_model, device)
    # axes1[2].set_ylabel("SPD model sum without cross terms", fontsize=8)
    # axes2[2].set_ylabel("SPD model sum without cross terms", fontsize=8)
    # axes1[2].set_xlabel("")
    # axes2[2].set_xlabel("")

    diag_relu_conns: Float[Tensor, "C n_features d_mlp"] = spd_calculate_virtual_weights(
        spd_model, device
    )["diag_relu_conns"][instance_idx]
    max_component_indices = diag_relu_conns.max(dim=-1).values.argmax(dim=0)
    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_features d_mlp"] = diag_relu_conns[
        max_component_indices, torch.arange(spd_model.config.n_features)
    ]
    feature_contribution_plot(
        ax=axes1[2],
        all_diag_relu_conns=max_component_contributions,
        model=spd_model,
        n_features=n_features,
    )
    axes1[2].set_ylabel("SPD model max component", fontsize=8)
    axes1[2].set_xlabel("")
    # Label the x axis with the subnets that have the largest neuron for each feature
    # Set xticks for every index
    axes1[2].set_xticks(range(n_features))
    axes1[2].set_xticklabels(max_component_indices.tolist())
    axes1[2].tick_params(axis="x", labelsize=6)

    relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select="sum_onlycrossterms")
    # relu_contribution_plot(axes1[3], relu_conns, spd_model, device)
    feature_contribution_plot(
        ax=axes1[3],
        all_diag_relu_conns=relu_conns[instance_idx],
        model=spd_model,
        n_features=n_features,
    )
    axes1[3].set_ylabel("SPD model sum only cross terms", fontsize=8)
    axes1[3].set_xlabel("")

    # Use the same y-axis max for all plots
    y_min = min([axes1[i].get_ylim()[0] for i in range(4)])
    y_max = max([axes1[i].get_ylim()[1] for i in range(4)])
    axes1[0].set_ylim(y_min, y_max)
    axes1[1].set_ylim(y_min, y_max)
    axes1[2].set_ylim(y_min, y_max)
    axes1[3].set_ylim(y_min, y_max)

    for c in range(k_plot_limit or spd_model.config.C):
        relu_conns = spd_calculate_diag_relu_conns(spd_model, device, k_select=c)
        # relu_contribution_plot(axes1[k + offset], relu_conns, spd_model, device)
        feature_contribution_plot(
            ax=axes1[c + offset],
            all_diag_relu_conns=relu_conns[instance_idx],
            model=spd_model,
            n_features=n_features,
        )
        axes1[c + offset].set_ylabel(f"k={c}")
        if (k_plot_limit or spd_model.config.C) - 1 > c:
            axes1[c + offset].set_xlabel("")
    return fig1


def plot_spd_feature_contributions_truncated(
    spd_model: ResidualMLPSPDModel,
    target_model: ResidualMLPModel,
    device: str = "cuda",
    n_features: int | None = 10,
    include_crossterms: bool = False,
):
    assert spd_model.config.n_instances == 1, "Only one instance supported for now"

    n_features = n_features or spd_model.config.n_features
    n_rows = 3 if include_crossterms else 2
    fig1, axes1 = plt.subplots(n_rows, 1, figsize=(10, 7), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)  # type: ignore

    # First plot: Target model
    virtual_weights = calculate_virtual_weights(target_model, device)
    relu_conns: Float[Tensor, "n_features d_mlp"] = virtual_weights["diag_relu_conns"][
        0, :n_features, :
    ]
    labelled_neurons = feature_contribution_plot(
        axes1[0],
        relu_conns,
        model=target_model,
        n_features=n_features,
        legend=True,
    )
    axes1[0].set_ylabel("Neuron contribution")
    axes1[0].set_xlabel(f"Input feature index (first {n_features} shown)")
    axes1[0].set_title("Target model")
    axes1[0].set_xticks(range(n_features))  # Ensure all xticks have labels

    # Second plot: SPD model (without cross terms max k)
    # Previously we would just use no_crossterms which sums over all k
    # spd_relu_conns: Float[Tensor, "n_features d_mlp"] = spd_calculate_diag_relu_conns(
    #     spd_model, device, k_select="sum_nocrossterms"
    # )[0, :n_features, :]
    # Instead, we want find the C which has the largest neuron for each feature index
    diag_relu_conns: Float[Tensor, "C n_features d_mlp"] = spd_calculate_virtual_weights(
        spd_model, device
    )["diag_relu_conns"][0, :, :n_features, :]
    max_component_indices = diag_relu_conns.max(dim=-1).values.argmax(dim=0)
    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_features d_mlp"] = diag_relu_conns[
        max_component_indices, torch.arange(n_features)
    ]
    feature_contribution_plot(
        axes1[1],
        max_component_contributions,
        model=spd_model,
        n_features=n_features,
        pre_labelled_neurons=labelled_neurons,
        legend=False,
    )
    axes1[1].set_ylabel("Neuron contribution")
    axes1[1].set_xlabel("Parameter component index")
    axes1[1].set_title("Individual APD parameter components")
    axes1[1].set_xticks(range(n_features))

    # Set the same y-axis limits for both plots
    y_min = min(axes1[0].get_ylim()[0], axes1[1].get_ylim()[0])
    y_max = max(axes1[0].get_ylim()[1], axes1[1].get_ylim()[1])
    axes1[0].set_ylim(y_min, y_max)
    axes1[1].set_ylim(y_min, y_max)

    # Label the x axis with the subnets that have the largest neuron for each feature
    axes1[1].set_xticklabels(max_component_indices.tolist())  # Labels are the subnet indices

    if include_crossterms:
        # Third plot: SPD model (with cross terms)
        spd_relu_conns: Float[Tensor, "n_features d_mlp"] = spd_calculate_diag_relu_conns(
            spd_model, device, k_select="sum_onlycrossterms"
        )[0, :n_features, :]
        feature_contribution_plot(
            axes1[2], spd_relu_conns, model=spd_model, n_features=n_features, legend=False
        )
        axes1[2].set_ylabel("Neuron contribution")
        axes1[2].set_xlabel("Input feature index")
        axes1[2].set_title("Input feature cross terms")
        axes1[2].set_xticks(range(n_features))

    return fig1


def collect_per_feature_losses(
    target_model: ResidualMLPModel,
    spd_model: ResidualMLPSPDModel,
    config: Config,
    dataset: ResidualMLPDataset,
    device: str,
    batch_size: int,
    n_samples: int,
) -> tuple[
    Float[Tensor, " n_features"], Float[Tensor, " n_features"], Float[Tensor, " n_features"]
]:
    """Collect the MSE losses for the target model, SPD batch topk SPD sample topk.

    Returns:
        loss_target: Float[Tensor, " n_features"]
        loss_spd_batch_topk: Float[Tensor, " n_features"]
        loss_spd_sample_topk: Float[Tensor, " n_features"]
    """
    print_every_counter = 10000
    n_samples_acc = 0
    feature_counts = torch.zeros(spd_model.config.n_features, device=device)
    loss_target = torch.zeros(spd_model.config.n_features, device=device)
    loss_spd_batch_topk = torch.zeros(spd_model.config.n_features, device=device)
    loss_spd_sample_topk = torch.zeros(spd_model.config.n_features, device=device)

    while n_samples_acc < n_samples:
        # Generate batch with no zero samples
        batch, labels = dataset.generate_batch(batch_size)
        batch = batch.to(device)
        labels = labels.to(device)

        # n_instances should be 1
        assert batch.shape[1] == 1

        # Forward pass on target model
        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            batch, names_filter=target_cache_filter
        )

        # Do a forward pass with all subnetworks
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        spd_out, spd_cache = spd_model.run_with_cache(batch, names_filter=spd_cache_filter)

        attribution_scores = calculate_attributions(
            model=spd_model,
            batch=batch,
            out=spd_out,
            target_out=target_out,
            pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
            post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
            component_acts={
                k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")
            },
            attribution_type=config.attribution_type,
        )

        with torch.inference_mode():
            assert config.batch_topk is True
            assert config.topk is not None
            # Get the topk mask for the batch topk model
            batch_topk_mask = calc_topk_mask(
                attribution_scores, topk=config.topk, batch_topk=config.batch_topk
            )
            # Get the topk mask for the sample topk model
            sample_topk_mask = calc_topk_mask(attribution_scores, topk=1, batch_topk=False)

            # Get the batch topk model output
            spd_out_batch_topk = spd_model(batch, topk_mask=batch_topk_mask)
            # Get the sample topk model output
            spd_out_sample_topk = spd_model(batch, topk_mask=sample_topk_mask)

            # Get rid of the n_instances dimension for simplicity
            batch: Float[Tensor, "batch n_features"] = batch.squeeze(1)
            batch_topk_mask: Float[Tensor, "batch C"] = batch_topk_mask.squeeze(1)
            target_out: Float[Tensor, "batch n_features"] = target_out.squeeze(1)
            spd_out_batch_topk: Float[Tensor, "batch n_features"] = spd_out_batch_topk.squeeze(1)
            spd_out_sample_topk: Float[Tensor, "batch n_features"] = spd_out_sample_topk.squeeze(1)
            labels: Float[Tensor, "batch n_features"] = labels.squeeze(1)

            # Get the indices of samples where there is exactly one feature active
            active_features_batch: Float[Tensor, "batch n_features"] = (batch != 0).float()
            exactly_one_active_features_batch: Float[Tensor, "batch n_features"] = (
                active_features_batch.sum(dim=-1) == 1
            )

            # Filter to only include the samples where there is exactly one feature active
            batch = batch[exactly_one_active_features_batch]
            labels = labels[exactly_one_active_features_batch]
            target_out = target_out[exactly_one_active_features_batch]
            spd_out_batch_topk = spd_out_batch_topk[exactly_one_active_features_batch]
            spd_out_sample_topk = spd_out_sample_topk[exactly_one_active_features_batch]
            filtered_active_features_batch: Float[Tensor, "sub_batch n_features"] = (
                active_features_batch[exactly_one_active_features_batch]
            )

            # Get the Squared error loss for each sample
            loss_target_batch_raw: Float[Tensor, "sub_batch 1"] = ((target_out - labels) ** 2).sum(
                dim=-1, keepdim=True
            )
            loss_spd_batch_topk_batch_raw: Float[Tensor, "sub_batch 1"] = (
                (spd_out_batch_topk - labels) ** 2
            ).sum(dim=-1, keepdim=True)
            loss_spd_sample_topk_batch_raw: Float[Tensor, "sub_batch 1"] = (
                (spd_out_sample_topk - labels) ** 2
            ).sum(dim=-1, keepdim=True)

            # Element-wise multiply the loss by the active features
            loss_target_batch: Float[Tensor, "sub_batch n_features"] = (
                loss_target_batch_raw * filtered_active_features_batch
            )
            loss_spd_batch_topk_batch: Float[Tensor, "sub_batch n_features"] = (
                loss_spd_batch_topk_batch_raw * filtered_active_features_batch
            )
            loss_spd_sample_topk_batch: Float[Tensor, "sub_batch n_features"] = (
                loss_spd_sample_topk_batch_raw * filtered_active_features_batch
            )

            # Count the number of times each feature was active
            feature_counts += filtered_active_features_batch.sum(dim=0)

            # Add to the losses
            loss_target += loss_target_batch.sum(dim=0)
            loss_spd_batch_topk += loss_spd_batch_topk_batch.sum(dim=0)
            loss_spd_sample_topk += loss_spd_sample_topk_batch.sum(dim=0)
            n_samples_acc += batch.shape[0]

            if n_samples_acc > print_every_counter:
                print(f"n_samples_acc: {n_samples_acc}, n_samples: {n_samples}")
                print_every_counter += 10_000

    # Normalize the losses by the number of samples each feature was active for and n_features
    loss_target /= feature_counts * spd_model.config.n_features
    loss_spd_batch_topk /= feature_counts * spd_model.config.n_features
    loss_spd_sample_topk /= feature_counts * spd_model.config.n_features

    print(f"n_samples_acc: {n_samples_acc}, n_samples: {n_samples}")
    return (
        loss_target.detach().cpu(),
        loss_spd_batch_topk.detach().cpu(),
        loss_spd_sample_topk.detach().cpu(),
    )


def collect_average_components_per_feature(
    model_fn: Callable[
        [Float[Tensor, "batch n_instances n_features"]],
        SPDOutputs,
    ],
    dataset: ResidualMLPDataset,
    device: str,
    n_features: int,
    batch_size: int,
    n_samples: int,
    exactly_one_active: bool = True,
) -> Float[Tensor, " n_features"]:
    """Collect the average number of components active per feature when that feature is active."""
    # Initialize counters
    feature_active_count = torch.zeros(n_features, device=device)
    component_active_count = torch.zeros(n_features, device=device)

    for _ in tqdm(range(n_samples // batch_size)):
        # Generate batch with no zero samples
        batch = dataset._generate_multi_feature_batch(batch_size)
        batch = batch.to(device)

        # n_instances should be 1
        assert batch.shape[1] == 1

        # Get which components were active for each feature
        topk_mask_raw: Float[Tensor, "batch n_instances C"] = model_fn(batch).topk_mask

        batch: Float[Tensor, "batch n_features"] = batch.squeeze(1)
        topk_mask: Float[Tensor, "batch C"] = topk_mask_raw.squeeze(1)

        active_features_batch: Float[Tensor, "batch n_features"] = (batch != 0).float()

        if exactly_one_active:
            # Get the indices of samples where there is exactly one feature active
            exactly_one_active_features_batch: Float[Tensor, "batch n_features"] = (
                active_features_batch.sum(dim=-1) == 1
            )
            # Filter the batch to only include the samples where there is exactly one feature active
            # batch = batch[exactly_one_active_features_batch]
            active_features_batch = active_features_batch[exactly_one_active_features_batch]
            topk_mask = topk_mask[exactly_one_active_features_batch]

        # Get the number of components active for each sample
        n_components_active: Float[Tensor, "batch 1"] = topk_mask.sum(dim=-1, keepdim=True)

        # Multiply the number of components active by the number of features active
        n_components_active_times_features_active_batch: Float[Tensor, "batch n_features"] = (
            n_components_active * active_features_batch
        )

        # Sum over the batch
        n_components_active_times_features_active = (
            n_components_active_times_features_active_batch.sum(dim=0)
        )

        # Get the number of features that were active
        n_features_active = active_features_batch.sum(dim=0)

        # Add to the counters
        feature_active_count += n_features_active
        component_active_count += n_components_active_times_features_active

    # Calculate average components per feature
    avg_components = component_active_count / feature_active_count
    return avg_components


def analyze_per_feature_performance(
    model_fn: Callable[[Float[Tensor, "batch n_instances"]], Float[Tensor, "batch n_instances"]],
    model_config: ResidualMLPConfig | ResidualMLPSPDConfig,
    device: str,
    batch_size: int = 128,
    target_model_fn: Callable[
        [Float[Tensor, "batch n_instances"]], Float[Tensor, "batch n_instances"]
    ]
    | None = None,
) -> Float[Tensor, " n_features"]:
    """For each feature, run a bunch where only that feature varies, then measure loss"""
    n_features = model_config.n_features
    n_instances = model_config.n_instances
    losses = torch.zeros(model_config.n_features)
    assert n_instances == 1, "Only one instance supported for now"
    label_fn = F.relu if model_config.act_fn_name == "relu" else F.gelu
    for i in range(model_config.n_features):
        batch_i = torch.zeros((batch_size, n_instances, n_features), device=device)
        batch_i[:, 0, i] = torch.linspace(-1, 1, batch_size)
        model_output = model_fn(batch_i)
        if target_model_fn is not None:
            # Get the labels from the target model if it's provided
            labels_i = target_model_fn(batch_i)
        else:
            labels_i = torch.zeros((batch_size, n_instances, n_features), device=device)
            labels_i[:, 0, i] = batch_i[:, 0, i] + label_fn(batch_i[:, 0, i])
        loss = F.mse_loss(model_output, labels_i)
        losses[i] = loss.item()
    losses = losses.detach().cpu()
    return losses


def plot_per_feature_performance(
    losses: Float[Tensor, " n_features"],
    sorted_indices: Float[Tensor, " n_features"] | None = None,
    ax: plt.Axes | None = None,
    label: str | None = None,
    color: str = "C0",
    zorder: int = 0,
    show_xticks: bool = False,
):
    sorted_indices = sorted_indices if sorted_indices is not None else losses.argsort()
    # Plot the losses as bar chart with x labels corresponding to feature index
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    features = torch.arange(losses.shape[0])
    ax.bar(features, losses[sorted_indices], label=label, zorder=zorder, color=color)
    if show_xticks:
        ax.set_xticks(features, features[sorted_indices].numpy(), fontsize=6, rotation=90)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_xlabel("Input feature index (sorted by target model MSE)")
    ax.set_ylabel("MSE w.r.t true labels")


def plot_virtual_weights_target_spd(
    target_model: ResidualMLPModel, model: ResidualMLPSPDModel, device: str
):
    target_virtual_weights = calculate_virtual_weights(target_model, device)
    spd_virtual_weights = spd_calculate_virtual_weights(model=model, device=device)
    instance_idx = 0
    fig = plt.figure(constrained_layout=True, figsize=(10, 2 * model.config.C + 8))
    gs = fig.add_gridspec(ncols=2, nrows=model.config.C + 1 + 2)
    ax_ID = fig.add_subplot(gs[:2, :])
    W_E_W_U = einops.einsum(
        target_virtual_weights["W_E"][instance_idx],
        target_virtual_weights["W_U"][instance_idx],
        "n_features1 d_embed, d_embed n_features2 -> n_features1 n_features2",
    )
    plot_matrix(
        ax_ID,
        W_E_W_U,
        "Virtual weights $W_E W_U$",
        "Features",
        "Features",
        colorbar_format="%.2f",
    )
    norm = Normalize(vmin=-1, vmax=1)
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    in_conns = target_virtual_weights["in_conns"][instance_idx].cpu().detach()
    out_conns = target_virtual_weights["out_conns"][instance_idx].cpu().detach()
    plot_matrix(
        ax1,
        in_conns.T,
        "Virtual input weights $(W_E W_{in})^T$",
        "Features",
        "(Target Model) Neurons",
        colorbar_format="%.2f",
        norm=norm,
    )
    plot_matrix(
        ax2,
        out_conns,
        "Virtual output weights $W_{out} W_U$",
        "Features",
        "Neurons",
        colorbar_format="%.2f",
        norm=norm,
    )
    for c in range(model.config.C):
        ax1 = fig.add_subplot(gs[3 + c, 0])
        ax2 = fig.add_subplot(gs[3 + c, 1])
        plot_matrix(
            ax1,
            spd_virtual_weights["in_conns"][instance_idx, c].T,
            "$(W_E W_{in})^T$",
            "Features",
            f"c={c} Neurons",
            colorbar_format="%.2f",
            norm=norm,
        )
        plot_matrix(
            ax2,
            spd_virtual_weights["out_conns"][instance_idx, c],
            "$W_{out} W_U$",
            "Features",
            "Neurons",
            colorbar_format="%.2f",
            norm=norm,
        )
    return fig


def plot_resid_vs_mlp_out(
    target_model: ResidualMLPModel,
    device: str,
    ax: plt.Axes,
    topk_model_fn: Callable[
        [
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances C"] | None,
        ],
        SPDOutputs,
    ]
    | None = None,
    subnet_indices: Float[Tensor, " C"] | None = None,
    instance_idx: int = 0,
    feature_idx: int = 0,
):
    tied_weights = True
    if not torch.allclose(target_model.W_U.data, target_model.W_E.data.transpose(-2, -1)):
        print("Warning: W_E and W_U are not tied")
        tied_weights = False
    batch_size = 1
    batch_idx = 0
    n_instances = target_model.config.n_instances
    n_features = target_model.config.n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[:, instance_idx, feature_idx] = 1
    # Target model full output
    out = target_model(batch)[batch_idx, instance_idx, :].cpu().detach()
    # Target model residual stream contribution
    W_E = target_model.W_E[instance_idx].cpu().detach()
    W_U = target_model.W_U[instance_idx].cpu().detach()
    W_EU = einops.einsum(W_E, W_U, "f1 d_mlp, d_mlp f2 -> f1 f2")[feature_idx, :]
    # Compute MLP-out
    mlp_out = out - W_EU
    # Mask for noise & correlation
    mask = torch.ones_like(out).bool()
    mask[feature_idx] = False
    noise_out = F.mse_loss(out[mask], torch.zeros_like(out[mask])).item()
    corr = np.corrcoef(mlp_out[mask], W_EU[mask])[0, 1]
    ax.axhline(0, color="grey", linestyle="-", lw=0.5)
    ax.plot([], [], c="white", label=f"Full target model noise level ~ {noise_out:.2e}")
    ax.plot(
        mlp_out,
        color="C0",
        label=f"Target MLP output.\n"
        f"Corr w/ resid (excluding feature {feature_idx}): {corr:.2f}",
        lw=2,
    )
    noise_W_EU = F.mse_loss(W_EU[mask], torch.zeros_like(W_EU[mask])).item()
    ax.plot(
        W_EU,
        color="C1",
        label=f"Target resid contribution (W_E W_U)\n" f"Noise level ~ {noise_W_EU:.2e}",
    )
    # If topk_model_fn is provided, use it to get the SPD model output
    if topk_model_fn is not None:
        # Get the SPD resid contribution by running with no subnetworks. This should be equivalent
        # to W_E W_U and but doesn't require access to the ResidMLP SPD model.
        topk_mask = torch.zeros_like(batch)
        spd_WEU = topk_model_fn(batch, topk_mask).spd_topk_model_output[batch_idx, instance_idx, :]
        spd_WEU = spd_WEU.detach().cpu()
        if tied_weights:
            assert torch.allclose(spd_WEU, W_EU), "Tied weights but W_EU != SPD resid contribution"
        else:
            ax.plot(
                spd_WEU,
                color="C4",
                label="SPD resid contribution (no subnets).\n"
                "Note that embeddings are untied and numbers in legend are not applicable",
                ls=":",
            )
        # Get SPD forward pass, either from subnet_indices or attribution-based topk_mask
        if subnet_indices is None:
            topk_mask = None
        else:
            topk_mask = torch.zeros_like(batch)
            topk_mask[:, :, subnet_indices] = 1
        topk_out = topk_model_fn(batch, topk_mask).spd_topk_model_output[batch_idx, instance_idx, :]
        topk_mlp_out = topk_out.detach().cpu() - spd_WEU
        topk_mlp_out_mse = F.mse_loss(topk_mlp_out, mlp_out).item()
        corr = np.corrcoef(topk_mlp_out[mask], W_EU[mask])[0, 1]
        ax.plot(
            topk_mlp_out,
            color="C2",
            label=f"SPD MLP output (topk) MSE: {topk_mlp_out_mse:.1e}.\n"
            f"Corr w/ resid (excluding feature {feature_idx}): {corr:.2f}",
            ls="--",
        )
        # Full forward pass
        topk_mask = torch.ones_like(batch)
        full_out = topk_model_fn(batch, topk_mask).spd_topk_model_output[batch_idx, instance_idx, :]
        full_mlp_out = full_out.detach().cpu() - spd_WEU
        full_mlp_out_mse = F.mse_loss(full_mlp_out, mlp_out).item()
        corr = np.corrcoef(full_mlp_out[mask], W_EU[mask])[0, 1]
        ax.plot(
            full_mlp_out,
            color="C3",
            label=f"SPD MLP output (full) MSE: {full_mlp_out_mse:.1e}.\n"
            f"Corr w/ resid (excluding feature {feature_idx}): {corr:.2f}",
            ls=":",
        )
    # Can we scale W_EU by a scalar to make it match the model output in mask?
    # def difference(alpha):
    #     return F.mse_loss( float(alpha) * W_EU[mask], out[mask])
    # from scipy.optimize import minimize
    # res = minimize(difference, x0=0.1, method="Nelder-Mead")
    # ax.plot(W_EU * float(res.x[0]), color="C2", label="Scaled W_E W_U")
    ax.legend()
    ax.set_title(f"Instance {instance_idx}, feature {feature_idx}")


def plot_per_feature_performance_fig(
    loss_target: Float[Tensor, " n_features"],
    loss_spd_batch_topk: Float[Tensor, " n_features"],
    loss_spd_sample_topk: Float[Tensor, " n_features"],
    config: Config,
    color_map: dict[str, str],
) -> plt.Figure:
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs = np.array(axs)

    indices = loss_target.argsort()
    topk = int(config.topk) if config.topk is not None and config.topk == 1 else config.topk
    plot_per_feature_performance(
        losses=loss_spd_batch_topk,
        sorted_indices=indices,
        ax=axs[1],
        label=f"APD (per-batch top-k={topk})",
        color=color_map["apd_topk"],
    )

    plot_per_feature_performance(
        losses=loss_target,
        sorted_indices=indices,
        ax=axs[1],
        label="Target model",
        color=color_map["target"],
    )
    axs[1].legend(loc="upper left")

    plot_per_feature_performance(
        losses=loss_spd_sample_topk,
        sorted_indices=indices,
        ax=axs[0],
        label="APD (per-sample top-k=1)",
        color=color_map["apd_topk"],
    )
    plot_per_feature_performance(
        losses=loss_target,
        sorted_indices=indices,
        ax=axs[0],
        label="Target model",
        color=color_map["target"],
    )

    axs[0].legend(loc="upper left", fontsize=12)
    axs[1].legend(loc="upper left", fontsize=12)

    # Use the max y-axis limit for both subplots
    max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim(0, max_ylim)
    axs[1].set_ylim(0, max_ylim)

    # Remove the top and right spines
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # # Increase the fontsize of the xlabel and ylabel
    for ax in axs:
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)

    return fig


def plot_avg_components_scatter(
    losses_spd_wrt_target: Float[Tensor, " n_features"],
    avg_components: Float[Tensor, " n_features"],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(
        losses_spd_wrt_target.abs().detach().cpu(),
        avg_components.detach().cpu(),
    )
    ax.set_xlabel("MSE between APD (per-sample top-k=1) and target model outputs")
    ax.set_ylabel("Average number of active components")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase the fontsize of the xlabel and ylabel
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    return fig


@dataclass
class ScrubbedLosses:
    loss_scrubbed: Float[Tensor, " n_features"]
    loss_antiscrubbed: Float[Tensor, " n_features"]
    loss_random: Float[Tensor, " n_features"]
    loss_spd: Float[Tensor, " n_features"]
    loss_zero: Float[Tensor, " n_features"]
    loss_monosemantic: Float[Tensor, " n_features"]
    n_samples: int


def get_scrubbed_losses(
    dataset: ResidualMLPDataset,
    model: ResidualMLPSPDModel,
    target_model: ResidualMLPModel,
    device: str,
    config: Config,
    top1_model_fn: Callable[
        [
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances C"] | None,
        ],
        SPDOutputs,
    ],
    spd_model_fn: Callable[
        [
            Float[Tensor, "batch n_instances n_features"],  # batch
            PositiveFloat | None,  # topk
            bool,  # batch_topk
        ],
        SPDOutputs,
    ],
    n_batches: int,
) -> ScrubbedLosses:
    assert model.config.n_instances == 1, "Can only handle n_instances = 1 for now"

    # Dictionary feature_idx -> subnet_idx
    subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

    batch_size = config.batch_size

    n_samples = 0
    # Initialize tensors to store all losses
    all_loss_scrubbed = []
    all_loss_antiscrubbed = []
    all_loss_random = []
    all_loss_spd = []
    all_loss_zero = []
    all_loss_monosemantic = []
    for _ in tqdm(range(n_batches)):
        # In the future this will be merged into generate_batch
        batch = dataset._generate_multi_feature_batch(batch_size)
        if isinstance(dataset, ResidualMLPDataset) and dataset.label_fn is not None:
            labels = dataset.label_fn(batch)
        else:
            labels = batch.clone().detach()

        # Count the number of samples in which there is at least one active feature
        n_samples += int((batch != 0).any(dim=-1).sum().item())

        batch = batch.to(device)
        active_features = torch.where(batch != 0)
        # Randomly assign 0 or 1 to topk mask
        random_topk_mask = torch.randint(
            0, 2, (batch_size, model.config.n_instances, model.config.C)
        )
        scrubbed_topk_mask = torch.randint(
            0, 2, (batch_size, model.config.n_instances, model.config.C)
        )
        antiscrubbed_topk_mask = torch.randint(
            0, 2, (batch_size, model.config.n_instances, model.config.C)
        )
        for b, i, f in zip(*active_features, strict=False):
            s = subnet_indices[f.item()]
            scrubbed_topk_mask[b, i, s] = 1
            antiscrubbed_topk_mask[b, i, s] = 0
        topk = config.topk
        batch_topk = config.batch_topk

        out_spd = spd_model_fn(batch, topk, batch_topk).spd_topk_model_output
        out_random = top1_model_fn(batch, random_topk_mask).spd_topk_model_output
        out_scrubbed = top1_model_fn(batch, scrubbed_topk_mask).spd_topk_model_output
        out_antiscrubbed = top1_model_fn(batch, antiscrubbed_topk_mask).spd_topk_model_output
        out_target = target_model(batch)
        # Monosemantic baseline
        out_monosemantic = batch.clone()
        d_mlp = target_model.config.d_mlp * target_model.config.n_layers  # type: ignore
        out_monosemantic[..., :d_mlp] = labels[..., :d_mlp]

        # Calc MSE losses
        all_loss_scrubbed.append(
            ((out_scrubbed - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_antiscrubbed.append(
            ((out_antiscrubbed - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_random.append(
            ((out_random - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_spd.append(((out_spd - out_target) ** 2).mean(dim=-1).flatten().detach().cpu())
        all_loss_zero.append(
            ((torch.zeros_like(out_target) - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )
        all_loss_monosemantic.append(
            ((out_monosemantic - out_target) ** 2).mean(dim=-1).flatten().detach().cpu()
        )

    # Concatenate all batches
    loss_scrubbed = torch.cat(all_loss_scrubbed)
    loss_antiscrubbed = torch.cat(all_loss_antiscrubbed)
    loss_random = torch.cat(all_loss_random)
    loss_spd = torch.cat(all_loss_spd)
    loss_zero = torch.cat(all_loss_zero)
    loss_monosemantic = torch.cat(all_loss_monosemantic)

    print(f"Loss SPD:           {loss_spd.mean().item():.6f}")
    print(f"Loss scrubbed:      {loss_scrubbed.mean().item():.6f}")
    print(f"Loss antiscrubbed:  {loss_antiscrubbed.mean().item():.6f}")
    print(f"Loss monosemantic:  {loss_monosemantic.mean().item():.6f}")
    print(f"Loss random:        {loss_random.mean().item():.6f}")
    print(f"Loss zero:          {loss_zero.mean().item():.6f}")
    return ScrubbedLosses(
        loss_scrubbed=loss_scrubbed,
        loss_antiscrubbed=loss_antiscrubbed,
        loss_random=loss_random,
        loss_spd=loss_spd,
        loss_zero=loss_zero,
        loss_monosemantic=loss_monosemantic,
        n_samples=n_samples,
    )


def plot_scrub_losses(
    losses: ScrubbedLosses, config: Config, color_map: dict[str, str], n_batches: int
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 5))
    log_bins: list[float] = np.geomspace(1e-7, losses.loss_zero.max().item(), 50).tolist()  # type: ignore
    ax.hist(
        losses.loss_spd,
        bins=log_bins,
        label="APD (top-k)",
        histtype="step",
        lw=2,
        color=color_map["apd_topk"],
    )
    ax.axvline(losses.loss_spd.mean().item(), color=color_map["apd_topk"], linestyle="--")
    ax.hist(
        losses.loss_scrubbed,
        bins=log_bins,
        label="APD (scrubbed)",
        histtype="step",
        lw=2,
        color=color_map["apd_scrubbed"],
    )
    ax.axvline(losses.loss_scrubbed.mean().item(), color=color_map["apd_scrubbed"], linestyle="--")
    ax.hist(
        losses.loss_antiscrubbed,
        bins=log_bins,
        label="APD (anti-scrubbed)",
        histtype="step",
        lw=2,
        color=color_map["apd_antiscrubbed"],
    )
    ax.axvline(
        losses.loss_antiscrubbed.mean().item(), color=color_map["apd_antiscrubbed"], linestyle="--"
    )
    # ax.hist(loss_random, bins=log_bins, label="APD (random)", histtype="step")
    # ax.hist(loss_zero, bins=log_bins, label="APD (zero)", histtype="step")
    ax.axvline(
        losses.loss_monosemantic.mean().item(),
        color=color_map["baseline_monosemantic"],
        linestyle="-",
        label="Monosemantic neuron solution",
    )
    ax.legend()
    ax.set_ylabel("Count")
    ax.set_xlabel("MSE loss with target model output")
    ax.set_xscale("log")

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase fontsize
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    return fig


def plot_feature_response_with_subnets(
    topk_model_fn: Callable[
        [Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances C"]],
        SPDOutputs,
    ],
    device: str,
    model_config: ResidualMLPSPDConfig,
    feature_idx: int = 0,
    subnet_idx: int = 0,
    instance_idx: int = 0,
    ax: plt.Axes | None = None,
    batch_size: int | None = None,
    plot_type: Literal["line", "errorbar"] = "errorbar",
    color_map: dict[str, str] | None = None,
) -> dict[str, plt.Figure]:
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = batch_size or n_features
    C = model_config.C

    if color_map is None:
        color_map = {
            "apd_topk": "C1",
            "apd_scrubbed": "C4",
            "apd_antiscrubbed": "C2",
            "target": "C0",
        }

    if ax is None:
        _, ax = plt.subplots(constrained_layout=True, figsize=(10, 5))
    fig = ax.figure

    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[:, instance_idx, feature_idx] = 1
    topk_mask_blue = torch.zeros(batch_size, n_instances, C, device=device)
    topk_mask_red = torch.zeros(batch_size, n_instances, C, device=device)
    topk_mask_blue[:, :, subnet_idx] = 1
    for s in range(batch_size):
        # Randomly ablate half the features
        half_n_features = n_features // 2
        choice = torch.randperm(n_features - 1)[:half_n_features]
        # Exclude feature_idx from choice
        choice[choice >= subnet_idx] += 1
        topk_mask_blue[s, :, choice] = 1
        topk_mask_red[s, :, choice] = 1
    assert torch.allclose(
        topk_mask_blue[:, :, subnet_idx], torch.ones_like(topk_mask_blue[:, :, subnet_idx])
    )
    assert torch.allclose(
        topk_mask_red[:, :, subnet_idx], torch.zeros_like(topk_mask_red[:, :, subnet_idx])
    )
    zeros_topk_mask = torch.zeros(batch_size, n_instances, C, device=device)
    ones_topk_mask = torch.ones(batch_size, n_instances, C, device=device)
    out_WE_WU_only = topk_model_fn(batch, zeros_topk_mask).spd_topk_model_output[:, instance_idx, :]

    out_red = topk_model_fn(batch, topk_mask_red)
    out_blue = topk_model_fn(batch, topk_mask_blue)
    out_spd = topk_model_fn(batch, ones_topk_mask).spd_topk_model_output[:, instance_idx, :]
    mlp_out_blue_spd = out_blue.spd_topk_model_output[:, instance_idx, :] - out_WE_WU_only
    mlp_out_red_spd = out_red.spd_topk_model_output[:, instance_idx, :] - out_WE_WU_only
    mlp_out_target = out_blue.target_model_output[:, instance_idx, :] - out_WE_WU_only
    mlp_out_spd = out_spd - out_WE_WU_only

    x = torch.arange(n_features)

    if plot_type == "errorbar":
        # Calculate means and stds across batch dimension
        blue_mean = mlp_out_blue_spd.mean(dim=0).detach().cpu()
        blue_std = mlp_out_blue_spd.std(dim=0).detach().cpu()
        red_mean = mlp_out_red_spd.mean(dim=0).detach().cpu()
        red_std = mlp_out_red_spd.std(dim=0).detach().cpu()
        mlp_out_spd_mean = mlp_out_spd.mean(dim=0).detach().cpu()

        # Plot errorbars
        ax.errorbar(
            x,
            blue_mean,
            yerr=blue_std,
            color=color_map["apd_scrubbed"],
            label="APD (scrubbed)",
            fmt="o",
            markersize=2,
        )
        ax.errorbar(
            x,
            red_mean,
            yerr=red_std,
            color=color_map["apd_antiscrubbed"],
            label="APD (anti-scrubbed)",
            fmt="o",
            markersize=2,
        )

        # Plot target model output
        yt = mlp_out_target[0, :].detach().cpu()
        ax.scatter(
            x, yt, color=color_map["target"], label="Target model\n≈APD (top-k)", marker="x", s=10
        )
        # Plot non-scrubbed SPD model output
        # ax.scatter(
        #     x,
        #     mlp_out_spd_mean.detach().cpu(),
        #     color=color_map["apd_topk"],
        #     label="APD (top-k)",
        #     marker=".",
        #     s=10,
        # )
        # Remove all axes lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
    elif plot_type == "line":
        cmap1 = plt.get_cmap("Purples")
        cmap2 = plt.get_cmap("Oranges")
        for s in range(batch_size):
            yb = mlp_out_blue_spd[s, :].detach().cpu()
            yr = mlp_out_red_spd[s, :].detach().cpu()
            if plot_type == "line":
                ax.plot(x, yb, color=cmap1(s / batch_size), lw=0.3)
                ax.plot(x, yr, color=cmap2(s / batch_size), lw=0.3)
        ax.plot([], [], color=cmap1(0), label="APD (scrubbed)")
        ax.plot([], [], color=cmap2(0), label="APD (anti-scrubbed)")
        yt = mlp_out_target[0, :].detach().cpu()
        ax.plot(x, yt, color="red", lw=0.5, label="Target model")
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")

    ax.set_ylabel("MLP output (forward pass minus W_E W_U contribution)")
    ax.set_xlabel("Output index")

    # I only need 0, feature_idx, and 100 as x ticks
    ax.set_xticks([0, feature_idx, 100])
    ax.set_xticklabels(["0", str(feature_idx), "100"])
    # ax.set_title(f"APD model when ablating parameter components. One-hot $x_{{{feature_idx}}}=1$")
    ax.legend()
    assert isinstance(fig, plt.Figure)
    return {"feature_response_with_subnets": fig}


def get_feature_subnet_map(
    top1_model_fn: Callable[
        [
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances C"] | None,
        ],
        SPDOutputs,
    ],
    device: str,
    model_config: ResidualMLPConfig | ResidualMLPSPDConfig,
    instance_idx: int = 0,
) -> dict[int, int]:
    n_instances = model_config.n_instances
    n_features = model_config.n_features
    batch_size = n_features
    batch = torch.zeros(batch_size, n_instances, n_features, device=device)
    batch[torch.arange(n_features), instance_idx, torch.arange(n_features)] = 1
    top1_out = top1_model_fn(batch, None)
    top1_mask = top1_out.topk_mask[:, instance_idx, :]
    subnet_indices = {
        int(feature_idx.item()): int(subnet_idx.item())
        for feature_idx, subnet_idx in top1_mask.nonzero()
    }
    return subnet_indices
