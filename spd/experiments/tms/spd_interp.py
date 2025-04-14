# %%

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.tms.models import TMSModel, TMSSPDModel
from spd.plotting import collect_sparse_dataset_mse_losses, plot_sparse_feature_mse_line_plot
from spd.run_spd import TMSTaskConfig
from spd.settings import REPO_ROOT
from spd.utils import COLOR_PALETTE, DataGenerationType, SparseFeatureDataset


def plot_vectors(
    subnets: Float[Tensor, "n_instances n_subnets n_features n_hidden"],
    axs: npt.NDArray[np.object_],
) -> None:
    """2D polygon plot of each subnetwork.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    n_instances, n_subnets, n_features, n_hidden = subnets.shape

    # Use different colors for each subnetwork if there's only one instance
    color_vals = np.linspace(0, 1, n_features) if n_instances == 1 else np.zeros(n_features)
    colors = plt.cm.viridis(color_vals)  # type: ignore

    for subnet_idx in range(n_subnets):
        for instance_idx, ax in enumerate(axs[:, subnet_idx]):
            arr = subnets[instance_idx, subnet_idx].cpu().detach().numpy()

            # Plot each feature with its unique color
            for j in range(n_features):
                ax.scatter(arr[j, 0], arr[j, 1], color=colors[j])
                ax.add_collection(
                    mc.LineCollection([[(0, 0), (arr[j, 0], arr[j, 1])]], colors=[colors[j]])
                )

            ax.set_aspect("equal")
            z = 1.3
            ax.set_facecolor("#f6f6f6")
            ax.set_xlim((-z, z))
            ax.set_ylim((-z, z))
            ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_position("center")

            if instance_idx == 0:  # Only add labels to the first row
                if subnet_idx == 0:
                    label = "Target model"
                elif subnet_idx == 1:
                    label = "Sum of components"
                else:
                    label = f"Component {subnet_idx - 2}"
                ax.set_title(label, pad=10, fontsize="large")


def plot_networks(
    subnets: Float[Tensor, "n_instances n_subnets n_features n_hidden"],
    axs: npt.NDArray[np.object_],
) -> None:
    """Plot neural network diagrams for each W matrix in the subnet variable.

    Args:
        subnets: Tensor of shape [n_instances, n_subnets, n_features, n_hidden].
        axs: Matplotlib axes to plot on.
    """

    n_instances, n_subnets, n_features, n_hidden = subnets.shape

    # Take the absolute value of the weights
    subnets_abs = subnets.abs()

    # Find the maximum weight across each instance
    max_weights = subnets_abs.amax(dim=(1, 2, 3))

    axs = np.atleast_2d(np.array(axs))

    # axs[0, 0].set_xlabel("Outputs (before ReLU and biases)")
    # Add the above but in text because the x-axis is killed
    axs[0, 0].text(
        0.05,
        0.05,
        "Outputs (before bias & ReLU)",
        ha="left",
        va="center",
        transform=axs[0, 0].transAxes,
    )
    # Also add "input label"
    axs[0, 0].text(
        0.05,
        0.95,
        "Inputs",
        ha="left",
        va="center",
        transform=axs[0, 0].transAxes,
    )

    # Grayscale colormap. darker for larger weight
    cmap = plt.get_cmap("gray_r")

    for subnet_idx in range(n_subnets):
        for instance_idx, ax in enumerate(axs[:, subnet_idx]):
            arr = subnets_abs[instance_idx, subnet_idx].cpu().detach().numpy()

            # Define node positions (top to bottom)
            y_input, y_hidden, y_output = 0, -1, -2
            x_input = np.linspace(0.05, 0.95, n_features)
            x_hidden = np.linspace(0.25, 0.75, n_hidden)
            x_output = np.linspace(0.05, 0.95, n_features)

            # Add transparent grey box around hidden layer
            box_width = 0.8
            box_height = 0.4
            box = plt.Rectangle(
                (0.5 - box_width / 2, y_hidden - box_height / 2),
                box_width,
                box_height,
                fill=True,
                facecolor="#e4e4e4",
                edgecolor="none",
                alpha=0.33,
                transform=ax.transData,
            )
            ax.add_patch(box)

            # Plot nodes
            ax.scatter(
                x_input, [y_input] * n_features, s=200, color="grey", edgecolors="k", zorder=3
            )
            ax.scatter(
                x_hidden, [y_hidden] * n_hidden, s=200, color="grey", edgecolors="k", zorder=3
            )
            ax.scatter(
                x_output, [y_output] * n_features, s=200, color="grey", edgecolors="k", zorder=3
            )

            # Plot edges from input to hidden layer
            for idx_input in range(n_features):
                for idx_hidden in range(n_hidden):
                    weight = arr[idx_input, idx_hidden]
                    norm_weight = weight / max_weights[instance_idx]
                    color = cmap(norm_weight)
                    ax.plot(
                        [x_input[idx_input], x_hidden[idx_hidden]],
                        [y_input, y_hidden],
                        color=color,
                        linewidth=1,
                    )

            # Plot edges from hidden to output layer
            arr_T = arr.T  # Transpose of W for W^T
            for idx_hidden in range(n_hidden):
                for idx_output in range(n_features):
                    weight = arr_T[idx_hidden, idx_output]
                    norm_weight = weight / max_weights[instance_idx]
                    color = cmap(norm_weight)
                    ax.plot(
                        [x_hidden[idx_hidden], x_output[idx_output]],
                        [y_hidden, y_output],
                        color=color,
                        linewidth=1,
                    )

            # Remove axes for clarity
            # ax.axis("off")
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(y_output - 0.5, y_input + 0.5)
            # Remove x and y ticks and bounding boxes
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ["top", "right", "bottom", "left"]:
                ax.spines[spine].set_visible(False)


def plot_combined(
    subnets: Float[Tensor, "n_instances n_subnets n_features n_hidden"],
    target_weights: Float[Tensor, "n_instances n_features n_hidden"],
    n_instances: int | None = None,
) -> plt.Figure:
    """Create a combined figure with both vector and network diagrams side by side."""
    if n_instances is not None:
        subnets = subnets[:n_instances]
        target_weights = target_weights[:n_instances]
    n_instances, n_subnets, n_features, n_hidden = subnets.shape

    # We wish to add two panels to the left: The target model weights and the sum of the subnets
    # Add an extra dimension to the target weights so we can concatenate them
    target_subnet = target_weights[:, None, :, :]
    summed_subnet = subnets.sum(dim=1, keepdim=True)
    subnets = torch.cat([target_subnet, summed_subnet, subnets], dim=1)
    n_subnets += 2

    # Create figure with two rows
    fig, axs = plt.subplots(
        nrows=n_instances * 2,
        ncols=n_subnets,
        figsize=(3 * n_subnets, 6 * n_instances),
    )

    plt.subplots_adjust(hspace=0)

    axs = np.atleast_2d(np.array(axs))

    # Split axes into left (vectors) and right (networks) sides
    axs_vectors = axs[:n_instances, :]
    axs_networks = axs[n_instances:, :]

    # Call existing plotting logic with the split axes
    plot_vectors(subnets=subnets, axs=axs_vectors)
    plot_networks(subnets=subnets, axs=axs_networks)

    return fig


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# path = "wandb:spd-tms/runs/bft0pgi8"  # Old 5-2 run with attributions from spd model # paper run
# instance_idx = 0
# path = "wandb:spd-tms/runs/sv9padmo"  # 10-5
# path = "wandb:spd-tms/runs/vt0i4a22"  # 20-5
# path = "wandb:spd-tms/runs/tyo4serm"  # 40-10 with topk=2, topk_recon_coeff=1e1, schatten_coeff=15# old paper run
# path = "wandb:spd-tms/runs/9zzp2s68"  # 40-10 with topk=2, topk_recon_coeff=1e1, schatten_coeff=20
path = "wandb:spd-tms/runs/08no00iq"  # 40-10 with topk=1, topk_recon_coeff=1e1, schatten_coeff=20# new paper run
instance_idx = 2
# path = "wandb:spd-tms/runs/014t4f9n"  # 40-10 with topk=1, topk_recon_coeff=1e1, schatten_coeff=1e1

run_id = path.split("/")[-1]

# Plot showing polygons for each subnet
model, config = TMSSPDModel.from_pretrained(path)
subnets = model.linear1.component_weights.detach().cpu()

assert isinstance(config.task_config, TMSTaskConfig)
target_model, target_model_train_config_dict = TMSModel.from_pretrained(
    config.task_config.pretrained_model_path
)

out_dir = REPO_ROOT / "spd/experiments/tms/out/figures/"
out_dir.mkdir(parents=True, exist_ok=True)


# %%
# Max cosine similarity between subnets and target model
def plot_max_cosine_sim(max_cosine_sim: Float[Tensor, " n_features"]) -> plt.Figure:
    fig, ax = plt.subplots()
    # Make a bar plot of the max cosine similarity for each feature
    ax.bar(range(max_cosine_sim.shape[0]), max_cosine_sim.cpu().detach().numpy())
    # Add a grey horizontal line at 1
    ax.axhline(1, color="grey", linestyle="--")
    ax.set_xlabel("Input feature index")
    ax.set_ylabel("Max cosine similarity")
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


cosine_sims = torch.einsum(
    "C f h, f h -> C f",
    subnets[instance_idx] / torch.norm(subnets[instance_idx], dim=-1, keepdim=True),
    target_model.linear1.weight[instance_idx]
    / torch.norm(target_model.linear1.weight[instance_idx], dim=-1, keepdim=True),
)
max_cosine_sim = cosine_sims.max(dim=0).values
print(f"Max cosine similarity:\n{max_cosine_sim}")
print(f"Mean max cosine similarity: {max_cosine_sim.mean()}")
print(f"std max cosine similarity: {max_cosine_sim.std()}")


# Get the subnet weights at the max cosine similarity
subnet_weights_at_max_cosine_sim: Float[Tensor, "n_features n_hidden"] = subnets[
    instance_idx, cosine_sims.max(dim=0).indices, torch.arange(target_model.config.n_features)
]
# Get the norm of the target model weights
target_model_weights_norm = torch.norm(
    target_model.linear1.weight[instance_idx], dim=-1, keepdim=True
)
# Get the norm of subnet_weights_at_max_cosine_sim
subnet_weights_at_max_cosine_sim_norm = torch.norm(
    subnet_weights_at_max_cosine_sim, dim=-1, keepdim=True
)
# Divide the subnet weights by the target model weights ratio
l2_ratio = subnet_weights_at_max_cosine_sim_norm / target_model_weights_norm
print(f"Mean L2 ratio: {l2_ratio.mean()}")
print(f"std L2 ratio: {l2_ratio.std()}")

# Mean bias
print(f"Mean bias: {target_model.b_final[instance_idx].mean()}")


# fig = plot_max_cosine_sim(max_cosine_sim)
# # Save figure
# fig.savefig(out_dir / f"tms_max_cosine_sim_{run_id}.png", bbox_inches="tight", dpi=400)
# print(f"Saved figure to {out_dir / f'tms_max_cosine_sim_{run_id}.png'}")
# %%
# Only plot if the hidden dimension is 2
if target_model.config.n_hidden == 2:
    # We only look at the first instance
    fig = plot_combined(subnets, target_model.linear1.weight.detach().cpu(), n_instances=1)
    fig.savefig(out_dir / f"tms_combined_diagram_{run_id}.png", bbox_inches="tight", dpi=400)
    print(f"Saved figure to {out_dir / f'tms_combined_diagram_{run_id}.png'}")

# %%
# This doesn't work for TMS.
# # Get the entries for the main loss table in the paper
dataset = SparseFeatureDataset(
    n_instances=target_model.config.n_instances,
    n_features=target_model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    data_generation_type="at_least_zero_active",  # This will be changed in collect_sparse_dataset_mse_losses
    value_range=(0.0, 1.0),
)
gen_types: list[DataGenerationType] = [
    "at_least_zero_active",
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
]
assert config.topk is not None
results = collect_sparse_dataset_mse_losses(
    dataset=dataset,
    target_model=target_model,
    spd_model=model,
    batch_size=10000,
    device=device,
    topk=config.topk,
    attribution_type=config.attribution_type,
    batch_topk=config.batch_topk,
    distil_from_target=config.distil_from_target,
    gen_types=gen_types,
)

# %%
# Option to plot a single instance
inst = None
if inst is not None:
    # We only plot the {inst}th instance
    plot_data = {
        gen_type: {k: float(v[inst].detach().cpu()) for k, v in results[gen_type].items()}
        for gen_type in gen_types
    }
else:
    # Take the mean over all instances
    plot_data = {
        gen_type: {k: float(v.mean(dim=0).detach().cpu()) for k, v in results[gen_type].items()}
        for gen_type in gen_types
    }

# %%
# Create line plot of results
color_map = {
    "target": COLOR_PALETTE[0],
    "apd_topk": COLOR_PALETTE[1],
    "baseline_monosemantic": "grey",
}
label_map = [
    ("target", "Target model", color_map["target"]),
    ("spd", "APD model", color_map["apd_topk"]),
    ("baseline_monosemantic", "Monosemantic baseline", color_map["baseline_monosemantic"]),
]

fig = plot_sparse_feature_mse_line_plot(plot_data, label_map=label_map, log_scale=False)
fig.show()
# fig.savefig(out_dir / f"tms_mse_{run_id}_inst{inst}.png", dpi=400)
# print(f"Saved figure to {out_dir / f'tms_mse_{run_id}_inst{inst}.png'}")
fig.savefig(out_dir / f"tms_mse_{run_id}.png", dpi=400)
print(f"Saved figure to {out_dir / f'tms_mse_{run_id}.png'}")

# %%
