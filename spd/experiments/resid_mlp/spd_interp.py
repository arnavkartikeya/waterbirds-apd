# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from pydantic import PositiveFloat
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDModel
from spd.experiments.resid_mlp.plotting import (
    analyze_per_feature_performance,
    collect_average_components_per_feature,
    collect_per_feature_losses,
    get_feature_subnet_map,
    get_scrubbed_losses,
    plot_avg_components_scatter,
    plot_feature_response_with_subnets,
    plot_per_feature_performance_fig,
    plot_scrub_losses,
    plot_spd_feature_contributions_truncated,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.resid_mlp_decomposition import plot_subnet_categories
from spd.run_spd import ResidualMLPTaskConfig
from spd.settings import REPO_ROOT
from spd.utils import (
    COLOR_PALETTE,
    SPDOutputs,
    run_spd_forward_pass,
    set_seed,
)

color_map = {
    "target": COLOR_PALETTE[0],
    "apd_topk": COLOR_PALETTE[1],
    "apd_scrubbed": COLOR_PALETTE[4],
    "apd_antiscrubbed": COLOR_PALETTE[2],  # alt: 3
    "baseline_monosemantic": "grey",
}

out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out/figures/"
out_dir.mkdir(parents=True, exist_ok=True)

# %% Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed

use_data_from_files = True
wandb_path = "wandb:spd-resid-mlp/runs/8qz1si1l"  # 1 layer 40k steps (R6) topk=1.28
# wandb_path = "wandb:spd-resid-mlp/runs/9a639c6w"  # 1 layer topk=1
# wandb_path = "wandb:spd-resid-mlp/runs/cb0ej7hj"  # 2 layer 2LR4 topk=1.28
# wandb_path = "wandb:spd-resid-mlp/runs/wbeghftm"  # 2 layer topk=1
# wandb_path = "wandb:spd-resid-mlp/runs/c1q3bs6f"  # 2 layer m=1 topk=1.28 (not in paper)

wandb_id = wandb_path.split("/")[-1]

# Load the pretrained SPD model
model, config, label_coeffs = ResidualMLPSPDModel.from_pretrained(wandb_path)
assert isinstance(config.task_config, ResidualMLPTaskConfig)

# Path must be local
target_model, target_model_train_config_dict, target_label_coeffs = (
    ResidualMLPModel.from_pretrained(config.task_config.pretrained_model_path)
)
# Print some basic information about the model
print(f"Number of features: {model.config.n_features}")
print(f"Feature probability: {config.task_config.feature_probability}")
print(f"Embedding dimension: {model.config.d_embed}")
print(f"MLP dimension: {model.config.d_mlp}")
print(f"Number of layers: {model.config.n_layers}")
print(f"Number of subnetworks (C): {model.config.C}")
model = model.to(device)
label_coeffs = label_coeffs.to(device)
target_model = target_model.to(device)
target_label_coeffs = target_label_coeffs.to(device)
assert torch.allclose(target_label_coeffs, label_coeffs)

n_layers = target_model.config.n_layers


# Functions used for various plots
def spd_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk: PositiveFloat | None = config.topk,
    batch_topk: bool = config.batch_topk,
) -> SPDOutputs:
    assert topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=batch_topk,
        topk=topk,
        distil_from_target=config.distil_from_target,
    )


def target_model_fn(batch: Float[Tensor, "batch n_instances"]):
    return target_model(batch)


def top1_model_fn(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk_mask: Float[Tensor, "batch n_instances C"] | None,
) -> SPDOutputs:
    """Top1 if topk_mask is None, else just use provided topk_mask"""
    topk_mask = topk_mask.to(device) if topk_mask is not None else None
    assert config.topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=False,
        topk=1,
        distil_from_target=config.distil_from_target,
        topk_mask=topk_mask,
    )


dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=True,
    label_type=target_model_train_config_dict["label_type"],
    act_fn_name=target_model.config.act_fn_name,
    label_coeffs=target_label_coeffs,
    data_generation_type="at_least_zero_active",  # We will change this in the for loop
)

# %% Plot how many subnets are monosemantic, etc.
fig = plot_subnet_categories(model, device, cutoff=4e-2)
# Save the figure
fig.savefig(out_dir / f"resid_mlp_subnet_categories_{n_layers}layers_{wandb_id}.png")
print(f"Saved figure to {out_dir / f'resid_mlp_subnet_categories_{n_layers}layers_{wandb_id}.png'}")


# %%
per_feature_losses_path = Path(out_dir) / f"resid_mlp_losses_{n_layers}layers_{wandb_id}.pt"
if not use_data_from_files or not per_feature_losses_path.exists():
    loss_target, loss_spd_batch_topk, loss_spd_sample_topk = collect_per_feature_losses(
        target_model=target_model,
        spd_model=model,
        config=config,
        dataset=dataset,
        device=device,
        batch_size=config.batch_size,
        n_samples=100_000,
    )
    # Save the losses to a file
    torch.save(
        (loss_target, loss_spd_batch_topk, loss_spd_sample_topk),
        per_feature_losses_path,
    )

# Load the losses from a file
loss_target, loss_spd_batch_topk, loss_spd_sample_topk = torch.load(
    per_feature_losses_path, weights_only=True, map_location="cpu"
)

fig = plot_per_feature_performance_fig(
    loss_target=loss_target,
    loss_spd_batch_topk=loss_spd_batch_topk,
    loss_spd_sample_topk=loss_spd_sample_topk,
    config=config,
    color_map=color_map,
)
fig.show()
fig.savefig(out_dir / f"resid_mlp_per_feature_performance_{n_layers}layers_{wandb_id}.png")
print(
    f"Saved figure to {out_dir / f'resid_mlp_per_feature_performance_{n_layers}layers_{wandb_id}.png'}"
)

# %%
# Scatter plot of avg active components vs loss difference
avg_components_path = Path(out_dir) / f"avg_components_{n_layers}layers_{wandb_id}.pt"
if not use_data_from_files or not avg_components_path.exists():
    avg_components = collect_average_components_per_feature(
        model_fn=spd_model_fn,
        dataset=dataset,
        device=device,
        n_features=model.config.n_features,
        batch_size=config.batch_size,
        n_samples=500_000,
    )
    # Save the avg_components to a file
    torch.save(avg_components.cpu(), avg_components_path)

# Load the avg_components from a file
avg_components = torch.load(avg_components_path, map_location=device, weights_only=True)

# Get the loss of the spd model w.r.t the target model
fn_without_batch_topk = lambda batch: spd_model_fn(
    batch, topk=1, batch_topk=False
).spd_topk_model_output  # type: ignore
losses_spd_wrt_target = analyze_per_feature_performance(
    model_fn=fn_without_batch_topk,
    target_model_fn=target_model_fn,
    model_config=model.config,
    device=device,
    batch_size=config.batch_size,
)

fig = plot_avg_components_scatter(
    losses_spd_wrt_target=losses_spd_wrt_target, avg_components=avg_components
)
fig.show()
# Save the figure
fig.savefig(out_dir / f"resid_mlp_avg_components_scatter_{n_layers}layers_{wandb_id}.png")
print(
    f"Saved figure to {out_dir / f'resid_mlp_avg_components_scatter_{n_layers}layers_{wandb_id}.png'}"
)

# %%
# Plot the main truncated feature contributions figure for the paper
fig = plot_spd_feature_contributions_truncated(
    spd_model=model,
    target_model=target_model,
    device=device,
    n_features=10,
    include_crossterms=False,
)
fig.savefig(out_dir / f"resid_mlp_weights_{n_layers}layers_{wandb_id}.png")
print(f"Saved figure to {out_dir / f'resid_mlp_weights_{n_layers}layers_{wandb_id}.png'}")

# Full figure for updating wandb report
# fig = plot_spd_feature_contributions(
#     spd_model=model,
#     target_model=target_model,
#     device=device,
# )
# fig.savefig(out_dir / f"resid_mlp_weights_full_{n_layers}layers_{wandb_id}.png")
# plt.close(fig)
# print(f"Saved figure to {out_dir / f'resid_mlp_weights_full_{n_layers}layers_{wandb_id}.png'}")
# import wandb

# # Restart the run and log the figure
# run = wandb.init(project="spd-resid-mlp", id=wandb_id, resume="must")
# run.log({"neuron_contributions": wandb.Image(fig)})
# run.finish()

# %%
# Plot causal scrubbing-esque test
n_batches = 100
losses = get_scrubbed_losses(
    top1_model_fn=top1_model_fn,
    spd_model_fn=spd_model_fn,
    target_model=target_model,
    dataset=dataset,
    model=model,
    device=device,
    config=config,
    n_batches=n_batches,
)

fig = plot_scrub_losses(losses, config, color_map, n_batches)
fig.savefig(
    out_dir / f"resid_mlp_scrub_hist_{n_layers}layers_{wandb_id}.png", bbox_inches="tight", dpi=300
)
print(f"Saved figure to {out_dir / f'resid_mlp_scrub_hist_{n_layers}layers_{wandb_id}.png'}")

# %% Linearity test: Enable one subnet after the other
# candlestick plot

# # Dictionary feature_idx -> subnet_idx
subnet_indices = get_feature_subnet_map(top1_model_fn, device, model.config, instance_idx=0)

n_features = model.config.n_features
feature_idx = 42
subtract_inputs = True  # TODO TRUE subnet


fig = plot_feature_response_with_subnets(
    topk_model_fn=top1_model_fn,
    device=device,
    model_config=model.config,
    feature_idx=feature_idx,
    subnet_idx=subnet_indices[feature_idx],
    batch_size=1000,
    plot_type="errorbar",
    color_map=color_map,
)["feature_response_with_subnets"]
fig.savefig(  # type: ignore
    out_dir / f"feature_response_with_subnets_{feature_idx}_{n_layers}layers_{wandb_id}.png",
    bbox_inches="tight",
    dpi=300,
)
print(
    f"Saved figure to {out_dir / f'feature_response_with_subnets_{feature_idx}_{n_layers}layers_{wandb_id}.png'}"
)
plt.show()
