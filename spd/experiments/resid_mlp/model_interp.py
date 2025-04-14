# %% Imports

import einops
import matplotlib.pyplot as plt
import torch

from spd.experiments.resid_mlp.models import (
    ResidualMLPModel,
)
from spd.experiments.resid_mlp.plotting import (
    calculate_virtual_weights,
    plot_2d_snr,
    plot_all_relu_curves,
    plot_individual_feature_response,
    plot_resid_vs_mlp_out,
    plot_single_feature_response,
    plot_single_relu_curve,
    relu_contribution_plot,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.train_resid_mlp import ResidMLPTrainConfig
from spd.plotting import plot_matrix
from spd.settings import REPO_ROOT
from spd.types import ModelPath
from spd.utils import set_seed

# %% Load model and config

out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out/figures/"
out_dir.mkdir(parents=True, exist_ok=True)

set_seed(0)
device = "cpu" if torch.cuda.is_available() else "cpu"
path: ModelPath = "wandb:spd-train-resid-mlp/runs/zas5yjdl"  # 1 layer
# path: ModelPath = "wandb:spd-train-resid-mlp/runs/sv23xrhj"  # 2 layers
model, train_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(path)
model = model.to(device)
train_config = ResidMLPTrainConfig(**train_config_dict)
dataset = ResidualMLPDataset(
    n_instances=train_config.resid_mlp_config.n_instances,
    n_features=train_config.resid_mlp_config.n_features,
    feature_probability=train_config.feature_probability,
    device=device,
    calc_labels=False,
    label_type=train_config.label_type,
    act_fn_name=train_config.resid_mlp_config.act_fn_name,
    label_fn_seed=train_config.label_fn_seed,
    label_coeffs=label_coeffs,
    data_generation_type=train_config.data_generation_type,
)
if train_config.data_generation_type == "at_least_zero_active":
    # In the future this will be merged into generate_batch
    batch = dataset._generate_multi_feature_batch_no_zero_samples(
        train_config.batch_size, buffer_ratio=2
    )
    if isinstance(dataset, ResidualMLPDataset) and dataset.label_fn is not None:
        labels = dataset.label_fn(batch)
    else:
        labels = batch.clone().detach()
else:
    batch, labels = dataset.generate_batch(train_config.batch_size)

n_layers = train_config.resid_mlp_config.n_layers
# %% Plot feature response with one active feature
fig = plot_individual_feature_response(
    lambda batch: model(batch),
    model_config=train_config.resid_mlp_config,
    device=device,
    sweep=False,
    plot_type="line",
)
fig = plot_individual_feature_response(
    lambda batch: model(batch),
    model_config=train_config.resid_mlp_config,
    device=device,
    sweep=True,
    plot_type="line",
)
plt.show()

# %% Simple plot for paper appendix

fig, axes = plt.subplots(ncols=2, figsize=(10, 5), constrained_layout=True, sharey=True)
ax1, ax2 = axes  # type: ignore
plot_single_feature_response(
    lambda batch: model(batch),
    model_config=train_config.resid_mlp_config,
    device=device,
    subtract_inputs=False,
    feature_idx=42,
    ax=ax1,
)
plot_single_relu_curve(
    lambda batch: model(batch),
    model_config=train_config.resid_mlp_config,
    device=device,
    subtract_inputs=False,
    feature_idx=42,
    ax=ax2,
)
fig.savefig(
    out_dir / f"resid_mlp_feature_response_single_{n_layers}layers.png",
    bbox_inches="tight",
    dpi=300,
)
print(f"Saved figure to {out_dir / f'resid_mlp_feature_response_single_{n_layers}layers.png'}")

fig, axes = plt.subplots(ncols=2, figsize=(10, 5), constrained_layout=True, sharey=True)
ax1, ax2 = axes  # type: ignore
plot_individual_feature_response(
    lambda batch: model(batch),
    model_config=train_config.resid_mlp_config,
    device=device,
    sweep=False,
    subtract_inputs=False,
    ax=ax1,
    cbar=False,
)
ax1.set_title("Outputs one-hot inputs (coloured by input index)")
plot_all_relu_curves(
    lambda batch: model(batch),
    model_config=train_config.resid_mlp_config,
    ax=ax2,
    device=device,
    subtract_inputs=False,
)
# Colorbar
cmap_viridis = plt.get_cmap("viridis")
sm = plt.cm.ScalarMappable(
    cmap=cmap_viridis, norm=plt.Normalize(0, train_config.resid_mlp_config.n_features)
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, orientation="vertical")
cbar.set_label("Active input feature index")

ax2.plot([], [], color="red", ls="--", label=r"Label ($x+\mathrm{ReLU}(x)$)")
ax2.legend(loc="upper left")

fig.savefig(
    out_dir / f"resid_mlp_feature_response_multi_{n_layers}layers.png", bbox_inches="tight", dpi=300
)
print(f"Saved figure to {out_dir / f'resid_mlp_feature_response_multi_{n_layers}layers.png'}")

# %%


instance_idx = 0
nrows = 10
fig, axs = plt.subplots(nrows=nrows, ncols=1, constrained_layout=True, figsize=(10, 3 + 4 * nrows))
fig.suptitle(f"Model {path}")
for i in range(nrows):
    ax = axs[i]  # type: ignore
    plot_resid_vs_mlp_out(
        target_model=model, device=device, ax=ax, instance_idx=instance_idx, feature_idx=i
    )
plt.show()


# %% Show connection strength between ReLUs and features
virtual_weights = calculate_virtual_weights(target_model=model, device=device)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)  # type: ignore

relu_contribution_plot(
    ax1=ax1,
    ax2=ax2,
    all_diag_relu_conns=virtual_weights["diag_relu_conns"],
    model=model,
    device=device,
    instance_idx=0,
)
plt.show()

# %% Calculate S/N ratio for 1 and 2 active features.
fig = plot_2d_snr(model, device)
plt.show()

# %% Plot virtual weights

fig = plt.figure(constrained_layout=True, figsize=(20, 20))
gs = fig.add_gridspec(ncols=2, nrows=3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1:, :])
virtual_weights = calculate_virtual_weights(target_model=model, device=device)
instance_idx = 0
in_conns = virtual_weights["in_conns"][instance_idx].cpu().detach()
out_conns = virtual_weights["out_conns"][instance_idx].cpu().detach()
W_E_W_U = einops.einsum(
    virtual_weights["W_E"][instance_idx],
    virtual_weights["W_U"][instance_idx],
    "n_features1 d_embed, d_embed n_features2 -> n_features1 n_features2",
)
plot_matrix(
    ax1,
    in_conns.T,
    "Virtual input weights $(W_E W_{in})^T$",
    "Features",
    "Neurons",
    colorbar_format="%.2f",
)
plot_matrix(
    ax2,
    out_conns,
    "Virtual output weights $W_{out} W_U$",
    "Features",
    "Neurons",
    colorbar_format="%.2f",
)
ax2.xaxis.set_label_position("top")
plot_matrix(
    ax3,
    W_E_W_U,
    "Virtual weights $W_E W_U$",
    "Features",
    "Features",
    colorbar_format="%.2f",
)
plt.show()

# %%
