from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.models import (
    ResidualMLPConfig,
    ResidualMLPModel,
    ResidualMLPSPDConfig,
    ResidualMLPSPDModel,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.module_utils import get_nested_module_attr
from spd.run_spd import Config, ResidualMLPTaskConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple ResidualMLP config that we can use in multiple tests
RESID_MLP_TASK_CONFIG = ResidualMLPTaskConfig(
    task_name="residual_mlp",
    feature_probability=0.333,
    init_scale=1.0,
    data_generation_type="at_least_zero_active",
    pretrained_model_path=Path(),  # We'll create this later
)


def test_resid_mlp_decomposition_happy_path() -> None:
    # Just noting that this test will only work on 98/100 seeds. So it's possible that future
    # changes will break this test.
    set_seed(0)
    resid_mlp_config = ResidualMLPConfig(
        n_instances=2,
        n_features=3,
        d_embed=2,
        d_mlp=3,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=True,
        out_bias=True,
    )

    device = "cpu"
    config = Config(
        seed=0,
        C=3,
        topk=1,
        batch_topk=True,
        param_match_coeff=1.0,
        topk_recon_coeff=1,
        schatten_pnorm=1,
        schatten_coeff=1,
        attribution_type="gradient",
        lr=1e-3,
        batch_size=32,
        steps=10,  # Run only a few steps for the test
        print_freq=2,
        image_freq=5,
        save_freq=None,
        lr_warmup_pct=0.01,
        lr_schedule="cosine",
        task_config=RESID_MLP_TASK_CONFIG,
    )

    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    # Create a pretrained model
    target_model = ResidualMLPModel(config=resid_mlp_config).to(device)

    # Create the SPD model
    spd_config = ResidualMLPSPDConfig(**resid_mlp_config.model_dump(), C=config.C)
    model = ResidualMLPSPDModel(config=spd_config).to(device)

    # Use the pretrained model's embedding matrices and don't train them further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False
    model.W_U.data[:, :] = target_model.W_U.data.detach().clone()
    model.W_U.requires_grad = False

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(resid_mlp_config.n_layers):
        if resid_mlp_config.in_bias:
            model.layers[i].bias1.data[:, :] = target_model.layers[i].bias1.data.detach().clone()
            model.layers[i].bias1.requires_grad = False
        if resid_mlp_config.out_bias:
            model.layers[i].bias2.data[:, :] = target_model.layers[i].bias2.data.detach().clone()
            model.layers[i].bias2.requires_grad = False

    # Create dataset and dataloader
    dataset = ResidualMLPDataset(
        n_instances=model.n_instances,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type="at_least_zero_active",
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Set up param_map
    param_map = {}
    for i in range(resid_mlp_config.n_layers):
        param_map[f"layers.{i}.mlp_in"] = f"layers.{i}.mlp_in"
        param_map[f"layers.{i}.mlp_out"] = f"layers.{i}.mlp_out"

    # Calculate initial loss
    with torch.inference_mode():
        batch, _ = next(iter(dataloader))
        initial_out = model(batch)
        labels = target_model(batch)
        initial_loss = torch.mean((labels - initial_out) ** 2).item()

    param_names = []
    for i in range(target_model.config.n_layers):
        param_names.append(f"layers.{i}.mlp_in")
        param_names.append(f"layers.{i}.mlp_out")
    # Run optimize function
    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=None,
        plot_results_fn=None,
    )

    # Calculate final loss
    with torch.inference_mode():
        final_out = model(batch)
        final_loss = torch.mean((labels - final_out) ** 2).item()

    print(f"Final loss: {final_loss}, initial loss: {initial_loss}")
    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"

    # Show that W_E is still the same as the target model's W_E
    assert torch.allclose(model.W_E, target_model.W_E, atol=1e-6)


def test_resid_mlp_equivalent_to_raw_model() -> None:
    device = "cpu"
    set_seed(0)
    resid_mlp_config = ResidualMLPConfig(
        n_instances=2,
        n_features=3,
        d_embed=2,
        d_mlp=3,
        n_layers=2,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=True,
        out_bias=True,
    )
    C = 2

    target_model = ResidualMLPModel(config=resid_mlp_config).to(device)

    # Create the SPD model with k=1
    resid_mlp_spd_config = ResidualMLPSPDConfig(**resid_mlp_config.model_dump(), C=C)
    spd_model = ResidualMLPSPDModel(config=resid_mlp_spd_config).to(device)

    # Init all params to random values
    for param in spd_model.parameters():
        param.data = torch.randn_like(param.data)

    # Copy the subnetwork params from the SPD model to the target model
    for i in range(target_model.config.n_layers):
        for pos in ["mlp_in", "mlp_out"]:
            target_pos: Tensor = get_nested_module_attr(target_model, f"layers.{i}.{pos}.weight")
            spd_pos: Tensor = get_nested_module_attr(spd_model, f"layers.{i}.{pos}.weight")
            target_pos.data[:, :, :] = spd_pos.data

    # Also copy the embeddings and biases
    target_model.W_E.data[:, :, :] = spd_model.W_E.data
    target_model.W_U.data[:, :, :] = spd_model.W_U.data
    for i in range(resid_mlp_config.n_layers):
        target_model.layers[i].bias1.data[:, :] = spd_model.layers[i].bias1.data
        target_model.layers[i].bias2.data[:, :] = spd_model.layers[i].bias2.data

    # Create a random input
    batch_size = 4
    input_data: Float[torch.Tensor, "batch n_instances n_features"] = torch.rand(
        batch_size, resid_mlp_config.n_instances, resid_mlp_config.n_features, device=device
    )

    with torch.inference_mode():
        # Forward pass on target model
        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            input_data, names_filter=target_cache_filter
        )
        # Forward pass with all subnetworks
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out, spd_cache = spd_model.run_with_cache(input_data, names_filter=spd_cache_filter)

    # Assert outputs are the same
    assert torch.allclose(target_out, out, atol=1e-6), "Outputs do not match"

    # Assert that all post-acts are the same
    target_post_weight_acts = {k: v for k, v in target_cache.items() if k.endswith(".hook_post")}
    spd_post_weight_acts = {k: v for k, v in spd_cache.items() if k.endswith(".hook_post")}
    for key_name in target_post_weight_acts:
        assert torch.allclose(
            target_post_weight_acts[key_name], spd_post_weight_acts[key_name], atol=1e-6
        ), f"post-acts do not match at layer {key_name}"
