import torch

from spd.experiments.resid_mlp.models import ResidualMLPSPDConfig, ResidualMLPSPDModel
from spd.experiments.tms.models import TMSSPDModel, TMSSPDModelConfig


def test_tms_set_and_restore_subnet():
    subnet_idx = 2
    config = TMSSPDModelConfig(
        n_instances=2,
        n_features=4,
        n_hidden=3,
        C=5,
        n_hidden_layers=1,
        bias_val=0.0,
        device="cpu",
    )
    model = TMSSPDModel(config)
    assert model.linear1.component_weights.shape == (2, 5, 4, 3)  # (n_instances, C, d_in, d_out)

    # Get the original values of the weight_matrix of subnet_idx
    original_vals = model.linear1.component_weights[:, subnet_idx, :, :].detach().clone()

    # Now set the 3rd subnet to zero
    stored_vals = model.set_subnet_to_zero(subnet_idx=subnet_idx, has_instance_dim=True)

    # Check that model.linear1.component_weights is zero for all instances
    assert model.linear1.component_weights[:, subnet_idx, :, :].allclose(
        torch.zeros_like(model.linear1.component_weights[:, subnet_idx, :, :])
    )
    assert subnet_idx != 0
    # Check that it's not zero in another component
    assert not model.linear1.component_weights[:, 0, :, :].allclose(
        torch.zeros_like(model.linear1.component_weights[:, 0, :, :])
    )

    # Now restore the subnet
    model.restore_subnet(subnet_idx=subnet_idx, stored_vals=stored_vals, has_instance_dim=True)
    assert model.linear1.component_weights[:, subnet_idx, :, :].allclose(original_vals)


def test_resid_mlp_set_and_restore_subnet():
    subnet_idx = 2
    config = ResidualMLPSPDConfig(
        n_instances=2,
        n_features=4,
        d_embed=6,
        d_mlp=8,
        n_layers=1,
        act_fn_name="gelu",
        apply_output_act_fn=False,
        in_bias=False,
        out_bias=False,
        init_scale=1.0,
        C=5,
        init_type="xavier_normal",
    )
    model = ResidualMLPSPDModel(config)

    # Check shapes of first layer's component weights
    assert model.layers[0].mlp_in.component_weights.shape == (2, 5, 6, 8)  # n_inst, C, d_in, d_out

    # Get the original values of the weight_matrix of subnet_idx for both mlp_in and mlp_out
    original_vals_in = (
        model.layers[0].mlp_in.component_weights[:, subnet_idx, :, :].detach().clone()
    )
    original_vals_out = (
        model.layers[0].mlp_out.component_weights[:, subnet_idx, :, :].detach().clone()
    )

    # Set the subnet to zero
    stored_vals = model.set_subnet_to_zero(subnet_idx=subnet_idx, has_instance_dim=True)

    # Check that component_weights are zero for all instances in both mlp_in and mlp_out
    assert (
        model.layers[0]
        .mlp_in.component_weights[:, subnet_idx, :, :]
        .allclose(torch.zeros_like(model.layers[0].mlp_in.component_weights[:, subnet_idx, :, :]))
    )
    assert (
        model.layers[0]
        .mlp_out.component_weights[:, subnet_idx, :, :]
        .allclose(torch.zeros_like(model.layers[0].mlp_out.component_weights[:, subnet_idx, :, :]))
    )

    assert subnet_idx != 0
    # Check that it's not zero in another component
    assert (
        not model.layers[0]
        .mlp_in.component_weights[:, 0, :, :]
        .allclose(torch.zeros_like(model.layers[0].mlp_in.component_weights[:, 0, :, :]))
    )
    assert (
        not model.layers[0]
        .mlp_out.component_weights[:, 0, :, :]
        .allclose(torch.zeros_like(model.layers[0].mlp_out.component_weights[:, 0, :, :]))
    )

    # Restore the subnet
    model.restore_subnet(subnet_idx=subnet_idx, stored_vals=stored_vals, has_instance_dim=True)

    # Verify restoration was successful
    assert model.layers[0].mlp_in.component_weights[:, subnet_idx, :, :].allclose(original_vals_in)
    assert (
        model.layers[0].mlp_out.component_weights[:, subnet_idx, :, :].allclose(original_vals_out)
    )
