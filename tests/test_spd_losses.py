import torch

from spd.run_spd import _calc_param_mse, calc_act_recon


class TestCalcParamMatchLoss:
    # Actually testing _calc_param_mse. calc_param_match_loss should fail hard in most cases, and
    # testing it would require lots of mocking the way it is currently written.
    def test_calc_param_match_loss_single_instance_single_param(self):
        A = torch.ones(2, 3)
        B = torch.ones(3, 2)
        n_params = 2 * 3 * 2
        spd_params = {"layer1": A @ B}
        target_params = {"layer1": torch.tensor([[1.0, 1.0], [1.0, 1.0]])}

        result = _calc_param_mse(
            params1=target_params,
            params2=spd_params,
            n_params=n_params,
            device="cpu",
        )

        # A: [2, 3], B: [3, 2], both filled with ones
        # AB: [[3, 3], [3, 3]]
        # (AB - pretrained_weights)^2: [[4, 4], [4, 4]]
        # Sum and divide by n_params: 16 / 12 = 4/3
        expected = torch.tensor(4.0 / 3.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_param_match_loss_single_instance_multiple_params(self):
        As = [torch.ones(2, 3), torch.ones(3, 3)]
        Bs = [torch.ones(3, 3), torch.ones(3, 2)]
        n_params = 2 * 3 * 3 + 3 * 3 * 2
        target_params = {
            "layer1": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            "layer2": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        }
        spd_params = {
            "layer1": As[0] @ Bs[0],
            "layer2": As[1] @ Bs[1],
        }
        result = _calc_param_mse(
            params1=target_params,
            params2=spd_params,
            n_params=n_params,
            device="cpu",
        )

        # First layer: AB1: [[3, 3, 3], [3, 3, 3]], diff^2: [[1, 1, 1], [1, 1, 1]]
        # Second layer: AB2: [[3, 3], [3, 3], [3, 3]], diff^2: [[4, 4], [4, 4], [4, 4]]
        # Add together 24 + 6 = 30
        # Divide by n_params: 30 / (18+18) = 5/6
        expected = torch.tensor(5.0 / 6.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_param_match_loss_multiple_instances(self):
        As = [torch.ones(2, 2, 3)]
        Bs = [torch.ones(2, 3, 2)]
        n_params = 2 * 3 * 2
        target_params = {
            "layer1": torch.tensor([[[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
        }
        spd_params = {"layer1": As[0] @ Bs[0]}
        result = _calc_param_mse(
            params1=target_params,
            params2=spd_params,
            n_params=n_params,
            device="cpu",
        )

        # AB [n_instances=2, d_in=2, d_out=2]: [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]
        # diff^2: [[[1, 1], [1, 1]], [[4, 4], [4, 4]]]
        # Sum together and divide by n_params: [4, 16] / 12 = [1/3, 4/3]
        expected = torch.tensor([1.0 / 3.0, 4.0 / 3.0])
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


class TestCalcActReconLoss:
    def test_calc_topk_act_recon_simple(self):
        # Batch size 2, d_out 2
        target_post_weight_acts = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        layer_acts_topk = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        expected = torch.tensor(0.0)

        result = calc_act_recon(target_post_weight_acts, layer_acts_topk)
        torch.testing.assert_close(result, expected)

    def test_calc_topk_act_recon_different_d_out(self):
        # Batch size 2, d_out 2/3
        target_post_weight_acts = {
            "layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer2": torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        }
        layer_acts_topk = {
            "layer1": torch.tensor([[1.5, 2.5], [4.0, 5.0]]),
            "layer2": torch.tensor([[5.5, 6.5, 7.5], [9.0, 10.0, 11.0]]),
        }
        expected = torch.tensor((0.25 + 1) / 2)  # ((0.5^2 * 5) / 5 + (1^2 * 5) / 5) / 2

        result = calc_act_recon(target_post_weight_acts, layer_acts_topk)
        torch.testing.assert_close(result, expected)

    def test_calc_topk_act_recon_with_n_instances(self):
        target_post_weight_acts = {
            "layer1": torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            "layer2": torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]),
        }
        layer_acts_topk = {
            "layer1": torch.tensor([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]]),
            "layer2": torch.tensor([[[9.5, 10.5], [11.5, 12.5]], [[13.5, 14.5], [15.5, 16.5]]]),
        }
        expected = torch.tensor([0.25, 0.25])  # (0.5^2 * 8) / 8 for each instance

        result = calc_act_recon(target_post_weight_acts, layer_acts_topk)
        torch.testing.assert_close(result, expected)
