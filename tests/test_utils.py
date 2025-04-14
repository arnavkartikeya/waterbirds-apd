from typing import Literal

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from spd.utils import (
    SparseFeatureDataset,
    calc_activation_attributions,
    calc_topk_mask,
    compute_feature_importances,
)


def test_calc_topk_mask_without_batch_topk():
    attribution_scores = torch.tensor([[1.0, 5.0, 2.0, 1.0, 2.0], [3.0, 3.0, 5.0, 4.0, 4.0]])
    topk = 3
    expected_mask = torch.tensor(
        [[False, True, True, False, True], [False, False, True, True, True]]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=False)
    torch.testing.assert_close(result, expected_mask)


def test_calc_topk_mask_with_batch_topk():
    attribution_scores = torch.tensor([[1.0, 5.0, 2.0, 1.0, 2.0], [3.0, 3.0, 5.0, 4.0, 4.0]])
    topk = 3  # mutliplied by batch size to get 6
    expected_mask = torch.tensor(
        [[False, True, False, False, False], [True, True, True, True, True]]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=True)
    torch.testing.assert_close(result, expected_mask)


def test_calc_topk_mask_without_batch_topk_n_instances():
    """attributions have shape [batch, n_instances, n_features]. We take the topk
    over the n_features dim for each instance in each batch."""
    attribution_scores = torch.tensor(
        [[[1.0, 5.0, 3.0, 4.0], [2.0, 4.0, 6.0, 1.0]], [[2.0, 1.0, 5.0, 9.5], [3.0, 4.0, 1.0, 5.0]]]
    )
    topk = 2
    expected_mask = torch.tensor(
        [
            [[False, True, False, True], [False, True, True, False]],
            [[False, False, True, True], [False, True, False, True]],
        ]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=False)
    torch.testing.assert_close(result, expected_mask)


def test_calc_topk_mask_with_batch_topk_n_instances():
    """attributions have shape [batch, n_instances, n_features]. We take the topk
    over the concatenated batch and n_features dim."""
    attribution_scores = torch.tensor(
        [[[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]], [[2.0, 1.0, 5.0], [3.0, 4.0, 1.0]]]
    )
    topk = 2  # multiplied by batch size to get 4
    expected_mask = torch.tensor(
        [[[False, True, True], [False, True, True]], [[True, False, True], [True, True, False]]]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=True)
    torch.testing.assert_close(result, expected_mask)


def test_calc_activation_attributions_obvious():
    component_acts = {"layer1": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    expected = torch.tensor([[1.0, 1.0]])

    result = calc_activation_attributions(component_acts)
    torch.testing.assert_close(result, expected)


def test_calc_activation_attributions_different_d_out():
    component_acts = {
        "layer1": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        "layer2": torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]),
    }
    expected = torch.tensor(
        [[1.0**2 + 2**2 + 5**2 + 6**2 + 7**2, 3**2 + 4**2 + 8**2 + 9**2 + 10**2]]
    )

    result = calc_activation_attributions(component_acts)
    torch.testing.assert_close(result, expected)


def test_calc_activation_attributions_with_n_instances():
    # Batch=1, n_instances=2, C=2, d_out=2
    component_acts = {
        "layer1": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
        "layer2": torch.tensor([[[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]),
    }
    expected = torch.tensor(
        [
            [
                [1.0**2 + 2**2 + 9**2 + 10**2, 3**2 + 4**2 + 11**2 + 12**2],
                [5**2 + 6**2 + 13**2 + 14**2, 7**2 + 8**2 + 15**2 + 16**2],
            ]
        ]
    )

    result = calc_activation_attributions(component_acts)
    torch.testing.assert_close(result, expected)


def test_dataset_at_least_zero_active():
    n_instances = 3
    n_features = 5
    feature_probability = 0.5
    device = "cpu"
    batch_size = 100

    dataset = SparseFeatureDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that the values are between 0 and 1
    assert torch.all((batch >= 0) & (batch <= 1)), "Values should be between 0 and 1"

    # Check that the proportion of non-zero elements is close to feature_probability
    non_zero_proportion = torch.count_nonzero(batch) / batch.numel()
    assert (
        abs(non_zero_proportion - feature_probability) < 0.05
    ), f"Expected proportion {feature_probability}, but got {non_zero_proportion}"


def test_generate_multi_feature_batch_no_zero_samples():
    n_instances = 3
    n_features = 5
    feature_probability = 0.05  # Low probability to increase chance of zero samples
    device = "cpu"
    batch_size = 100
    buffer_ratio = 1.5

    dataset = SparseFeatureDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
    )

    batch = dataset._generate_multi_feature_batch_no_zero_samples(batch_size, buffer_ratio)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that the values are between 0 and 1
    assert torch.all((batch >= 0) & (batch <= 1)), "Values should be between 0 and 1"

    # Check that there are no all-zero samples
    zero_samples = (batch.sum(dim=-1) == 0).sum()
    assert zero_samples == 0, f"Found {zero_samples} samples with all zeros"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_dataset_exactly_n_active(n: int):
    n_instances = 3
    n_features = 10
    feature_probability = 0.5  # This won't be used when data_generation_type="exactly_one_active"
    device = "cpu"
    batch_size = 10
    value_range = (0.0, 1.0)

    n_map: dict[
        int,
        Literal[
            "exactly_one_active",
            "exactly_two_active",
            "exactly_three_active",
            "exactly_four_active",
            "exactly_five_active",
        ],
    ] = {
        1: "exactly_one_active",
        2: "exactly_two_active",
        3: "exactly_three_active",
        4: "exactly_four_active",
        5: "exactly_five_active",
    }
    dataset = SparseFeatureDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type=n_map[n],
        value_range=value_range,
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that there's exactly one non-zero value per sample and instance
    for sample in batch:
        for instance in sample:
            non_zero_count = torch.count_nonzero(instance)
            assert non_zero_count == n, f"Expected {n} non-zero values, but found {non_zero_count}"

    # Check that the non-zero values are in the value_range
    non_zero_values = batch[batch != 0]
    assert torch.all(
        (non_zero_values >= value_range[0]) & (non_zero_values <= value_range[1])
    ), f"Non-zero values should be between {value_range[0]} and {value_range[1]}"


@pytest.mark.parametrize(
    "importance_val, expected_tensor",
    [
        (
            1.0,
            torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        ),
        (
            0.5,
            torch.tensor(
                [[[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]], [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]]]
            ),
        ),
        (
            0.0,
            torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        ),
    ],
)
def test_compute_feature_importances(
    importance_val: float, expected_tensor: Float[Tensor, "batch_size n_instances n_features"]
):
    importances = compute_feature_importances(
        batch_size=2, n_instances=2, n_features=3, importance_val=importance_val, device="cpu"
    )
    torch.testing.assert_close(importances, expected_tensor)


def test_sync_inputs_non_overlapping():
    dataset = SparseFeatureDataset(
        n_instances=1,
        n_features=6,
        feature_probability=0.5,
        device="cpu",
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
        synced_inputs=[[0, 1], [2, 3, 4]],
    )

    batch, _ = dataset.generate_batch(5)
    # Ignore the n_instances dimension
    batch = batch[:, 0, :]
    for sample in batch:
        # If there is a value in 0 or 1, there should be a value in 1 or
        if sample[0] != 0.0:
            assert sample[1] != 0.0
        if sample[1] != 0.0:
            assert sample[0] != 0.0
        if sample[2] != 0.0:
            assert sample[3] != 0.0 and sample[4] != 0.0
        if sample[3] != 0.0:
            assert sample[2] != 0.0 and sample[4] != 0.0
        if sample[4] != 0.0:
            assert sample[2] != 0.0 and sample[3] != 0.0


def test_sync_inputs_overlapping():
    dataset = SparseFeatureDataset(
        n_instances=1,
        n_features=6,
        feature_probability=0.5,
        device="cpu",
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
        synced_inputs=[[0, 1], [1, 2, 3]],
    )
    # Should raise an assertion error with the word "overlapping"
    with pytest.raises(AssertionError, match="overlapping"):
        dataset.generate_batch(5)
