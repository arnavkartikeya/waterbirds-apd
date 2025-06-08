import math
import random
import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from wilds import get_dataset

from models import ColorMNISTConvNetGAP, SPDTwoLayerFC
from spd.run_spd import get_lr_schedule_fn, get_lr_with_warmup
from spd.hooks import HookedRootModule
from spd.log import logger
from spd.models.base import SPDModel
from spd.module_utils import (
    get_nested_module_attr,
    collect_nested_module_attrs,
)
from spd.types import Probability
from spd.utils import set_seed
from train_mnist import SpuriousMNIST

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field
import torch.nn.functional as F


import einops

class SPDConfig(BaseModel):
    # Basic
    seed: int = 0
    batch_size: PositiveInt = 32
    steps: PositiveInt = 500
    lr: float = 1e-3
    print_freq: int = 50
    save_freq: Optional[int] = None
    out_dir: Optional[str] = None

    # Distillation
    distill_coeff: float = 1.0  # how strongly we do MSE with teacher
    # If you want param match
    param_match_coeff: float = 0.0
    # Relative parameter matching
    relative_param_match_coeff: float = 0.0

    # For subcomponent #0 background detection
    alpha_condition: float = 1.0
    cond_coeff: float = 1.0 
    lambda_r: float = 1.0
    mu_r: float = 1.0

    routing_weight_start: float = 0.1
    routing_weight_max: float = 8.0

    # SPD subcomponent config
    C: PositiveInt = 40
    m_fc1: PositiveInt = 16
    m_fc2: PositiveInt = 16

    # LR schedule
    lr_schedule: str = "constant"  # or "linear", "cosine", "exponential"
    lr_exponential_halflife: float | None = None
    lr_warmup_pct: float = 0.0
    concept_corr_target_component_idx: int = 0
    concept_corr_loss_coeff: float = 1.0

    unit_norm_matrices: bool = False
    schatten_coeff: float | None = None
    schatten_pnorm: float | None = None
    relative_schatten_coeff: float | None = None
    # teacher ckpt
    teacher_ckpt: str = "waterbird_resnet18_best.pth"

    # topk config
    topk: float | None = None
    batch_topk: bool = True
    topk_recon_coeff: float | None = None
    distil_from_target: bool = True

    # lp sparsity
    lp_sparsity_coeff: float | None = None
    pnorm: float | None = None
    warmup_steps: int = 1000

    # attribution type
    attribution_type: str = "gradient"  # or "activation"


def save_config_to_yaml(config: SPDConfig, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config.dict(), f)

def load_config_from_yaml(path: str) -> SPDConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SPDConfig(**data)

# def calc_routing_loss(delta_attrib, delta_attrib_d_cf, lambda_r, device):
#     return ((delta_attrib[:, 0].abs()).mean() * lambda_r)/(delta_attrib_d_cf[:, 0].abs().mean() + 1e-8) + \
#          (delta_attrib_d_cf[:, 1].abs().mean() * lambda_r)/ (delta_attrib[:, 1].abs().mean() + 1e-8)  # reduce l1 norm for now 
import torch.nn.functional as F
import torch
import torch.autograd as autograd

def calculate_S_concept_colormnist_spurious(
    batch_x: torch.Tensor,
    batch_background_labels: torch.Tensor, # 0 for red, 1 for green
    spd_model_output_logits: torch.Tensor # Output of P_0 + P_1
) -> torch.Tensor:
    """
    Calculates S_concept for ColorMNIST's spurious correlation.
    Assumes red (0) correlates with low digits (0-4), green (1) with high (5-9).
    S_concept measures how much the model's output aligns with this spurious expectation.

    Args:
        batch_x: Input features (not directly used here but often part of signature).
        batch_background_labels: Tensor of background labels (0 or 1).
        spd_model_output_logits: Logits from the full SPD model (sum of components).
                                 Shape: (batch_size, num_classes)

    Returns:
        torch.Tensor: S_concept scores for each item in the batch. Shape: (batch_size,)
    """
    batch_size, num_classes = spd_model_output_logits.shape
    s_concept_scores = torch.zeros(batch_size, device=spd_model_output_logits.device)

    # Define which logits correspond to "spuriously expected" classes
    # For red background (label 0), expect low digits (e.g., classes 0-4)
    is_red_bg = (batch_background_labels == 0)
    if torch.any(is_red_bg):
        logits_for_red = spd_model_output_logits[is_red_bg]
        # Sum of logits for classes 0, 1, 2, 3, 4
        # Adjust class indices if your num_classes is different or mapping is different
        s_concept_scores[is_red_bg] = logits_for_red[:, 0:5].sum(dim=1)

    # For green background (label 1), expect high digits (e.g., classes 5-9)
    is_green_bg = (batch_background_labels == 1)
    if torch.any(is_green_bg):
        logits_for_green = spd_model_output_logits[is_green_bg]
        # Sum of logits for classes 5, 6, 7, 8, 9
        s_concept_scores[is_green_bg] = logits_for_green[:, 5:10].sum(dim=1)

    return s_concept_scores

def pearson_correlation_loss(tensor1: torch.Tensor, tensor2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if tensor1.numel() < 2 or tensor2.numel() < 2:
        return torch.tensor(0.0, device=tensor1.device, requires_grad=True)
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)
    std1 = torch.std(tensor1, unbiased=False)
    std2 = torch.std(tensor2, unbiased=False)
    cov = torch.mean((tensor1 - mean1) * (tensor2 - mean2))
    correlation = cov / (std1 * std2 + eps)
    return -correlation # Returns -Correlation, so a smaller loss means higher positive correlation

def calculate_concept_correlation_loss(
    spd_model: SPDTwoLayerFC,
    feats: torch.Tensor,                       # Input features for SPD model, shape (batch_size, feature_dim)
    background_labels: torch.Tensor,           # Metadata, e.g., background color, shape (batch_size,)
    target_component_idx: int,                 # Index of P_k to condition (e.g., 0 for P0)
    loss_coefficient: float,                   # Coefficient for this loss term
    device: torch.device
) -> torch.Tensor:
    """
    Calculates a loss to encourage P_k's attribution (A_k) to S_concept
    to be highly correlated with S_concept itself.

    Args:
        spd_model: The SPD model.
        feats: Input features to the SPD model.
        background_labels: Metadata for calculating S_concept.
        target_component_idx: The index of the component (P_k) to condition.
        loss_coefficient: Multiplier for the final loss.
        device: Torch device.

    Returns:
        Scalar loss tensor.
    """
    if loss_coefficient == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- 1. Get S_concept (per-sample) ---
    # Ensure spd_model parameters require grad for this forward pass
    for param in spd_model.parameters():
        param.requires_grad_(True)
    
    # This forward pass needs to build a graph connecting spd_model params to spd_logits_full_model
    spd_logits_full_model = spd_model(feats) 
    
    S_concept_batch_vector = calculate_S_concept_colormnist_spurious(
        feats, background_labels, spd_logits_full_model
    ) # Shape: (batch_size,)

    if S_concept_batch_vector.numel() < 2: # Correlation needs at least 2 points
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- 2. Get P_k's parameters (full tensors) ---
    # These are the leaf nodes we'll differentiate S_concept with respect to
    params_for_grad_computation = []
    param_names_for_grad_computation = [] 

    # Assuming spd_model has fc1 and fc2 attributes which are SPDLinear-like
    # And SPDLinear has A and B attributes which are nn.Parameter tensors of shape (C, ...)
    if hasattr(spd_model, 'fc1') and spd_model.fc1 is not None:
        if hasattr(spd_model.fc1, 'A') and spd_model.fc1.A is not None and spd_model.fc1.A.requires_grad:
            params_for_grad_computation.append(spd_model.fc1.A)
            param_names_for_grad_computation.append('fc1.A')
        if hasattr(spd_model.fc1, 'B') and spd_model.fc1.B is not None and spd_model.fc1.B.requires_grad:
            params_for_grad_computation.append(spd_model.fc1.B)
            param_names_for_grad_computation.append('fc1.B')
        # Add global bias if present and learnable
        if hasattr(spd_model.fc1, 'bias') and isinstance(spd_model.fc1.bias, torch.nn.Parameter) and spd_model.fc1.bias.requires_grad:
             params_for_grad_computation.append(spd_model.fc1.bias)
             param_names_for_grad_computation.append('fc1.bias')


    if hasattr(spd_model, 'fc2') and spd_model.fc2 is not None:
        if hasattr(spd_model.fc2, 'A') and spd_model.fc2.A is not None and spd_model.fc2.A.requires_grad:
            params_for_grad_computation.append(spd_model.fc2.A)
            param_names_for_grad_computation.append('fc2.A')
        if hasattr(spd_model.fc2, 'B') and spd_model.fc2.B is not None and spd_model.fc2.B.requires_grad:
            params_for_grad_computation.append(spd_model.fc2.B)
            param_names_for_grad_computation.append('fc2.B')
        if hasattr(spd_model.fc2, 'bias') and isinstance(spd_model.fc2.bias, torch.nn.Parameter) and spd_model.fc2.bias.requires_grad:
             params_for_grad_computation.append(spd_model.fc2.bias)
             param_names_for_grad_computation.append('fc2.bias')

    if not params_for_grad_computation:
        # print("Warning: No learnable full parameter tensors found for differentiation in calculate_concept_correlation_loss.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- 3. Calculate per-sample A_k ---
    A_k_batch_vector_list = []
    for i in range(feats.shape[0]): # Iterate over batch samples
        s_concept_single_sample = S_concept_batch_vector[i]

        for p in params_for_grad_computation: # Zero grads before each .grad call
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        
        grads_S_wrt_full_params_tuple = autograd.grad(
            outputs=s_concept_single_sample,
            inputs=params_for_grad_computation,
            retain_graph=True, # Crucial for loop, S_concept_batch_vector graph is reused
            create_graph=True, # A_k will be part of the loss, so P_k needs to get grads from it
            allow_unused=True 
        )
        
        grads_S_wrt_full_params_dict = dict(zip(param_names_for_grad_computation, grads_S_wrt_full_params_tuple))

        # This A_k needs to be differentiable wrt P_k's parameters
        # So, param_val_pk should NOT be detached here.
        current_A_k_sample_i = torch.tensor(0.0, device=device, dtype=feats.dtype).requires_grad_() # Ensure it's part of graph

        # Layer fc1 contribution to A_k
        if 'fc1.A' in grads_S_wrt_full_params_dict and grads_S_wrt_full_params_dict['fc1.A'] is not None:
            # P_k's actual parameters for fc1.A (these are part of the model, require grad)
            Pk_slice_A_fc1 = spd_model.fc1.A[target_component_idx] 
            grad_slice_A_fc1 = grads_S_wrt_full_params_dict['fc1.A'][target_component_idx]
            current_A_k_sample_i = current_A_k_sample_i + (grad_slice_A_fc1.flatten() * Pk_slice_A_fc1.flatten()).sum()

        if 'fc1.B' in grads_S_wrt_full_params_dict and grads_S_wrt_full_params_dict['fc1.B'] is not None:
            Pk_slice_B_fc1 = spd_model.fc1.B[target_component_idx]
            grad_slice_B_fc1 = grads_S_wrt_full_params_dict['fc1.B'][target_component_idx]
            current_A_k_sample_i = current_A_k_sample_i + (grad_slice_B_fc1.flatten() * Pk_slice_B_fc1.flatten()).sum()

        # Layer fc2 contribution to A_k
        if 'fc2.A' in grads_S_wrt_full_params_dict and grads_S_wrt_full_params_dict['fc2.A'] is not None:
            Pk_slice_A_fc2 = spd_model.fc2.A[target_component_idx]
            grad_slice_A_fc2 = grads_S_wrt_full_params_dict['fc2.A'][target_component_idx]
            current_A_k_sample_i = current_A_k_sample_i + (grad_slice_A_fc2.flatten() * Pk_slice_A_fc2.flatten()).sum()

        if 'fc2.B' in grads_S_wrt_full_params_dict and grads_S_wrt_full_params_dict['fc2.B'] is not None:
            Pk_slice_B_fc2 = spd_model.fc2.B[target_component_idx]
            grad_slice_B_fc2 = grads_S_wrt_full_params_dict['fc2.B'][target_component_idx]
            current_A_k_sample_i = current_A_k_sample_i + (grad_slice_B_fc2.flatten() * Pk_slice_B_fc2.flatten()).sum()
        
        # Bias contribution to A_k (if global biases are part of Pk's "responsibility")
        # This is conceptually tricky. If biases are global, Pk doesn't "own" a slice.
        # If Pk is meant to modulate its A,B matrices to achieve the S_concept effect,
        # then including global biases in Pk's attribution might not be standard.
        # For now, focusing on A, B matrices.

        A_k_batch_vector_list.append(current_A_k_sample_i)

    if not A_k_batch_vector_list or len(A_k_batch_vector_list) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    A_k_batch_tensor = torch.stack(A_k_batch_vector_list) # Shape: (batch_size,)
    
    # --- 4. Calculate Correlation Loss ---
    # pearson_correlation_loss returns -correlation. We want to minimize this.
    # (i.e., maximize correlation)
    correlation_loss_val = pearson_correlation_loss(A_k_batch_tensor, S_concept_batch_vector.detach())
    
    return loss_coefficient * correlation_loss_val

def calc_routing_loss(delta_attrib, delta_attrib_d_cf, lambda_r, device):
    # For digit counterfactuals: we want component 0 to have high attribution
    # So we want softmax to put more probability on component 0
    digit_probs = F.softmax(delta_attrib_d_cf, dim=1)  # shape: [batch, 2]
    digit_loss = -torch.log(digit_probs[:, 0] + 1e-8).mean()  # negative log likelihood of component 0
    
    # For background counterfactuals: we want component 1 to have high attribution  
    # So we want softmax to put more probability on component 1
    bg_probs = F.softmax(delta_attrib, dim=1)  # shape: [batch, 2]
    bg_loss = -torch.log(bg_probs[:, 1] + 1e-8).mean()  # negative log likelihood of component 1
    
    return lambda_r * (digit_loss + bg_loss)

def new_calc_routing_loss(
     delta: torch.Tensor,
     delta_d: torch.Tensor,
     lambda_r: float,
     mu_r: float,
     *,
     low: float = 1.0,
     high: float = 7.0
) -> torch.Tensor:
    # always index with literal ints 0 and 1
    diff_c0   = torch.abs(delta[:, 0])
    diff_d_c0 = torch.abs(delta_d[:, 0])
    diff_c1   = torch.abs(delta[:, 1])
    diff_d_c1 = torch.abs(delta_d[:, 1])

    loss_c0_bg = lambda_r * F.relu(diff_c0   - low).mean()
    loss_c0_d  = mu_r     * F.relu(high - diff_d_c0).mean()
    loss_c1_bg = lambda_r * F.relu(high - diff_c1).mean()
    loss_c1_d  = mu_r     * F.relu(diff_d_c1 - low).mean()

    return loss_c0_bg + loss_c0_d + loss_c1_bg + loss_c1_d


def calc_concept_conditioning_loss(
    spd_model: SPDTwoLayerFC,
    feats_orig: torch.Tensor,        # Features from original input x
    feats_cf_bg: torch.Tensor,       # Features from background counterfactual x_cf
    digit_labels: torch.Tensor,      # True digit labels for x (and x_cf_bg)
    config_lambda_invariance: float, # Coefficient for the invariance loss
    config_mu_task_correctness: float, # Coefficient for the task correctness loss
    device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates a loss to condition component P_1 (index 1) to be the "digit concept"
    and P_0 (index 0) to be the "background concept".

    This is achieved by:
    1. L_invariance: Ensuring f(x | P_1) is invariant to background changes.
       i.e., output of P_1 on original input approx. output of P_1 on background_cf input.
    2. L_ablated_task_correctness: Ensuring f(x | P_1) correctly classifies the digit.

    Args:
        spd_model: The SPDTwoLayerFC model.
        feats_orig: Features from original input (B, feature_dim).
        feats_cf_bg: Features from background counterfactual input (B, feature_dim).
        digit_labels: Digit labels (B,).
        config_lambda_invariance: Weight for the invariance loss.
        config_mu_task_correctness: Weight for the task correctness loss.
        device: Torch device.

    Returns:
        total_loss: The combined conditioning loss.
        loss_invariance: The invariance part of the loss.
        loss_task_orig: Task loss on original features using P_1.
        loss_task_cf_bg: Task loss on background_cf features using P_1.
    """
    # Component P_1 (index 1) is intended for the true digit concept.
    # Component P_0 (index 0) is intended for the spurious background concept.
    # We evaluate the network using only P_1.
    component_idx_digit = 1

    # Get weights for P_1 (the digit component)
    # fc1 layer, component P_1
    # A shape: (C, d_in, m), B shape: (C, m, d_out)
    A_fc1_p1 = spd_model.fc1.A[component_idx_digit]  # Shape: (d_in, m_fc1)
    B_fc1_p1 = spd_model.fc1.B[component_idx_digit]  # Shape: (m_fc1, hidden_dim)
    W_fc1_p1 = torch.matmul(A_fc1_p1, B_fc1_p1)     # Shape: (d_in, hidden_dim)

    # fc2 layer, component P_1
    A_fc2_p1 = spd_model.fc2.A[component_idx_digit]  # Shape: (hidden_dim, m_fc2)
    B_fc2_p1 = spd_model.fc2.B[component_idx_digit]  # Shape: (m_fc2, num_classes)
    W_fc2_p1 = torch.matmul(A_fc2_p1, B_fc2_p1)     # Shape: (hidden_dim, num_classes)

    # --- Forward pass using only P_1 for original input ---
    h1_abl_P0_orig = torch.matmul(feats_orig, W_fc1_p1)
    # Assuming SPDLinear does not have per-component biases or shared biases are handled elsewhere/ignored for P_c
    # If spd_model.fc1.bias exists and is per-component: h1_abl_P0_orig += spd_model.fc1.bias[component_idx_digit]
    # If shared: h1_abl_P0_orig += spd_model.fc1.bias
    h_relu_abl_P0_orig = F.relu(h1_abl_P0_orig)
    out_abl_P0_orig = torch.matmul(h_relu_abl_P0_orig, W_fc2_p1)
    # Handle fc2 bias similarly if it exists

    # --- Forward pass using only P_1 for background counterfactual input ---
    h1_abl_P0_cf_bg = torch.matmul(feats_cf_bg, W_fc1_p1)
    h_relu_abl_P0_cf_bg = F.relu(h1_abl_P0_cf_bg)
    out_abl_P0_cf_bg = torch.matmul(h_relu_abl_P0_cf_bg, W_fc2_p1)

    # 1. L_invariance: Output of P_1 should be similar for x and x_cf_bg
    # Using MSE on logits. Could also use KL divergence on softmax probabilities.
    loss_invariance = F.mse_loss(out_abl_P0_orig, out_abl_P0_cf_bg)

    # 2. L_ablated_task_correctness: P_1 should correctly classify the digit
    criterion_task = nn.CrossEntropyLoss()
    loss_task_orig = criterion_task(out_abl_P0_orig, digit_labels)
    loss_task_cf_bg = criterion_task(out_abl_P0_cf_bg, digit_labels) # y_digit is same for x_cf_bg

    combined_task_loss = loss_task_orig + loss_task_cf_bg
    
    total_conditioning_loss = (config_lambda_invariance * loss_invariance +
                               config_mu_task_correctness * combined_task_loss)
    
    return total_conditioning_loss, loss_invariance, loss_task_orig, loss_task_cf_bg



# def calc_routing_loss(delta_attrib, delta_attrib_d_cf, lambda_r, mu_r, device):
#     """
#     Dual objective routing loss:
#     1. Minimize delta_attrib[:, 0] (background color swap should not affect component 0)
#     2. Maximize delta_attrib_d_cf[:, 0] (digit swap should strongly affect component 0)
    
#     Args:
#         delta_attrib: Attribution difference for color swap (background change)
#         delta_attrib_d_cf: Attribution difference for digit swap (content change)
#         lambda_r: Weight for minimizing background attribution
#         mu_r: Weight for maximizing digit attribution
#         device: Computing device
    
#     Returns:
#         Combined routing loss (lower is better)
#     """
#     # Minimize background attribution (push delta_attrib[:, 0] to zero)
#     background_loss = (delta_attrib[:, 0]**2).mean() * lambda_r
    
#     # Maximize digit attribution (encourage large delta_attrib_d_cf[:, 0])
#     # Using negative exponential to create diminishing returns for very large values
#     # This avoids the model just making this term arbitrarily large
#     digit_loss = torch.exp(-torch.clamp(delta_attrib_d_cf[:, 0]**2, min=0, max=10)).mean() * mu_r

#     background_c1_loss = torch.exp(-torch.clamp(delta_attrib[:, 1]**2, min=0, max=10)).mean() * lambda_r
#     digit_c1_loss = (delta_attrib_d_cf[:, 1]**2).mean() * mu_r
    
#     # Total loss (both terms are positive, lower is better)
#     return background_loss + digit_loss + background_c1_loss + digit_c1_loss


def make_cf_digit_swap(imgs: torch.Tensor, digit_label: torch.Tensor, background_label: torch.Tensor) -> torch.Tensor:
    """
    Colour-MNIST counter-factual: replace each digit with a different digit that has
    the same background color.
    
    Args:
        imgs: (B,3,28,28) – RGB images in [0,1]
        digit_label: (B,) – labels for each digit (0-9)
        background_label: (B,) – labels for background color (0=red, 1=green)
    
    Returns:
        A tensor of counterfactual images with different digits but same background colors
    """
    batch_size = imgs.shape[0]
    cf_imgs = imgs.clone()
    
    # Create groups of images by background color
    red_indices = (background_label == 0).nonzero(as_tuple=True)[0]
    green_indices = (background_label == 1).nonzero(as_tuple=True)[0]
    
    # Process each image in the batch
    for i in range(batch_size):
        current_digit = digit_label[i]
        current_bg = background_label[i]
        
        # Get pool of candidates with same background but different digit
        if current_bg == 0:  # Red background
            candidates = [idx for idx in red_indices if digit_label[idx] != current_digit]
        else:  # Green background
            candidates = [idx for idx in green_indices if digit_label[idx] != current_digit]
        
        # If we have candidates, randomly select one
        if candidates:
            # Choose a random candidate
            cf_idx = random.choice(candidates)
            # Replace the digit (keep the background)
            cf_imgs[i] = imgs[cf_idx]
        # If no candidates (unlikely but possible), keep original image
    
    return cf_imgs

def calc_grad_attributions(
    model_out: torch.Tensor,  # teacher or spd output
    post_weight_acts: dict[str, torch.Tensor],
    pre_weight_acts: dict[str, torch.Tensor],
    component_weights: dict[str, torch.Tensor],
    C: int,
) -> torch.Tensor:
    """
    Like TMS's gradient approach: for each output dimension,
    grad wrt post acts * subcomponent partial forward -> subcomponent attribution
    Summed across output dims and squared.
    """
    import torch.autograd as autograd

    # unify keys
    post_names = [k.removesuffix(".hook_post") for k in post_weight_acts.keys()]
    pre_names = [k.removesuffix(".hook_pre") for k in pre_weight_acts.keys()]
    comp_names = list(component_weights.keys())
    assert set(post_names) == set(pre_names) == set(comp_names), "layer name mismatch"

    batch_prefix = model_out.shape[:-1]  # e.g. (batch,) or (batch, n_inst)
    out_dim = model_out.shape[-1]
    attribution_scores = torch.zeros((*batch_prefix, C), device=model_out.device)

    # get subcomponent partial forward
    component_acts = {}
    for nm in pre_names:
        # shape pre: (batch..., d_in), comp_W: (C, d_in, d_out)
        # => (batch..., C, d_out)
        pre_ = pre_weight_acts[nm + ".hook_pre"].detach()
        w_ = component_weights[nm]
        partial = einops.einsum(
            pre_, w_, "... d_in, C d_in d_out -> ... C d_out"
        )
        component_acts[nm] = partial

    for feature_idx in range(out_dim):
        # sum up that scalar
        grads = autograd.grad(
            model_out[..., feature_idx].sum(),
            list(post_weight_acts.values()),
            retain_graph=True,
        )
        # grads is tuple of same length as post_weight_acts
        feature_attrib = torch.zeros((*batch_prefix, C), device=model_out.device)
        for grad_val, nm_post in zip(grads, post_weight_acts.keys()):
            nm_clean = nm_post.removesuffix(".hook_post")
            feature_attrib += einops.einsum(
                grad_val, component_acts[nm_clean],
                "... d_out, ... C d_out -> ... C"
            )
        # square then accumulate
        attribution_scores += feature_attrib**2

    return attribution_scores

def calc_relative_param_match_loss(
    param_names,
    target_model,
    spd_model,
    device,
):
    target_params_layers = {} 
    spd_params_layers = {}
    for param_name in param_names:
        target_params_layers[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params_layers[param_name] = get_nested_module_attr(spd_model, param_name + ".weight").transpose(-1, -2)

    total_norm_diff = torch.tensor(0.0, device=device)
    total_norm_target = torch.tensor(0.0, device=device)

    for param in param_names:
        target = target_params_layers[param]
        spd = spd_params_layers[param]

        norm_diff = torch.norm(target - spd)
        norm_target = torch.norm(target)

        total_norm_diff += norm_diff
        total_norm_target += norm_target

    return total_norm_diff / (total_norm_target + 1e-8)

def calculate_attributions(
    model: SPDTwoLayerFC,
    input_x: torch.Tensor,
    out: torch.Tensor,
    teacher_out: torch.Tensor,
    pre_acts: dict[str, torch.Tensor],
    post_acts: dict[str, torch.Tensor],
    component_acts: dict[str, torch.Tensor],
    attribution_type: str = "gradient",
) -> torch.Tensor:
    """
    The snippet used in TMS: if attribution_type="gradient", we do
    gradient w.r.t. teacher_out, else we can do activation-based, etc.
    For clarity, we show gradient-based here.
    """
    if attribution_type == "gradient":
        # We'll do gradient-based attributions wrt teacher_out
        # teacher_out shape e.g. [B, 2]
        import torch.autograd as autograd

        # We basically replicate your snippet from TMS code:
        #   grad_post_weight_acts => multiply subcomponent partial forward => sum => square
        # We'll just define a small function inline:
        post_names = [k.removesuffix(".hook_post") for k in post_acts]
        pre_names  = [k.removesuffix(".hook_pre") for k in pre_acts]
        comp_names = list(component_acts.keys())  # these are like "fc1", "fc2"
        # check they match
        assert set(post_names) == set(pre_names) == set(comp_names), \
            f"Mismatch: {post_names}, {pre_names}, {comp_names}"

        # Build partial forward for SPD subcomponents
        # i.e. (input_x @ A[k]) @ B[k]
        # but we already have it in "component_acts" if the hooking is done.
        # Actually, hooking is "hook_component_acts"? let's see. For TMS, we do an alternate approach:
        # We'll do "manual partial forward" or rely on hooking. We'll do hooking if you want. We'll do it manually:
        # But let's do it simpler: we reuse 'component_acts'
        # The shape is (batch, C, d_out).
        # We'll do a single-later approach. For multi-layers, you'd sum them, etc.

        # We'll unify them into one big "component_acts" for each layer name:
        # Then do the gradient wrt teacher_out dimension by dimension
        batch_shape = teacher_out.shape[:-1]
        c_dim = model.C  # subcomponent dim
        attributions = torch.zeros((*batch_shape, c_dim), device=teacher_out.device)

        out_dim = teacher_out.shape[-1]
        # 1) get grad wrt post_acts
        # Because we only have teacher_out -> post_acts for each layer
        grad_list = autograd.grad(
            teacher_out.sum(),  # sum => scalar
            list(post_acts.values()),
            retain_graph=True
        )
        # grad_list is the same length as # of post_acts
        # Now multiply each grad by subcomponent_acts for that layer
        for grad_val, post_name in zip(grad_list, post_acts.keys()):
            lay_name = post_name.removesuffix(".hook_post")
            # subcomponent partial: comp_acts[lay_name] => shape (batch, C, d_out)
            # multiply grad_val shape (batch, d_out) => sum over d_out => (batch, C)
            # Then we square or accumulate?
            # TMS code squares across output dims. But let's replicate your TMS approach dimension by dimension
            # We'll do a simpler approach: multiply them once, sum over d_out => we get (batch, C).
            partial_contrib = einops.einsum(
                grad_val, component_acts[lay_name], "... d_out, ... C d_out -> ... C"
            )
            # TMS: squares then sums across output dims. If you want each dimension, you'd do it dimension by dimension
            # We'll keep it simple and just add squares:
            attributions += partial_contrib # shoudl be squared but lets see what happens
        return attributions

    elif attribution_type == "activation":
        # Summation of L2 norms for each subcomponent in 'component_acts'
        # shape is (batch, C, d_out). sum over d_out => (batch, C)
        # plus for each layer
        attributions = torch.zeros(
            (input_x.shape[0], model.C), device=input_x.device
        )
        for layername, acts in component_acts.items():
            attributions += acts.pow(2).sum(dim=-1)
        return attributions
    else:
        raise ValueError(f"Invalid attribution_type={attribution_type}")




def calc_relative_schatten_loss(
    As: dict[str, torch.Tensor],
    Bs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    p: float,
    device: str,
) -> torch.Tensor:
    """
    Relative Schatten‑p loss:
        ρ = 1 −  ||P_c||_p / ||P_c||_p(max‑entropy)
    averaged over batch and layers.
    """
    assert As.keys() == Bs.keys()
    batch = mask.shape[0]
    n_instances = mask.shape[1] if mask.ndim == 3 else None
    acc = torch.zeros((n_instances,) if n_instances else (), device=device)

    for name in As:
        A, B = As[name], Bs[name]           # shapes as in your docstring
        m = A.shape[-1]                     # rank dimension

        S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")
        S_AB = S_A * S_B                    # proxy for s_i^2

        S_AB = einops.einsum(
            S_AB, mask,
            "... C m, batch ... C -> batch ... C m"
        )

        actual = (S_AB + 1e-16).pow(0.5 * p).sum(dim=-1)        # Σ s_i^{p/2}
        total = S_AB.sum(dim=-1)                                 # Σ s_i
        max_ent = (m ** (1 - 0.5 * p)) * (total + 1e-16).pow(0.5 * p)

        ratio = torch.clamp(actual / max_ent, 0.0, 1.0)
        rel = ratio                                        # lower ⇒ simpler
        acc += rel.sum(dim=(0, -1))                              # sum batch + C

    return acc / (batch * len(As))


def make_cf(imgs: torch.Tensor) -> torch.Tensor:
    cf = imgs.clone()
    cf[:, 0], cf[:, 1] = imgs[:, 1], imgs[:, 0]
    return cf

def run_spd(config, train_loader, device):

    teacher_model = ColorMNISTConvNetGAP(num_classes=10, hidden_dim=128)
    teacher_model.load_state_dict(torch.load(config.teacher_ckpt, map_location="cpu"))
    teacher_model.to(device)
    teacher_model.eval()

    spd_fc = SPDTwoLayerFC(
        in_features=28*28,
        hidden_dim=128,
        num_classes=10,
        C=config.C,
        m_fc1=config.m_fc1,
        m_fc2=config.m_fc2,
    )
    spd_fc.to(device)

    opt = optim.AdamW(spd_fc.parameters(), lr=config.lr, weight_decay=0.0)
    lr_sched_fn = get_lr_schedule_fn(
        config.lr_schedule,
        config.lr_exponential_halflife,
    )
    mse = nn.MSELoss()

    trunk = teacher_model.conv
    teacher_fc1 = teacher_model.fc1
    teacher_fc2 = teacher_model.fc2

    best_loss = float('inf')

    for step in tqdm(range(config.steps+1), ncols=100):
        opt.zero_grad(set_to_none=True)
        batch = next(iter(train_loader))
        x, y, m = batch
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        #generate counterfactuals 
        x_cf = make_cf(x)
        x_d_cf = make_cf_digit_swap(x, y, m)

        # calculate faithfulness loss 
        relative_param_match_loss = calc_relative_param_match_loss(
            param_names=["fc1", "fc2"],
            target_model=teacher_model,
            spd_model=spd_fc,
            device=device
        )

        # calculate simplicity loss 
        As = {"fc1": spd_fc.fc1.A, "fc2": spd_fc.fc2.A}
        Bs = {"fc1": spd_fc.fc1.B, "fc2": spd_fc.fc2.B}
        mask = torch.ones((x.size(0), 1), device=device, dtype=torch.bool)
        relative_schatten_loss = calc_relative_schatten_loss(
            As=As,
            Bs=Bs,
            mask=mask,
            p=config.schatten_pnorm,
            device=device
        )

        # calculate attribution loss 
        with torch.no_grad():
            feats_cf = trunk(x_cf).mean(1).flatten(1) 
            feats_d_cf = trunk(x_d_cf).mean(1).flatten(1)

        with torch.no_grad():
            feats = trunk(x)
            feats = feats.mean(dim=1)
            feats = feats.flatten(1)  # [B, 512]

        #=========================
        # (2) feats_with_grad
        #=========================
        feats_with_grad = feats.detach().clone().requires_grad_(True)
        feats_d_cf_grad = feats_d_cf.detach().clone().requires_grad_(True)

        #=========================
        # (3) teacher forward pass manually, storing in a teacher_cache
        #=========================
        feats_cf_grad = feats_cf.detach().clone().requires_grad_(True)
        t_cache_cf = {}
        t_cache_cf_d_cf = {}

        t_cache_cf["fc1.hook_pre"]  = feats_cf_grad
        t_cache_cf_d_cf["fc1.hook_pre"]  = feats_d_cf_grad
        h1_cf = teacher_fc1(feats_cf_grad)
        h1_d_cf = teacher_fc1(feats_d_cf_grad)
        t_cache_cf["fc1.hook_post"] = h1_cf
        t_cache_cf_d_cf["fc1.hook_post"] = h1_d_cf
        h_relu_cf = torch.relu(h1_cf)
        h_relu_d_cf = torch.relu(h1_d_cf)
        t_cache_cf["fc2.hook_pre"]  = h_relu_cf
        t_cache_cf_d_cf["fc2.hook_pre"]  = h_relu_d_cf
        teacher_out_cf              = teacher_fc2(h_relu_cf)
        teacher_out_d_cf = teacher_fc2(h_relu_d_cf)
        t_cache_cf["fc2.hook_post"] = teacher_out_cf
        t_cache_cf_d_cf["fc2.hook_post"] = teacher_out_d_cf

        teacher_cache = {}

        teacher_cache["fc1.hook_pre"] = feats_with_grad
        teacher_h_pre = teacher_fc1(feats_with_grad)
        teacher_cache["fc1.hook_post"] = teacher_h_pre

        teacher_h = torch.relu(teacher_h_pre)
        teacher_cache["fc2.hook_pre"] = teacher_h

        teacher_out = teacher_fc2(teacher_h)
        teacher_cache["fc2.hook_post"] = teacher_out

        #=========================
        # (4) SPD forward pass with hooking
        #=========================
        spd_fc.reset_hooks()
        cache_d_cf, fwd_hooks_d_cf, _ = spd_fc.get_caching_hooks()
        with spd_fc.hooks(fwd_hooks_d_cf, [], reset_hooks_end=True):
            h1_spd_d_cf = spd_fc.fc1(feats_d_cf)
            h_spd_d_cf  = torch.relu(h1_spd_d_cf)
            spd_out_d_cf = spd_fc.fc2(h_spd_d_cf)
        spd_fc.reset_hooks()

        spd_fc.reset_hooks()
        cache_cf, fwd_hooks_cf, _ = spd_fc.get_caching_hooks()
        with spd_fc.hooks(fwd_hooks_cf, [], reset_hooks_end=True):
            h1_spd_cf = spd_fc.fc1(feats_cf)
            h_spd_cf  = torch.relu(h1_spd_cf)
            spd_out_cf = spd_fc.fc2(h_spd_cf)
        spd_fc.reset_hooks()
        cache_dict, fwd_hooks, _ = spd_fc.get_caching_hooks()
        with spd_fc.hooks(fwd_hooks, [], reset_hooks_end=True):
            spd_h_pre = spd_fc.fc1(feats)
            spd_h = torch.relu(spd_h_pre)
            spd_out = spd_fc.fc2(spd_h)

        #=========================
        # (5) gather SPD activations
        #=========================
        pre_d_cf  = {k:v for k,v in cache_d_cf.items() if k.endswith("hook_pre")}
        post_d_cf = {k:v for k,v in cache_d_cf.items() if k.endswith("hook_post")}
        comp_d_cf = {k.removesuffix(".hook_component_acts"):v
                for k,v in cache_d_cf.items() if k.endswith("hook_component_acts")}


        pre_cf  = {k:v for k,v in cache_cf.items() if k.endswith("hook_pre")}
        post_cf = {k:v for k,v in cache_cf.items() if k.endswith("hook_post")}
        comp_cf = {k.removesuffix(".hook_component_acts"):v
                for k,v in cache_cf.items() if k.endswith("hook_component_acts")}
        pre_weight_acts = {}
        post_weight_acts = {}
        comp_acts = {}
        for k,v in cache_dict.items():
            if k.endswith("hook_pre"):
                pre_weight_acts[k] = v
            elif k.endswith("hook_post"):
                post_weight_acts[k] = v
            elif k.endswith("hook_component_acts"):
                comp_acts[k.removesuffix(".hook_component_acts")] = v  # e.g. "fc1", "fc2"

        #=========================
        # (6) teacher pre/post from teacher_cache
        #=========================
        teacher_pre_acts = {k:v for k,v in teacher_cache.items() if k.endswith("hook_pre")}
        teacher_post_acts= {k:v for k,v in teacher_cache.items() if k.endswith("hook_post")}

        feats_for_concept_loss = feats.detach().clone() 

        concept_correlation_loss_p0 = calculate_concept_correlation_loss(
            spd_model=spd_fc,
            feats=feats_for_concept_loss,
            background_labels=m, # background_label from batch
            target_component_idx=config.concept_corr_target_component_idx, # e.g., 0
            loss_coefficient=config.concept_corr_loss_coeff,
            device=device
        )


        #=========================
        # (7) calculate attributions
        #=========================
        # attrib_d_cf = calculate_attributions(
        #     model          = spd_fc,
        #     input_x        = feats_d_cf,
        #     out            = spd_out_d_cf,
        #     teacher_out    = teacher_out_d_cf if config.distil_from_target else spd_out_d_cf,
        #     pre_acts       = {k:v for k,v in t_cache_cf_d_cf.items() if k.endswith("hook_pre")},
        #     post_acts      = {k:v for k,v in t_cache_cf_d_cf.items() if k.endswith("hook_post")},
        #     component_acts = comp_d_cf,
        #     attribution_type = config.attribution_type,
        # )

        # attrib_cf = calculate_attributions(
        #     model          = spd_fc,
        #     input_x        = feats_cf,
        #     out            = spd_out_cf,
        #     teacher_out    = teacher_out_cf if config.distil_from_target else spd_out_cf,
        #     pre_acts       = {k:v for k,v in t_cache_cf.items() if k.endswith("hook_pre")},
        #     post_acts      = {k:v for k,v in t_cache_cf.items() if k.endswith("hook_post")},
        #     component_acts = comp_cf,
        #     attribution_type = config.attribution_type,
        # )

        # attributions = calculate_attributions(
        #     model=spd_fc,
        #     input_x=feats,
        #     out=spd_out,
        #     teacher_out=teacher_out if getattr(config,"distil_from_target",True) else spd_out,
        #     pre_acts=teacher_pre_acts,
        #     post_acts=teacher_post_acts,
        #     component_acts=comp_acts,
        #     attribution_type=config.attribution_type
        # )

        # delta_attrib = attributions - attrib_cf 
        # delta_attrib_d_cf = attrib_d_cf - attrib_cf
        # routing_loss = calc_routing_loss(delta_attrib, delta_attrib_d_cf, config.lambda_r, device) * min(1.0, step / config.warmup_steps)
        concept_loss, inv_loss_val, task_orig_val, task_cf_val = calc_concept_conditioning_loss(
            spd_model=spd_fc,
            feats_orig=feats,       # Features from original x
            feats_cf_bg=feats_cf,   # Features from background counterfactual x_cf
            digit_labels=y,         # Digit labels
            config_lambda_invariance=config.lambda_r, # Use lambda_r from your config
            config_mu_task_correctness=config.mu_r,   # Use mu_r from your config
            device=device
        )
        
        concept_loss_weighted = concept_loss * min(1.0, step / config.warmup_steps)


        distill_loss = mse(spd_out, teacher_out) * config.distill_coeff

        # total_loss = relative_param_match_loss + relative_schatten_loss + routing_loss + distill_loss
        # Replace your old routing_loss with concept_loss_weighted
        total_loss = (relative_param_match_loss +    # Added coefficient from config
                    relative_schatten_loss +        # Added coefficient from config
                    distill_loss + concept_loss_weighted)                                      # distill_coeff already applied
                    # concept_correlation_loss_p0) 

        proper_norm = 1.0 


        # with torch.no_grad():
        #     for layer in (spd_fc.fc1, spd_fc.fc2):
        #         A0, B0 = layer.A[0], layer.B[0]           # tensors to rescale
        #         frob = torch.norm(torch.matmul(A0, B0))    # ‖P₀‖_F

        #         target =  layer.d_in ** 0.5    # e.g. keep entries ~O(1)
        #         if frob > 0:
        #             scale = target / frob
        #             layer.A[0].mul_(scale)
        #             # or equally distribute:  A√s, B√s

        if step % config.print_freq == 0: 
            logger.info(f"Step {step} | Total Loss: {total_loss.item()} | Distill Loss: {distill_loss.item()} | Relative Param Match Loss: {relative_param_match_loss.item()} | Relative Schatten Loss: {relative_schatten_loss.item()} | Concept Loss: {concept_loss.item()}")

        if step % config.save_freq == 0:
            out_dir = getattr(config,"out_dir",None)
            if out_dir:
                Path(out_dir).mkdir(parents=True,exist_ok=True)
                savepth = Path(out_dir)/f"waterbird_spd_step{step}.pth"
                torch.save(spd_fc.state_dict(), savepth)
                logger.info(f"Saved SPD checkpoint step={step} => {savepth}")
        
        # save the best model in out directory
        if total_loss < best_loss:
            savepth = Path(out_dir)/f"waterbird_spd_best.pth"
            torch.save(spd_fc.state_dict(), savepth)
            best_loss = total_loss

        total_loss.backward()
        opt.step()

    out_dir = getattr(config,"out_dir",None)
    if out_dir:
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        finalpth = Path(out_dir)/"waterbird_spd_final.pth"
        torch.save(spd_fc.state_dict(), finalpth)
        print(f'Saved final spd to {finalpth}')
        logger.info(f"Saved final SPD => {finalpth}")
    
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform = T.Compose([
        T.ToTensor()
    ])
    
    train_subset = SpuriousMNIST(root_dir="colorized-MNIST/training", transform=train_transform, colour_prior=0.5)

    loader = DataLoader(train_subset, batch_size=128, shuffle=True)

    cfg = SPDConfig(
        seed            = 0,
        out_dir         = "spd_models_routing_loss",
        batch_size      = 64,
        lr              = 3e-4,         
        lr_schedule      = "cosine",
        steps            = 15000,     
        lr_warmup_pct   = 0.05,

        C               = 2,              
        m_fc1           = 320,            
        m_fc2           = 320,

        teacher_ckpt    = "checkpoints/best_model.pth",

        param_match_coeff           = 0.0,    
        relative_param_match_coeff  = 2.0,    
        distill_coeff               = 1.0,    

        alpha_condition     = 1.0,            
        cond_coeff          = 0.0,            

        topk                = None,              
        batch_topk          = False,
        topk_recon_coeff    = 0.0,            

        schatten_coeff      = 0.0,
        schatten_pnorm      = 0.9,
        unit_norm_matrices  = False,
        relative_schatten_coeff = 0.0,
        lp_sparsity_coeff   = 0.0,            
        pnorm               = None,
        lambda_r            = 3.00,
        mu_r                = 1.00,
        warmup_steps        = 1,
        routing_weight_start = 0.1,
        routing_weight_max = 8.0,   
        concept_corr_target_component_idx = 0,
        concept_corr_loss_coeff = 1.0,

        print_freq      = 50,
        save_freq       = 1000,
    )


    run_spd(cfg, loader, device)

