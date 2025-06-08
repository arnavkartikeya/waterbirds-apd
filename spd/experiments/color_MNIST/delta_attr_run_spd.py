"""
File: spd/experiments/waterbird/run_spd.py

Trains SPD final layers to match a pretrained ResNet's final 2 FC layers for Waterbirds,
with:
1) Distillation from the teacher model's final FC output
2) Conditioning on subcomponent #0 for background detection
3) Top-K subnetwork selection (topk) + topk_recon
4) LP sparsity penalty

Heavily modeled after the TMS "optimize" code with attributions, etc.
"""

import math
import random
import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import yaml
from spd.utils import calc_ablation_attributions

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

###############################
# Additional SPD-style methods
###############################

import einops

# ─── add at top of file (or wherever you keep helpers) ──────────────────────────
def make_cf(imgs: torch.Tensor) -> torch.Tensor:
    """
    Colour‑MNIST counter‑factual: swap the pure‑red and pure‑green background
    while keeping the white digit pixels unchanged.

    imgs : (B,3,28,28) – RGB in [0,1]
    returns a cloned tensor with R and G channels swapped
    """
    cf = imgs.clone()
    cf[:, 0], cf[:, 1] = imgs[:, 1], imgs[:, 0]        # swap R↔G
    return cf

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
# ────────────────────────────────────────────────────────────────────────────────


# These are the SPD/TMS-like helper methods – topk, Lp-sparsity, etc.
def calc_topk_mask(attribution_scores: torch.Tensor, topk: float, batch_topk: bool) -> torch.Tensor:
    batch_size = attribution_scores.shape[0]
    k = int(topk * batch_size) if batch_topk else int(topk)
    if batch_topk:
        reshaped = einops.rearrange(attribution_scores, "b ... c -> ... (b c)")
        topk_indices = torch.topk(reshaped, k, dim=-1).indices
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        mask = einops.rearrange(mask, "... (b c) -> b ... c", b=batch_size)
        return mask
    else:
        topk_indices = attribution_scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(attribution_scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        return mask

def calc_lp_sparsity_loss(
    out: torch.Tensor,
    attributions: torch.Tensor,
    step_pnorm: float,
) -> torch.Tensor:
    d_model_out = out.shape[-1]
    scaled_attrib = attributions / d_model_out
    return (scaled_attrib.abs() + 1e-16) ** (0.5 * step_pnorm)

@torch.inference_mode()
def calc_activation_attributions(
    component_acts: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Example "activation" approach: sum of L2 over the subcomponent_acts
    shape: (batch, C) or (batch, n_instances, C).
    """
    # Just sum up squares of each subcomponent
    # e.g. each entry in component_acts is (batch, C, d_out)
    # sum over d_out dimension
    first_key = next(iter(component_acts.keys()))
    out_shape = component_acts[first_key].shape[:-1]  # strip d_out
    attributions = torch.zeros(out_shape, device=component_acts[first_key].device)
    for val in component_acts.values():
        attributions += val.pow(2).sum(dim=-1)
    return attributions


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

        batch_shape = teacher_out.shape[:-1]
        c_dim = model.C  # subcomponent dim
        attributions = torch.zeros((*batch_shape, c_dim), device=teacher_out.device)

        out_dim = teacher_out.shape[-1]
        grad_list = autograd.grad(
            teacher_out.sum(),  # sum => scalar
            list(post_acts.values()),
            retain_graph=True
        )
        for grad_val, post_name in zip(grad_list, post_acts.keys()):
            lay_name = post_name.removesuffix(".hook_post")
            partial_contrib = einops.einsum(
                grad_val, component_acts[lay_name], "... d_out, ... C d_out -> ... C"
            )
            attributions += partial_contrib # shoudl be squared but lets see what happens
        return attributions
    elif attribution_type == "ablation":
        attributions = calc_ablation_attributions(spd_model=model, batch=input_x, out=out)
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



def calc_recon_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """MSE across batch, shape e.g. (batch, #classes) => scalar"""
    return ((pred - ref) ** 2).mean(dim=-1).mean()


###############################
# The actual training script
###############################

def _calc_param_mse(
    params1,
    params2, 
    n_params,
    device,
):
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    param_match_loss = torch.tensor(0.0, device=device)
    for name in params1:
        param_match_loss = param_match_loss + ((params2[name] - params1[name]) ** 2).sum(
            dim=(-2, -1)
        )
    return param_match_loss / n_params

def _calc_relative_param_match(
    params1,
    params2,
    device,
):
    """Calculate the relative parameter matching loss between params1 and params2.
    
    Args:
        params1: The first set of parameters (target)
        params2: The second set of parameters (component sum)
        device: The device to use for calculations
    """
    total_norm_diff = torch.tensor(0.0, device=device)
    total_norm_target = torch.tensor(0.0, device=device)
    
    for name in params1:
        # Calculate component sum for this layer
        component_sum = params2[name]
        target = params1[name]
        
        # Calculate norms
        norm_diff = torch.norm(target - component_sum)
        norm_target = torch.norm(target)
        
        # Add to totals
        total_norm_diff += norm_diff
        total_norm_target += norm_target
    
    # Return relative difference
    return total_norm_diff / (total_norm_target + 1e-8)

def calc_param_match_loss(
    param_names,
    target_model,
    spd_model,
    n_params,
    device,
):
    """Calculate the MSE between the target model weights and the SPD model weights.

    Args:
        param_names: The names of the parameters to be matched.
        target_model: The target model to match.
        spd_model: The SPD model to match.
        n_params: The number of parameters in the model. Used for normalization.
        device: The device to use for calculations.
    """
    target_params = {}
    spd_params = {}
    for param_name in param_names:
        target_params[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params[param_name] = get_nested_module_attr(spd_model, param_name + ".weight").transpose(-1, -2)
    return _calc_param_mse(
        params1=target_params,
        params2=spd_params,
        n_params=n_params,
        device=device,
    )

def calc_relative_param_match_loss(
    param_names,
    target_model,
    spd_model,
    device,
):
    """Calculate the relative parameter matching loss between target model and SPD model.
    
    Args:
        param_names: The names of the parameters to be matched.
        target_model: The target model to match.
        spd_model: The SPD model to match.
        device: The device to use for calculations.
    """
    target_params = {}
    spd_params = {}
    for param_name in param_names:
        target_params[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params[param_name] = get_nested_module_attr(spd_model, param_name + ".weight").transpose(-1, -2)
    return _calc_relative_param_match(
        params1=target_params,
        params2=spd_params,
        device=device,
    )


from pydantic import BaseModel, PositiveInt, PositiveFloat, Field


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

    # SPD subcomponent config
    C: PositiveInt = 40
    m_fc1: PositiveInt = 16
    m_fc2: PositiveInt = 16

    # LR schedule
    lr_schedule: str = "constant"  # or "linear", "cosine", "exponential"
    lr_exponential_halflife: float | None = None
    lr_warmup_pct: float = 0.0

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

    # attribution type
    attribution_type: str = "gradient"  # or "activation"


def save_config_to_yaml(config: SPDConfig, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config.dict(), f)

def load_config_from_yaml(path: str) -> SPDConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return SPDConfig(**data)



def set_As_and_Bs_to_unit_norm(spd_fc: torch.nn.Module):
    """
    In-place normalization so that each (A_c, B_c) has ||A_c||_F = 1 and/or ||B_c||_F = 1,
    preventing unbounded scale growth in factorized weights.
    """
    with torch.no_grad():
        # FC1
        A1 = spd_fc.fc1.A  # shape [C, d_in, m_fc1]
        B1 = spd_fc.fc1.B  # shape [C, m_fc1, hidden_dim]
        # Normalize each subcomponent's A and B
        for c in range(A1.shape[0]):
            normA = A1[c].norm(p=2)
            if normA > 1e-9:
                A1[c] /= normA
            normB = B1[c].norm(p=2)
            if normB > 1e-9:
                B1[c] /= normB

        # FC2 (same logic; if you only want to do it for one layer, remove these lines)
        A2 = spd_fc.fc2.A  # shape [C, hidden_dim, m_fc2]
        B2 = spd_fc.fc2.B  # shape [C, m_fc2, num_classes]
        for c in range(A2.shape[0]):
            normA = A2[c].norm(p=2)
            if normA > 1e-9:
                A2[c] /= normA
            normB = B2[c].norm(p=2)
            if normB > 1e-9:
                B2[c] /= normB


def fix_normalized_adam_gradients(spd_fc: torch.nn.Module):
    """
    Removes the gradient component corresponding to pure scale changes in each factor pair (A_c, B_c).
    In rank factorization, scaling one factor by alpha and dividing the other by alpha is a no-op.
    This prevents Adam from chasing that redundant direction.
    """
    with torch.no_grad():
        # For each layer in SPD, do an orthogonal projection of grads
        # that removes the direction that changes the norm scale of (A_c, B_c).

        def _proj_out_scale_direction(A, B, gradA, gradB):
            """
            If d d/d(alpha) [A*alpha, B/alpha] = 0 means no scale changes:
            We'll find the derivative direction that changes scale
            and zero it out from (gradA, gradB).
            """
            # Flatten
            A_vec = A.view(-1)
            B_vec = B.view(-1)
            gA_vec = gradA.view(-1)
            gB_vec = gradB.view(-1)

            # The "scale" direction is something like: d/d(alpha) [A*alpha, B/alpha],
            # which at alpha=1 is (A, -B).
            # So the direction vector is v = [A, -B].
            # We compute the component of (gA, gB) in that direction and remove it.
            v = torch.cat([A_vec, -B_vec], dim=0)
            v_norm_sq = v.dot(v) + 1e-12

            g = torch.cat([gA_vec, gB_vec], dim=0)
            scale_coeff = g.dot(v) / v_norm_sq  # how much of g is in direction of v

            # new_g = g - scale_coeff * v
            new_g = g - scale_coeff * v

            # put back
            new_gA = new_g[:A_vec.shape[0]].view_as(gradA)
            new_gB = new_g[A_vec.shape[0]:].view_as(gradB)

            gradA.copy_(new_gA)
            gradB.copy_(new_gB)

        # FC1
        A1 = spd_fc.fc1.A
        B1 = spd_fc.fc1.B
        if A1.requires_grad and B1.requires_grad:
            for c in range(A1.shape[0]):
                if A1.grad is not None and B1.grad is not None:
                    gradA1 = A1.grad[c]
                    gradB1 = B1.grad[c]
                    _proj_out_scale_direction(A1[c], B1[c], gradA1, gradB1)

        # FC2
        A2 = spd_fc.fc2.A
        B2 = spd_fc.fc2.B
        if A2.requires_grad and B2.requires_grad:
            for c in range(A2.shape[0]):
                if A2.grad is not None and B2.grad is not None:
                    gradA2 = A2.grad[c]
                    gradB2 = B2.grad[c]
                    _proj_out_scale_direction(A2[c], B2[c], gradA2, gradB2)

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



def calc_schatten_loss(
    As: dict[str, torch.Tensor],
    Bs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    p: float,
    n_params: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Approximate rank penalty: sum_{c} ||P_c||_p^p, where P_c = A_c * B_c^T,
    implemented in a factorized form. 
    We multiply by a topk or attributions-based mask if desired (like in TMS).
    """
    # We assume all layers in As and Bs have the same shape for dimension 0 (the "C" dimension).
    # If you use topk_mask, shape should be (batch, C). We'll average over batch below.

    assert As.keys() == Bs.keys(), "As and Bs must have identical keys"
    batch_size = mask.shape[0]

    schatten_penalty = torch.zeros((), device=device)  # scalar
    for layer_name in As.keys():
        # A: shape [C, d_in, m], B: shape [C, m, d_out]
        # or possibly [batch, C, ...] if you handle multi-instance
        A = As[layer_name]
        B = Bs[layer_name]

        # S_A = sum_{i,j} A^2 over d_in
        # S_B = sum_{k,l} B^2 over d_out
        # We'll do something like TMS: 
        #    S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        #    S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")
        # Then multiply S_AB = S_A * S_B, apply the topk mask, etc.

        S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")
        S_AB = S_A * S_B  # shape [batch..., C, m] or [C, m] if no batch dimension

        # Now apply mask along the "C" dimension
        # shape of mask is [batch, C]. We broadcast over the "m" dimension.
        # We'll do an einsum:
        # S_AB_topk = einops.einsum(S_AB, mask, "... C m, batch C -> batch ... C m")
        # Then sum and do the p/2 exponent.
        if S_AB.ndim == 2:
            # no extra batch dimension on the parameter side => broadcast
            # reshape so S_AB => (1, C, m) for broadcasting
            S_AB = S_AB.unsqueeze(0)  # shape (1, C, m)

        S_AB_topk = einops.einsum(S_AB, mask, "b C m, b C -> b C m")
        # Now apply ( +1e-16 )^(0.5 * p)
        schatten_penalty += ((S_AB_topk + 1e-16) ** (0.5 * p)).sum()

    # normalizations
    # 1) divide by number of parameters n_params
    # 2) divide by batch_size
    schatten_penalty = schatten_penalty / (n_params * batch_size)
    return schatten_penalty


def run_spd_waterbird(config, device):
    """
    Incorporates your 7-step snippet for teacher manual caching + SPD hooking,
    plus top-k, LP-sparsity, condition subcomponent #0, etc.
    """
    set_seed(config.seed)
    logger.info(f"Running SPD Waterbird with manual teacher caching, config={config}")

    # 1) Teacher load
    ckpt = torch.load(config.teacher_ckpt, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt 
    teacher_model = ColorMNISTConvNetGAP(num_classes=10, hidden_dim=128)
    # teacher_model.load_state_dict(torch.load(config.teacher_ckpt, map_location="cpu"))
    missing, unexpected = teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.to(device)
    teacher_model.eval()

    trunk = teacher_model.conv  # up to conv part
    teacher_fc1 = teacher_model.fc1
    teacher_fc2 = teacher_model.fc2

    # 2) SPD final-layers
    spd_fc = SPDTwoLayerFC(
        in_features=28*28,
        hidden_dim=128,
        num_classes=10,
        C=config.C,
        m_fc1=config.m_fc1,
        m_fc2=config.m_fc2
    ).to(device)

    # all_indices = np.arange(dataset_size)
    # np.random.shuffle(all_indices)
    # train_indices = all_indices[:2000].tolist()

    train_transform = T.Compose([
        T.ToTensor()
    ])
    
    train_subset = SpuriousMNIST(root_dir="colorized-MNIST/training", transform=train_transform)

    loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)

    # # 3) Data
    # waterbird_ds = get_dataset("waterbirds", download=False)
    # tform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    # def collate_fn(idx):
    #     x,y,m = waterbird_ds[idx]
    #     x = tform(x)
    #     return x,y,m

    # indices = list(range(2000))  # subset
    # loader = DataLoader(indices, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # 4) Optim + LR schedule
    opt = optim.AdamW(spd_fc.parameters(), lr=config.lr, weight_decay=0.0)
    lr_sched_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    # Loss objects
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    data_iter = iter(loader)
    epoch = 0

    # If you want param matching:
    param_names = ["fc1","fc2"]
    n_params = 0
    for param_name in param_names:
        n_params += get_nested_module_attr(teacher_model, param_name + ".weight").numel()

    import math

    for step in tqdm(range(config.steps+1), ncols=100):
        # set LR
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_sched_fn,
            lr_warmup_pct=config.lr_warmup_pct
        )
        for g in opt.param_groups:
            g["lr"] = step_lr

        # fetch batch
        try:
            batch_data = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            batch_data = next(data_iter)

        imgs, digit_label, background_label = batch_data
        imgs = imgs.to(device)
        digit_label = digit_label.to(device)
        background_label = background_label.to(device)
        imgs_cf = make_cf(imgs)
        # imgs, bird_label, meta = batch_data
        # imgs = imgs.to(device)
        # bird_label = bird_label.to(device)
        # background_label = meta.float().to(device)  # shape [B], 0 or 1

        opt.zero_grad(set_to_none=True)

        # # if we do "unit_norm_matrices"
        # if getattr(config, "unit_norm_matrices", False):
        #     set_As_and_Bs_to_unit_norm(spd_fc)  # you can define or import your own

        #=========================
        # (1) trunk features
        #=========================
        with torch.no_grad():
            feats_cf = trunk(imgs_cf).mean(1).flatten(1) 

        with torch.no_grad():
            feats = trunk(imgs)
            feats = feats.mean(dim=1)
            feats = feats.flatten(1)  # [B, 512]

        #=========================
        # (2) feats_with_grad
        #=========================
        feats_with_grad = feats.detach().clone().requires_grad_(True)

        #=========================
        # (3) teacher forward pass manually, storing in a teacher_cache
        #=========================
        feats_cf_grad = feats_cf.detach().clone().requires_grad_(True)
        t_cache_cf = {}

        t_cache_cf["fc1.hook_pre"]  = feats_cf_grad
        h1_cf = teacher_fc1(feats_cf_grad)
        t_cache_cf["fc1.hook_post"] = h1_cf
        h_relu_cf = torch.relu(h1_cf)
        t_cache_cf["fc2.hook_pre"]  = h_relu_cf
        teacher_out_cf              = teacher_fc2(h_relu_cf)
        t_cache_cf["fc2.hook_post"] = teacher_out_cf

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

        #=========================
        # (7) calculate attributions
        #=========================
        attrib_cf = calculate_attributions(
            model          = spd_fc,
            input_x        = feats_cf,
            out            = spd_out_cf,
            teacher_out    = teacher_out_cf if config.distil_from_target else spd_out_cf,
            pre_acts       = {k:v for k,v in t_cache_cf.items() if k.endswith("hook_pre")},
            post_acts      = {k:v for k,v in t_cache_cf.items() if k.endswith("hook_post")},
            component_acts = comp_cf,
            attribution_type = config.attribution_type,
        )

        attributions = calculate_attributions(
            model=spd_fc,
            input_x=feats,
            out=spd_out,
            teacher_out=teacher_out if getattr(config,"distil_from_target",True) else spd_out,
            pre_acts=teacher_pre_acts,
            post_acts=teacher_post_acts,
            component_acts=comp_acts,
            attribution_type=config.attribution_type
        )

        delta_attrib = attributions - attrib_cf 

        # optionally do top-k
        out_topk = None
        topk_mask = None
        if config.topk is not None:
            topk_attrs = attributions
            if config.distil_from_target and topk_attrs.shape[-1]>1:
                # remove last subcomp if you want
                topk_attrs = topk_attrs[..., :-1]

            topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=getattr(config,"batch_topk",True))
            if config.distil_from_target:
                # add final subcomp back
                last_submask = torch.ones((*topk_mask.shape[:-1],1), dtype=torch.bool, device=device)
                topk_mask = torch.cat((topk_mask, last_submask), dim=-1)

            # forward pass with topk
            spd_fc.reset_hooks()
            with spd_fc.hooks(*spd_fc.get_caching_hooks()[:2], reset_hooks_end=True):
                hpre_topk = spd_fc.fc1(feats, topk_mask=topk_mask)
                h_topk = torch.relu(hpre_topk)
                out_topk = spd_fc.fc2(h_topk, topk_mask=topk_mask)

        # Distill loss
        distill_loss = mse(spd_out, teacher_out) * config.distill_coeff 

        # param match
        param_match_loss = torch.tensor(0.0, device=device)
        if getattr(config,"param_match_coeff",0.0)>0:
            # from spd.run_spd import calc_param_match_loss
            param_val = calc_param_match_loss(
                param_names=param_names,
                target_model=teacher_model,
                spd_model=spd_fc,
                n_params=n_params,
                device=device
            )
            param_match_loss = param_val.mean()*config.param_match_coeff

        # relative param match
        relative_param_match_loss = torch.tensor(0.0, device=device)
        if getattr(config,"relative_param_match_coeff",0.0)>0:
            relative_param_val = calc_relative_param_match_loss(
                param_names=param_names,
                target_model=teacher_model,
                spd_model=spd_fc,
                device=device
            )
            relative_param_match_loss = relative_param_val * config.relative_param_match_coeff

        # topk recon
        topk_recon_loss = torch.tensor(0.0, device=device)
        if getattr(config,"topk_recon_coeff",None) is not None and out_topk is not None:
            # MSE
            topk_recon_loss = ((out_topk - teacher_out)**2).mean() * config.topk_recon_coeff

        # lp_sparsity
        lp_sparsity_loss = torch.tensor(0.0, device=device)
        if getattr(config,"lp_sparsity_coeff",None) is not None and getattr(config,"pnorm",None) is not None:
            lps = calc_lp_sparsity_loss(spd_out, attributions, config.pnorm)
            # shape (batch, C)
            lp_sparsity_loss = lps.sum(dim=-1).mean(dim=0)*config.lp_sparsity_coeff

        # ─── add *after* lp_sparsity_loss is computed ──────────────────────────────────
        # (i.e. just before you build total_loss)

        # ─── ❶ Schatten‑rank loss ──────────────────────────────────────────────────────
        schatten_loss = torch.tensor(0.0, device=device)
        if (getattr(config, "schatten_coeff", None) is not None 
                and getattr(config, "schatten_pnorm", None) is not None
                and config.schatten_coeff != 0):
            
            # Build per‑layer dicts of A and B factors
            As = {
                "fc1": spd_fc.fc1.A[0:1],   # shape [C, d_in, m_fc1]
                "fc2": spd_fc.fc2.A[0:1],   # shape [C, hidden_dim, m_fc2]
            }
            Bs = {
                "fc1": spd_fc.fc1.B[0:1],   # shape [C, m_fc1, hidden_dim]
                "fc2": spd_fc.fc2.B[0:1],   # shape [C, m_fc2, num_classes]
            }

            # A (batch,C) mask must be supplied; if you're not using top‑k,
            # just make it all‑ones.
            if topk_mask is None:
                mask = torch.ones((imgs.size(0), 1), device=device, dtype=torch.bool)
            else:
                mask = topk_mask          # already shape (batch, C)

            # Number of parameters in the factorised layers (for normalisation)
            # n_params = sum(A.numel() + B.numel() for A, B in zip(As.values(), Bs.values()))

            schatten_loss = calc_schatten_loss(
                As      = As,
                Bs      = Bs,
                mask    = mask,
                p       = config.schatten_pnorm,
                n_params= n_params,
                device  = device,
            ) * config.schatten_coeff
        # ────────────────────────────────────────────────────────────────────────────────
        relative_schatten_loss = torch.tensor(0.0, device=device)
        #mask should make it so that the relative schatten loss is only for component 0
        if getattr(config, "relative_schatten_coeff", None) is not None:
            As = {
                "fc1": spd_fc.fc1.A,   # shape [C, d_in, m_fc1]
                "fc2": spd_fc.fc2.A,   # shape [C, hidden_dim, m_fc2]
            }
            Bs = {
                "fc1": spd_fc.fc1.B,   # shape [C, m_fc1, hidden_dim]
                "fc2": spd_fc.fc2.B,   # shape [C, m_fc2, num_classes]
            }
            mask = torch.ones((imgs.size(0), 1), device=device, dtype=torch.bool)
            relative_schatten_loss = calc_relative_schatten_loss(
                As=As,
                Bs=Bs,
                mask=mask,
                p=config.schatten_pnorm,
                device=device
            ) * config.relative_schatten_coeff


        # condition subcomp #0
        # subcomp #0 => spd_fc.fc1.A[0], spd_fc.fc1.B[0]
        attr0_logit = attributions[:, 0] * config.alpha_condition 
        condition_loss = bce(attr0_logit, background_label.float()) * getattr(config, "cond_coeff", None)

        # routing_loss = (  config.lambda_r * delta_attrib[:,0].abs().mean()
        #                 - config.mu_r     * delta_attrib[:,1].abs().mean() )
        routing_loss = delta_attrib[:,1].abs().mean()/delta_attrib[:,0].abs().mean()

        total_loss = 0.0
        total_loss = (
            distill_loss 
            + param_match_loss
            + relative_param_match_loss
            + topk_recon_loss
            + lp_sparsity_loss
            + condition_loss
            + schatten_loss
            + relative_schatten_loss
            + routing_loss 
        )

        total_loss.backward()
        opt.step()
        # set_As_and_Bs_to_unit_norm(spd_fc)

        if step % config.print_freq==0:
            logger.info(
                f"Step {step} | total={total_loss.item():.4f} "
                f"distill={distill_loss.item():.4f}, param={param_match_loss.item():.4f}, "
                f"rel_param={relative_param_match_loss.item():.4f}, topk={topk_recon_loss.item():.4f}, "
                f"condition={condition_loss.item():.4f}, "
                f"relative_schatten={relative_schatten_loss.item():.4f}, "
                f"routing={routing_loss.item():.4f}, "
                f"schatten={schatten_loss.item():.4f}, lr={step_lr:.2e}"
            )

        if getattr(config,"save_freq",None) and step>0 and step%config.save_freq==0:
            out_dir = getattr(config,"out_dir",None)
            if out_dir:
                Path(out_dir).mkdir(parents=True,exist_ok=True)
                savepth = Path(out_dir)/f"waterbird_spd_step{step}.pth"
                torch.save(spd_fc.state_dict(), savepth)
                logger.info(f"Saved SPD checkpoint step={step} => {savepth}")

    # final
    out_dir = getattr(config,"out_dir",None)
    if out_dir:
        Path(out_dir).mkdir(parents=True,exist_ok=True)
        finalpth = Path(out_dir)/"waterbird_spd_final.pth"
        torch.save(spd_fc.state_dict(), finalpth)
        print(f'Saved final spd to {finalpth}')
        logger.info(f"Saved final SPD => {finalpth}")

        


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # getting from best optuna trial 
    cfg = SPDConfig(
        # ─── bookkeeping ──────────────────────────────────────────────────────────────
        seed            = 0,
        out_dir         = "spd_models",
        steps           = 20000,            # ≈3–4 epochs on full train set
        batch_size      = 64,
        lr              = 1e-3,            # decays via cosine scheduler below
        lr_schedule     = "linear",
        lr_warmup_pct   = 0.05,

        # ─── decomposition geometry ───────────────────────────────────────────────────
        C               = 16,               # exactly two components
        m_fc1           = 8,             # give factors enough rank to match weights
        m_fc2           = 8,

        # ─── teacher model ------------------------------------------------------------
        teacher_ckpt    = "checkpoints/best_model.pth",

        # ─── loss coefficients --------------------------------------------------------
        # 1️⃣  faithfulness
        param_match_coeff           = 0.0,    # disable absolute parameter matching
        relative_param_match_coeff  = 5.0,    # use relative parameter matching instead
        distill_coeff               = 1.0,    # logits of sum(comp0+comp1) ≈ teacher

        # 2️⃣  background channel (component 0)
        alpha_condition     = 1.0,            # scale attribution → logit
        cond_coeff          = 0.0,            # BCE loss weight on bg component

        # 3️⃣  “self-reconstruction” of logits w/out comp0  (forces debias split)
        topk                = None,              # select **only comp1** in sparse pass
        batch_topk          = False,
        topk_recon_coeff    = 0.0,            # MSE( logits(masked) , logits(full) )

        # 4️⃣  capacity / simplicity regularisers
        schatten_coeff      = 0.0,
        schatten_pnorm      = 0.9,
        unit_norm_matrices  = False,
        relative_schatten_coeff = 1.0,
        # ─── losses kept off for clarity ---------------------------------------------
        lp_sparsity_coeff   = 0.0,            # not needed
        pnorm               = None,
        lambda_r            = 0.001,
        mu_r                = 0.001,

        # ─── checkpointing / logging --------------------------------------------------
        print_freq      = 50,
        save_freq       = 1000,
    )


    run_spd_waterbird(cfg, device)
