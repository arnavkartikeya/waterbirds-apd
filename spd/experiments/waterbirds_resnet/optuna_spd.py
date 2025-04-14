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
import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from wilds import get_dataset

from models import WaterbirdResNet18, SPDTwoLayerFC
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
from train_resnet import WaterbirdsSubset

###############################
# Additional SPD-style methods (same as before)
###############################

import einops

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
    first_key = next(iter(component_acts.keys()))
    out_shape = component_acts[first_key].shape[:-1]
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
    import torch.autograd as autograd
    post_names = [k.removesuffix(".hook_post") for k in post_weight_acts.keys()]
    pre_names = [k.removesuffix(".hook_pre") for k in pre_weight_acts.keys()]
    comp_names = list(component_weights.keys())
    assert set(post_names) == set(pre_names) == set(comp_names), "layer name mismatch"

    batch_prefix = model_out.shape[:-1]
    out_dim = model_out.shape[-1]
    attribution_scores = torch.zeros((*batch_prefix, C), device=model_out.device)

    component_acts = {}
    for nm in pre_names:
        pre_ = pre_weight_acts[nm + ".hook_pre"].detach()
        w_ = component_weights[nm]
        partial = einops.einsum(
            pre_, w_, "... d_in, C d_in d_out -> ... C d_out"
        )
        component_acts[nm] = partial

    for feature_idx in range(out_dim):
        grads = autograd.grad(
            model_out[..., feature_idx].sum(),
            list(post_weight_acts.values()),
            retain_graph=True,
        )
        feature_attrib = torch.zeros((*batch_prefix, C), device=model_out.device)
        for grad_val, nm_post in zip(grads, post_weight_acts.keys()):
            nm_clean = nm_post.removesuffix(".hook_post")
            feature_attrib += einops.einsum(
                grad_val, component_acts[nm_clean],
                "... d_out, ... C d_out -> ... C"
            )
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
    if attribution_type == "gradient":
        import torch.autograd as autograd
        post_names = [k.removesuffix(".hook_post") for k in post_acts]
        pre_names  = [k.removesuffix(".hook_pre") for k in pre_acts]
        comp_names = list(component_acts.keys())
        assert set(post_names) == set(pre_names) == set(comp_names), \
            f"Mismatch: {post_names}, {pre_names}, {comp_names}"
        batch_shape = teacher_out.shape[:-1]
        c_dim = model.C
        attributions = torch.zeros((*batch_shape, c_dim), device=teacher_out.device)
        out_dim = teacher_out.shape[-1]
        grad_list = autograd.grad(
            teacher_out.sum(),
            list(post_acts.values()),
            retain_graph=True
        )
        for grad_val, post_name in zip(grad_list, post_acts.keys()):
            lay_name = post_name.removesuffix(".hook_post")
            partial_contrib = einops.einsum(
                grad_val, component_acts[lay_name], "... d_out, ... C d_out -> ... C"
            )
            attributions += partial_contrib**2
        return attributions

    elif attribution_type == "activation":
        attributions = torch.zeros((input_x.shape[0], model.C), device=input_x.device)
        for layername, acts in component_acts.items():
            attributions += acts.pow(2).sum(dim=-1)
        return attributions
    else:
        raise ValueError(f"Invalid attribution_type={attribution_type}")

def calc_recon_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return ((pred - ref) ** 2).mean(dim=-1).mean()

###############################
# The training script (modified to return a validation metric)
###############################

def _calc_param_mse(
    params1,
    params2, 
    n_params,
    device,
):
    param_match_loss = torch.tensor(0.0, device=device)
    for name in params1:
        param_match_loss = param_match_loss + ((params2[name] - params1[name]) ** 2).sum(dim=(-2, -1))
    return param_match_loss / n_params

def calc_param_match_loss(
    param_names,
    target_model,
    spd_model,
    n_params,
    device,
):
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

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field

class WaterbirdSPDConfig(BaseModel):
    seed: int = 0
    batch_size: PositiveInt = 32
    steps: PositiveInt = 500
    lr: float = 1e-3
    print_freq: int = 50
    save_freq: Optional[int] = None
    out_dir: Optional[str] = None

    distill_coeff: float = 1.0
    param_match_coeff: float = 0.0

    alpha_condition: float = 1.0

    C: PositiveInt = 40
    m_fc1: PositiveInt = 16
    m_fc2: PositiveInt = 16

    lr_schedule: str = "constant"
    lr_exponential_halflife: float | None = None
    lr_warmup_pct: float = 0.0

    unit_norm_matrices: bool = False
    schatten_coeff: float | None = None
    schatten_pnorm: float | None = None
    teacher_ckpt: str = "waterbird_resnet18_best.pth"

    topk: float | None = None
    batch_topk: bool = True
    topk_recon_coeff: float | None = None
    distil_from_target: bool = True

    lp_sparsity_coeff: float | None = None
    pnorm: float | None = None

    attribution_type: str = "gradient"

def set_As_and_Bs_to_unit_norm(spd_fc: torch.nn.Module):
    with torch.no_grad():
        A1 = spd_fc.fc1.A
        B1 = spd_fc.fc1.B
        for c in range(A1.shape[0]):
            normA = A1[c].norm(p=2)
            if normA > 1e-9:
                A1[c] /= normA
            normB = B1[c].norm(p=2)
            if normB > 1e-9:
                B1[c] /= normB

        A2 = spd_fc.fc2.A
        B2 = spd_fc.fc2.B
        for c in range(A2.shape[0]):
            normA = A2[c].norm(p=2)
            if normA > 1e-9:
                A2[c] /= normA
            normB = B2[c].norm(p=2)
            if normB > 1e-9:
                B2[c] /= normB

def fix_normalized_adam_gradients(spd_fc: torch.nn.Module):
    with torch.no_grad():
        def _proj_out_scale_direction(A, B, gradA, gradB):
            A_vec = A.view(-1)
            B_vec = B.view(-1)
            gA_vec = gradA.view(-1)
            gB_vec = gradB.view(-1)
            v = torch.cat([A_vec, -B_vec], dim=0)
            v_norm_sq = v.dot(v) + 1e-12
            g = torch.cat([gA_vec, gB_vec], dim=0)
            scale_coeff = g.dot(v) / v_norm_sq
            new_g = g - scale_coeff * v
            new_gA = new_g[:A_vec.shape[0]].view_as(gradA)
            new_gB = new_g[A_vec.shape[0]:].view_as(gradB)
            gradA.copy_(new_gA)
            gradB.copy_(new_gB)

        A1 = spd_fc.fc1.A
        B1 = spd_fc.fc1.B
        if A1.requires_grad and B1.requires_grad:
            for c in range(A1.shape[0]):
                if A1.grad is not None and B1.grad is not None:
                    gradA1 = A1.grad[c]
                    gradB1 = B1.grad[c]
                    _proj_out_scale_direction(A1[c], B1[c], gradA1, gradB1)

        A2 = spd_fc.fc2.A
        B2 = spd_fc.fc2.B
        if A2.requires_grad and B2.requires_grad:
            for c in range(A2.shape[0]):
                if A2.grad is not None and B2.grad is not None:
                    gradA2 = A2.grad[c]
                    gradB2 = B2.grad[c]
                    _proj_out_scale_direction(A2[c], B2[c], gradA2, gradB2)

def calc_schatten_loss(
    As: dict[str, torch.Tensor],
    Bs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    p: float,
    n_params: int,
    device: torch.device,
) -> torch.Tensor:
    assert As.keys() == Bs.keys(), "As and Bs must have identical keys"
    batch_size = mask.shape[0]
    schatten_penalty = torch.zeros((), device=device)
    for layer_name in As.keys():
        A = As[layer_name]
        B = Bs[layer_name]
        S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")
        S_AB = S_A * S_B
        if S_AB.ndim == 2:
            S_AB = S_AB.unsqueeze(0)
        S_AB_topk = einops.einsum(S_AB, mask, "b C m, b C -> b C m")
        schatten_penalty += ((S_AB_topk + 1e-16) ** (0.5 * p)).sum()
    schatten_penalty = schatten_penalty / (n_params * batch_size)
    return schatten_penalty

def run_spd_waterbird(config, device):
    """
    Trains the SPD model and then evaluates it over the training subset to return an average loss.
    This evaluation metric serves as the objective for hyperparameter tuning.
    """
    set_seed(config.seed)
    logger.info(f"Running SPD Waterbird with manual teacher caching, config={config}")

    # 1) Teacher load
    ckpt = torch.load(config.teacher_ckpt, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt 
    teacher_model = WaterbirdResNet18(num_classes=2, hidden_dim=512)
    missing, unexpected = teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.to(device)
    teacher_model.eval()

    trunk = teacher_model.features  
    teacher_fc1 = teacher_model.fc1
    teacher_fc2 = teacher_model.fc2

    # 2) SPD final-layers
    spd_fc = SPDTwoLayerFC(
        in_features=512,
        hidden_dim=512,
        num_classes=2,
        C=config.C,
        m_fc1=config.m_fc1,
        m_fc2=config.m_fc2
    ).to(device)

    waterbird_dataset = get_dataset(dataset="waterbirds", download=False)
    dataset_size = len(waterbird_dataset)
    print(f"Total dataset size: {dataset_size}")
    
    all_indices = np.arange(dataset_size)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:2000].tolist()

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    train_subset = WaterbirdsSubset(
        waterbird_dataset, 
        indices=train_indices,
        transform=train_transform
    )
    loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)

    # 4) Optimizer & LR schedule
    opt = optim.AdamW(spd_fc.parameters(), lr=config.lr, weight_decay=0.0)
    lr_sched_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    data_iter = iter(loader)
    epoch = 0

    param_names = ["fc1", "fc2"]

    for step in tqdm(range(config.steps+1), ncols=100):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_sched_fn,
            lr_warmup_pct=config.lr_warmup_pct
        )
        for g in opt.param_groups:
            g["lr"] = step_lr

        try:
            batch_data = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            batch_data = next(data_iter)

        imgs, bird_label, meta = batch_data
        imgs = imgs.to(device)
        bird_label = bird_label.to(device)
        background_label = meta.float().to(device)

        opt.zero_grad(set_to_none=True)

        if getattr(config, "unit_norm_matrices", False):
            set_As_and_Bs_to_unit_norm(spd_fc)

        with torch.no_grad():
            feats = trunk(imgs)
            feats = feats.flatten(1)

        feats_with_grad = feats.detach().clone().requires_grad_(True)

        teacher_cache = {}
        teacher_cache["fc1.hook_pre"] = feats_with_grad
        teacher_h_pre = teacher_fc1(feats_with_grad)
        teacher_cache["fc1.hook_post"] = teacher_h_pre

        teacher_h = torch.relu(teacher_h_pre)
        teacher_cache["fc2.hook_pre"] = teacher_h

        teacher_out = teacher_fc2(teacher_h)
        teacher_cache["fc2.hook_post"] = teacher_out

        spd_fc.reset_hooks()
        cache_dict, fwd_hooks, _ = spd_fc.get_caching_hooks()
        with spd_fc.hooks(fwd_hooks, [], reset_hooks_end=True):
            spd_h_pre = spd_fc.fc1(feats)
            spd_h = torch.relu(spd_h_pre)
            spd_out = spd_fc.fc2(spd_h)

        pre_weight_acts = {}
        post_weight_acts = {}
        comp_acts = {}
        for k, v in cache_dict.items():
            if k.endswith("hook_pre"):
                pre_weight_acts[k] = v
            elif k.endswith("hook_post"):
                post_weight_acts[k] = v
            elif k.endswith("hook_component_acts"):
                comp_acts[k.removesuffix(".hook_component_acts")] = v

        teacher_pre_acts = {k: v for k, v in teacher_cache.items() if k.endswith("hook_pre")}
        teacher_post_acts = {k: v for k, v in teacher_cache.items() if k.endswith("hook_post")}

        attributions = calculate_attributions(
            model=spd_fc,
            input_x=feats,
            out=spd_out,
            teacher_out=teacher_out if getattr(config, "distil_from_target", True) else spd_out,
            pre_acts=teacher_pre_acts,
            post_acts=teacher_post_acts,
            component_acts=comp_acts,
            attribution_type=config.attribution_type
        )

        out_topk = None
        topk_mask = None
        if config.topk is not None:
            topk_attrs = attributions
            if config.distil_from_target and topk_attrs.shape[-1] > 1:
                topk_attrs = topk_attrs[..., :-1]

            topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=getattr(config, "batch_topk", True))
            if config.distil_from_target:
                last_submask = torch.ones((*topk_mask.shape[:-1], 1), dtype=torch.bool, device=device)
                topk_mask = torch.cat((topk_mask, last_submask), dim=-1)

            spd_fc.reset_hooks()
            with spd_fc.hooks(*spd_fc.get_caching_hooks()[:2], reset_hooks_end=True):
                hpre_topk = spd_fc.fc1(feats, topk_mask=topk_mask)
                h_topk = torch.relu(hpre_topk)
                out_topk = spd_fc.fc2(h_topk, topk_mask=topk_mask)

        distill_loss = mse(spd_out, teacher_out)

        param_match_loss = torch.tensor(0.0, device=device)
        if getattr(config, "param_match_coeff", 0.0) > 0:
            param_val = calc_param_match_loss(
                param_names=param_names,
                target_model=teacher_model,
                spd_model=spd_fc,
                n_params=1,
                device=device
            )
            param_match_loss = param_val.mean() * config.param_match_coeff

        topk_recon_loss = torch.tensor(0.0, device=device)
        if getattr(config, "topk_recon_coeff", None) is not None and out_topk is not None:
            topk_recon_loss = ((out_topk - teacher_out) ** 2).mean() * config.topk_recon_coeff

        lp_sparsity_loss = torch.tensor(0.0, device=device)
        if getattr(config, "lp_sparsity_coeff", None) is not None and getattr(config, "pnorm", None) is not None:
            lps = calc_lp_sparsity_loss(spd_out, attributions, config.pnorm)
            lp_sparsity_loss = lps.sum(dim=-1).mean(dim=0) * config.lp_sparsity_coeff

        A0 = spd_fc.fc1.A[0]
        B0 = spd_fc.fc1.B[0]
        sc0_feats = feats @ A0
        sc0_logits = sc0_feats @ B0
        h0 = sc0_logits[:, 0]
        condition_loss = bce(h0, background_label) * getattr(config, "alpha_condition", 1.0)

        total_loss = distill_loss + param_match_loss + topk_recon_loss + lp_sparsity_loss + condition_loss

        total_loss.backward()
        opt.step()

        if step % config.print_freq == 0:
            logger.info(
                f"Step {step} | total={total_loss.item():.4f} "
                f"distill={distill_loss.item():.4f}, param={param_match_loss.item():.4f}, "
                f"topk={topk_recon_loss.item():.4f}, lp={lp_sparsity_loss.item():.4f}, cond={condition_loss.item():.4f}, lr={step_lr:.2e}"
            )

        if getattr(config, "save_freq", None) and step > 0 and step % config.save_freq == 0:
            out_dir = getattr(config, "out_dir", None)
            if out_dir:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                savepth = Path(out_dir) / f"waterbird_spd_step{step}.pth"
                torch.save(spd_fc.state_dict(), savepth)
                logger.info(f"Saved SPD checkpoint step={step} => {savepth}")

    if getattr(config, "out_dir", None):
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        finalpth = Path(config.out_dir) / "waterbird_spd_final.pth"
        torch.save(spd_fc.state_dict(), finalpth)
        logger.info(f"Saved final SPD => {finalpth}")

    # --- Evaluation Phase ---
    spd_fc.eval()
    total_eval_loss = 0.0
    count = 0
    # We reuse the training loader (or you could use a dedicated validation set)
    for batch in loader:
        imgs, bird_label, meta = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = trunk(imgs)
            feats = feats.flatten(1)
            teacher_h_pre = teacher_fc1(feats)
            teacher_h = torch.relu(teacher_h_pre)
            teacher_out = teacher_fc2(teacher_h)
            spd_h_pre = spd_fc.fc1(feats)
            spd_h = torch.relu(spd_h_pre)
            spd_out = spd_fc.fc2(spd_h)
            loss = mse(spd_out, teacher_out)
        total_eval_loss += loss.item()
        count += 1
    avg_eval_loss = total_eval_loss / count if count > 0 else float('inf')
    logger.info(f"Validation loss: {avg_eval_loss:.4f}")
    return avg_eval_loss  # This evaluation metric is returned for hyperparameter optimization

###############################
# Optuna integration in __main__
###############################

if __name__ == "__main__":
    import optuna

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial):
        # Define the hyperparameter search space
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        steps = trial.suggest_int('steps', 500, 1500)
        distill_coeff = trial.suggest_float('distill_coeff', 0.5, 2.0)
        param_match_coeff = trial.suggest_float('param_match_coeff', 0.0, 2.0)
        topk = trial.suggest_float('topk', 1.0, 5.0)
        lp_sparsity_coeff = trial.suggest_loguniform('lp_sparsity_coeff', 1e-3, 1e-1)

        cfg = WaterbirdSPDConfig(
            batch_size=batch_size,
            steps=steps,
            lr=lr,
            print_freq=50,
            save_freq=None,         # Disable saving during hyperparameter trials
            out_dir=None,           # Optionally disable output saving
            seed=trial.number,
            distill_coeff=distill_coeff,
            param_match_coeff=param_match_coeff,
            alpha_condition=1.0,
            C=40,
            m_fc1=16,
            m_fc2=16,
            lp_sparsity_coeff=lp_sparsity_coeff,
            pnorm=2.0,
            topk=topk,
            batch_topk=True,
            topk_recon_coeff=0.1,
            teacher_ckpt="checkpoints/waterbird_resnet18_best.pth",
            attribution_type="gradient",
            lr_schedule="constant",
        )

        final_loss = run_spd_waterbird(cfg, device)
        return final_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(trial.params)
