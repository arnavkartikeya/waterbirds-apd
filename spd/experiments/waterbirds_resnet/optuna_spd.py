"""
File: spd/experiments/waterbird/run_spd.py

Trains SPD final layers to match a pretrained ResNet's final 2 FC layers for Waterbirds,
with:
1) Distillation from the teacher model's final FC output
2) Conditioning on subcomponent #0 for background detection
3) Top‑K subnetwork selection (topk) + topk_recon
4) LP sparsity penalty

Heavily modeled after the TMS "optimize" code with attributions, etc.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
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

import einops
import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt, Field

from models import WaterbirdResNet18, SPDTwoLayerFC
from spd.hooks import HookedRootModule
from spd.log import logger
from spd.models.base import SPDModel
from spd.module_utils import (
    collect_nested_module_attrs,
    get_nested_module_attr,
)
from spd.run_spd import get_lr_schedule_fn, get_lr_with_warmup
from spd.types import Probability
from spd.utils import set_seed
from train_resnet import WaterbirdsSubset

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
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
        pre_names = [k.removesuffix(".hook_pre") for k in pre_acts]
        comp_names = list(component_acts.keys())
        assert set(post_names) == set(pre_names) == set(comp_names), \
            f"Mismatch: {post_names}, {pre_names}, {comp_names}"

        batch_shape = teacher_out.shape[:-1]
        c_dim = model.C
        attributions = torch.zeros((*batch_shape, c_dim), device=teacher_out.device)

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
            attributions += partial_contrib ** 2
        return attributions

    elif attribution_type == "activation":
        attributions = torch.zeros((input_x.shape[0], model.C), device=input_x.device)
        for acts in component_acts.values():
            attributions += acts.pow(2).sum(dim=-1)
        return attributions
    else:
        raise ValueError(f"Invalid attribution_type={attribution_type}")


def calc_recon_mse(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return ((pred - ref) ** 2).mean(dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
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
    cond_coeff: float = 1.0
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


# ──────────────────────────────────────────────────────────────────────────────
# Training routine
# ──────────────────────────────────────────────────────────────────────────────
def run_spd_waterbird(config: WaterbirdSPDConfig, device: torch.device):
    """
    Trains the SPD model and returns:
        • total loss from the **last** training step (used as the Optuna objective)
        • a dict with the last values of every individual loss term
        • the trained SPD model
    """
    set_seed(config.seed)
    logger.info(f"Running SPD Waterbird | config={config}")

    # 1) Teacher
    ckpt = torch.load(config.teacher_ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    teacher_model = WaterbirdResNet18(num_classes=2, hidden_dim=512)
    teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.to(device).eval()

    trunk = teacher_model.features
    teacher_fc1, teacher_fc2 = teacher_model.fc1, teacher_model.fc2

    # 2) SPD head
    spd_fc = SPDTwoLayerFC(
        in_features=512,
        hidden_dim=512,
        num_classes=2,
        C=config.C,
        m_fc1=config.m_fc1,
        m_fc2=config.m_fc2,
    ).to(device)

    # 3) Dataset (tiny subset for speed)
    waterbird_dataset = get_dataset("waterbirds", download=False)
    all_idx = np.random.permutation(len(waterbird_dataset))
    train_idx = all_idx[:2000].tolist()

    train_subset = WaterbirdsSubset(
        waterbird_dataset,
        indices=train_idx,
        transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]),
    )
    loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)

    # 4) Optimiser / schedule
    opt = optim.AdamW(spd_fc.parameters(), lr=config.lr, weight_decay=0.0)
    lr_sched_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    data_iter, epoch = iter(loader), 0
    param_names = ["fc1", "fc2"]

    last_losses: dict[str, float] | None = None

    for step in tqdm(range(config.steps + 1), ncols=100):
        # LR schedule
        step_lr = get_lr_with_warmup(
            step, config.steps, config.lr, lr_sched_fn, config.lr_warmup_pct
        )
        for g in opt.param_groups:
            g["lr"] = step_lr

        # Batch
        try:
            imgs, bird_label, meta = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            imgs, bird_label, meta = next(data_iter)

        imgs = imgs.to(device)
        background_label = meta.float().to(device)

        opt.zero_grad(set_to_none=True)

        # Teacher forward
        with torch.no_grad():
            feats = trunk(imgs).flatten(1)

        feats_g = feats.detach().clone().requires_grad_(True)
        teacher_cache = {
            "fc1.hook_pre": feats_g,
            "fc1.hook_post": teacher_fc1(feats_g),
        }
        teacher_h = torch.relu(teacher_cache["fc1.hook_post"])
        teacher_cache["fc2.hook_pre"] = teacher_h
        teacher_out = teacher_fc2(teacher_h)
        teacher_cache["fc2.hook_post"] = teacher_out

        # SPD forward (+ hook capture)
        spd_fc.reset_hooks()
        cache_dict, fwd_hooks, _ = spd_fc.get_caching_hooks()
        with spd_fc.hooks(fwd_hooks, [], reset_hooks_end=True):
            spd_h_pre = spd_fc.fc1(feats)
            spd_out = spd_fc.fc2(torch.relu(spd_h_pre))

        pre_weight_acts = {k: v for k, v in cache_dict.items() if k.endswith("hook_pre")}
        post_weight_acts = {k: v for k, v in cache_dict.items() if k.endswith("hook_post")}
        comp_acts = {
            k.removesuffix(".hook_component_acts"): v
            for k, v in cache_dict.items()
            if k.endswith("hook_component_acts")
        }

        teacher_pre = {k: v for k, v in teacher_cache.items() if k.endswith("hook_pre")}
        teacher_post = {k: v for k, v in teacher_cache.items() if k.endswith("hook_post")}

        attributions = calculate_attributions(
            model=spd_fc,
            input_x=feats,
            out=spd_out,
            teacher_out=teacher_out
            if config.distil_from_target
            else spd_out,
            pre_acts=teacher_pre,
            post_acts=teacher_post,
            component_acts=comp_acts,
            attribution_type=config.attribution_type,
        )

        # Optional top‑k mask / recon
        out_topk = topk_mask = None
        if config.topk is not None:
            topk_attrs = attributions[..., :-1] if config.distil_from_target else attributions
            topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)
            if config.distil_from_target:  # keep last component always on
                last = torch.ones((*topk_mask.shape[:-1], 1), dtype=torch.bool, device=device)
                topk_mask = torch.cat((topk_mask, last), dim=-1)
            spd_fc.reset_hooks()
            with spd_fc.hooks(*spd_fc.get_caching_hooks()[:2], reset_hooks_end=True):
                h_topk = torch.relu(spd_fc.fc1(feats, topk_mask=topk_mask))
                out_topk = spd_fc.fc2(h_topk, topk_mask=topk_mask)

        # ──────────────── loss terms ────────────────
        distill_loss = mse(spd_out, teacher_out) * config.distill_coeff

        param_match_loss = torch.tensor(0.0, device=device)
        if config.param_match_coeff > 0:
            target_params = {
                p: get_nested_module_attr(teacher_model, p + ".weight") for p in param_names
            }
            spd_params = {
                p: get_nested_module_attr(spd_fc, p + ".weight").transpose(-1, -2)
                for p in param_names
            }
            diff = sum(((spd_params[p] - target_params[p]) ** 2).sum() for p in param_names)
            param_match_loss = diff / len(param_names) * config.param_match_coeff

        topk_recon_loss = torch.tensor(0.0, device=device)
        if config.topk_recon_coeff is not None and out_topk is not None:
            topk_recon_loss = ((out_topk - teacher_out) ** 2).mean() * config.topk_recon_coeff

        lp_sparsity_loss = torch.tensor(0.0, device=device)
        if config.lp_sparsity_coeff is not None and config.pnorm is not None:
            lp_vals = calc_lp_sparsity_loss(spd_out, attributions, config.pnorm)
            lp_sparsity_loss = lp_vals.sum(dim=-1).mean() * config.lp_sparsity_coeff

        # Condition subcomponent 0
        attr0_logit = attributions[:, 0] * config.alpha_condition
        condition_loss = bce(attr0_logit, background_label) * config.cond_coeff

        total_loss = (
            distill_loss
            + param_match_loss
            + topk_recon_loss
            + lp_sparsity_loss
            + condition_loss
        )

        # ──────────────── optimise ────────────────
        total_loss.backward()
        opt.step()

        # Remember the latest losses (for Optuna return & logging)
        last_losses = {
            "total": total_loss.item(),
            "distill": distill_loss.item(),
            "param_match": param_match_loss.item(),
            "topk_recon": topk_recon_loss.item(),
            "lp_sparsity": lp_sparsity_loss.item(),
            "condition": condition_loss.item(),
        }

        # Console logger
        if step % config.print_freq == 0:
            logger.info(
                f"Step {step:>5} | "
                + ", ".join(f"{k}={v:.4f}" for k, v in last_losses.items())
                + f", lr={step_lr:.2e}"
            )

        # Check‑pointing (optional)
        if config.save_freq and step and step % config.save_freq == 0:
            if config.out_dir:
                Path(config.out_dir).mkdir(parents=True, exist_ok=True)
                torch.save(
                    spd_fc.state_dict(),
                    Path(config.out_dir) / f"waterbird_spd_step{step}.pth",
                )

    # Save final model if desired
    if config.out_dir:
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(spd_fc.state_dict(), Path(config.out_dir) / "waterbird_spd_final.pth")

    return last_losses["total"], last_losses, spd_fc


# ──────────────────────────────────────────────────────────────────────────────
# YAML helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_config_to_yaml(config: WaterbirdSPDConfig, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config.dict(), f)


def load_config_from_yaml(path: str) -> WaterbirdSPDConfig:
    with open(path, "r") as f:
        return WaterbirdSPDConfig(**yaml.safe_load(f))


# ──────────────────────────────────────────────────────────────────────────────
# Optuna entry‑point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import optuna

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.trial.Trial):
        # Hyper‑parameter search space
        cfg = WaterbirdSPDConfig(
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            lr=trial.suggest_loguniform("lr", 1e-4, 1e-2),
            steps=trial.suggest_int("steps", 500, 1500),
            distill_coeff=trial.suggest_float("distill_coeff", 0.5, 2.0),
            param_match_coeff=trial.suggest_float("param_match_coeff", 1.0, 5.0),
            topk=trial.suggest_float("topk", 1.0, 5.0),
            lp_sparsity_coeff=trial.suggest_loguniform("lp_sparsity_coeff", 1e-3, 1e-1),
            cond_coeff=trial.suggest_loguniform("cond_coeff", 0.5, 5.0),
            topk_recon_coeff=trial.suggest_loguniform("topk_recon_coeff", 0.05, 0.5),
            seed=trial.number,
            pnorm=2.0,
            teacher_ckpt="checkpoints/waterbird_resnet18_best.pth",
        )

        # Save config
        trial_dir = Path("optuna_trials")
        trial_dir.mkdir(exist_ok=True)
        save_config_to_yaml(cfg, trial_dir / f"trial_{trial.number}_config.yaml")

        # Train
        total, losses, spd_model = run_spd_waterbird(cfg, device)

        # Store breakdown so we can access later
        for k, v in losses.items():
            trial.set_user_attr(k, v)

        # Append to log file
        with open(trial_dir / "total_losses.txt", "a") as f:
            f.write(f"Trial {trial.number}: total_loss = {total:.6f}\n")

        # Save model weights
        torch.save(spd_model.state_dict(), trial_dir / f"trial_{trial.number}_spd.pth")

        return total  # ← Optuna minimises total loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # ─── Summary ───
    best = study.best_trial
    print("\nBest trial:")
    print(f"  number       = {best.number}")
    print("  losses:")
    for k in ("total", "distill", "param_match", "topk_recon", "lp_sparsity", "condition"):
        print(f"    {k:<12}= {best.user_attrs[k]:.6f}")
    print("  hyper‑params =", best.params)
