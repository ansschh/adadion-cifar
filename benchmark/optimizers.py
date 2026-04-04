"""
Unified optimizer factory for CIFAR-10 benchmarking.

Implements the hybrid param-grouping pattern from torchtitan:
  - 2D+ weight matrices -> spectral optimizer (Muon/Dion/Dion2/AdaDion)
  - 1D params (norms, biases) -> AdamW with no weight decay
  - Embedding-like params -> AdamW with separate LR

For AdamW baseline, all params go through standard AdamW.
"""

import sys
import os
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer


def group_params_for_hybrid(model: nn.Module):
    """
    Split model parameters into matrix params (2D+) and scalar params (1D).
    Mirrors the torchtitan param_grouper logic adapted for vision models.

    Returns:
        dict with keys: matrix_params, norm_params, embed_params, other_params
    """
    matrix_params = []
    norm_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D params: norms, biases
        if param.ndim <= 1:
            norm_params.append(param)
        # Embedding-like: cls_token, pos_embed (ViT specific)
        elif "cls_token" in name or "pos_embed" in name:
            embed_params.append(param)
        # Everything else with ndim >= 2: conv weights, linear weights
        else:
            matrix_params.append(param)

    return {
        "matrix_params": matrix_params,
        "norm_params": norm_params,
        "embed_params": embed_params,
    }


def create_optimizer(model: nn.Module, opt_config) -> Optimizer:
    """
    Create an optimizer from config, applying hybrid param grouping
    for spectral optimizers.
    """
    name = opt_config.name

    if name == "adamw":
        return _create_adamw(model, opt_config)
    elif name == "muon":
        return _create_muon(model, opt_config)
    elif name == "dion":
        return _create_dion(model, opt_config)
    elif name == "dion2":
        return _create_dion2(model, opt_config)
    elif name == "adadion":
        return _create_adadion(model, opt_config)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _create_adamw(model: nn.Module, config) -> Optimizer:
    """Standard AdamW with weight decay only on 2D+ params."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )


def _create_muon(model: nn.Module, config) -> Optimizer:
    """Muon for matrix params + AdamW for scalars, using the dion package."""
    from dion import Muon

    groups = group_params_for_hybrid(model)
    param_groups = []

    # Matrix params -> Muon
    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})

    # Scalar / norm params -> AdamW
    if groups["norm_params"]:
        param_groups.append({
            "params": groups["norm_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": 0.0,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    # Embedding params -> AdamW
    if groups["embed_params"]:
        param_groups.append({
            "params": groups["embed_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": config.scalar_weight_decay,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    adjust_lr = config.adjust_lr if config.adjust_lr != "none" else None
    return Muon(
        param_groups,
        lr=config.lr,
        mu=config.mu,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
        adjust_lr=adjust_lr,
        flatten=config.flatten,
    )


def _create_dion(model: nn.Module, config) -> Optimizer:
    """Dion for matrix params + AdamW for scalars."""
    from dion import Dion

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})

    if groups["norm_params"]:
        param_groups.append({
            "params": groups["norm_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": 0.0,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    if groups["embed_params"]:
        param_groups.append({
            "params": groups["embed_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": config.scalar_weight_decay,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    return Dion(
        param_groups,
        lr=config.lr,
        rank_fraction=config.rank_fraction,
        weight_decay=config.weight_decay,
    )


def _create_dion2(model: nn.Module, config) -> Optimizer:
    """Dion2 for matrix params + AdamW for scalars."""
    from dion import Dion2

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})

    if groups["norm_params"]:
        param_groups.append({
            "params": groups["norm_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": 0.0,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    if groups["embed_params"]:
        param_groups.append({
            "params": groups["embed_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": config.scalar_weight_decay,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    adjust_lr = getattr(config, "adjust_lr", "spectral_norm")
    if adjust_lr == "none":
        adjust_lr = None

    return Dion2(
        param_groups,
        lr=config.lr,
        fraction=config.fraction,
        ef_decay=config.ef_decay,
        weight_decay=config.weight_decay,
        adjust_lr=adjust_lr,
        flatten=getattr(config, "flatten", True),
    )


def _create_adadion(model: nn.Module, config) -> Optimizer:
    """AdaDion V2 for matrix params + AdamW for scalars."""
    # Import from the cloned torchtitan repo
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(os.path.dirname(benchmark_dir), "torchtitan")
    experiments_dir = os.path.join(repo_root, "torchtitan", "experiments")

    # Add to path so we can import the module
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)

    from ortho_matrix.ada_dion_v2.adadion_v2 import AdaDionV2

    groups = group_params_for_hybrid(model)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})

    if groups["norm_params"]:
        param_groups.append({
            "params": groups["norm_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": 0.0,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    if groups["embed_params"]:
        param_groups.append({
            "params": groups["embed_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": config.scalar_weight_decay,
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })

    return AdaDionV2(
        param_groups,
        lr=config.lr,
        mu=config.mu,
        rank_fraction_max=config.rank_fraction_max,
        weight_decay=config.weight_decay,
        adaptive_rank=config.adaptive_rank,
        init_rank_fraction=config.init_rank_fraction,
        erank_ema_beta=config.erank_ema_beta,
        rank_scale=config.rank_scale,
        rank_min=config.rank_min,
        rank_quantize=config.rank_quantize,
        rank_step_up=config.rank_step_up,
        rank_step_down=config.rank_step_down,
    )
