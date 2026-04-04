"""
Unified optimizer factory for CIFAR-10 benchmarking.

Implements the hybrid param-grouping pattern from torchtitan:
  - 2D weight matrices -> spectral optimizer (Muon/Dion/Dion2/AdaDion)
  - 4D conv weights -> spectral optimizer with flatten (Muon/Dion2) or AdamW (Dion)
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

# Increase torch.compile recompile limit for spectral optimizers
# They use torch.compile internally and hit recompile limits with varied batch shapes
try:
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 64
except Exception:
    pass


def group_params_for_hybrid(model: nn.Module, flatten_supported: bool = True):
    """
    Split model parameters into groups for hybrid spectral + AdamW optimization.

    Args:
        flatten_supported: If True, 3D+ tensors go to matrix_params (optimizer will flatten).
                          If False, only 2D tensors go to matrix_params; 3D+ go to conv_params.

    Returns:
        dict with keys: matrix_params, conv_params, norm_params, embed_params
    """
    matrix_params = []  # 2D params -> always spectral
    conv_params = []    # 4D params -> spectral (if flatten) or AdamW
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
        # 2D: linear weights -> always spectral
        elif param.ndim == 2:
            matrix_params.append(param)
        # 3D+: conv weights
        else:
            if flatten_supported:
                matrix_params.append(param)
            else:
                conv_params.append(param)

    return {
        "matrix_params": matrix_params,
        "conv_params": conv_params,
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


def _build_scalar_groups(groups, config):
    """Build AdamW param groups for non-matrix parameters."""
    param_groups = []
    # Conv params that can't be handled by spectral optimizer
    if groups["conv_params"]:
        param_groups.append({
            "params": groups["conv_params"],
            "algorithm": "adamw",
            "lr": config.scalar_lr,
            "weight_decay": getattr(config, "scalar_weight_decay", config.weight_decay),
            "betas": config.scalar_betas,
            "eps": config.scalar_eps,
        })
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
    return param_groups


def _create_muon(model: nn.Module, config) -> Optimizer:
    """Muon for matrix params + AdamW for scalars, using the dion package."""
    from dion import Muon

    # Muon supports flatten=True, so 4D conv params go to matrix_params
    groups = group_params_for_hybrid(model, flatten_supported=True)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

    adjust_lr = config.adjust_lr if config.adjust_lr != "none" else None
    return Muon(
        param_groups,
        lr=config.lr,
        mu=config.mu,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
        adjust_lr=adjust_lr,
        flatten=config.flatten,
        use_triton=False,
    )


def _create_dion(model: nn.Module, config) -> Optimizer:
    """Dion for 2D matrix params + AdamW for conv/scalars (Dion doesn't support flatten)."""
    from dion import Dion

    # Dion does NOT support flatten, so 4D conv params go to AdamW
    groups = group_params_for_hybrid(model, flatten_supported=False)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

    return Dion(
        param_groups,
        lr=config.lr,
        rank_fraction=config.rank_fraction,
        weight_decay=config.weight_decay,
    )


def _create_dion2(model: nn.Module, config) -> Optimizer:
    """Dion2 for matrix params + AdamW for scalars."""
    from dion import Dion2

    # Dion2 supports flatten=True
    groups = group_params_for_hybrid(model, flatten_supported=True)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

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
        use_triton=False,
    )


def _create_adadion(model: nn.Module, config) -> Optimizer:
    """AdaDion V2 for 2D matrix params + AdamW for conv/scalars (no flatten support)."""
    from adadion_v2.adadion_v2 import AdaDionV2

    # AdaDion V2 does NOT support flatten, conv params go to AdamW
    groups = group_params_for_hybrid(model, flatten_supported=False)
    param_groups = []

    if groups["matrix_params"]:
        param_groups.append({"params": groups["matrix_params"]})
    param_groups.extend(_build_scalar_groups(groups, config))

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
