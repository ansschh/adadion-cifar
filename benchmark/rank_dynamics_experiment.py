#!/usr/bin/env python3
"""
Track effective rank and actual rank per step for different rank_scale values.

Produces two plots:
  1. Steps vs effective rank (erank), with legend = rank_scale gamma
  2. Steps vs actual rank used by AdaDion, with legend = rank_scale gamma

Uses ResNet-18 CIFAR-10, 50 epochs (enough to see dynamics), seed 42.
Logs every step for fine-grained tracking.
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import create_model
from optimizers import create_optimizer
from configs import BaseConfig, AdaDionConfig
from metrics import compute_accuracy


def get_loaders(batch_size=128, num_workers=4, seed=42):
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_t)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_t)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def run_single(rank_scale, epochs, output_dir, gpu_id=0):
    """Run AdaDion with a specific rank_scale and log rank dynamics every step."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train_loader, test_loader = get_loaders()
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    model = create_model("resnet18").to(device)

    opt_config = AdaDionConfig(
        lr=0.01, mu=0.95, rank_fraction_max=1.0, weight_decay=0.1,
        adaptive_rank=True, init_rank_fraction=1.0,
        erank_ema_beta=0.5, rank_scale=rank_scale,
        rank_min=16, rank_quantize=8, rank_step_up=16, rank_step_down=8,
        scalar_lr=0.01, scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1, scalar_eps=1e-8,
    )
    optimizer, flatten_wrapper = create_optimizer(model, opt_config)

    warmup_steps = 5 * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Per-step tracking
    step_data = []  # list of {step, erank, rank, loss}
    global_step = 0

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if flatten_wrapper:
                flatten_wrapper.flatten_for_optimizer()
            optimizer.step()

            # Collect rank info WHILE PARAMS ARE STILL FLATTENED
            # so the optimizer can find all params in its state dict
            entry = {"step": global_step, "loss": loss.item()}

            if hasattr(optimizer, "get_rank"):
                ranks = optimizer.get_rank()
                if ranks:
                    entry["rank"] = np.mean(list(ranks.values()))
                    entry["rank_per_param"] = ranks

            if hasattr(optimizer, "get_effective_rank"):
                eranks = optimizer.get_effective_rank()
                if eranks:
                    entry["erank"] = np.mean(list(eranks.values()))
                    entry["erank_per_param"] = eranks

            scheduler.step()
            if flatten_wrapper:
                flatten_wrapper.restore_shapes()

            step_data.append(entry)
            global_step += 1

        # Epoch summary
        if epoch % 10 == 0:
            last = step_data[-1]
            logger.info(
                f"[gamma={rank_scale}] Epoch {epoch}/{epochs} step {global_step} | "
                f"loss={loss.item():.4f} | "
                f"rank={last.get('rank', 'N/A')} | "
                f"erank={last.get('erank', 'N/A')}"
            )

    # Save
    run_name = f"gamma_{rank_scale}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save compact version (step, erank, rank, loss only)
    compact = [{"step": d["step"], "loss": d["loss"],
                "rank": d.get("rank"), "erank": d.get("erank")} for d in step_data]
    with open(os.path.join(run_dir, "rank_dynamics.json"), "w") as f:
        json.dump(compact, f)

    logger.info(f"[gamma={rank_scale}] DONE. {len(step_data)} steps saved to {run_dir}")
    return compact


def run_all(output_dir, epochs=50):
    os.makedirs(output_dir, exist_ok=True)
    gammas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    all_data = {}
    for gamma in gammas:
        data = run_single(gamma, epochs, output_dir)
        all_data[gamma] = data

    # Save combined
    with open(os.path.join(output_dir, "all_dynamics.json"), "w") as f:
        json.dump({str(g): d for g, d in all_data.items()}, f)

    logger.info(f"All rank dynamics saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results/rank_dynamics")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=None, help="Single gamma to run")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.gamma is not None:
        run_single(args.gamma, args.epochs, args.output_dir, args.gpu)
    else:
        run_all(args.output_dir, args.epochs)
