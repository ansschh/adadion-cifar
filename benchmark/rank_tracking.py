#!/usr/bin/env python3
"""
Track AdaDion V2 rank evolution during training.

Logs per-layer rank, effective rank, and approximation error at every epoch.
Produces JSON output that can be plotted to show rank dynamics.

Usage:
    python rank_tracking.py --output_dir ./results/rank_tracking
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cifar10_benchmark import train_one_run, get_cifar10_loaders, create_lr_scheduler, evaluate
from configs import BaseConfig, AdaDionConfig
from models import create_model, get_param_summary
from optimizers import create_optimizer, group_params_for_hybrid
from metrics import MetricsCollector, compute_accuracy, compute_gradient_norm

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_rank_tracking(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data
    train_loader, test_loader = get_cifar10_loaders(128, 4, seed)
    steps_per_epoch = len(train_loader)

    # Model
    model = create_model("resnet18").to(device)
    logger.info(f"Model: ResNet-18, Params: {sum(p.numel() for p in model.parameters()):,}")

    # AdaDion V2 at best config
    opt_config = AdaDionConfig(
        lr=0.01, mu=0.95, rank_fraction_max=0.7, weight_decay=0.1,
        adaptive_rank=True, init_rank_fraction=0.5,
        erank_ema_beta=0.5, rank_scale=1.5,
        rank_min=16, rank_quantize=8, rank_step_up=16, rank_step_down=8,
        scalar_lr=0.01, scalar_betas=(0.95, 0.95), scalar_weight_decay=0.1,
    )

    optimizer, flatten_wrapper = create_optimizer(model, opt_config)

    base_config = BaseConfig(
        epochs=100, warmup_epochs=5, batch_size=128, num_workers=4,
        log_interval=100, save_checkpoints=False, model_name="resnet18",
        device="cuda", mixed_precision=False, seed=42, gradient_clip=1.0,
        label_smoothing=0.1,
    )
    scheduler = create_lr_scheduler(optimizer, base_config, steps_per_epoch)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Rank tracking data
    rank_history = []  # per-epoch snapshots
    step_rank_history = []  # per-step for fine-grained tracking

    global_step = 0
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = train_criterion(output, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if flatten_wrapper:
                flatten_wrapper.flatten_for_optimizer()
            optimizer.step()
            scheduler.step()
            if flatten_wrapper:
                flatten_wrapper.restore_shapes()

            with torch.no_grad():
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(targets).sum().item()
            epoch_loss += loss.item() * targets.size(0)
            epoch_samples += targets.size(0)

            # Log rank every 50 steps
            if global_step % 50 == 0:
                rank_data = {"step": global_step, "epoch": epoch}
                if hasattr(optimizer, "get_rank"):
                    rank_data["ranks"] = optimizer.get_rank()
                if hasattr(optimizer, "get_effective_rank"):
                    rank_data["effective_ranks"] = optimizer.get_effective_rank()
                if hasattr(optimizer, "get_aerr"):
                    rank_data["approx_errors"] = optimizer.get_aerr()
                step_rank_history.append(rank_data)

            global_step += 1

        # Epoch-level metrics
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples * 100
        val = evaluate(model, test_loader, device, use_amp=False)

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
        }
        if hasattr(optimizer, "get_rank"):
            epoch_data["ranks"] = optimizer.get_rank()
        if hasattr(optimizer, "get_effective_rank"):
            epoch_data["effective_ranks"] = optimizer.get_effective_rank()
        if hasattr(optimizer, "get_aerr"):
            epoch_data["approx_errors"] = optimizer.get_aerr()
        rank_history.append(epoch_data)

        logger.info(
            f"Epoch {epoch}/100 | Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | "
            f"Val: {val['acc']:.1f}% | "
            f"Ranks: {epoch_data.get('ranks', {})}"
        )

        # Save periodically
        if epoch % 10 == 0:
            with open(os.path.join(output_dir, "rank_history.json"), "w") as f:
                json.dump(rank_history, f, indent=2)
            with open(os.path.join(output_dir, "step_rank_history.json"), "w") as f:
                json.dump(step_rank_history, f, indent=2)

    # Final save
    with open(os.path.join(output_dir, "rank_history.json"), "w") as f:
        json.dump(rank_history, f, indent=2)
    with open(os.path.join(output_dir, "step_rank_history.json"), "w") as f:
        json.dump(step_rank_history, f, indent=2)

    logger.info(f"Rank tracking complete. Results in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results/rank_tracking")
    args = parser.parse_args()
    run_rank_tracking(args.output_dir)
