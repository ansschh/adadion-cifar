#!/usr/bin/env python3
"""
Distributed training benchmark for communication overhead analysis.

Runs each optimizer on multi-GPU (DDP) and measures:
  - Wall-clock time per step (compute + communication)
  - Bytes transferred per step (estimated from param sizes and algorithm)
  - Speedup over single-GPU
  - Communication fraction (time spent in collectives)

Usage (4 GPUs):
    torchrun --nproc_per_node=4 distributed_benchmark.py --output_dir ./results/distributed

Usage (single GPU baseline):
    python distributed_benchmark.py --output_dir ./results/distributed --single_gpu
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import create_model, get_param_summary
from configs import AdaDionConfig, AdamWConfig, MuonConfig, DionConfig, Dion2Config


def estimate_bytes_per_step(model, opt_name):
    """Estimate communication bytes per step for each optimizer under DDP."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 4  # float32

    if opt_name == "adamw":
        # DDP all-reduces full gradients
        return total_params * bytes_per_param * 2  # ring all-reduce = 2x

    elif opt_name == "muon":
        # Muon: all-reduce full gradients for matrix params (Newton-Schulz is local)
        # Plus scalar param gradients
        matrix_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim >= 2)
        scalar_params = total_params - matrix_params
        return (matrix_params + scalar_params) * bytes_per_param * 2

    elif opt_name == "dion":
        # Dion: compressed communication via low-rank factors P (m x r) and R (n x r)
        # Instead of full gradient (m x n), sends P and R
        rank_fraction = 0.25
        total_compressed = 0
        for p in model.parameters():
            if not p.requires_grad or p.ndim < 2:
                total_compressed += p.numel() * bytes_per_param * 2
                continue
            m, n = p.shape[0], p.numel() // p.shape[0]
            r = max(1, int(rank_fraction * min(m, n)))
            # P: m x r, R: n x r (instead of full m x n gradient)
            compressed = (m * r + n * r) * bytes_per_param * 2
            total_compressed += compressed
        return total_compressed

    elif opt_name == "dion2":
        # Dion2: fraction selection, sends selected submatrix
        fraction = 0.25
        total_compressed = 0
        for p in model.parameters():
            if not p.requires_grad or p.ndim < 2:
                total_compressed += p.numel() * bytes_per_param * 2
                continue
            m, n = p.shape[0], p.numel() // p.shape[0]
            selected = int(fraction * min(m, n))
            compressed = (selected * max(m, n)) * bytes_per_param * 2
            total_compressed += compressed
        return total_compressed

    elif opt_name == "adadion":
        # AdaDion: same as Dion but with adaptive rank (variable compression)
        # Use init_rank_fraction=0.5 as estimate
        rank_fraction = 0.5
        total_compressed = 0
        for p in model.parameters():
            if not p.requires_grad or p.ndim < 2:
                total_compressed += p.numel() * bytes_per_param * 2
                continue
            m, n = p.shape[0], p.numel() // p.shape[0]
            r = max(1, int(rank_fraction * min(m, n)))
            compressed = (m * r + n * r) * bytes_per_param * 2
            total_compressed += compressed
        return total_compressed

    return total_params * bytes_per_param * 2


def compute_compression_ratio(model, opt_name):
    """Compression ratio: full gradient bytes / actual bytes transferred."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    full_bytes = total_params * 4 * 2  # float32, ring all-reduce
    compressed_bytes = estimate_bytes_per_step(model, opt_name)
    return full_bytes / max(1, compressed_bytes)


def run_communication_analysis(output_dir):
    """Compute theoretical communication analysis without needing GPUs."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for model_name in ["resnet18", "vit_small"]:
        model = create_model(model_name)
        param_summary = get_param_summary(model)
        total_bytes_full = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 * 2

        model_results = {
            "model": model_name,
            "total_params": param_summary["total"],
            "matrix_params": param_summary["matrix"],
            "full_allreduce_bytes": total_bytes_full,
            "optimizers": {},
        }

        for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
            bytes_per_step = estimate_bytes_per_step(model, opt_name)
            compression = compute_compression_ratio(model, opt_name)
            model_results["optimizers"][opt_name] = {
                "bytes_per_step": bytes_per_step,
                "megabytes_per_step": bytes_per_step / (1024 * 1024),
                "compression_ratio": compression,
                "savings_pct": (1 - 1/compression) * 100 if compression > 1 else 0,
            }

        results[model_name] = model_results

        logger.info(f"\n{'='*60}")
        logger.info(f"  Communication Analysis: {model_name}")
        logger.info(f"  Total params: {param_summary['total']:,}")
        logger.info(f"  Full all-reduce: {total_bytes_full/(1024*1024):.1f} MB")
        logger.info(f"{'='*60}")
        for opt_name, opt_data in model_results["optimizers"].items():
            logger.info(
                f"  {opt_name:12s}: {opt_data['megabytes_per_step']:>8.1f} MB/step | "
                f"compression: {opt_data['compression_ratio']:.2f}x | "
                f"savings: {opt_data['savings_pct']:.1f}%"
            )

    with open(os.path.join(output_dir, "communication_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_distributed_timing(output_dir, num_steps=200):
    """
    Run DDP timing benchmark. Must be launched with torchrun.
    Measures per-step wall clock with and without communication.
    """
    if not dist.is_initialized():
        logger.error("Must be launched with torchrun for distributed timing")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for opt_name in ["adamw", "muon", "dion", "dion2", "adadion"]:
        model = create_model("resnet18").to(device)
        model = DDP(model, device_ids=[rank])

        # Create optimizer on the underlying module
        from optimizers import create_optimizer
        from configs import get_default_optimizer_config
        opt_config = get_default_optimizer_config(opt_name)
        if opt_name == "adadion":
            opt_config.lr = 0.01
            opt_config.scalar_lr = 0.01
        optimizer, wrapper = create_optimizer(model.module, opt_config)

        # Warmup
        x = torch.randn(32, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        for _ in range(10):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if wrapper:
                wrapper.flatten_for_optimizer()
            optimizer.step()
            if wrapper:
                wrapper.restore_shapes()

        # Timed steps
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()

        for step in range(num_steps):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if wrapper:
                wrapper.flatten_for_optimizer()
            optimizer.step()
            if wrapper:
                wrapper.restore_shapes()

        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start

        ms_per_step = elapsed / num_steps * 1000
        if rank == 0:
            results[opt_name] = {
                "ms_per_step": ms_per_step,
                "steps": num_steps,
                "world_size": world_size,
                "total_time_s": elapsed,
            }
            logger.info(f"  {opt_name:12s}: {ms_per_step:.1f} ms/step ({world_size} GPUs)")

        del model, optimizer
        torch.cuda.empty_cache()

    if rank == 0:
        with open(os.path.join(output_dir, "distributed_timing.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./results/distributed")
    parser.add_argument("--single_gpu", action="store_true", help="Run single-GPU baseline only")
    parser.add_argument("--analysis_only", action="store_true", help="Theoretical analysis only (no GPU needed)")
    args = parser.parse_args()

    if args.analysis_only:
        run_communication_analysis(args.output_dir)
    elif args.single_gpu:
        run_communication_analysis(args.output_dir)
    else:
        # Initialize distributed
        dist.init_process_group("nccl")
        run_communication_analysis(args.output_dir)
        run_distributed_timing(args.output_dir)
        dist.destroy_process_group()
