"""
Metrics collection, logging, and checkpointing for CIFAR-10 benchmarks.

Tracks:
  - Training loss, accuracy per step
  - Validation loss, accuracy per epoch
  - Learning rate schedule
  - Wall-clock time and throughput
  - GPU memory usage
  - Gradient norms
  - Per-optimizer diagnostic metrics (rank, effective rank for AdaDion)
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int = 0
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    step_time_ms: float = 0.0
    gpu_mem_mb: float = 0.0
    throughput_samples_sec: float = 0.0


@dataclass
class EpochMetrics:
    """Metrics for a full epoch."""
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    val_top5_acc: float = 0.0
    epoch_time_sec: float = 0.0
    best_val_acc: float = 0.0
    lr: float = 0.0


class MetricsCollector:
    """Collects and stores metrics throughout training."""

    def __init__(self, output_dir: str, run_name: str):
        self.output_dir = output_dir
        self.run_name = run_name
        self.run_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.step_history: list[dict] = []
        self.epoch_history: list[dict] = []
        self.best_val_acc = 0.0
        self.best_val_epoch = 0
        self.total_train_time = 0.0
        self._epoch_start = None
        self._step_start = None

        # Extra optimizer diagnostics
        self.optimizer_metrics: list[dict] = []

    def start_epoch(self):
        self._epoch_start = time.perf_counter()

    def start_step(self):
        self._step_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def end_step(self, step: int, epoch: int, loss: float, acc: float,
                 lr: float, grad_norm: float, batch_size: int):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._step_start) * 1000  # ms

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        throughput = batch_size / (elapsed / 1000) if elapsed > 0 else 0

        metrics = {
            "step": step,
            "epoch": epoch,
            "train_loss": loss,
            "train_acc": acc,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time_ms": elapsed,
            "gpu_mem_mb": gpu_mem,
            "throughput_samples_sec": throughput,
        }
        self.step_history.append(metrics)
        return metrics

    def end_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, val_top5_acc: float, lr: float):
        epoch_time = time.perf_counter() - self._epoch_start
        self.total_train_time += epoch_time

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_epoch = epoch

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_top5_acc": val_top5_acc,
            "epoch_time_sec": epoch_time,
            "best_val_acc": self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "lr": lr,
        }
        self.epoch_history.append(metrics)
        return metrics

    def log_optimizer_metrics(self, step: int, opt_metrics: dict):
        """Log optimizer-specific diagnostics (rank, erank, etc.)."""
        entry = {"step": step}
        entry.update(opt_metrics)
        self.optimizer_metrics.append(entry)

    def save(self):
        """Save all metrics to JSON files."""
        with open(os.path.join(self.run_dir, "step_metrics.json"), "w") as f:
            json.dump(self.step_history, f, indent=2)
        with open(os.path.join(self.run_dir, "epoch_metrics.json"), "w") as f:
            json.dump(self.epoch_history, f, indent=2)
        if self.optimizer_metrics:
            with open(os.path.join(self.run_dir, "optimizer_metrics.json"), "w") as f:
                json.dump(self.optimizer_metrics, f, indent=2)

        # Save summary
        summary = self.get_summary()
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def get_summary(self) -> dict:
        """Get a concise summary of the run."""
        if not self.epoch_history:
            return {}

        final = self.epoch_history[-1]
        avg_step_time = 0.0
        if self.step_history:
            avg_step_time = sum(s["step_time_ms"] for s in self.step_history) / len(self.step_history)

        avg_throughput = 0.0
        if self.step_history:
            avg_throughput = sum(s["throughput_samples_sec"] for s in self.step_history) / len(self.step_history)

        peak_mem = 0.0
        if self.step_history:
            peak_mem = max(s["gpu_mem_mb"] for s in self.step_history)

        return {
            "run_name": self.run_name,
            "final_train_loss": final["train_loss"],
            "final_train_acc": final["train_acc"],
            "final_val_loss": final["val_loss"],
            "final_val_acc": final["val_acc"],
            "final_val_top5_acc": final["val_top5_acc"],
            "best_val_acc": self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "total_epochs": len(self.epoch_history),
            "total_train_time_sec": self.total_train_time,
            "avg_step_time_ms": avg_step_time,
            "avg_throughput_samples_sec": avg_throughput,
            "peak_gpu_mem_mb": peak_mem,
        }


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.item() / batch_size * 100)
        return res


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.float().norm().item() ** 2
    return total_norm_sq ** 0.5
