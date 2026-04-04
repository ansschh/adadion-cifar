#!/usr/bin/env python3
"""
CPU smoke test — verifies the full pipeline works on a laptop without GPU/triton.

Tests:
  1. Data loading (CIFAR-10 download + augmentation)
  2. All model architectures (ResNet-18, ResNet-34, VGG-16-BN, ViT-Small)
  3. Training loop (forward, backward, optimizer step, LR schedule)
  4. Metrics collection and saving
  5. Evaluation loop
  6. AdamW optimizer (native PyTorch, always works)
  7. Structural test of spectral optimizer param grouping

Since triton is GPU-only, Dion/Dion2/Muon/AdaDion cannot be imported on CPU.
We validate the param grouping and config logic, then run a real training loop
with AdamW to verify end-to-end correctness.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add benchmark dir to path
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)

from configs import BaseConfig, get_default_optimizer_config, MODEL_CONFIGS
from models import create_model, count_parameters, get_param_summary
from metrics import MetricsCollector, compute_accuracy, compute_gradient_norm
from optimizers import group_params_for_hybrid


def test_models():
    """Test all model architectures create correctly and forward pass works."""
    print("=" * 60)
    print("TEST 1: Model Architectures")
    print("=" * 60)

    dummy_input = torch.randn(2, 3, 32, 32)

    for name in ["resnet18", "resnet34", "vgg16_bn", "vit_small"]:
        kwargs = MODEL_CONFIGS.get(name, {"num_classes": 10})
        model = create_model(name, **kwargs)
        summary = get_param_summary(model)
        out = model(dummy_input)

        assert out.shape == (2, 10), f"{name}: wrong output shape {out.shape}"
        print(f"  {name:12s} | params: {summary['total']:>10,} | "
              f"matrix: {summary['matrix_pct']:.1f}% | "
              f"output: {out.shape} | OK")

    print()
    return True


def test_param_grouping():
    """Test hybrid param grouping for all models."""
    print("=" * 60)
    print("TEST 2: Hybrid Parameter Grouping")
    print("=" * 60)

    for name in ["resnet18", "vit_small"]:
        kwargs = MODEL_CONFIGS.get(name, {"num_classes": 10})
        model = create_model(name, **kwargs)
        groups = group_params_for_hybrid(model)

        n_matrix = len(groups["matrix_params"])
        n_norm = len(groups["norm_params"])
        n_embed = len(groups["embed_params"])

        matrix_numel = sum(p.numel() for p in groups["matrix_params"])
        norm_numel = sum(p.numel() for p in groups["norm_params"])
        embed_numel = sum(p.numel() for p in groups["embed_params"])

        total = matrix_numel + norm_numel + embed_numel
        total_params = count_parameters(model)
        assert total == total_params, (
            f"{name}: grouping missed params ({total} vs {total_params})"
        )

        print(f"  {name:12s} | matrix: {n_matrix} tensors ({matrix_numel:,}) | "
              f"norm: {n_norm} ({norm_numel:,}) | "
              f"embed: {n_embed} ({embed_numel:,}) | "
              f"total: {total:,} | OK")

        # Verify all matrix params are 2D+
        for p in groups["matrix_params"]:
            assert p.ndim >= 2, f"matrix param has ndim={p.ndim}"
        # Verify all norm params are 1D
        for p in groups["norm_params"]:
            assert p.ndim <= 1, f"norm param has ndim={p.ndim}"

    print()
    return True


def test_data_loading():
    """Test CIFAR-10 download and data loading."""
    print("=" * 60)
    print("TEST 3: CIFAR-10 Data Loading")
    print("=" * 60)

    # Import here to avoid circular issues
    from cifar10_benchmark import get_cifar10_loaders

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32, num_workers=0, seed=42,
    )

    # Check one batch
    images, targets = next(iter(train_loader))
    assert images.shape == (32, 3, 32, 32), f"wrong image shape: {images.shape}"
    assert targets.shape == (32,), f"wrong target shape: {targets.shape}"
    assert targets.min() >= 0 and targets.max() <= 9

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Batch shape:   {images.shape}")
    print(f"  Target range:  [{targets.min()}, {targets.max()}]")
    print(f"  Image stats:   mean={images.mean():.3f}, std={images.std():.3f}")
    print()
    return True


def test_adamw_training():
    """Run 2 epochs of AdamW training on ResNet-18, CPU."""
    print("=" * 60)
    print("TEST 4: AdamW Training Loop (2 epochs, ResNet-18, CPU)")
    print("=" * 60)

    from cifar10_benchmark import train_one_run

    output_dir = os.path.join(BENCH_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    config = BaseConfig(
        epochs=2,
        warmup_epochs=1,
        batch_size=64,
        num_workers=0,
        log_interval=20,
        save_checkpoints=False,
        model_name="resnet18",
        device="cpu",
        mixed_precision=False,
        seed=42,
        label_smoothing=0.1,
    )

    opt_config = get_default_optimizer_config("adamw")

    start = time.perf_counter()
    summary = train_one_run(config, opt_config, output_dir, "cpu_smoke_adamw")
    elapsed = time.perf_counter() - start

    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Final train loss: {summary['final_train_loss']:.4f}")
    print(f"  Final train acc:  {summary['final_train_acc']:.1f}%")
    print(f"  Final val acc:    {summary['final_val_acc']:.1f}%")
    print(f"  Best val acc:     {summary['best_val_acc']:.1f}%")
    print(f"  Avg step time:    {summary['avg_step_time_ms']:.1f}ms")

    # Sanity checks
    assert summary['final_train_loss'] < 3.0, "Loss too high — training not working"
    assert summary['final_train_acc'] > 10, "Accuracy below random — training broken"
    assert summary['total_epochs'] == 2

    # Check metrics files were saved
    run_dir = os.path.join(output_dir, "cpu_smoke_adamw")
    for fname in ["step_metrics.json", "epoch_metrics.json", "summary.json", "config.json"]:
        fpath = os.path.join(run_dir, fname)
        assert os.path.exists(fpath), f"Missing {fname}"
        with open(fpath) as f:
            data = json.load(f)
            assert data, f"{fname} is empty"

    print(f"  Metrics saved:    OK (4 JSON files)")
    print()
    return True


def test_vit_training():
    """Quick ViT-Small forward/backward sanity on CPU."""
    print("=" * 60)
    print("TEST 5: ViT-Small Forward/Backward (CPU)")
    print("=" * 60)

    kwargs = MODEL_CONFIGS["vit_small"]
    model = create_model("vit_small", **kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    grad_norm = compute_gradient_norm(model)
    opt.step()

    assert out.shape == (4, 10)
    assert loss.item() > 0
    assert grad_norm > 0

    print(f"  Output shape: {out.shape}")
    print(f"  Loss:         {loss.item():.4f}")
    print(f"  Grad norm:    {grad_norm:.4f}")
    print(f"  Step:         OK")
    print()
    return True


def test_metrics_collection():
    """Test metrics collector independently."""
    print("=" * 60)
    print("TEST 6: Metrics Collector")
    print("=" * 60)

    output_dir = os.path.join(BENCH_DIR, "test_results")
    mc = MetricsCollector(output_dir, "test_metrics")

    mc.start_epoch()
    for step in range(5):
        mc.start_step()
        time.sleep(0.01)
        mc.end_step(step=step, epoch=0, loss=2.0 - step * 0.1,
                    acc=20.0 + step * 5, lr=0.001, grad_norm=1.5,
                    batch_size=64)

    mc.end_epoch(epoch=0, train_loss=1.5, train_acc=40.0,
                 val_loss=1.8, val_acc=35.0, val_top5_acc=80.0, lr=0.001)
    mc.save()

    assert len(mc.step_history) == 5
    assert len(mc.epoch_history) == 1
    summary = mc.get_summary()
    assert summary["best_val_acc"] == 35.0
    assert summary["total_epochs"] == 1

    print(f"  Steps recorded:  {len(mc.step_history)}")
    print(f"  Epochs recorded: {len(mc.epoch_history)}")
    print(f"  Summary:         OK")
    print(f"  Files saved:     OK")
    print()
    return True


def test_lr_scheduler():
    """Test LR scheduler warmup + cosine decay."""
    print("=" * 60)
    print("TEST 7: LR Scheduler (warmup + cosine)")
    print("=" * 60)

    from cifar10_benchmark import create_lr_scheduler

    model = nn.Linear(10, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = BaseConfig(epochs=10, warmup_epochs=2)
    steps_per_epoch = 100

    scheduler = create_lr_scheduler(opt, config, steps_per_epoch)

    lrs = []
    for step in range(1000):
        lrs.append(scheduler.get_last_lr()[0])
        opt.step()
        scheduler.step()

    # Warmup: LR should increase
    assert lrs[50] > lrs[0], "LR not increasing during warmup"
    # Post-warmup peak
    assert lrs[200] > lrs[500], "LR not decaying after warmup"
    # End: should be near min_lr_factor (0.0)
    assert lrs[-1] < lrs[200] * 0.1, "LR not decaying to near zero"

    print(f"  Start LR:     {lrs[0]:.6f}")
    print(f"  After warmup: {lrs[200]:.6f}")
    print(f"  Mid training: {lrs[500]:.6f}")
    print(f"  End LR:       {lrs[-1]:.6f}")
    print(f"  Schedule:     OK")
    print()
    return True


def test_configs():
    """Test all optimizer configs instantiate correctly."""
    print("=" * 60)
    print("TEST 8: Optimizer Configs")
    print("=" * 60)

    for name in ["adamw", "muon", "dion", "dion2", "adadion"]:
        config = get_default_optimizer_config(name)
        assert config.name == name
        assert hasattr(config, "lr")
        print(f"  {name:12s} | lr={config.lr} | OK")

    print()
    return True


def main():
    print()
    print("CIFAR-10 Optimizer Benchmark — CPU Smoke Test")
    print("=" * 60)
    print(f"PyTorch:  {torch.__version__}")
    print(f"Device:   CPU (no CUDA)")
    print(f"Platform: {sys.platform}")
    print()

    tests = [
        ("Model architectures", test_models),
        ("Param grouping", test_param_grouping),
        ("Data loading", test_data_loading),
        ("Configs", test_configs),
        ("Metrics collector", test_metrics_collection),
        ("LR scheduler", test_lr_scheduler),
        ("ViT forward/backward", test_vit_training),
        ("AdamW training (2 epochs)", test_adamw_training),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((name, f"FAIL: {e}"))

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, status in results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}")
        if status != "PASS":
            all_pass = False

    print()
    if all_pass:
        print("All tests passed! The benchmark is ready for GPU/RunPod.")
        print()
        print("Note: Dion, Dion2, Muon, and AdaDion require triton (GPU-only).")
        print("They will work on RunPod but cannot be tested on CPU.")
    else:
        print("Some tests failed. Check output above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
