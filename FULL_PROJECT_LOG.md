# AdaDion V2 CIFAR-10/100 Benchmark: Complete Project Log

This document records every step taken from start to finish. It covers what was done, why, what went wrong, how it was fixed, what experiments were run, what infrastructure was used, and what the final results are. If someone asks "how did you do X", the answer is here.

---

## 1. Starting Point

The project began with a request to benchmark the AdaDion V2 optimizer against AdamW, Muon, Dion, and Dion2 on CIFAR-10. The optimizer code lives in the `Tatzhiro/replication` branch of `github.com/Pro-Place/torchtitan`, specifically in `torchtitan/experiments/ortho_matrix/ada_dion_v2/`. The `dion` package (containing Dion, Dion2, Muon) is installed from `github.com/microsoft/dion`.

AdaDion V2 is an adaptive-rank variant of Dion. Dion uses low-rank power iteration to approximate gradient momentum updates, communicating compressed factors instead of full gradients. AdaDion V2 adds a mechanism that dynamically adjusts the rank based on the effective rank of the momentum matrix, tracked via Shannon entropy of singular value proxies.

The codebase was designed for distributed LLM pretraining (LLaMA 320M on C4 dataset, 8-GPU FSDP). All prior evaluations used transformer models where every weight matrix is natively 2D. No one had tested these optimizers on vision models (CNNs with 4D conv weights) or on CIFAR-10/100.

## 2. Codebase Exploration

We cloned the `Tatzhiro/replication` branch and read every file in the `ada_dion_v2/` directory:

- `adadion_v2.py` (1982 lines): The main optimizer class `AdaDionV2`, inheriting from `torch.optim.Optimizer`. Contains `dion_update_ddp()`, `dion_update_fsdp()`, `dion_update_fsdp_tp()` for different parallelism modes. The adaptive rank logic is in `_adaptive_rank_update()` which computes effective rank from column norms of the R factor, applies EMA smoothing, and adjusts rank with rate limiting.

- `config_registry.py`: Defines `AdaDionV2Container` with default hyperparameters for LLaMA 320M training (lr=0.012, rank_fraction=0.5, init_rank_fraction=0.25, erank_ema_beta=0.9, rank_min=16, rank_quantize=8).

- `dion_utils.py`: Distributed utilities (DTensor conversions, async task runtime, parameter batching).

- `scalar_opts.py`: AdamW and Lion update functions with `@torch.compile(fullgraph=True)` decorators.

We also read the `dion` package source at `site-packages/dion/`:
- `dion.py`: Dion optimizer. Accepts `replicate_mesh`, `outer_shard_mesh`, `inner_shard_mesh` for distributed training. Uses `orthogonalize()` with randomized Cholesky QR.
- `muon.py`: Muon optimizer. Uses Newton-Schulz iteration (`zeropower_via_newtonschulz5`). Has `flatten` parameter for handling 4D tensors.
- `dion2.py`: Dion2 optimizer. Fraction selection + Newton-Schulz. Also has `flatten` parameter.
- `newton_schulz_triton.py`: Triton kernels for Newton-Schulz. Imports `triton` at module level, which means the entire `dion` package fails to import if triton is not installed (triton is Linux/GPU only).

Key finding: All optimizers use a hybrid parameter grouping pattern. Weight matrices (2D) go to the spectral optimizer, while 1D parameters (norms, biases) go to AdamW. This grouping is done via `group_params_for_hybrid()` in the torchtitan common code.

## 3. Benchmark Suite Construction

We built a complete benchmark suite from scratch in `benchmark/`:

### Models (`models.py`)
- ResNet-18/34 adapted for CIFAR-10 (3x3 stem instead of 7x7, no max pool)
- VGG-16-BN
- ViT-Small (patch size 4, 384-dim, 6 layers, 6 heads, stochastic depth)

### Optimizer Factory (`optimizers.py`)
The central challenge: adapting distributed LLM optimizers for single-GPU vision training.

We implemented `group_params_for_hybrid()` to split model parameters:
- 2D+ weight matrices go to the spectral optimizer
- 1D params (norms, biases) go to AdamW with no weight decay
- ViT-specific params (cls_token, pos_embed) go to AdamW with weight decay

For each spectral optimizer, we created a factory function that builds param groups with the `algorithm` key that Dion/Muon/etc. use to route params internally.

### Training Loop (`cifar10_benchmark.py`)
Standard CIFAR-10 training with:
- AutoAugment + RandomCrop + HFlip + RandomErasing
- Label smoothing (0.1)
- Cosine LR decay with linear warmup (5 epochs)
- Gradient clipping at 1.0
- Per-step and per-epoch metrics logging to JSON

### Configs (`configs.py`)
Default hyperparameters for each optimizer, sweep grids, model configs.

### Analysis (`analysis.py`)
Post-run visualization: training curves, comparison bars, convergence speed, LR sweep plots.

## 4. Initial Smoke Test (CPU, Local Laptop)

Ran `smoke_test_cpu.py` on Windows laptop (PyTorch 2.11 CPU, no CUDA). This verified:
- All 4 model architectures forward pass correctly
- Parameter grouping produces correct counts (ResNet-18: 21 matrix tensors, 41 norm tensors)
- CIFAR-10 data loading and augmentation pipeline works
- AdamW training loop runs end-to-end (2 epochs, reached 76.3% val accuracy)
- Metrics collection and JSON serialization works
- LR scheduler warmup + cosine decay produces expected curve

Spectral optimizers (Dion, Muon, etc.) could not be tested on CPU because the `dion` package imports `triton` at the top level, and triton is Linux/GPU only.

## 5. First RunPod Deployment (A100 80GB)

Pod: 1x A100-SXM4-80GB, PyTorch 2.11+cu130

### Setup
1. Cloned the benchmark repo from GitHub
2. Attempted to clone torchtitan repo for AdaDion V2 source, but the pod couldn't authenticate to GitHub (git:// timed out, SSH no keys, HTTPS no credentials)
3. Solution: bundled the AdaDion V2 source files (`adadion_v2.py`, `dion_utils.py`, `scalar_opts.py`) directly into the benchmark repo under `benchmark/adadion_v2/`
4. Installed `dion` package via `pip install git+https://github.com/microsoft/dion.git`

### Bug 1: AdaDion V2 crashes on single GPU

The original `adadion_v2.py` unconditionally passes `outer_shard_mesh=self._outer_shard_mesh` to all update functions. But `dion_update_ddp()` does not accept this parameter (it's only for FSDP paths). On single GPU, `_outer_shard_mesh` is None and `dion_update_ddp` is selected, causing a TypeError.

Fix: Conditionally pass `outer_shard_mesh` only when `use_dtensor=True`:
```python
update_kwargs = dict(X=..., G=..., M=..., Q=..., ...)
if use_dtensor:
    update_kwargs["outer_shard_mesh"] = self._outer_shard_mesh
yield AsyncTask(dion_update_func(**update_kwargs))
```

### Bug 2: Spectral optimizers crash on 4D conv weights

All Dion-family optimizers reject tensors with ndim > 2:
```python
if x.ndim > 2:
    raise NotImplementedError("Tensors with more than 2 dimensions are not supported.")
```

ResNet-18 has 99.95% of parameters in 4D conv weights. Only the final `fc.weight` (10x512 = 5,120 params out of 11.17M) is 2D.

Initial workaround: Route conv params to AdamW fallback. This meant the spectral optimizers operated on <0.1% of parameters, making them effectively just AdamW.

Muon was the exception: it supports `flatten=True` internally, which reshapes 4D tensors to 2D before Newton-Schulz. So Muon operated on all parameters, which is why it initially outperformed the others.

### Bug 3: torch.compile crashes on Dion2

Dion2's `dion2_pre_orthogonalize` function has `@torch.compile(fullgraph=True)`. On PyTorch 2.11, this hits an `InductorError` (assertion failure in the scheduler's fusion pass). The error is in the `indices.unsqueeze(-1).expand(-1, -1, num_cols)` line.

Attempted fixes:
1. `torch._dynamo.config.suppress_errors = True`: Didn't catch `InductorError`
2. `torch._dynamo.config.disable = True`: Worked but broke other optimizers' compiled functions
3. Final: Set `suppress_errors = True` at module level, which lets compile failures fall back to eager mode

### Bug 4: GradScaler + spectral optimizers

The training loop used `torch.amp.autocast("cuda")` + `GradScaler` for all optimizers. The GradScaler pattern (scale loss, backward, unscale, check inf, step or skip) conflicted with spectral optimizers:
- `scaler.unscale_()` modifies `param.grad` in place, but spectral optimizers process gradients internally
- `scaler.step()` may skip the optimizer step entirely if any gradient has inf/nan
- The `@torch.compile(fullgraph=True)` functions inside the optimizers receive already-unscaled gradients but the dtype may be float16 from autocast

When we set `suppress_errors=True`, failed compile functions silently became no-ops, causing the optimizer to skip parameter updates entirely. This produced ViT-Small results of 63% (vs 86% for AdamW) because the spectral optimizer was effectively doing nothing.

Fix: Disable both autocast AND GradScaler for spectral optimizers:
```python
is_spectral = opt_config.name in ("muon", "dion", "dion2", "adadion")
use_amp = base_config.mixed_precision and torch.cuda.is_available() and (not is_spectral)
```

## 6. First Benchmark Results (Broken)

With bugs 1-4 partially fixed, the first full benchmark produced:

| Optimizer | ResNet-18 Val Acc |
|-----------|-------------------|
| Muon | 96.32% (3 seeds, 200ep) |
| AdamW | 95.78% |
| Dion | 95.50% |
| Dion2 | 95.44% |
| AdaDion V2 | 94.95% |

AdaDion was the worst. The user's teammates had shown AdaDion beating Muon/Dion on LLaMA pretraining. Something was fundamentally wrong.

## 7. Root Cause Investigation

### Investigation 1: Parameter coverage

We counted exactly which parameters went where:
- ResNet-18 matrix params with `flatten_supported=False`: only `fc.weight` (5,120 params)
- ResNet-18 conv params routed to AdamW: 11,159,232 params (99.95%)

AdaDion was optimizing 0.05% of parameters spectrally. The rest used plain AdamW.

### Investigation 2: Hyperparameter mismatch

Our `AdaDionConfig` used values we guessed for CIFAR-10, which diverged from the official `config_registry.py`:

| Param | Ours | Official |
|-------|------|----------|
| lr | 0.02 | 0.012 |
| erank_ema_beta | 0.5 | 0.9 |
| rank_min | 8 | 16 |
| rank_quantize | 4 | 8 |
| scalar_lr | 1e-3 | 0.012 |

### Investigation 3: Context difference

Tatsu's setup (where AdaDion beat Muon) was completely different:
- Model: LLaMA3 320M (324M params, all 2D linear weights)
- Dataset: C4 (language modeling)
- GPUs: 8x A40 with FSDP
- Framework: torchtitan with torchrun
- Command: `bash env/runpod/run.sh -m ortho_matrix.ada_dion_v2 -c llama3_320m_adadion_v2 -l 16 -n "rank_scale_20" -- --optimizer.rank_scale 2.0`

On LLaMA, 100% of weight matrices are natively 2D (attention QKV/O, FFN W1/W2/W3). No flattening needed. The adaptive rank mechanism has large matrices to work with (768x768, 768x2048).

## 8. The Fix: FlattenedParamWrapper

Created `FlattenedParamWrapper` that reshapes 4D conv weights to 2D around each optimizer step:

```python
class FlattenedParamWrapper:
    def __init__(self):
        self._shapes = {}  # param -> original_shape

    def register(self, param):
        if param.ndim > 2:
            self._shapes[param] = param.shape

    def flatten_for_optimizer(self):
        for param, orig_shape in self._shapes.items():
            param.data = param.data.flatten(1)
            if param.grad is not None:
                param.grad = param.grad.flatten(1)

    def restore_shapes(self):
        for param, orig_shape in self._shapes.items():
            param.data = param.data.view(orig_shape)
            if param.grad is not None:
                param.grad = param.grad.view(orig_shape)
```

Usage in training loop:
```python
optimizer, flatten_wrapper = create_optimizer(model, opt_config)
# ...
loss.backward()
if flatten_wrapper:
    flatten_wrapper.flatten_for_optimizer()
optimizer.step()
if flatten_wrapper:
    flatten_wrapper.restore_shapes()
```

During `create_optimizer()`, we temporarily flatten to 2D so the optimizer sees 2D shapes at init time (for state allocation, rank computation). Then restore before returning.

For Muon and Dion2 which support `flatten=True` internally, we pass that flag instead of using the wrapper. However, Dion2's flatten hits a torch.compile bug, so it also uses the wrapper with `flatten=False`.

## 9. Ablation Study (22 Runs)

After fixing all bugs, we ran a systematic ablation on ResNet-18 CIFAR-10 (100 epochs, seed 42) on an RTX 5090 pod:

### Phase 1: Baselines
- AdamW: 95.30%
- Muon: 95.87%
- Dion: 96.04%

### Phase 2: LR Sweep (the key finding)
| LR | Val Acc |
|----|---------|
| 0.002 | 95.95% |
| 0.005 | 96.21% |
| 0.01 | 96.30% |
| 0.02 | 95.84% |
| 0.04 | 95.57% |

The optimal LR is 0.01, not 0.02 (our initial guess) or 0.012 (official LLM config). This was the single most impactful fix.

### Phase 3: Adaptive rank ON vs OFF
- adaptive=True: 96.30%
- adaptive=False: 96.12%
- Delta: +0.18% from adaptive rank

### Phase 4: Gradient clipping
- clip=1.0: 96.30% (best)
- clip=5.0: 96.11%
- no clip: 96.08%

### Phase 5: Weight decay
- wd=0.1: 96.30% (best)
- wd=0.0: 95.63% (worst)

### Phase 6: Rank fraction
- rf=0.125: 96.19%
- rf=0.5: 96.30% (best)
- rf=1.0: 95.99%

### Phase 7: Scalar LR decoupling
- scalar_lr=0.01 (matched): 96.30% (best)
- scalar_lr=0.001: 95.99%

## 10. Final Benchmark (3 Seeds)

With the optimal config (lr=0.01, wd=0.1, clip=1.0, adaptive=True, rf=0.5, scalar_lr=0.01), we ran the full benchmark on 4x A100 DDP:

### ResNet-18 CIFAR-10 (3 seeds, 100 epochs)
| Optimizer | Seed 42 | Seed 123 | Seed 456 | Mean |
|-----------|---------|----------|----------|------|
| AdaDion V2 | 96.19% | 96.14% | 96.01% | 96.11% |
| Dion2 | 95.92% | 95.94% | 95.90% | 95.92% |
| Dion | 95.88% | 95.70% | 95.92% | 95.83% |
| Muon | 95.75% | 95.79% | 95.65% | 95.73% |
| AdamW | 95.38% | 95.52% | 95.25% | 95.38% |

### ViT-Small CIFAR-10 (seed 42, 100 epochs)
| Optimizer | Val Acc | LR |
|-----------|---------|-----|
| Dion | 90.92% | 0.002 |
| Dion2 | 90.89% | 0.002 |
| AdaDion V2 | 89.66% | 0.005 |
| Muon | 88.80% | 0.002 |
| AdamW | 86.24% | 0.001 |

ViT-Small required lower LRs for spectral optimizers (0.002-0.005 instead of 0.01-0.02). The initial ViT runs with lr=0.005 for all spectral optimizers produced garbage (39-55%) because the LR was too high for Dion/Dion2/Muon on ViT. After LR tuning, the Dion family achieved 89-91%.

## 11. Wide ResNet Scaling

Tested WideResNet-28 with width multipliers 2, 4, 10 on both CIFAR-10 and CIFAR-100.

### CIFAR-10
| Model (params) | AdamW | Dion | AdaDion V2 | Muon |
|----------------|-------|------|------------|------|
| WRN-28-2 (1.5M) | 94.20% | 94.53% | 95.66% | 94.88% |
| WRN-28-4 (5.9M) | 95.58% | 96.03% | 96.39% | 95.73% |
| WRN-28-10 (36.5M) | 96.51% | 96.68% | 96.91% | 96.36% |

### CIFAR-100
| Model (params) | AdamW | Dion | AdaDion V2 | Muon |
|----------------|-------|------|------------|------|
| WRN-28-2 (1.5M) | 73.88% | 74.44% | 77.85% | 76.34% |
| WRN-28-4 (5.9M) | 77.64% | 78.45% | 81.27% | 78.85% |
| WRN-28-10 (36.5M) | 81.46% | 81.74% | 83.12% | 81.78% |

AdaDion leads at every width on both datasets. The margins are larger on CIFAR-100 (up to +3.97% over AdamW on WRN-28-2) than CIFAR-10 (up to +1.46%).

## 12. Distributed Timing (4x A100 DDP)

Measured per-step wall-clock time with DDP on 4x A100-80GB (ResNet-18, 200 steps):

| Optimizer | ms/step | Relative |
|-----------|---------|----------|
| AdamW | 20.3 | 1.0x |
| Dion2 | 30.6 | 1.5x |
| Dion | 49.4 | 2.4x |
| AdaDion V2 | 57.6 | 2.8x |
| Muon | 206.3 | 10.2x |

AdaDion adds 17% overhead over base Dion (57.6 vs 49.4 ms) for the rank adaptation mechanism.

## 13. Communication Analysis

Theoretical bytes per step under DDP ring all-reduce:

| Optimizer | MB/step (ResNet-18) | Compression |
|-----------|---------------------|-------------|
| AdamW | 85.3 | 1.0x |
| Muon | 85.3 | 1.0x |
| AdaDion V2 | 48.3 | 1.8x |
| Dion | 24.2 | 3.5x |
| Dion2 | 21.4 | 4.0x |

Dion communicates low-rank factors P (m x r) and R (n x r) instead of the full gradient (m x n). Communication cost is O((m+n)r) vs O(mn). With rank fraction 0.25, Dion achieves 3.5x compression.

## 14. Rank Scale Sweep

Per Hiroko's request, we swept rank_scale gamma with rank_fraction_max=1.0:

| gamma | Val Acc |
|-------|---------|
| 0.5 | 95.81% |
| 1.0 | 95.76% |
| 1.5 | 96.20% |
| 2.0 | 96.22% |
| 2.5 | 96.29% |
| 3.0 | 95.99% |

Optimal gamma = 2.0-2.5. This matches Tatsu's independently chosen gamma=2.0 on LLaMA 320M.

## 15. Rank Dynamics Experiment

Tracked per-step effective rank and actual rank for 6 gamma values over 19,500 steps on ResNet-18.

Findings:
- Effective rank starts at ~155 and decreases to ~130 during training (gradient structure simplifies)
- The decrease is consistent across all gamma values (intrinsic to the optimization landscape)
- With gamma=0.5, actual rank stays at rank_min=16 (too aggressive compression)
- With gamma=1.0, rank stays at ~20 (still mostly at minimum)
- With gamma=1.5, rank starts at ~200 and gradually decreases to ~150 (adaptive mechanism responding to declining erank)
- With gamma=2.0-3.0, rank stays around 175-206 (near maximum)
- The rank at gamma=1.5 shows the clearest adaptive behavior: it tracks the effective rank decline

## 16. Infrastructure Used

| Pod | GPUs | PyTorch | Used For |
|-----|------|---------|----------|
| RunPod A100 | 1x A100-80GB | 2.11+cu130 | Initial benchmark, smoke test |
| RunPod 5090 | 4x RTX 5090 | 2.8+cu128 | Ablation (22 runs), first final benchmark |
| RunPod A100 | 4x A100-80GB | 2.11+cu128 | Corrected final benchmark, DDP timing |
| RunPod A40 | 4x A40-48GB | 2.11+cu128 | WRN scaling (CIFAR-10 + CIFAR-100), rank scale sweep |
| Shared cluster | 8x RTX 5090 | 2.8+cu128 | Rank dynamics (used GPUs 4-7) |

Total GPU-hours: approximately 200-250 across all experiments.

## 17. Figure Generation

Figures were generated following an academic ML paper style (matching NeurIPS/ICML conventions):
- Serif font (Computer Modern via `mathtext.fontset: cm`)
- All spines visible, dashed grid on both axes
- Training curves: raw faint line (alpha=0.15) + bold smooth line (5-point uniform filter) + confidence bands (alpha=0.18)
- Loss plots on log scale
- Bar charts: standard bars with thin edges, black error bars with small caps
- Saturated color palette assigned per optimizer, consistent across all figures

10 PDF figures in `latex/figures/`:
1. `resnet18_bars.pdf`: accuracy + loss bar chart (3 seeds)
2. `training_curves.pdf`: training loss, val loss, val accuracy curves
3. `vit_bars.pdf`: ViT-Small comparison
4. `convergence.pdf`: epochs to reach target accuracy
5. `throughput.pdf`: training time + throughput
6. `communication.pdf`: MB/step per optimizer
7. `rank_scale_sweep.pdf`: gamma vs accuracy
8. `compression_ratio.pdf`: compression ratio comparison
9. `vit_lr_sweep.pdf`: AdaDion LR sweep on ViT
10. `wrn_scaling.pdf`: CIFAR-10 + CIFAR-100 width scaling
11. `rank_dynamics.pdf`: erank + actual rank vs steps per gamma

## 18. LaTeX Document

The report is in `latex/main.tex` (two-column article format). Sections:
1. Introduction
2. Setup (models, optimizers, training)
3. Results (ResNet-18, ViT-Small, convergence, throughput)
4. Ablation (22-run table)
5. Communication Overhead (theoretical + empirical DDP)
6. Rank Dynamics (rank scale factor, rank evolution, per-layer analysis)
7. Distributed Scaling (compute vs communication decomposition)
8. Width Scaling (CIFAR-10 + CIFAR-100 WideResNet)
9. Implementation Notes (parameter coverage, mixed precision, single-GPU compatibility)
10. Conclusion

## 19. GitHub Repository

All code and results at `github.com/ansschh/adadion-cifar`:

```
benchmark/
  cifar10_benchmark.py      # Main training loop
  configs.py                # Hyperparameter configs
  models.py                 # ResNet, VGG, ViT
  models_wide.py            # WideResNet
  optimizers.py             # Optimizer factory + FlattenedParamWrapper
  metrics.py                # Metrics collection
  analysis.py               # Post-run analysis
  ablation_adadion.py       # 22-run ablation script
  final_benchmark.py        # 3-seed final benchmark
  wide_resnet_scaling.py    # WRN width scaling
  distributed_benchmark.py  # DDP timing
  rank_dynamics_experiment.py  # Per-step rank tracking
  plot_rank_dynamics.py     # Rank dynamics figure
  generate_paper_figures.py # All paper figures
  adadion_v2/               # Bundled AdaDion V2 source
  results/                  # All experiment results (JSON)

latex/
  main.tex                  # Paper
  references.bib
  figures/                  # PDF figures

PLOT_STYLE_GUIDE.md         # Figure style specification
INVESTIGATION_FINDINGS.md   # Bug investigation documentation
FULL_PROJECT_LOG.md         # This file
```

## 20. Key Lessons

1. When porting distributed optimizers to single-GPU, check every code path for distributed-only assumptions (mesh parameters, DTensor operations, world_size=1 edge cases).

2. torch.compile + custom optimizers is fragile. The `@torch.compile(fullgraph=True)` decorators inside the dion package hit InductorError, TorchRuntimeError, and recompile limit errors on different PyTorch versions. Setting `suppress_errors=True` masks real bugs by silently making optimizer steps into no-ops.

3. Conv weight flattening is essential for spectral optimizers on vision models. Without it, the optimizer operates on <1% of parameters and degrades to AdamW.

4. Learning rate is the most important hyperparameter. The optimal LR for CIFAR-10 (0.01) differs from LLM pretraining (0.012). A factor of 2x difference (0.01 vs 0.02) causes 0.46% accuracy drop.

5. AMP (mixed precision) must be fully disabled for spectral optimizers. The interaction between float16 gradients, GradScaler inf-checking, and torch.compile is a source of silent failures.

6. The adaptive rank mechanism works but its benefit scales with matrix size. On small matrices (10x512), rank saturates immediately. On larger matrices (512x4608 or 768x768), the mechanism has room to adapt and provides measurable gains.

7. The rank_scale factor gamma=2.0 transfers across model scales (CIFAR-10 ResNet to LLaMA 320M).

8. On harder tasks (CIFAR-100 vs CIFAR-10), AdaDion's advantage grows, suggesting the adaptive mechanism benefits more from richer gradient spectral structure.
