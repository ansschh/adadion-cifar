# CIFAR-10 Optimizer Benchmark Suite

Comprehensive benchmarking of Dion1, Dion2, AdaDion V2, Muon, and AdamW on CIFAR-10.

## Quick Start (RunPod)

```bash
bash setup_runpod.sh
bash run_benchmark.sh
```

## Structure

- `cifar10_benchmark.py` — Main benchmark runner
- `models.py` — Model architectures (ResNet, VGG, ViT)  
- `optimizers.py` — Unified optimizer factory with hybrid param grouping
- `metrics.py` — Metrics collection and logging
- `analysis.py` — Post-run analysis and visualization
- `configs.py` — Hyperparameter configurations and sweep grids
- `setup_runpod.sh` — RunPod environment setup
- `run_benchmark.sh` — Full benchmark launch script
