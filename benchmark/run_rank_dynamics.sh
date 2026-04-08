#!/bin/bash
set -euo pipefail
cd /workspace/adadion-cifar/benchmark
RD="./results/rank_dynamics"
mkdir -p "$RD"

echo "=== Rank Dynamics Experiment (6 gammas, 4 GPUs, 50 epochs) ==="
echo "Started: $(date)"

# 6 gammas on 4 GPUs
CUDA_VISIBLE_DEVICES=0 python3 rank_dynamics_experiment.py --gamma 0.5 --epochs 50 --gpu 0 --output_dir "$RD" &
CUDA_VISIBLE_DEVICES=1 python3 rank_dynamics_experiment.py --gamma 1.0 --epochs 50 --gpu 0 --output_dir "$RD" &
CUDA_VISIBLE_DEVICES=2 python3 rank_dynamics_experiment.py --gamma 1.5 --epochs 50 --gpu 0 --output_dir "$RD" &
CUDA_VISIBLE_DEVICES=3 python3 rank_dynamics_experiment.py --gamma 2.0 --epochs 50 --gpu 0 --output_dir "$RD" &
wait

CUDA_VISIBLE_DEVICES=0 python3 rank_dynamics_experiment.py --gamma 2.5 --epochs 50 --gpu 0 --output_dir "$RD" &
CUDA_VISIBLE_DEVICES=1 python3 rank_dynamics_experiment.py --gamma 3.0 --epochs 50 --gpu 0 --output_dir "$RD" &
wait

# Combine all
python3 -c "
import json, glob, os
combined = {}
for d in sorted(glob.glob('$RD/gamma_*/rank_dynamics.json')):
    gamma = os.path.basename(os.path.dirname(d)).replace('gamma_','')
    with open(d) as f:
        combined[gamma] = json.load(f)
with open('$RD/all_dynamics.json','w') as f:
    json.dump(combined, f)
print(f'Combined {len(combined)} gammas')
"

echo "=== ALL DONE: $(date) ==="
