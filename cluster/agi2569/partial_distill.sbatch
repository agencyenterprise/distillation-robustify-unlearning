#!/bin/bash
#SBATCH --job-name=agi2569-partial
#SBATCH --wckey=gradient-routing
#SBATCH --gpus=8
#SBATCH --cpus-per-task=192
#SBATCH --mem-per-gpu=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%A_%a.out
# A sweep of partial-distillation runs, each on 4 GPUs as in the paper (2 runs at a time on a full node).
# Array tasks split the sweep (task i runs every n-th run). Arguments go to run_partial_distill_language.py, e.g.
#   sbatch --array=0-1 cluster/agi2569/partial_distill.sbatch --arms A --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
source cluster/agi2569/env.sh
$PY run_partial_distill_language.py --run_all --gpus_per_run 4 \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
