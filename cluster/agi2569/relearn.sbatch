#!/bin/bash
#SBATCH --job-name=agi2569-relearn
#SBATCH --wckey=gradient-routing
#SBATCH --gpus=8
#SBATCH --cpus-per-task=192
#SBATCH --mem-per-gpu=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%A_%a.out
# Relearning attacks (500 steps on Korean, one run per (model, LR)), packed one per GPU on a full node.
# Array tasks split the (model, LR) list. Arguments go to run_relearn_language.py, e.g.
#   sbatch --array=0-1 cluster/agi2569/relearn.sbatch --groups refs
#   sbatch --array=0-7%2 cluster/agi2569/relearn.sbatch --groups partial --arms A B C D E F
# Preview the list on the login node first: ... run_relearn_language.py <same args> --list
source cluster/agi2569/env.sh
$PY run_relearn_language.py \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
