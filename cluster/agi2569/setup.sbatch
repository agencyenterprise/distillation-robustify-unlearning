#!/bin/bash
#SBATCH --job-name=agi2569-setup
#SBATCH --wckey=gradient-routing
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Builds .venv from uv.lock on a compute node (not the login node). CPU only.
ONLINE=1 source cluster/agi2569/env.sh
$UV sync --locked --python 3.10
$PY -c "import torch, transformers, datasets, accelerate; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
mkdir -p "$CACHE_DIR" "$DATASET_DIR" "$MODEL_DIR"
