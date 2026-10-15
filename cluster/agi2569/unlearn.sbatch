#!/bin/bash
#SBATCH --job-name=agi2569-unlearn
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# MaxEnt unlearning of the base model. Default: upstream's 8-LR sweep (1e-5..8e-5, App. B.1's rows), one run per GPU
# on a full node. The UNDO teacher is the 8e-5 run. One LR only: sbatch --gpus=1 cluster/agi2569/unlearn.sbatch --lrs 8e-5
# Exits non-zero if any run has no final_model, so afterok dependents wait.
source cluster/agi2569/env.sh
$PY run_unlearn_language.py --methods MaxEnt "$@"
