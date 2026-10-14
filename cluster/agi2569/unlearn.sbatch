#!/bin/bash
#SBATCH --job-name=agi2569-unlearn
#SBATCH --wckey=gradient-routing
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# MaxEnt unlearning of the base model. Default: the UNDO teacher only (lr 3e-5, the rate upstream hard-codes).
# The full upstream sweep (8 LRs, one per GPU): sbatch --gpus=8 cluster/agi2569/unlearn.sbatch --lrs 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5
source cluster/agi2569/env.sh
if [ $# -eq 0 ]; then set -- --lrs 3e-5; fi
$PY run_unlearn_language.py --methods MaxEnt "$@"
