#!/bin/bash
#SBATCH --job-name=agi2569-distill
#SBATCH --wckey=gradient-routing
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Full distillation for Fig 4: MaxEnt teacher (lr 3e-5) into the random init, 1000 steps of 240 sequences.
source cluster/agi2569/env.sh
$PY run_distill_language.py --setups gemma-2-0.1B_MaxEnt
