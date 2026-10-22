#!/bin/bash
#SBATCH --job-name=agi2569-distill
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Full distillation for Fig 4: MaxEnt teacher (lr 8e-5 by default; --teacher_lr 3e-5 for upstream's) into the
# random init, on 4 GPUs as in the paper (960 sequences per step; one epoch, ~800 steps, ends before the 1000-step cap).
# bf16 autocast unless PRECISION=fp32 (env.sh).
source cluster/agi2569/env.sh
$UV run --no-sync accelerate launch --multi_gpu --num_processes 4 --mixed_precision "$ACCELERATE_MIXED_PRECISION" \
    run_distill_language.py --setups gemma-2-0.1B_MaxEnt "$@"
