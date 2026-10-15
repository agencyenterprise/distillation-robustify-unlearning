#!/bin/bash
#SBATCH --job-name=agi2569-pretrain
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=96G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Usage: sbatch cluster/agi2569/pretrain.sbatch gemma-2-0.1B_eng+kor   (base: English + Korean)
#        sbatch cluster/agi2569/pretrain.sbatch gemma-2-0.1B_eng       (data-filtered oracle: English only)
# 4 GPUs (the upstream scripts' documented 4xH200 server); per-GPU batch 5 x 80 accumulation.
source cluster/agi2569/env.sh
SETUP=${1:?setup id}
$UV run --no-sync accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    run_pretrain_language.py --setups "$SETUP"
