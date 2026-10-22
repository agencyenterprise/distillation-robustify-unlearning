#!/bin/bash
#SBATCH --job-name=agi2569-pretrain-v2
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=120G
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Usage: sbatch cluster/agi2569/pretrain_v2.sbatch [setup id]. See pretrain_v2.py. Outputs to $WORK/models_v2 so the v1
# models are untouched; the random init (seeded, identical to v1's) is linked from $WORK/models.
source cluster/agi2569/env.sh
export DATASET_DIR=$WORK/datasets_v2 MODEL_DIR=$WORK/models_v2 ACCELERATE_MIXED_PRECISION=no
mkdir -p $MODEL_DIR; [ -e $MODEL_DIR/random_init_models ] || ln -s $WORK/models/random_init_models $MODEL_DIR/random_init_models
$UV run --no-sync accelerate launch --num_processes 1 --mixed_precision no cluster/agi2569/pretrain_v2.py "$@"
