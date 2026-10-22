#!/bin/bash
#SBATCH --job-name=agi2569-pretrain-geom
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=120G
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Usage: sbatch --gpus=N --time=... --export=ALL,BATCH=b,GA=g,TAG=t cluster/agi2569/pretrain_geom.sbatch [setup id]
# Data v2, fp32 + TF32, outputs pretrained_models/<setup>_<TAG> in models_v2 (see pretrain_geom.py).
AGI2569_V=2 PRECISION=fp32 source cluster/agi2569/env.sh
N=${SLURM_GPUS_ON_NODE:-1}
MULTI=""; [ "$N" -gt 1 ] && MULTI="--multi_gpu"
$UV run --no-sync accelerate launch $MULTI --num_processes $N --mixed_precision no cluster/agi2569/pretrain_geom.py "$@"
