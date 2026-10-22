#!/bin/bash
#SBATCH --job-name=agi2569-point
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-gpu=96G
#SBATCH --time=06:30:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# One data point in one allocation: ONE early-stop partial-distillation run (4 GPUs, as partial_distill.sbatch), then
# that model's 10 relearn attacks (eval every 25), one per GPU, 4 at a time (as relearn.sbatch). Scheduling only: the
# same scripts and arguments as the two separate jobs. Arguments: arm, alpha, then flags both scripts take, e.g.
#   sbatch --export=ALL,AGI2569_V=2,PRECISION=fp32,MATMUL_PRECISION=high cluster/agi2569/point.sbatch D 0.6 --kor_targets uniform
# PRECISION/MATMUL_PRECISION apply to the partial stage only; the relearn stage runs with relearn.sbatch's defaults
# (bf16 mixed precision, PyTorch's matmul precision). Each stage skips finished runs, so a rerun picks up where it
# stopped. Exits non-zero if the partial run fails or any attack has no record reaching step 500.
source cluster/agi2569/env.sh
arm=$1 alpha=$2; shift 2

echo "[point] partial distillation: arm $arm, alpha $alpha, early stop $*"
$PY run_partial_distill_language.py --run_all --gpus_per_run 4 --arms "$arm" --alphas "$alpha" --stop_mode early "$@"

echo "[point] relearning: 10 attacker LRs, eval every 25, bf16 (as relearn.sbatch)"
env -u MATMUL_PRECISION PRECISION=bf16 ACCELERATE_MIXED_PRECISION=bf16 \
    $PY run_relearn_language.py --groups partial --arms "$arm" --alphas "$alpha" --stop_modes early --eval_every 25 "$@"
echo "[point] done: arm $arm, alpha $alpha"
