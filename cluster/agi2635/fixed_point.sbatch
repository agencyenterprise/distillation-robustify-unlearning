#!/bin/bash
#SBATCH --job-name=agi2635-point
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-gpu=96G
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2635/logs/%x-%j.out
# AGI-2635: one fixed-compute Fig 7 point (AGI-2569 v4) in one allocation, then its records to S3.
#   1. partial distillation, --stop_mode fixed_steps (4 GPUs, fp32 + matmul "high", as the v4 Fig 7 runs)
#   2. its 10 relearn attacks (eval every 25, bf16, as point.sbatch), --stop_modes fixed_steps with the same budget
#   3. sync this point's records and this job's log to s3://ae-gr/agi-2635/v<V>/ and verify count + bytes
# Unlike point.sbatch, every stop flag is passed once and explicitly (no '--stop_mode early ... --stop_mode
# fixed_steps' override, and no reliance on argparse prefix-matching --stop_mode to the relearn's --stop_modes).
# Arguments: arm alpha beta fixed_steps warmup_frac|none noise_target [noise_frac]. Pass --time; submit from the repo
# root. E.g. the 2% budget, 5% of neurons by activation saliency:
#   sbatch --time=01:30:00 -J agi2635-fx16-act0.05 cluster/agi2635/fixed_point.sbatch F 1.0 1.0 16 0.1 neuron_act 0.05
# and the no-noise baseline shape: ... fixed_point.sbatch F 0.0 0.1 403 none all
# Defaults: AGI2569_V=4, PRECISION=fp32, MATMUL_PRECISION=high (override with sbatch --export=ALL,...).
export AGI2569_V=${AGI2569_V:-4} PRECISION=${PRECISION:-fp32} MATMUL_PRECISION=${MATMUL_PRECISION:-high}
source cluster/agi2569/env.sh
arm=$1 alpha=$2 beta=$3 steps=$4 wf=$5 nt=$6 nf=${7:-}

common=(--arms "$arm" --alphas "$alpha" --betas "$beta" --kor_targets uniform --fixed_steps "$steps")
[ "$wf" = none ] || common+=(--warmup_frac "$wf")
[ "$nt" = all ] || common+=(--noise_target "$nt" --noise_frac "$nf")

NAME=$($PY -c "import sys; from src.utils.run_names import partial_distill_name as n
a = sys.argv[1:]; print(n(a[0], float(a[1]), float(a[2]), 111, 'fixed_steps', int(a[3]), 'uniform',
  warmup_frac=None if a[4] == 'none' else float(a[4]), noise_target=a[5], noise_frac=float(a[6]) if a[6] else None))" \
  "$arm" "$alpha" "$beta" "$steps" "$wf" "$nt" "$nf")
NAME=${NAME#partial_distill_models/}
echo "[agi2635] model $NAME"

status=0
fail() { [ "$status" != 0 ] || status=1; }
echo "[agi2635] partial distillation: fixed_steps $steps, ${common[*]}"
$PY run_partial_distill_language.py --run_all --gpus_per_run 4 --stop_mode fixed_steps "${common[@]}" || status=$?
if [ "$status" = 0 ] && [ -f "$MODEL_DIR/local_records/partial_distill_models/$NAME.txt" ]; then
    echo "[agi2635] relearning: 10 attacker LRs, eval every 25, bf16"
    env -u MATMUL_PRECISION PRECISION=bf16 ACCELERATE_MIXED_PRECISION=bf16 \
        $PY run_relearn_language.py --groups partial --stop_modes fixed_steps --eval_every 25 "${common[@]}" || status=$?
else
    echo "[agi2635] partial stage failed or wrote no record; skipping relearn" >&2; fail
fi

# S3 (always, also after a failure): this point's records (partial record, noise selection, relearn records), the
# fork commit, and this job's log. Model weights stay on the cluster, as for AGI-2569.
DEST=s3://ae-gr/agi-2635/v$AGI2569_V
REC=$MODEL_DIR/local_records
LOG=/mnt/polished-lake/home/allegralaro/agi-2635/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
echo "[agi2635] syncing to $DEST"
aws s3 sync "$REC" "$DEST/local_records" --exclude "*" --include "*$NAME*" --only-show-errors || fail
git rev-parse HEAD | aws s3 cp - "$DEST/fork_commit/$SLURM_JOB_ID.txt" --only-show-errors || fail
read -r n_local b_local < <(cd "$REC" && find partial_distill_models relearned_models -path "*$NAME*" -type f \
    -printf '%s\n' | awk '{n++; s+=$1} END {print n+0, s+0}')
read -r n_s3 b_s3 < <(aws s3 ls "$DEST/local_records/" --recursive | { grep -F "$NAME" || true; } \
    | awk '{n++; s+=$3} END {print n+0, s+0}')
echo "[agi2635] verify: local $n_local files / $b_local B, S3 $n_s3 files / $b_s3 B"
if [ "$n_local" -gt 0 ] && [ "$n_local" = "$n_s3" ] && [ "$b_local" = "$b_s3" ]; then
    echo "[agi2635] S3 upload verified"
else
    echo "[agi2635] S3 VERIFY FAILED" >&2; fail
fi
echo "[agi2635] done: $NAME, exit $status"
aws s3 cp "$LOG" "$DEST/logs/" --only-show-errors || fail   # last, so the log is (nearly) complete
exit $status
