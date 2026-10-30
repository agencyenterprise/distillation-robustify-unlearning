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
# Arguments: arm alpha beta fixed_steps warmup_frac|none noise_target [noise_spec] [seed=N] [seqs=N]. Pass --time; submit
# from the repo root. noise_spec by noise_target: all: none; neuron_act: the fraction; neuron_rand: the fraction, or
# n<N> for exactly N neurons (--noise_count); neuron_rule: q<q>_r<r> (--p_kor_min q --p_eng_max r). seed=N (default
# 111, the runs' seed) goes to both stages as --seeds N; seqs=N as --saliency_seqs N (neuron_act / neuron_rule; default
# 64; other values add '_s<N>' to the name). E.g. the 2% budget, 5% of neurons by activation saliency:
#   sbatch --time=01:30:00 -J agi2635-fx16-act0.05 cluster/agi2635/fixed_point.sbatch F 1.0 1.0 16 0.1 neuron_act 0.05
# the firing-probability rule and its size-matched random control, seed 112:
#   sbatch --time=02:00:00 cluster/agi2635/fixed_point.sbatch F 1.0 1.0 81 none neuron_rule q0.2_r0.02 seqs=256 seed=112
#   sbatch --time=02:00:00 cluster/agi2635/fixed_point.sbatch F 1.0 1.0 81 none neuron_rand n9 seed=112
# and the no-noise baseline shape: ... fixed_point.sbatch F 0.0 0.1 403 none all
# Defaults: AGI2569_V=4, PRECISION=fp32, MATMUL_PRECISION=high (override with sbatch --export=ALL,...).
export AGI2569_V=${AGI2569_V:-4} PRECISION=${PRECISION:-fp32} MATMUL_PRECISION=${MATMUL_PRECISION:-high}
source cluster/agi2569/env.sh
arm=$1 alpha=$2 beta=$3 steps=$4 wf=$5 nt=$6
shift 6
spec="" seed=111 seqs=""
for a in "$@"; do
    case $a in
        seed=*) seed=${a#seed=} ;;
        seqs=*) seqs=${a#seqs=} ;;
        *) [ -z "$spec" ] || { echo "[agi2635] unexpected argument $a" >&2; exit 2; }; spec=$a ;;
    esac
done
nf="" nc="" q="" r=""
case $nt:$spec in
    all:) ;;
    neuron_rand:n*) nc=${spec#n} ;;
    neuron_act:?*|neuron_rand:?*) nf=$spec ;;
    neuron_rule:q*_r*) q=${spec#q}; q=${q%%_r*}; r=${spec##*_r} ;;
    *) echo "[agi2635] noise_target $nt does not take noise_spec '$spec'" >&2; exit 2 ;;
esac

common=(--arms "$arm" --alphas "$alpha" --betas "$beta" --seeds "$seed" --kor_targets uniform --fixed_steps "$steps")
[ "$wf" = none ] || common+=(--warmup_frac "$wf")
[ "$nt" = all ] || common+=(--noise_target "$nt")
[ -z "$nf" ] || common+=(--noise_frac "$nf")
[ -z "$nc" ] || common+=(--noise_count "$nc")
[ -z "$q" ] || common+=(--p_kor_min "$q" --p_eng_max "$r")
[ -z "$seqs" ] || common+=(--saliency_seqs "$seqs")

NAME=$($PY -c "import sys; from src.utils.run_names import SALIENCY_SEQS, partial_distill_name as n
a = sys.argv[1:]; f = lambda x, t=float: t(x) if x else None
print(n(a[0], float(a[1]), float(a[2]), int(a[7]), 'fixed_steps', int(a[3]), 'uniform', warmup_frac=f(a[4]) if a[4] != 'none'
  else None, noise_target=a[5], noise_frac=f(a[6]), saliency_seqs=f(a[8], int) or SALIENCY_SEQS, noise_count=f(a[9], int),
  p_kor_min=f(a[10]), p_eng_max=f(a[11])))" \
  "$arm" "$alpha" "$beta" "$steps" "$wf" "$nt" "$nf" "$seed" "$seqs" "$nc" "$q" "$r")
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
# Exactly this run's files: its record and noise selection, and its relearn records (another run's name can extend
# this one, e.g. -nt_rand_n9 / -nt_rand_n90, so no '*$NAME*' globs).
ours=(--exclude "*" --include "partial_distill_models/$NAME.txt" --include "partial_distill_models/${NAME}_noise_selection.json"
      --include "relearned_models/relearned_partial_distill_models_$NAME/*")
aws s3 sync "$REC" "$DEST/local_records" "${ours[@]}" --only-show-errors || fail
git rev-parse HEAD | aws s3 cp - "$DEST/fork_commit/$SLURM_JOB_ID.txt" --only-show-errors || fail
mine() { awk -v n="$NAME" '{p = $NF} p == "partial_distill_models/" n ".txt" || p == "partial_distill_models/" n "_noise_selection.json" ||
    index(p, "relearned_models/relearned_partial_distill_models_" n "/") == 1 {k++; s += $(NF-1)} END {print k+0, s+0}'; }
read -r n_local b_local < <(cd "$REC" && find partial_distill_models relearned_models -type f -printf '%s %p\n' | mine)
read -r n_s3 b_s3 < <(aws s3 ls "$DEST/local_records/" --recursive | awk '{k = $4; sub(/^.*\/local_records\//, "", k); print $3, k}' | mine)
echo "[agi2635] verify: local $n_local files / $b_local B, S3 $n_s3 files / $b_s3 B"
if [ "$n_local" -gt 0 ] && [ "$n_local" = "$n_s3" ] && [ "$b_local" = "$b_s3" ]; then
    echo "[agi2635] S3 upload verified"
else
    echo "[agi2635] S3 VERIFY FAILED" >&2; fail
fi
echo "[agi2635] done: $NAME, exit $status"
aws s3 cp "$LOG" "$DEST/logs/" --only-show-errors || fail   # last, so the log is (nearly) complete
exit $status
