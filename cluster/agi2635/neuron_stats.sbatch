#!/bin/bash
#SBATCH --job-name=agi2635-neuron-stats
#SBATCH --wckey=gradient-routing
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=96G
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2635/logs/%x-%j.out
# AGI-2635: per-MLP-neuron activation stats (mean |h|, mean h, firing probability) of the v<V> base model on Korean and
# English train sequences (run_neuron_stats_language.py), then the .npz, the fork commit and this job's log to
# s3://ae-gr/agi-2635/v<V>/neuron_stats/, verified (count + bytes). Pass --time; submit from the repo root:
#   sbatch --time=00:45:00 cluster/agi2635/neuron_stats.sbatch [n_seqs, default 256]
# Defaults: AGI2569_V=4, PRECISION=fp32, MATMUL_PRECISION=high (as the v4 partial runs that scored neuron_act).
export AGI2569_V=${AGI2569_V:-4} PRECISION=${PRECISION:-fp32} MATMUL_PRECISION=${MATMUL_PRECISION:-high}
source cluster/agi2569/env.sh
n_seqs=${1:-256}
OUT_DIR=/mnt/polished-lake/home/allegralaro/agi-2635/neuron_stats
mkdir -p "$OUT_DIR"
OUT=$OUT_DIR/base_v${AGI2569_V}_n${n_seqs}.npz

status=0
fail() { [ "$status" != 0 ] || status=1; }
echo "[agi2635] neuron stats: base model, $n_seqs sequences per language -> $OUT"
$PY run_neuron_stats_language.py --n_seqs "$n_seqs" --dtype "$PRECISION" --out "$OUT" || status=$?

# S3 (always, also after a failure): the stats, the fork commit and this job's log.
DEST=s3://ae-gr/agi-2635/v$AGI2569_V/neuron_stats
LOG=/mnt/polished-lake/home/allegralaro/agi-2635/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
echo "[agi2635] syncing $OUT_DIR to $DEST"
aws s3 sync "$OUT_DIR" "$DEST/data" --only-show-errors || fail
git rev-parse HEAD | aws s3 cp - "$DEST/fork_commit/$SLURM_JOB_ID.txt" --only-show-errors || fail
read -r n_local b_local < <(cd "$OUT_DIR" && find . -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n+0, s+0}')
read -r n_s3 b_s3 < <(aws s3 ls "$DEST/data/" --recursive | awk '{n++; s+=$3} END {print n+0, s+0}')
echo "[agi2635] verify: local $n_local files / $b_local B, S3 $n_s3 files / $b_s3 B"
if [ "$n_local" -gt 0 ] && [ "$n_local" = "$n_s3" ] && [ "$b_local" = "$b_s3" ]; then
    echo "[agi2635] S3 upload verified"
else
    echo "[agi2635] S3 VERIFY FAILED" >&2; fail
fi
echo "[agi2635] done: $OUT, exit $status"
aws s3 cp "$LOG" "$DEST/logs/" --only-show-errors || fail   # last, so the log is (nearly) complete
exit $status
