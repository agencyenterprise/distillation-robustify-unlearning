#!/bin/bash
#SBATCH --job-name=agi2569-sync
#SBATCH --wckey=gradient-routing
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Copies results to s3://ae-gr/agi-2569/ (the system of record; /mnt/polished-lake is swept).
# Light enough for the login node: bash cluster/agi2569/sync_s3.sh [--models]. Or sbatch it.
#   always:   local_records/ (every JSONL record: pretrain, unlearn, distill, partial distill, relearn),
#             logs/, kor_unigram_counts.pt, and the fork commit
#   --models: also the four reference models (base, oracle, MaxEnt teacher, full distill; ~0.5 GB each), for the
#             default teacher LR (src/utils/run_names.py) or the one given: sync_s3.sh --models 3e-5
#   AGI2569_V=2 bash cluster/agi2569/sync_s3.sh syncs the v2 models and records to s3://ae-gr/agi-2569/v2/ instead
#             (same relative paths as v1, so they must not share a prefix).
ONLINE=1 source cluster/agi2569/env.sh
DEST=s3://ae-gr/agi-2569
[ "$AGI2569_V" = 1 ] || DEST=$DEST/v$AGI2569_V
aws sts get-caller-identity > /dev/null && echo "S3 credentials valid" || { echo "S3 credentials INVALID"; exit 1; }

aws s3 sync "$MODEL_DIR/local_records" "$DEST/local_records" --only-show-errors
aws s3 sync "$WORK/logs" "$DEST/logs" --only-show-errors
aws s3 cp "$DATASET_DIR/pretrain/kor_unigram_counts.pt" "$DEST/datasets/kor_unigram_counts.pt" --only-show-errors || true
git rev-parse HEAD | aws s3 cp - "$DEST/fork_commit.txt"

if [ "${1:-}" = "--models" ]; then
    REFS=$($PY -c "import sys; from src.utils.run_names import TEACHER_LR, reference_models
print(' '.join(reference_models(float(sys.argv[1]) if len(sys.argv) > 1 else TEACHER_LR).values()))" ${2:-})
    for m in $REFS; do
        for final in final_model final_student; do
            if [ -d "$MODEL_DIR/$m/$final" ]; then
                aws s3 sync "$MODEL_DIR/$m/$final" "$DEST/models/$m/$final" --only-show-errors
            fi
        done
    done
fi
aws s3 ls "$DEST/" --recursive --summarize | tail -2
