# Sourced by every AGI-2569 job. Submit jobs from the repo root, so $SLURM_SUBMIT_DIR is the repo.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

export WORK=/mnt/polished-lake/home/allegralaro/agi-2569
export CACHE_DIR=$WORK/cache          # datasets cache (load_dataset / map); never the shared datasets dir
export DATASET_DIR=$WORK/datasets
export MODEL_DIR=$WORK/models         # models and local_records/ (the JSONL records that go to S3)
export NUM_PROC=24                    # dataset workers; 24 CPUs come with each GPU. Keep it fixed:
                                      # it is part of the datasets cache fingerprint.
export ACCELERATE_MIXED_PRECISION=bf16
# HF_HUB_CACHE is set cluster-wide; do not set HF_HOME or HF_HUB_CACHE here.

# HF_TOKEN (Gemma licence) lives in ~/.env; huggingface_hub reads it from the environment.
if [ -f "$HOME/.env" ]; then set -a; . "$HOME/.env"; set +a; fi

# Training reads only local files. Jobs that download (data.sbatch) set ONLINE=1.
if [ "${ONLINE:-0}" != 1 ]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

UV=$(command -v uv || echo "$HOME/.local/bin/uv")
PY="$UV run --no-sync python"   # setup.sbatch builds .venv once; jobs never re-sync it

echo "[env] $(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} commit=$(git rev-parse --short HEAD) gpus=${CUDA_VISIBLE_DEVICES:-none}"
