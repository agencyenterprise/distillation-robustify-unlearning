# Sourced by every AGI-2569 job. Submit jobs from the repo root, so $SLURM_SUBMIT_DIR is the repo.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

export WORK=/mnt/polished-lake/home/allegralaro/agi-2569
export CACHE_DIR=$WORK/cache          # datasets cache (load_dataset / map); never the shared datasets dir
# Data/model version. AGI2569_V=2 (sbatch --export=ALL,AGI2569_V=2): data v2 (Korean pinned to fineweb-2 v2.0.1) and
# its models; the random init (seeded, identical across versions) is linked from v1.
export AGI2569_V=${AGI2569_V:-1}
case $AGI2569_V in
    1) export DATASET_DIR=$WORK/datasets MODEL_DIR=$WORK/models ;;   # MODEL_DIR also holds local_records/ (to S3)
    2) export DATASET_DIR=$WORK/datasets_v2 MODEL_DIR=$WORK/models_v2
       mkdir -p "$MODEL_DIR"
       [ -e "$MODEL_DIR/random_init_models" ] || ln -s "$WORK/models/random_init_models" "$MODEL_DIR/random_init_models" \
           2>/dev/null || [ -e "$MODEL_DIR/random_init_models" ] ;;
    3) # v3: data v2 with the base and oracle pretrained at the distillation geometry (4 GPUs x 4 x 60 = 960 sequences,
       # ~808 steps; paper App. B.2). models_v3 links them under the standard names, so every later stage finds them.
       export DATASET_DIR=$WORK/datasets_v2 MODEL_DIR=$WORK/models_v3
       mkdir -p "$MODEL_DIR/pretrained_models"
       for l in "random_init_models:$WORK/models/random_init_models" \
                "pretrained_models/gemma-2-0.1B_eng+kor:$WORK/models_v2/pretrained_models/gemma-2-0.1B_eng+kor_g4x4x60" \
                "pretrained_models/gemma-2-0.1B_eng:$WORK/models_v2/pretrained_models/gemma-2-0.1B_eng_g4x4x60"; do
           [ -e "$MODEL_DIR/${l%%:*}" ] || ln -s "${l#*:}" "$MODEL_DIR/${l%%:*}" 2>/dev/null || [ -e "$MODEL_DIR/${l%%:*}" ]
       done ;;
    *) echo "[env] unknown AGI2569_V=$AGI2569_V" >&2; exit 1 ;;
esac
export NUM_PROC=24                    # dataset workers; 24 CPUs come with each GPU. Keep it fixed:
                                      # it is part of the datasets cache fingerprint.
# PRECISION=fp32 runs without mixed precision (fp32 weights; matmul precision from MATMUL_PRECISION, PyTorch's default
# "highest" if unset, as upstream's distillation). Honoured by distill, partial distill and warm_cache; keep the
# default for unlearning and relearning (they load bf16 weights, as upstream).
export PRECISION=${PRECISION:-bf16}
case $PRECISION in
    bf16) export ACCELERATE_MIXED_PRECISION=bf16 ;;
    fp32) export ACCELERATE_MIXED_PRECISION=no ;;
    *) echo "[env] unknown PRECISION=$PRECISION" >&2; exit 1 ;;
esac
# HF_HUB_CACHE is set cluster-wide; do not set HF_HOME or HF_HUB_CACHE here.

# HF_TOKEN (Gemma licence) lives in ~/.env; huggingface_hub reads it from the environment.
if [ -f "$HOME/.env" ]; then set -a; . "$HOME/.env"; set +a; fi

# Training reads only local files. Jobs that download (data.sbatch) set ONLINE=1.
if [ "${ONLINE:-0}" != 1 ]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

UV=$(command -v uv || echo "$HOME/.local/bin/uv")
PY="$UV run --no-sync python"   # setup.sbatch builds .venv once; jobs never re-sync it

echo "[env] $(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none} commit=$(git rev-parse --short HEAD) v=$AGI2569_V precision=$PRECISION matmul=${MATMUL_PRECISION:-default} gpus=${CUDA_VISIBLE_DEVICES:-none}"
