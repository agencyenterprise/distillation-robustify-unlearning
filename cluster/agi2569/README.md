# AGI-2569 cluster launchers (language setting)

Slurm jobs for the AGI-2569 ablation: does UNDO need distillation? Arms A-F are described in
`run_partial_distill_language.py`. Every GPU job carries `--wckey=gradient-routing` and an explicit
`--time` and `--mem-per-gpu`.

## Layout

| What | Where |
|---|---|
| Fork checkout (submit from here) | `/mnt/polished-lake/home/allegralaro/agi-2569/distillation-robustify-unlearning` |
| Working storage | `/mnt/polished-lake/home/allegralaro/agi-2569/{cache,datasets,models,logs}` (set in `env.sh`) |
| Records (JSONL) | `$MODEL_DIR/local_records/...`, synced to `s3://ae-gr/agi-2569/local_records/` |
| HF token | `HF_TOKEN` in `~/.env` (needs the Gemma licence); `env.sh` loads it |

The home directory is swept after 30 days without writes, so S3 is the record: run `sync_s3.sh` after
every phase. W&B is off (no account); the local JSONL records are the logs.

Compute nodes have internet: the org guide downloads models and builds venvs on compute nodes. Only
`setup.sbatch`, `data.sbatch` and `sync_s3.sh` use it; training jobs run with `HF_HUB_OFFLINE=1`.

## Submit order

From the repo root on the login node, in tmux. Create the log directory first: Slurm does not create it, and
a job whose log directory is missing fails without a message.

```bash
mkdir -p /mnt/polished-lake/home/allegralaro/agi-2569/logs
J=cluster/agi2569

# 1. Environment and data (CPU only)
setup=$(sbatch --parsable $J/setup.sbatch)
data=$(sbatch --parsable -d afterok:$setup $J/data.sbatch)

# 2. Pretraining: base (English + Korean) and the data-filtered oracle (English only), 4 GPUs each
base=$(sbatch --parsable -d afterok:$data $J/pretrain.sbatch gemma-2-0.1B_eng+kor)
oracle=$(sbatch --parsable -d afterok:$data $J/pretrain.sbatch gemma-2-0.1B_eng)

# 3. MaxEnt unlearning (teacher, lr 3e-5), then full distillation (Fig 4)
unlearn=$(sbatch --parsable -d afterok:$base $J/unlearn.sbatch)
distill=$(sbatch --parsable -d afterok:$unlearn $J/distill.sbatch)

# 4. Partial distillation, fixed 500 steps, alpha 0.0-0.8, beta 0.1, seed 111
#    (7 variants x 9 alphas = 63 single-GPU runs over 2 nodes; E also needs only the base model)
partial=$(sbatch --parsable -d afterok:$unlearn --array=0-1 $J/partial_distill.sbatch --arms A B C D E F)
unigram=$(sbatch --parsable -d afterok:$unlearn --array=0 $J/partial_distill.sbatch --arms D E --kor_targets unigram)

# 5. Relearning attacks: 500 steps on Korean x 10 attacker LRs per model
sbatch -d afterok:$oracle:$distill --array=0 $J/relearn.sbatch --groups refs
sbatch -d afterok:$partial:$unigram --array=0-7%2 $J/relearn.sbatch --groups partial --arms A B C D E F --kor_targets uniform unigram

# 6. After each phase (light; login node is fine)
bash $J/sync_s3.sh            # records + logs
bash $J/sync_s3.sh --models   # once the reference models exist
```

`%2` keeps relearning at 2 nodes (16 GPUs, the weekday cap); raise to `%4` on weekends (cap 32). Preview any
relearn or partial-distill list with `--list` (relearn) before submitting. Reruns are safe: finished runs
are skipped, and a crashed run restarts from scratch with its partial record removed.

Optional phase 2, upstream's stopping rule (English validation loss within 5% of the teacher's), which gives
the paper's compute axis for Fig 5. Cap the steps so a run that never recovers still ends:

```bash
sbatch -d afterok:$unlearn --array=0-1 $J/partial_distill.sbatch --arms A C --stop_mode early --max_steps 2000
sbatch --array=0-3%2 $J/relearn.sbatch --groups partial --arms A C --stop_modes early
```

## GPU-hour estimates (H200)

Rough, from FLOP counts at ~100 TFLOP/s effective (bf16, eager attention, 256k vocab) and the AGI-2569
code audit. Check the first run of each kind and update.

| Job | Runs | GPU-h per run | GPU-h total | Wall clock |
|---|---|---|---|---|
| setup, data | CPU only | 0 | 0 | ~1 h, ~4-8 h |
| pretrain base (2B tokens) | 1 x 4 GPUs | 6-12 | 6-12 | 2-3 h |
| pretrain oracle (1B tokens) | 1 x 4 GPUs | 3-6 | 3-6 | 1-2 h |
| unlearn MaxEnt lr 3e-5 | 1 | 0.3 | 0.3 | 20 min |
| (unlearn LR sweep, optional) | 8 | 0.3 | 3 | 20 min |
| full distill (1000 steps, 0.5B tokens) | 1 | 2-4 | 2-4 | 2-4 h |
| partial distill, fixed 500 steps (0.25B tokens) | 63 | 0.8 (CE) - 1.1 (KL) | 55-70 | 4-5 h on 16 GPUs |
| relearn, references | 40 | 0.25 | 10 | 1.5 h on 8 GPUs |
| relearn, partial | 630 | 0.25 | 160 | 10 h on 16 GPUs |
| **fixed-step program** | | | **~240-270** | **~1.5 days** |
| phase 2: early stop, arms A and C | 18 + 180 relearn | | ~60-80 | |

Relearning dominates. Its cost is mostly validation (every 5 steps, ~1M tokens), so trimming the attacker LR
grid (`--lrs`) after the first arm cuts it proportionally.

## Choices to know about

- **GPUs per run.** Pretraining uses 4 GPUs (upstream documents a 4xH200 server). Partial distillation, full
  distillation, unlearning and relearning run one process per GPU, so the effective batch is 4 x 60 = 240
  sequences per step for both distillations. Upstream's GPU count for the paper's partial distillation is
  unknown; compute is compared through `train/tokens_seen`, which does not depend on it.
- **Mixed precision.** bf16 autocast everywhere (`ACCELERATE_MIXED_PRECISION=bf16`). Unlearning and relearning
  already load bf16 weights, as upstream.
- **Pretraining packs** documents into 2048-token sequences (`join_or_subsequence=True`), like every later
  stage. Upstream's pretraining call did not run (it passed an unknown `truncate=False`).
- **Datasets cache.** File locks do not work across nodes, so `data.sbatch` builds the JSON-to-Arrow cache
  once. The per-model `map` caches are written by the first job that needs them; two nodes racing on the same
  model write identical files through an atomic rename.
