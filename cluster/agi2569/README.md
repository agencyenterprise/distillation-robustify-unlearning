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

## Phase 1: calibration

Reproduce the paper's Fig 4 (MaxEnt, language) and Fig 5 (language) before running anything for arms B-F.
Targets and definitions are in `analysis/agi2569/paper_targets.md`. Submit from the repo root on the login
node, in tmux. Create the log directory first: Slurm does not create it, and a job whose log directory is
missing fails without a message.

The MaxEnt teacher is lr **8e-5**, the paper's (App. B.1 bold, App. C), not upstream's hard-coded 3e-5. It is
`TEACHER_LR` in `src/utils/run_names.py`. Distillation, partial distillation, relearning, `aggregate.py` and
`plot.py` take `--teacher_lr` (3e-5 reproduces upstream), and every path downstream of the teacher encodes it.

Setup and data (68507, running) and pretraining (base 68508 and oracle 68509, held) are already submitted.

```bash
mkdir -p /mnt/polished-lake/home/allegralaro/agi-2569/logs
J=cluster/agi2569
A="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"   # Fig 5 alphas (beta 0.1, seed 111)
base=68508 oracle=68509               # pretraining: base (English + Korean), data-filtered oracle (English only)

# 1. MaxEnt unlearning: upstream's 8-LR sweep (1e-5..8e-5, App. B.1), 1 GPU per LR on one 8-GPU node.
#    The 8e-5 run is the teacher of everything below.
unlearn=$(sbatch --parsable -d afterok:$base $J/unlearn.sbatch)

# 2. Full distillation of the 8e-5 teacher into the random init (Fig 4), 4 GPUs
distill=$(sbatch --parsable -d afterok:$unlearn $J/distill.sbatch)

# 3. Partial distillation, arm A, upstream's early stop (English within 5% of the teacher; at most one epoch),
#    alpha 0.1-0.8, 4 GPUs per run (2 per node)
early=$(sbatch --parsable -d afterok:$unlearn --array=0-1 $J/partial_distill.sbatch --arms A --alphas $A --stop_mode early)

# 4. Relearning, 500 steps on Korean x 10 attacker LRs, 1 GPU per run:
#    references (base, data-filtered, MaxEnt 8e-5, full distill), then arm A
sbatch -d afterok:$oracle:$distill --array=0 $J/relearn.sbatch --groups refs
sbatch -d afterok:$early --array=0-1 $J/relearn.sbatch --groups partial --arms A --alphas $A --stop_modes early

# 5. After each phase (light; login node is fine)
bash $J/sync_s3.sh            # records + logs
bash $J/sync_s3.sh --models   # once the reference models exist
```

Then compare with the paper, locally: `python analysis/agi2569/aggregate.py <records>` (prints the MaxEnt sweep
against App. B.1, then the robustness tables) and `python analysis/agi2569/plot.py <records> --paper` (Fig 4b,
Fig 5a axes, Fig 5b).

Calibration checks, in order:
1. Base English and Korean losses: 4.94 and 3.47.
2. MaxEnt final losses at all 8 LRs against App. B.1 (`maxent_sweep.csv`); 8e-5 should give 4.99 / 12.38.
3. Full distillation length: about 807 steps.
4. The Fig 4b curves.
5. The Fig 5b curves and the Fig 5a points.

Jobs over the 16-GPU weekday cap wait in the queue; Slurm paces them. Preview a relearn list with `--list`.
Reruns are safe: finished runs are skipped, and a crashed run restarts from scratch with its partial record
removed. The unlearning job and each partial-distill sweep job exit non-zero if any of their runs failed, so
`afterok` dependents wait.

## Phase 2: arms B-F (only after Phase 1 matches)

```bash
partial=$(sbatch --parsable --array=0-3 $J/partial_distill.sbatch --arms B C D E F --alphas 0.0 $A)
unigram=$(sbatch --parsable --array=0-1 $J/partial_distill.sbatch --arms D E --kor_targets unigram --alphas 0.0 $A)
sbatch -d afterok:$partial:$unigram --array=0-7 $J/relearn.sbatch --groups partial --arms B C D E F --kor_targets uniform unigram
sbatch --array=0-1 $J/partial_distill.sbatch --arms C --stop_mode early   # the paper's compute axis
```

Arm C at alpha 0.0 ("unlearn, then fine-tune on English") is the extra Fig 4 curve. At alpha 0.0, arms A and B are
no-ops.

## GPU geometry per stage

Conclusion: match upstream at every stage. Each stage below gives the evidence: the paper is arXiv 2506.06278v3,
App. E, and the code is upstream `edbf111`.

| Stage | GPUs per run | Evidence |
|---|---|---|
| Pretraining | 4 | App. E: "Language ... pretraining took 4xH200". Batch 5 x 80 per process (Table 2), so 1600 sequences per step. |
| MaxEnt unlearning | 1 | `run_unlearn_language.py:312` runs one process per GPU; the 8-LR sweep packs one node. The paper does not say otherwise. |
| Full distillation (Fig 4) | 4 | App. E: "All distillation and UNDO runs for language are run on 4xH200 GPUs". Fig 9: the language distillations ran 807 steps, which is one epoch at 4 x 4 x 60 = 960 sequences per step. On 1 GPU the 1000-step cap would stop them at about a third of an epoch. `src/tools/distill.py:271` logs a per-process step. |
| Partial distillation (UNDO) | 4 | App. E (as above). Header of `run_partial_distill_language.py:2`: "Configured for a 4xH200 GPU server". The sweep calls `accelerate launch` with the machine's default config (`:193`). `src/utils/parallel_launch.py:9-41` launches `--multi_gpu --num_processes 4` per run. |
| Relearning | 1 | App. E: "relearning took one H200 GPU per setup"; `run_relearn_language.py:197` runs one process per GPU. Table 4: batch 8, about 8M tokens per run. |

**What 4 processes do to the step and schedule code.** We keep upstream's code, and under 4 processes it behaves
as upstream's did. `tests/smoke_partial_distill_ddp_cpu.py` checks this with 2 processes.

- `total_steps` comes from the unsharded loader before `accelerator.prepare`: one epoch's batches / 60, about 4x
  the optimizer steps each rank takes per epoch, then `min(·, max_steps)`.
- accelerate's prepared scheduler advances `num_processes` ticks per optimizer step, so `total_steps` and the 50
  warmup steps are counted in ticks. That makes 12.5 optimizer steps of warmup.
- Early stop (`max_steps=-1`): the cosine spans the epoch, as intended.
- Fixed 500 steps: the run stops after 500 optimizer steps, but the 500-tick cosine ends at step 125. After that
  the LR cycles between `min_lr` (7e-4) and the peak (9e-4) with a period of 225 steps.
- `--max_steps` in early mode therefore also shortens the schedule. Leave it unset: the epoch is the paper's cap.

**Other DDP properties, all checked by the DDP smoke test:**
- The noised student is identical on every rank: same seed, same RNG sequence, noise applied before the DDP wrap,
  and DDP also broadcasts rank 0's weights.
- Validation and the teacher-baseline evaluation are reduced across ranks, so the early-stop decision agrees on
  every rank.
- Random Korean targets use one generator per rank (seed + 1000 x rank).
- `tokens_seen` and the per-domain counts are gathered from all ranks.
- Records and models are written by rank 0 only.
- The rerun check reads the shared filesystem on every rank before anything is written.
- Rank 0 builds the datasets caches first.

## GPU-hour estimates (H200)

Rough, from FLOP counts at about 100 TFLOP/s effective (bf16, eager attention, 256k vocab) and the AGI-2569
code audit. Check the first run of each kind and update. App. E reports two to three days for pretraining on
4xH200, against our 2-3 h: check the first pretraining log's throughput before trusting the pretraining rows.

| Job | Runs | GPU-h per run | Wall clock per run |
|---|---|---|---|
| setup, data | CPU only | 0 | ~1 h, ~4-8 h |
| pretrain base (2B tokens) | 4 GPUs | 6-12 | 2-3 h |
| pretrain oracle (1B tokens) | 4 GPUs | 3-6 | 1-2 h |
| unlearn MaxEnt, one LR (8 per node) | 1 GPU | 0.3 | 20 min |
| full distill (~807 steps x 960 seqs, ~1.6B tokens) | 4 GPUs | 8-16 | 2-4 h |
| partial, fixed 500 steps (0.98B tokens) | 4 GPUs | 3.2 (CE) - 4.4 (KL) | ~1-1.2 h |
| partial, early stop | 4 GPUs | 4.4 (KL) x steps / 500, plus loading and evaluation | 1.1 h x steps / 500 |
| relearn, one (model, LR) | 1 GPU | 0.25 | 15 min |

| Program | GPU-h |
|---|---|
| **Phase 1**: pretrain 9-18 (submitted), unlearn sweep 8 x 0.3 = ~3, full distill 8-16, arm A early stop at 8 alphas ~30, relearn 12 models x 10 LRs x 0.25 = 30 | **~80-100** (~70-80 after pretraining; ~6-8 h wall after the base model) |
| Phase 2, fixed: arms B-F at alpha 0.0-0.8, D/E with both target kinds (63 runs, ~215) + relearn (630 x 0.25 = ~160) | ~375 |
| Phase 2, arm C early stop: 9 runs (~15) + relearn (90 x 0.25) | ~40 |

Arm A early stop: the paper's stopping steps (Fig 5a) sum to about 1,950. Two array tasks hold a full node each
while their two lanes drain (task 0: alpha 0.1/0.3/0.5/0.7, ~580 steps of wall; task 1: 0.2/0.4/0.6/0.8, ~770),
so about 2 x 8 GPUs x 1.5-2 h = ~30 GPU-h. If every run hits the one-epoch cap (~807 steps), ~60.

Relearning cost is mostly validation (every 5 steps, ~1M tokens), so trimming the attacker LR grid (`--lrs`)
cuts it proportionally.

## Choices to know about

- **Mixed precision.** bf16 autocast everywhere (`ACCELERATE_MIXED_PRECISION=bf16`). Unlearning and relearning
  already load bf16 weights, as upstream.
- **Pretraining packs** documents into 2048-token sequences (`join_or_subsequence=True`), like every later
  stage. Upstream's pretraining call did not run (it passed an unknown `truncate=False`).
- **Datasets cache.** File locks do not work across nodes, so `data.sbatch` builds the JSON-to-Arrow cache
  once. The per-model `map` caches are written by the first job that needs them; two nodes racing on the same
  model write identical files through an atomic rename.
