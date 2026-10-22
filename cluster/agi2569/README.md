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
removed. The unlearning, partial-distill and relearn jobs exit non-zero if any of their runs failed, so `afterok`
dependents wait.

## Phase 2, Wave 1: arms B, C, D

Approved after calibration: arms B, C and D (uniform Korean targets) at alpha 0.1-0.8 with upstream's early stop, as
arm A in Phase 1 (teacher lr 8e-5, beta 0.1, seed 111), plus arm C at alpha 0 in epoch mode, "unlearn, then CE
fine-tune on English" (the extra Fig 4 curve; early stop would end it at step 1). Then relearning on each, validating
every 25 steps instead of 5. One run per array task. The job shapes on the command line override the sbatch
defaults.

```bash
mkdir -p /mnt/polished-lake/home/allegralaro/agi-2569/logs
J=cluster/agi2569
A="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
PD="--gpus=4 --cpus-per-task=80 --mem-per-gpu=96G"   # one partial-distill run per task: half a node
RL="--gpus=1 --cpus-per-task=20 --mem-per-gpu=96G"   # one relearn run per task

# 0. Datasets caches, built once on CPU before the arrays fan out (loads only, if calibration's caches are intact)
warm_pd=$(sbatch --parsable $J/warm_cache.sbatch run_partial_distill_language.py --arms B C D)
warm_rl=$(sbatch --parsable $J/warm_cache.sbatch run_relearn_language.py)

# 1. Partial distillation, early stop, alpha 0.1-0.8 (task i = alpha i)
pd_b=$(sbatch --parsable -d afterok:$warm_pd $PD --time=03:30:00 --array=0-7 $J/partial_distill.sbatch --arms B --alphas $A --stop_mode early)
pd_c=$(sbatch --parsable -d afterok:$warm_pd $PD --time=03:00:00 --array=0-7 $J/partial_distill.sbatch --arms C --alphas $A --stop_mode early)
pd_d=$(sbatch --parsable -d afterok:$warm_pd $PD --time=04:30:00 --array=0-7 $J/partial_distill.sbatch --arms D --kor_targets uniform --alphas $A --stop_mode early)

# 2. Arm C at alpha 0, 2 epochs of English (see the note below)
pd_c0=$(sbatch --parsable -d afterok:$warm_pd $PD --time=06:00:00 --array=0 $J/partial_distill.sbatch --arms C --alphas 0.0 --stop_mode epoch --epochs 2)

# 3. Relearning, 10 attacker LRs per model, validation every 25 steps (8 alphas: task i = LR i // 8, alpha i % 8)
sbatch -d afterok:$warm_rl:$pd_b $RL --time=01:00:00 --array=0-79 $J/relearn.sbatch --groups partial --arms B --alphas $A --stop_modes early --eval_every 25
sbatch -d afterok:$warm_rl:$pd_c $RL --time=01:00:00 --array=0-79 $J/relearn.sbatch --groups partial --arms C --alphas $A --stop_modes early --eval_every 25
sbatch -d afterok:$warm_rl:$pd_d $RL --time=01:00:00 --array=0-79 $J/relearn.sbatch --groups partial --arms D --kor_targets uniform --alphas $A --stop_modes early --eval_every 25
sbatch -d afterok:$warm_rl:$pd_c0 $RL --time=01:00:00 --array=0-9 $J/relearn.sbatch --groups partial --arms C --alphas 0.0 --stop_modes epoch --epochs 2 --eval_every 25

# 4. When done
bash $J/sync_s3.sh
```

Runs: 25 partial distillations (100 GPUs over 25 tasks), then 250 relearns. Every partial-distill and relearn
job exits non-zero if any of its runs failed or (relearning) has no record reaching step 500, so `afterok`
dependents wait. Rerunning a block line is safe: finished runs are skipped.

**Epoch length of the arm C alpha 0 run.** Full distillation's 808 steps are one epoch of English + Korean
interleaved (~775K packed rows at 960 per step). Arm C trains on English only, so one epoch is about half that:
~404 steps if English is the smaller packed set, at most ~509 (all 1B tokens, which packing does not keep).
`--epochs 2` gives ~808-1,018 steps, the closest to full distillation's step budget, and sees each English row twice;
`--epochs 1` sees each once, the English that full distillation saw. The partial-distill warm job prints the packed
English row count (`eng, seed 111, N rows`): steps per epoch = N / 960. The epoch count is in the run name
(`epochs_2`), so the two never collide.

**Time limits**, from calibration (arm A early stop: about 17 s per optimizer step on 4 GPUs, all in, from 2 min at
alpha 0.1 to 3 h 51 min for 800 steps at alpha 0.8; full distillation 808 steps in 4 h 26 min). CE arms skip the
teacher forward pass, so perhaps 13 s per step (unmeasured; the limits also hold at 17 s). Each limit covers the
arm's one-epoch cap with a margin: B (KL, English only) <=509 steps, C <=509, D (English + Korean) ~808, C at alpha
0 <=1,018. Relearning at 5-step validation took ~50 min; by FLOP count ~90% of that is validation (101 passes over
~3M tokens, each document padded to 2048). At 25 steps that is 21 passes, so an estimated ~15-20 min per run.

**Wave 1 GPU-hours (H200).**

| Job | Runs | GPU-h expected | GPU-h if every run hits its epoch cap |
|---|---|---|---|
| warm caches | 2 CPU jobs | 0 | 0 |
| partial B (KL, English) | 8 x 4 GPUs | ~35 | ~78 |
| partial C (CE, English) | 8 x 4 GPUs | ~27 | ~59 |
| partial D (CE, English + random Korean targets) | 8 x 4 GPUs | ~37 | ~93 |
| partial C, alpha 0, 2 epochs | 1 x 4 GPUs | ~12 | ~15 |
| relearn, 25 models x 10 LRs, eval every 25 | 250 x 1 GPU | ~65-85 | ~210 if eval_every saves nothing |
| **Total** | | **~175-200** | **~310-330** (~455 if eval_every saves nothing) |

"Expected" assumes each arm stops where arm A did (1-800 steps, 2,551 in all), capped at its epoch. At the 16-GPU
weekday cap that is ~11-13 h of wall clock with an empty queue.

## Fixed-compute check

Every run trains a fixed number of optimizer steps with no early stop, at equal compute: CE arms 403 steps (half
of full distillation's 806), KL arms round(403 x 3/4) = 302 (a KL step also runs the teacher forward: 4 units per
token against CE's 3, as `compute_ratio_flops` in `aggregate.py`). Warmup and cosine span exactly those steps.
Arms A, B at alpha 0.6; C, D, E, F at alpha 0.0 and 0.6 (A and B at alpha 0 are the unlearned model). D and E use
uniform Korean targets. One partial job per model and its 10 relearns after it, validating every 20 steps (Fig 7b
reads step 40, the tables step 500).

```bash
mkdir -p /mnt/polished-lake/home/allegralaro/agi-2569/logs
J=cluster/agi2569
PD="--gpus=4 --cpus-per-task=80 --mem-per-gpu=96G --time=02:30:00"   # one partial-distill run per job
RL="--gpus=1 --cpus-per-task=20 --mem-per-gpu=96G --time=00:40:00"   # one relearn run per task

# 0. Datasets caches (loads only, if Wave 1's caches are intact)
warm_pd=$(sbatch --parsable $J/warm_cache.sbatch run_partial_distill_language.py --arms A B C D E F)
warm_rl=$(sbatch --parsable $J/warm_cache.sbatch run_relearn_language.py)

# 1. run <arm> <alpha> <fixed steps>: the partial run, then its 10 relearns (task i = attacker LR i)
run() {
  pd=$(sbatch --parsable -d afterok:$warm_pd $PD --array=0 $J/partial_distill.sbatch        --arms $1 --alphas $2 --kor_targets uniform --stop_mode fixed_steps --fixed_steps $3)
  sbatch -d afterok:$warm_rl:$pd $RL --array=0-9 $J/relearn.sbatch --groups partial        --arms $1 --alphas $2 --kor_targets uniform --stop_modes fixed_steps --fixed_steps $3 --eval_every 20
}
run A 0.6 302; run B 0.6 302
for arm in C D E F; do for alpha in 0.0 0.6; do run $arm $alpha 403; done; done

# 2. When done
bash $J/sync_s3.sh
```

Runs: 10 partial distillations (`...-arm_{A,B}-alpha_0.6-beta_0.1-seed_111-fixed_steps_302`,
`...-arm_{C,D_uniform,E_uniform,F}-alpha_{0.0,0.6}-beta_0.1-seed_111-fixed_steps_403`), then 100 relearns (records
`<lr>_eval_every_20_only_forget_data.txt`). Time limits: 302 KL steps at ~17 s and 403 CE steps at ~13 s are both
~1.5 h with evaluation (403 steps at 17 s, ~2 h); a relearn validating every 20 steps is ~15 min (12.6 at 25).
About 60 partial GPU-h (10 x 4 GPUs x 1.5 h) and 25 relearn GPU-h.

Wave 2: arm E (D from the base model), arm F (C from the base model), arm G (sanity: CE on English + Korean) at two alphas; arm D with unigram targets later.

## v2 Fig 5a/5b points

Data v2 (Korean pinned to fineweb-2 v2.0.1) and its models: base and oracle pretrained by `pretrain_v2.sbatch`
(packed, 1 GPU, full precision). Every job carries `--export=ALL,AGI2569_V=2`, so `env.sh` reads and writes only
`$WORK/datasets_v2` and `$WORK/models_v2` (v1 is untouched; run and record names are v1's, under the v2 root). Partial
distillation runs without mixed precision (`PRECISION=fp32`: `--mixed_precision no`) with TF32 matmuls
(`MATMUL_PRECISION=high`). MaxEnt and relearning stay as they are (they load bf16 weights, as upstream).
Display letters: A = arm A, C = arm D (uniform Korean targets), E = arm E (uniform). Arm E's student is the base,
but every arm loads the MaxEnt teacher (early stop compares English to it), so E also waits for unlearning.

Each point is one `point.sbatch` job on 4 GPUs: the early-stop partial run, then its 10 relearns (eval every 25), one
per GPU. Order is submission order plus priority: references first, then the 8 arm A points (nice 0), then D and E
interleaved by alpha (nice 100), shortest first within each, so backfill can start short points in gaps.

Everything is submitted on hold. Release it once the base checks out (4.94 / 3.47): `scontrol release $(cat $HELD)`.

```bash
mkdir -p /mnt/polished-lake/home/allegralaro/agi-2569/logs
J=cluster/agi2569
base=76534 oracle=76599                               # v2 pretraining: base (English + Korean), oracle (English only)
HELD=/mnt/polished-lake/home/allegralaro/agi-2569/logs/v2_points_jobs.txt; : > $HELD
q() { sbatch --parsable --hold "$@" | tee -a $HELD; }   # submit on hold, record the id
V2="--export=ALL,AGI2569_V=2"
FP32="--export=ALL,AGI2569_V=2,PRECISION=fp32,MATMUL_PRECISION=high"   # point.sbatch resets it for its relearn stage
RL="--gpus=1 --cpus-per-task=20 --mem-per-gpu=96G --time=00:45:00"   # one relearn run per task

# 1. MaxEnt unlearning at lr 8e-5 only (the teacher), 1 GPU
unlearn=$(q $V2 -d afterok:$base --gpus=1 --cpus-per-gpu=8 --mem-per-gpu=96G --time=02:00:00 $J/unlearn.sbatch --lrs 8e-5)

# 2. Datasets caches (partial distillation packs with the teacher's tokenizer, relearning with the base's)
warm_pd=$(q $FP32 -d afterok:$unlearn $J/warm_cache.sbatch run_partial_distill_language.py --arms A D E)
warm_rl=$(q $V2 -d afterok:$base $J/warm_cache.sbatch run_relearn_language.py)

# 3. Reference attacks, 10 attacker LRs (task i = LR i), eval every 25: unlearn-only, data filtering, base
q $V2 $RL -d afterok:$warm_rl:$unlearn --array=0-9 $J/relearn.sbatch --groups refs --refs unlearn --eval_every 25
q $V2 $RL -d afterok:$warm_rl:$oracle  --array=0-9 $J/relearn.sbatch --groups refs --refs filter  --eval_every 25
q $V2 $RL -d afterok:$warm_rl          --array=0-9 $J/relearn.sbatch --groups refs --refs base    --eval_every 25

# 4. Points: point <nice> <time> <arm> <alpha>. Arm A first (nice 0), then D and E interleaved (nice 100)
point() { q $FP32 --nice=$1 --time=$2 -J agi2569-pt-$3-$4 -d afterok:$warm_pd:$warm_rl $J/point.sbatch $3 $4 --kor_targets uniform; }
for p in 0.1=01:45:00 0.2=02:30:00 0.3=03:00:00 0.4=03:30:00 0.5=04:00:00 0.6=04:45:00 0.7=06:30:00 0.8=06:30:00; do
  point 0 ${p#*=} A ${p%=*}
done
for p in 0.1=01:30:00 0.2=02:15:00 0.3=02:45:00 0.4=03:00:00 0.5=03:15:00 0.6=03:45:00 0.7=05:15:00 0.8=05:15:00; do
  point 100 ${p#*=} D ${p%=*}; point 100 ${p#*=} E ${p%=*}
done

wc -l $HELD   # 30 held jobs (arrays count once)

# 5. Later, NOT with the block above. Once the base checks out: scontrol release $(cat $HELD)
#    When done: AGI2569_V=2 bash $J/sync_s3.sh   (to s3://ae-gr/agi-2569/v2/)
```

Runs: 1 MaxEnt, 30 reference relearns, 24 points (24 partial distillations, 240 relearns). **Time limits** per point:
v1's early-stop steps (bf16) x 1.5 + 50, capped at one epoch (808), at 22 s per step for arm A (v1's 17 s x 1.3 for
TF32) and 17 s for D/E (13 s x 1.3; unmeasured), plus 10 min loading and ~60 min of attacks (3 waves of 4 on
4 GPUs). v1 steps per alpha 0.1-0.8: A 1/100/150/200/250/350/700/800, D 1/100/150/150/250/300/500/600, E
-/100/150/200/200/300/450/600. A job that hits its limit loses only its unfinished stage: a rerun skips the finished
partial run and finished attacks. Expected ~235 GPU-h (points ~225: A ~90, D ~68, E ~67; references ~8); ~370 if
every job uses its full limit. Analysis reads v2 from its own copy:
`aws s3 sync s3://ae-gr/agi-2569/v2/local_records <dir>`, then `aggregate.py <dir>` / `plot.py <dir>`. Precision is
not in the run names, so a later bf16 v2 run with the same arguments would find the TF32 model and skip.

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
- Fixed steps: upstream's 500-tick cosine ended at optimizer step 125 of 500 and then cycled between `min_lr`
  (7e-4) and the peak (9e-4). This fork counts fixed mode's `total_steps` as `fixed_steps x num_processes` ticks,
  so the cosine reaches `min_lr` at the last optimizer step on any process count; the warmup stays 50 ticks.
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
| Phase 2, Wave 1 (see its section) | ~175-200 |

Arm A early stop: the paper's stopping steps (Fig 5a) sum to about 1,950. Two array tasks hold a full node each
while their two lanes drain (task 0: alpha 0.1/0.3/0.5/0.7, ~580 steps of wall; task 1: 0.2/0.4/0.6/0.8, ~770),
so about 2 x 8 GPUs x 1.5-2 h = ~30 GPU-h. If every run hits the one-epoch cap (~807 steps), ~60.

Relearning cost is mostly validation (every 5 steps upstream, ~1M real tokens padded to ~3M), so `--eval_every 25`
cuts it ~3x and trimming the attacker LR grid (`--lrs`) proportionally.

## Choices to know about

- **Mixed precision.** bf16 autocast everywhere (`ACCELERATE_MIXED_PRECISION=bf16`) unless a job sets
  `PRECISION=fp32` (distillation and partial distillation, as upstream; see the v2 points). Unlearning and relearning
  already load bf16 weights, as upstream.
- **Pretraining packs** documents into 2048-token sequences (`join_or_subsequence=True`), like every later
  stage. Upstream's pretraining call did not run (it passed an unknown `truncate=False`).
- **Datasets cache.** File locks do not work across nodes, so `data.sbatch` builds the JSON-to-Arrow cache
  once, and `warm_cache.sbatch` builds a stage's packed and validation caches before its array fans out (in
  calibration, relearn tasks building the same cache at once crashed). A packing cache's fingerprint includes the
  tokenizer's path: partial distillation packs with the teacher's tokenizer, and relearning packs every model's
  Korean with the base model's tokenizer (the same Gemma tokenizer), so all relearns share one cache.
