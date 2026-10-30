"""
Partial distillation (UNDO) on the language setting, with the AGI-2569 ablation arms.

Every arm noises the student once, (1 - alpha) * theta + alpha * beta * N, then retrains it:
  A  KL to the teacher on English + Korean (upstream UNDO; the default)
  B  KL to the teacher on English only
  C  CE on real labels, English only (no teacher forward pass)
  D  CE: English rows on real labels, Korean rows on real inputs with random targets (--kor_targets)
  E  as D, but the student starts from the pretrained base model instead of the MaxEnt-unlearned one
  F  as C, but the student starts from the pretrained base model (no unlearning)
  G  CE on real labels, English + Korean (sanity check: should relearn Korean)
The teacher is always the MaxEnt-unlearned model, at --teacher_lr (default 8e-5, the paper's; upstream hard-coded
3e-5). The LR is part of every run name. In C-F it is evaluated once, as the reference for the
early-stop condition, and never run during training.

Stopping: --stop_mode fixed_steps (default; exactly --fixed_steps optimizer steps, default 500 as in the paper's relearn
model names, crossing epochs if needed; warmup + cosine to min_lr over exactly those steps on any GPU count),
--stop_mode early (upstream: stop once English validation loss is within 5% of the teacher's; at most one epoch), or
--stop_mode epoch (train --epochs full passes, default 1, with no early stop; the cosine spans them, as early mode's
spans its epoch). Epoch mode is for runs that early stop would end at once, e.g. arm C at alpha 0.
--stop_rule (early mode): 'pooled' (default) tests English loss over the whole validation set; 'any_shard' is upstream's
rule, stopping when any rank's own validation shard is within 5% of the teacher on that shard. any_shard runs get a
'-stop_any_shard' name suffix and log every shard's loss at each check.
--warmup_frac (fixed_steps mode): warm up over max(1, round(warmup_frac x fixed_steps)) optimizer steps instead of the
setup's 50 scheduler ticks (12.5 optimizer steps on 4 GPUs, longer than a very short run), then cosine to min_lr at the
last step. Runs get a '-warmup_frac_<f>' name suffix; unset, names and schedule are unchanged.
--noise_target (AGI-2635): 'all' (default) noises every 2-D parameter. 'neuron_act' / 'neuron_rand' noise only
round(--noise_frac x all MLP neurons) neurons (gate/up_proj rows, down_proj column), the top ones by mean |down_proj
input| on Korean / English train sequences (--saliency_seqs of each, from the student before noising) or a random
set (seeded by --seed); the whole model is then retrained as usual. Runs get a '-nt_act_f<f>' / '-nt_rand_f<f>'
name suffix, and the selection is written next to the run's record as <record>_noise_selection.json.
'neuron_rand' with --noise_count N instead of --noise_frac noises exactly N random neurons ('-nt_rand_n<N>'; size-matched
controls). 'neuron_rule' noises every neuron with firing probability P_kor >= --p_kor_min and P_eng <= --p_eng_max on
--saliency_seqs Korean / English train sequences (p_fire as run_neuron_stats_language.py; '-nt_rule_q<q>_r<r>').
--dry_select: only that selection (single-experiment flags; no training, nothing written), printing its size.

GPUs: the paper ran every language UNDO run on 4 H200s (arXiv 2506.06278, App. E): per-process batch 4 x 60
accumulation, so 960 sequences per optimizer step. --gpus_per_run 4 reproduces this; the step and schedule
code is upstream's, and under 4 processes it behaves as upstream's did (see cluster/agi2569/README.md).

Launch Command (Single):     accelerate launch --multi_gpu --num_processes 4 run_partial_distill_language.py --arm C --alpha 0.3 --beta 0.1 --seed 111
Launch Command (Sweep):      python run_partial_distill_language.py --run_all --gpus_per_run 4 --arms A --alphas 0.1 0.2
"""

from src.tools.partial_distill_langarith import load_tokenizer, load_train_dataset, partial_distill, select_mlp_neurons
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, USE_WANDB, WANDB_API_KEY
from src.utils.run_names import (
    ARMS, EPOCHS, KOR_TARGETS, NOISE_TARGETS, SALIENCY_SEQS, STOP_MODES, STOP_RULES, SWEEP_ALPHAS, SWEEP_BETAS, SWEEP_SEEDS, FIXED_STEPS, TEACHER_LR,
    noise_suffix, partial_distill_name, reference_models, warmup_optimizer_steps,
)
from accelerate import Accelerator
from utils.loss_functions import print_acc, custom_login
from src.utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_accelerate_groups
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

# MATMUL_PRECISION (env): torch.set_float32_matmul_precision for fp32 runs. Unset keeps PyTorch's "highest" (no TF32),
# which is what upstream's distillation ran without mixed precision; "high" allows TF32. Only fp32 matmuls are affected.
if os.environ.get("MATMUL_PRECISION"):
    torch.set_float32_matmul_precision(os.environ["MATMUL_PRECISION"])

# Configuration constants
SETUPS_TO_RUN = ["gemma-2-0.1B_MaxEnt"]
OVERWRITE_OK = True

# Early stopping configuration
STOP_CONDITION = "retain_only"  # Options: "retain_only", "forget_only", "both"
RETAIN_THRESHOLD = 0.05  # Threshold for English (retain domain)
FORGET_THRESHOLD = None  # Threshold for Korean (forget domain)

# Initialize parameters
custom_login()

ENG_TRAIN_FILE = f"{DATASET_DIR}/pretrain/train_eng.jsonl"
KOR_TRAIN_FILE = f"{DATASET_DIR}/pretrain/train_kor.jsonl"
KOR_UNIGRAM_COUNTS = f"{DATASET_DIR}/pretrain/kor_unigram_counts.pt"  # from src/prepare_data/kor_unigram.py


def model_path(role, teacher_lr=TEACHER_LR):
    return f"{MODEL_DIR}/{reference_models(teacher_lr)[role]}/final_model"

# What each arm trains on and with which loss. 'student': which reference model the student starts from.
ARM_CONFIGS = {
    #       train files (and domain names)                 loss   Korean targets  student
    'A': {'domains': ['eng', 'kor'], 'loss_type': 'kl', 'random_kor': False, 'student': 'unlearn'},
    'B': {'domains': ['eng'],        'loss_type': 'kl', 'random_kor': False, 'student': 'unlearn'},
    'C': {'domains': ['eng'],        'loss_type': 'ce', 'random_kor': False, 'student': 'unlearn'},
    'D': {'domains': ['eng', 'kor'], 'loss_type': 'ce', 'random_kor': True,  'student': 'unlearn'},
    'E': {'domains': ['eng', 'kor'], 'loss_type': 'ce', 'random_kor': True,  'student': 'base'},
    'F': {'domains': ['eng'],        'loss_type': 'ce', 'random_kor': False, 'student': 'base'},
    'G': {'domains': ['eng', 'kor'], 'loss_type': 'ce', 'random_kor': False, 'student': 'unlearn'},
}
TRAIN_FILES = {'eng': ENG_TRAIN_FILE, 'kor': KOR_TRAIN_FILE}

setups = {
    "gemma-2-0.1B_MaxEnt": {
        'teacher_model_name': None,  # set per run from --teacher_lr
        'student_model_name': None,
        'eng_train_file'    : ENG_TRAIN_FILE,
        'kor_train_file'    : KOR_TRAIN_FILE,
        'eng_valid_file'    : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'    : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'        : f"{MODEL_DIR}/model_name",
        'cache_dir'         : CACHE_DIR,  # upstream: "hf_cache"
        'dataset_cache_dir' : CACHE_DIR,

        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 4,
        'gradient_accumulation_steps' : 60,
        'epochs'                      : 1,
        'learning_rate'               : 9e-4,
        'max_steps'                   : -1,
        'num_warmup_steps'            : 50,
        'validation_steps'            : 50,
        'save_checkpoint_steps'       : 9999,
        'scheduler_type'              : "cosine",
        'min_lr'                      : 7e-4,
        'weight_decay'                : 0,
        'gradient_clipping_threshold' : 1.0,
        'max_length'                  : 2048,

        'use_wandb'        : USE_WANDB,
        'wandb_project'    : "gemma-2-0.1B_eng+kor_ga_distill",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/model_name.txt",

        'shrink_perturb_repeat' : False
    }
}

def korean_english_stop_cond_fn(student_eval_dict, teacher_eval_dict, stop_condition=STOP_CONDITION, retain_threshold=RETAIN_THRESHOLD, forget_threshold=FORGET_THRESHOLD):
    """Stop condition function for Korean and English evaluation"""
    # English is the retain domain
    student_eng_loss = student_eval_dict.get("val/eng_ce_loss", float('inf'))
    teacher_eng_loss = teacher_eval_dict.get("val/eng_ce_loss", 0)

    # Korean is the forget domain
    student_kor_loss = student_eval_dict.get("val/kor_ce_loss", float('inf'))
    teacher_kor_loss = teacher_eval_dict.get("val/kor_ce_loss", 0)

    # Calculate the relative difference in losses
    eng_diff = (student_eng_loss - teacher_eng_loss) / max(teacher_eng_loss, 1e-6)
    kor_diff = (student_kor_loss - teacher_kor_loss) / max(teacher_kor_loss, 1e-6)

    # Apply stopping condition based on configuration
    if stop_condition == "retain_only":
        # Stop if the student is within threshold of the teacher on English only
        return eng_diff < retain_threshold
    elif stop_condition == "forget_only":
        # Stop if the student is within threshold of the teacher on Korean only
        return kor_diff < forget_threshold
    else:  # "both"
        # Stop if the student is within threshold of the teacher on both languages
        return eng_diff < retain_threshold and kor_diff < forget_threshold

def any_shard_stop_cond_fn(accelerator):
    """Upstream's rule: each rank tests its own validation shard (student and teacher on the same shard) with
    korean_english_stop_cond_fn; partial_distill then stops if any rank says stop (its gather + sum > 0, as upstream).
    Needs eval_fn built with per_shard=True. Rank 0 logs every shard's English loss and verdict."""
    rank, n = accelerator.process_index, accelerator.num_processes

    def shard_view(d, i):
        return {"val/eng_ce_loss": d[f"val/eng_ce_loss_shard{i}"], "val/kor_ce_loss": d[f"val/kor_ce_loss_shard{i}"]}

    def stop_cond(student_eval_dict, teacher_eval_dict):
        verdicts = [korean_english_stop_cond_fn(shard_view(student_eval_dict, i), shard_view(teacher_eval_dict, i))
                    for i in range(n)]
        shards = ", ".join(
            f"shard{i} {student_eval_dict[f'val/eng_ce_loss_shard{i}']:.6f}/{teacher_eval_dict[f'val/eng_ce_loss_shard{i}']:.6f}"
            f"{' STOP' if v else ''}" for i, v in enumerate(verdicts))
        print_acc(f"[stop_rule any_shard] step {student_eval_dict.get('train/step')}: English student/teacher "
                  f"pooled {student_eval_dict['val/eng_ce_loss']:.6f}/{teacher_eval_dict['val/eng_ce_loss']:.6f}; "
                  f"{shards}", accelerator.is_main_process)
        return verdicts[rank]
    return stop_cond

def never_stop_cond_fn(student_eval_dict, teacher_eval_dict):
    """fixed_steps mode: train for exactly max_steps. epoch mode: for exactly the epochs."""
    return False

def run_experiment(setup_id, alpha, beta, seed=None, arm='A', kor_targets='uniform', stop_mode='fixed_steps',
                   fixed_steps=FIXED_STEPS, max_steps=None, teacher_lr=TEACHER_LR, epochs=EPOCHS, stop_rule='pooled',
                   warmup_frac=None, noise_target='all', noise_frac=None, saliency_seqs=SALIENCY_SEQS,
                   noise_count=None, p_kor_min=None, p_eng_max=None):
    """Run a single experiment. max_steps caps early-stop runs (default: none, as upstream)."""
    current_setup = setups[setup_id].copy()
    arm_config = ARM_CONFIGS[arm]

    # Use provided seed or keep default
    if seed is not None:
        current_setup['seed'] = seed

    model_name = partial_distill_name(arm, alpha, beta, current_setup['seed'], stop_mode, fixed_steps, kor_targets,
                                      teacher_lr, epochs, stop_rule, warmup_frac, noise_target, noise_frac,
                                      saliency_seqs, noise_count, p_kor_min, p_eng_max)
    current_setup['output_dir'] = current_setup['output_dir'].replace('model_name', model_name)
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('model_name', model_name)
    current_setup['teacher_model_name'] = model_path('unlearn', teacher_lr)
    current_setup['student_model_name'] = model_path(arm_config['student'], teacher_lr)
    current_setup['noise_alpha'] = alpha
    current_setup['noise_beta'] = beta
    if stop_mode == 'fixed_steps':
        # Exactly fixed_steps optimizer steps, crossing into further epochs if one is shorter (the loop stops at
        # max_steps), with the cosine over exactly those steps (schedule_over_max_steps).
        current_setup['max_steps'] = fixed_steps
        current_setup['epochs'] = 1000
    elif stop_mode == 'epoch':
        # No step cap (max_steps -1): total_steps, and so the cosine, span all the epochs, as in early mode.
        if max_steps is not None:
            raise ValueError("--max_steps does not apply to --stop_mode epoch (it would also shorten the schedule)")
        current_setup['epochs'] = epochs
    else:
        # Early stop: --epochs caps the run (and the cosine spans them); 1 is upstream's cap.
        current_setup['epochs'] = epochs
        if max_steps is not None:
            current_setup['max_steps'] = max_steps

    # Update wandb run name
    if current_setup['wandb_run_name'] is None:
        current_setup['wandb_run_name'] = model_name.split('/')[-1]

    accelerator = Accelerator()

    if os.path.exists(os.path.join(current_setup['output_dir'], 'final_model', 'config.json')):
        print_acc(f"[run_partial_distill_language.py] {model_name} already has a final_model; skipping.", accelerator.is_main_process)
        return
    # A rerun after a crash starts over, so drop the partial record rather than appending to it.
    if accelerator.is_main_process and os.path.exists(current_setup['path_local_record']):
        os.remove(current_setup['path_local_record'])

    # Get evaluation function for Korean and English (rank 0 builds the datasets cache first)
    with accelerator.main_process_first():
        eval_fn = get_korean_and_english_evalaution_fn(
            model_name=current_setup['student_model_name'],
            max_length=current_setup['max_length'],
            eng_valid_file=current_setup['eng_valid_file'],
            kor_valid_file=current_setup['kor_valid_file'],
            dataset_cache_dir=current_setup['dataset_cache_dir'],
            cache_dir=current_setup['cache_dir'],
            batch_size=current_setup['batch_size'],
            accelerator=accelerator,
            per_shard=(stop_rule == 'any_shard'),
        )

    if stop_mode != 'early':
        stop_cond_fn = never_stop_cond_fn
    elif stop_rule == 'any_shard':
        stop_cond_fn = any_shard_stop_cond_fn(accelerator)
    else:
        stop_cond_fn = korean_english_stop_cond_fn

    domains = arm_config['domains']
    partial_distill(
        teacher_model_name=current_setup['teacher_model_name'],
        student_model_name=current_setup['student_model_name'],
        train_files=[TRAIN_FILES[d] for d in domains],
        interleave_probs=[1.0 / len(domains)] * len(domains),  # Equal probabilities for eng and kor
        stopping_strategy='first_exhausted',  # Add missing parameter
        join_or_subsequence=True,
        eval_fn=eval_fn,
        stop_cond_fn=stop_cond_fn,
        accelerator=accelerator,
        output_dir=current_setup['output_dir'],
        cache_dir=current_setup['cache_dir'],
        dataset_cache_dir=current_setup['dataset_cache_dir'],
        seed=current_setup['seed'],
        device=current_setup['device'],
        batch_size=current_setup['batch_size'],
        gradient_accumulation_steps=current_setup['gradient_accumulation_steps'],
        epochs=current_setup['epochs'],
        learning_rate=current_setup['learning_rate'],
        max_steps=current_setup['max_steps'],
        num_warmup_steps=current_setup['num_warmup_steps'],
        validation_steps=current_setup['validation_steps'],
        save_checkpoint_steps=current_setup['save_checkpoint_steps'],
        scheduler_type=current_setup['scheduler_type'],
        min_lr=current_setup['min_lr'],
        weight_decay=current_setup['weight_decay'],
        gradient_clipping_threshold=current_setup['gradient_clipping_threshold'],
        max_length=current_setup['max_length'],
        use_wandb=current_setup['use_wandb'],
        wandb_project=current_setup['wandb_project'],
        wandb_run_name=current_setup['wandb_run_name'],
        use_local_record=current_setup['use_local_record'],
        path_local_record=current_setup['path_local_record'],
        noise_alpha=current_setup['noise_alpha'],
        noise_beta=current_setup['noise_beta'],  # Add beta parameter here
        shrink_perturb_repeat=current_setup['shrink_perturb_repeat'],
        overwrite_ok=OVERWRITE_OK,
        loss_type=arm_config['loss_type'],
        domain_names=domains,
        random_target_domain='kor' if arm_config['random_kor'] else None,
        random_target_counts_path=KOR_UNIGRAM_COUNTS if (arm_config['random_kor'] and kor_targets == 'unigram') else None,
        schedule_over_max_steps=(stop_mode == 'fixed_steps'),
        warmup_optimizer_steps=None if warmup_frac is None else warmup_optimizer_steps(warmup_frac, fixed_steps),
        noise_target=noise_target,
        noise_frac=noise_frac,
        noise_count=noise_count,
        p_kor_min=p_kor_min,
        p_eng_max=p_eng_max,
        saliency_seqs=saliency_seqs,
        # Forget, retain: train files (valid_kor is the relearn attack's eval set), loaded here for every arm.
        saliency_files=(KOR_TRAIN_FILE, ENG_TRAIN_FILE),
        # The local_records copy is what sync_s3.sh uploads; the output_dir copy sits with the model.
        noise_record_paths=(current_setup['path_local_record'][:-len('.txt')] + '_noise_selection.json',
                            os.path.join(current_setup['output_dir'], 'noise_selection.json')),
    )

def arrow_files():
    """The datasets cache's Arrow files, to report what --warm_cache built."""
    return {os.path.join(d, f) for d, _, files in os.walk(CACHE_DIR) for f in files if f.endswith(".arrow")}

def warm_cache(args):
    """Build (or load) the datasets caches the sweep's runs read, in one process, then return. Run it once before
    fanning out: file locks do not work across nodes. Packing depends on the teacher's tokenizer and the arm's train
    files; interleaving also on the seed; validation on neither."""
    before = arrow_files()
    current_setup = setups[SETUPS_TO_RUN[0]]
    teacher = model_path('unlearn', args.teacher_lr)
    accelerator = Accelerator(cpu=True)  # only builds datasets; nothing runs on it
    get_korean_and_english_evalaution_fn(
        model_name=teacher, max_length=current_setup['max_length'],
        eng_valid_file=current_setup['eng_valid_file'], kor_valid_file=current_setup['kor_valid_file'],
        dataset_cache_dir=current_setup['dataset_cache_dir'], cache_dir=current_setup['cache_dir'],
        batch_size=current_setup['batch_size'], accelerator=accelerator)
    tokenizer = load_tokenizer(teacher, current_setup['cache_dir'])
    domain_sets = {(tuple(ARM_CONFIGS[arm]['domains']), seed) for arm in args.arms for seed in args.seeds}
    if getattr(args, 'noise_target', 'all') in ('neuron_act', 'neuron_rule'):  # saliency reads each train file on its own (seed unused for one file)
        domain_sets |= {(('kor',), seed) for seed in args.seeds} | {(('eng',), seed) for seed in args.seeds}
    for domains, seed in sorted(domain_sets):
        train_ds = load_train_dataset([TRAIN_FILES[d] for d in domains], tokenizer, current_setup['max_length'], True,
                                      [1.0 / len(domains)] * len(domains), seed, 'first_exhausted',
                                      current_setup['dataset_cache_dir'])
        print(f"[run_partial_distill_language.py] datasets cache ready: {'+'.join(domains)}, seed {seed}, "
              f"{len(train_ds)} rows")
    print(f"[warm_cache] {len(arrow_files() - before)} new cache files in {CACHE_DIR} (0: every cache existed)")

def experiment_cli_args(experiment):
    (setup_id, alpha, beta, seed, arm, kor_targets, stop_mode, fixed_steps, max_steps, teacher_lr, epochs, stop_rule,
     warmup_frac, noise_target, noise_frac, saliency_seqs, noise_count, p_kor_min, p_eng_max) = experiment
    cmd = ["--setup", setup_id, "--alpha", str(alpha), "--beta", str(beta), "--seed", str(seed), "--arm", arm,
           "--kor_targets", kor_targets, "--stop_mode", stop_mode, "--fixed_steps", str(fixed_steps),
           "--teacher_lr", str(teacher_lr), "--epochs", str(epochs), "--stop_rule", stop_rule]
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    if warmup_frac is not None:
        cmd.extend(["--warmup_frac", str(warmup_frac)])
    if noise_target != 'all':
        cmd.extend(["--noise_target", noise_target, "--saliency_seqs", str(saliency_seqs)])
        for flag, value in (("--noise_frac", noise_frac), ("--noise_count", noise_count),
                            ("--p_kor_min", p_kor_min), ("--p_eng_max", p_eng_max)):
            if value is not None:
                cmd.extend([flag, str(value)])
    return cmd

def launch_single_experiment(experiment):
    """Launch a single experiment in a separate process using accelerate"""
    print(f"\n{'='*80}")
    print(f"LAUNCHING EXPERIMENT: {experiment}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Build the command to run this same script with arguments to run a single experiment
    cmd = ["accelerate", "launch", sys.argv[0]] + experiment_cli_args(experiment)

    # Run the command as a separate process
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY: {experiment}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT FAILED: {experiment}")
        print(f"  Error: {e}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        return False

def dry_select(args):
    """--dry_select: the targeted-noise selection run_experiment would make for the single-experiment flags (--arm's
    student, --seed, --noise_* flags), on one process (CUDA if available), with the same tokenizer, train files,
    packing, batch size and model loading as partial_distill. Prints the selection's size; trains and writes nothing
    unless --dry_select_out is given (the selection JSON)."""
    from transformers import AutoModelForCausalLM
    setup = setups[args.setup]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = model_path('unlearn', args.teacher_lr)
    student = model_path(ARM_CONFIGS[args.arm]['student'], args.teacher_lr)
    tokenizer = load_tokenizer(teacher, setup['cache_dir'])
    model = AutoModelForCausalLM.from_pretrained(student, cache_dir=setup['cache_dir'], attn_implementation='eager')
    saliency_ds = None
    if args.noise_target in ('neuron_act', 'neuron_rule'):
        saliency_ds = [load_train_dataset([f], tokenizer, setup['max_length'], True, [1.0], args.seed, 'first_exhausted',
                                          setup['dataset_cache_dir']) for f in (KOR_TRAIN_FILE, ENG_TRAIN_FILE)]
    _, record = select_mlp_neurons(
        model, args.noise_target, args.noise_frac, args.seed, saliency_ds=saliency_ds, saliency_seqs=args.saliency_seqs,
        batch_size=setup['batch_size'], tokenizer=tokenizer, max_length=setup['max_length'],
        noise_count=args.noise_count, p_kor_min=args.p_kor_min, p_eng_max=args.p_eng_max, device=device)
    record.update(noise_target=args.noise_target, seed=args.seed, saliency_seqs=args.saliency_seqs, student=student,
                  tokenizer=teacher, device=str(device), matmul_precision=torch.get_float32_matmul_precision())
    print(f"[dry_select] student {student} on {device}, matmul {torch.get_float32_matmul_precision()}")
    suffix = noise_suffix(args.noise_target, args.noise_frac, args.saliency_seqs, args.noise_count, args.p_kor_min,
                          args.p_eng_max)
    print(f"[dry_select] {suffix}, seed {args.seed}")
    print(f"[dry_select] per layer {record['per_layer_counts']}")
    print(f"[dry_select] SELECTED {record['k']} of {record['total_neurons']} MLP neurons")
    if args.dry_select_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.dry_select_out)), exist_ok=True)
        with open(args.dry_select_out, "w", encoding="utf-8") as f:
            json.dump(record, f)
    return record

def run_all_experiments(args):
    """Run all experiments in the sweep, optionally starting from a specific alpha"""
    alphas = [a for a in args.alphas if args.start_alpha is None or a >= args.start_alpha]
    experiments = [(setup_id, alpha, beta, seed, arm, args.kor_targets, args.stop_mode, args.fixed_steps, args.max_steps,
                    args.teacher_lr, args.epochs, args.stop_rule, args.warmup_frac, args.noise_target,
                    args.noise_frac, args.saliency_seqs, args.noise_count, args.p_kor_min, args.p_eng_max)
                   for setup_id in SETUPS_TO_RUN
                   for arm in args.arms
                   for alpha in alphas
                   for beta in args.betas
                   for seed in args.seeds]
    # Shard the sweep across Slurm array tasks.
    experiments = experiments[args.shard::args.num_shards]

    print(f"Running {len(experiments)} experiments (shard {args.shard}/{args.num_shards})")
    print(f"Arms: {args.arms}, alphas: {alphas}, betas: {args.betas}, seeds: {args.seeds}, stop: {args.stop_mode}, "
          f"teacher lr: {args.teacher_lr}")

    if args.gpus_per_run:
        # One accelerate launch per run on its own group of GPUs (2 runs per 8-GPU node at 4 GPUs per run).
        port_base = 29500 + 8 * (int(os.environ.get("SLURM_JOB_ID", 0)) % 400)
        failed = launch_in_parallel_accelerate_groups(
            [experiment_cli_args(e) for e in experiments], sys.argv[0], args.gpus_per_run, port_base)
        if failed:
            sys.exit(f"{failed} of {len(experiments)} runs failed")
        return

    # Run all experiments
    successful = 0
    failed = 0

    for i, experiment in enumerate(experiments):
        print(f"Running experiment {i+1}/{len(experiments)}")

        # Run the experiment in a separate process
        success = launch_single_experiment(experiment)

        if success:
            successful += 1
        else:
            failed += 1

        # Add a short delay between experiments to ensure cleanup
        time.sleep(5)

    print(f"\nAll experiments completed!")
    print(f"Successful experiments: {successful}/{len(experiments)}")
    print(f"Failed experiments: {failed}/{len(experiments)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run partial_distill language experiments')
    parser.add_argument('--setup', type=str, default="gemma-2-0.1B_MaxEnt", help='Setup ID to run')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for experiment')
    parser.add_argument('--beta', type=float, default=None, help='Beta value for experiment')
    parser.add_argument('--seed', type=int, default=SWEEP_SEEDS[0], help='Seed value for experiment')
    parser.add_argument('--arm', choices=ARMS, default='A', help='Single experiment: which arm')
    parser.add_argument('--kor_targets', choices=KOR_TARGETS, default='uniform', help='Arms D/E: random Korean targets')
    parser.add_argument('--stop_mode', choices=STOP_MODES, default='fixed_steps')
    parser.add_argument('--fixed_steps', type=int, default=FIXED_STEPS)
    parser.add_argument('--stop_rule', choices=STOP_RULES, default='pooled',
                        help="early mode: 'pooled' (default; whole validation set) or 'any_shard' (upstream: any "
                             "rank's own shard within 5%%); any_shard adds '-stop_any_shard' to the run name")
    parser.add_argument('--max_steps', type=int, default=None,
                        help='early mode only: cap on optimizer steps. Like upstream max_steps it also shortens the '
                             'cosine schedule, so leave it unset to match the paper (an epoch is the cap there)')
    parser.add_argument('--warmup_frac', type=float, default=None,
                        help="fixed_steps mode only: warm up over max(1, round(warmup_frac * fixed_steps)) optimizer "
                             "steps instead of the setup's 50 ticks, then cosine to min_lr; adds "
                             "'-warmup_frac_<f>' to the run name. Unset: the setup's warmup, names unchanged")
    parser.add_argument('--noise_target', choices=NOISE_TARGETS, default='all',
                        help="'all' (default): noise every 2-D parameter. 'neuron_act' / 'neuron_rand': noise only "
                             "--noise_frac of the MLP neurons (by Korean/English saliency, or at random); adds "
                             "'-nt_act_f<f>' / '-nt_rand_f<f>' to the run name")
    parser.add_argument('--noise_frac', type=float, default=None,
                        help='targeted noise: fraction of all MLP neurons (all layers) to noise, e.g. 0.05')
    parser.add_argument('--noise_count', type=int, default=None,
                        help="neuron_rand: noise exactly this many random neurons instead of --noise_frac (one of the "
                             "two); adds '-nt_rand_n<N>' to the run name")
    parser.add_argument('--p_kor_min', type=float, default=None,
                        help='neuron_rule: noise every neuron firing on >= this fraction of Korean tokens ...')
    parser.add_argument('--p_eng_max', type=float, default=None,
                        help="... and on <= this fraction of English tokens; adds '-nt_rule_q<q>_r<r>' to the name")
    parser.add_argument('--saliency_seqs', type=int, default=SALIENCY_SEQS,
                        help='neuron_act / neuron_rule: Korean and English train sequences each to score neurons on '
                             "(default 64; other values add '_s<n>' to the name)")
    parser.add_argument('--dry_select', action='store_true',
                        help='Only compute and print the targeted-noise selection of the single-experiment flags '
                             '(--arm, --seed, --noise_*), then exit: no training, nothing written')
    parser.add_argument('--dry_select_out', default=None, help='--dry_select: also write the selection JSON here')
    parser.add_argument('--teacher_lr', type=float, default=TEACHER_LR,
                        help='MaxEnt teacher LR (default 8e-5, the paper\'s; upstream hard-coded 3e-5)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='epoch mode: passes over the training data; early mode: the cap on passes (default 1)')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments in the sweep')
    parser.add_argument('--start_alpha', type=float, default=None, help='Alpha value to start from (inclusive)')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=['A'], help='Sweep: arms')
    parser.add_argument('--alphas', nargs='+', type=float, default=SWEEP_ALPHAS, help='Sweep: alphas')
    parser.add_argument('--betas', nargs='+', type=float, default=SWEEP_BETAS, help='Sweep: betas')
    parser.add_argument('--seeds', nargs='+', type=int, default=SWEEP_SEEDS, help='Sweep: seeds')
    parser.add_argument('--gpus_per_run', type=int, default=None,
                        help='Sweep: run each experiment as its own accelerate launch on this many GPUs, packing '
                             'the visible GPUs (4 = the paper). Unset: one run at a time with the default config')
    parser.add_argument('--shard', type=int, default=0, help='Sweep: run every num_shards-th experiment from here')
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--warm_cache', action='store_true',
                        help='Sweep: only build the datasets caches its runs read (one process, no GPU), then exit')
    args = parser.parse_args()
    if args.warmup_frac is not None and args.stop_mode != 'fixed_steps':
        parser.error("--warmup_frac applies only to --stop_mode fixed_steps")
    try:   # the name's rules: --noise_frac (act; rand), --noise_count (rand), --p_kor_min/--p_eng_max (rule), else none
        noise_suffix(args.noise_target, args.noise_frac, args.saliency_seqs, args.noise_count, args.p_kor_min,
                     args.p_eng_max)
    except ValueError as e:
        parser.error(f"targeted noise flags: {e}")

    if args.dry_select:
        if args.noise_target == 'all':
            parser.error("--dry_select needs a targeted --noise_target")
        dry_select(args)
        raise SystemExit

    if args.warm_cache:
        warm_cache(args)
        raise SystemExit

    # If specific experiment parameters are provided, run that single experiment
    if args.alpha is not None and args.beta is not None:
        print(f"Running single experiment: arm {args.arm}, alpha {args.alpha}, beta {args.beta}, seed {args.seed}, {args.stop_mode}")
        run_experiment(args.setup, args.alpha, args.beta, args.seed, args.arm, args.kor_targets,
                       args.stop_mode, args.fixed_steps, args.max_steps, args.teacher_lr, args.epochs, args.stop_rule,
                       args.warmup_frac, args.noise_target, args.noise_frac, args.saliency_seqs, args.noise_count,
                       args.p_kor_min, args.p_eng_max)

    # If run_all flag is set, run all experiments
    elif args.run_all or args.start_alpha is not None:
        run_all_experiments(args)

    # If no valid arguments provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run a single experiment directly:")
        print("  accelerate launch run_partial_distill_language.py --arm A --alpha 0.3 --beta 0.1 --seed 111")
        print("  # Run a sweep, 4 GPUs per run:")
        print("  python run_partial_distill_language.py --run_all --gpus_per_run 4 --arms A B C D")
