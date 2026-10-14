"""
Partial distillation (UNDO) on the language setting, with the AGI-2569 ablation arms.

Every arm noises the student once, (1 - alpha) * theta + alpha * beta * N, then retrains it:
  A  KL to the teacher on English + Korean (upstream UNDO; the default)
  B  KL to the teacher on English only
  C  CE on real labels, English only (no teacher forward pass)
  D  CE: English rows on real labels, Korean rows on real inputs with random targets (--kor_targets)
  E  as D, but the student starts from the pretrained base model instead of the MaxEnt-unlearned one
  F  CE on real labels, English + Korean (sanity check: should relearn Korean)
The teacher is always the MaxEnt-unlearned model. In C-F it is evaluated once, as the reference for the
early-stop condition, and never run during training.

Stopping: --stop_mode fixed_steps (default; --fixed_steps 500, as in the paper's relearn model names)
or --stop_mode early (upstream: stop once English validation loss is within 5% of the teacher's).

Launch Command (Single):     accelerate launch run_partial_distill_language.py --arm C --alpha 0.3 --beta 0.1 --seed 111
Launch Command (Sweep):      python run_partial_distill_language.py --run_all --arms A B C
Launch Command (One per GPU): python run_partial_distill_language.py --run_all --one_per_gpu --arms D --kor_targets unigram
"""

from src.tools.partial_distill_langarith import partial_distill
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, USE_WANDB, WANDB_API_KEY
from src.utils.run_names import (
    ARMS, KOR_TARGETS, STOP_MODES, SWEEP_ALPHAS, SWEEP_BETAS, SWEEP_SEEDS, FIXED_STEPS, REFERENCE_MODELS,
    partial_distill_name,
)
from accelerate import Accelerator
from utils.loss_functions import print_acc, custom_login
from src.utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

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
UNLEARNED_MODEL = f"{MODEL_DIR}/{REFERENCE_MODELS['unlearn']}/final_model"
BASE_MODEL = f"{MODEL_DIR}/{REFERENCE_MODELS['base']}/final_model"

# What each arm trains on and with which loss.
ARM_CONFIGS = {
    #       train files (and domain names)                 loss   Korean targets  student
    'A': {'domains': ['eng', 'kor'], 'loss_type': 'kl', 'random_kor': False, 'student': UNLEARNED_MODEL},
    'B': {'domains': ['eng'],        'loss_type': 'kl', 'random_kor': False, 'student': UNLEARNED_MODEL},
    'C': {'domains': ['eng'],        'loss_type': 'ce', 'random_kor': False, 'student': UNLEARNED_MODEL},
    'D': {'domains': ['eng', 'kor'], 'loss_type': 'ce', 'random_kor': True,  'student': UNLEARNED_MODEL},
    'E': {'domains': ['eng', 'kor'], 'loss_type': 'ce', 'random_kor': True,  'student': BASE_MODEL},
    'F': {'domains': ['eng', 'kor'], 'loss_type': 'ce', 'random_kor': False, 'student': UNLEARNED_MODEL},
}
TRAIN_FILES = {'eng': ENG_TRAIN_FILE, 'kor': KOR_TRAIN_FILE}

setups = {
    "gemma-2-0.1B_MaxEnt": {
        'teacher_model_name': UNLEARNED_MODEL,
        'student_model_name': UNLEARNED_MODEL,
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

def never_stop_cond_fn(student_eval_dict, teacher_eval_dict):
    """fixed_steps mode: train for exactly max_steps."""
    return False

def run_experiment(setup_id, alpha, beta, seed=None, arm='A', kor_targets='uniform', stop_mode='fixed_steps',
                   fixed_steps=FIXED_STEPS, max_steps=None):
    """Run a single experiment. max_steps caps early-stop runs (default: none, as upstream)."""
    current_setup = setups[setup_id].copy()
    arm_config = ARM_CONFIGS[arm]

    # Use provided seed or keep default
    if seed is not None:
        current_setup['seed'] = seed

    model_name = partial_distill_name(arm, alpha, beta, current_setup['seed'], stop_mode, fixed_steps, kor_targets)
    current_setup['output_dir'] = current_setup['output_dir'].replace('model_name', model_name)
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('model_name', model_name)
    current_setup['student_model_name'] = arm_config['student']
    current_setup['noise_alpha'] = alpha
    current_setup['noise_beta'] = beta
    if stop_mode == 'fixed_steps':
        current_setup['max_steps'] = fixed_steps
    elif max_steps is not None:
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

    # Get evaluation function for Korean and English
    eval_fn = get_korean_and_english_evalaution_fn(
        model_name=current_setup['student_model_name'],
        max_length=current_setup['max_length'],
        eng_valid_file=current_setup['eng_valid_file'],
        kor_valid_file=current_setup['kor_valid_file'],
        dataset_cache_dir=current_setup['dataset_cache_dir'],
        cache_dir=current_setup['cache_dir'],
        batch_size=current_setup['batch_size'],
        accelerator=accelerator
    )

    domains = arm_config['domains']
    partial_distill(
        teacher_model_name=current_setup['teacher_model_name'],
        student_model_name=current_setup['student_model_name'],
        train_files=[TRAIN_FILES[d] for d in domains],
        interleave_probs=[1.0 / len(domains)] * len(domains),  # Equal probabilities for eng and kor
        stopping_strategy='first_exhausted',  # Add missing parameter
        join_or_subsequence=True,
        eval_fn=eval_fn,
        stop_cond_fn=korean_english_stop_cond_fn if stop_mode == 'early' else never_stop_cond_fn,
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
    )

def experiment_cli_args(experiment):
    setup_id, alpha, beta, seed, arm, kor_targets, stop_mode, fixed_steps, max_steps = experiment
    cmd = ["--setup", setup_id, "--alpha", str(alpha), "--beta", str(beta), "--seed", str(seed), "--arm", arm,
           "--kor_targets", kor_targets, "--stop_mode", stop_mode, "--fixed_steps", str(fixed_steps)]
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
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

def run_all_experiments(args):
    """Run all experiments in the sweep, optionally starting from a specific alpha"""
    alphas = [a for a in args.alphas if args.start_alpha is None or a >= args.start_alpha]
    experiments = [(setup_id, alpha, beta, seed, arm, args.kor_targets, args.stop_mode, args.fixed_steps, args.max_steps)
                   for setup_id in SETUPS_TO_RUN
                   for arm in args.arms
                   for alpha in alphas
                   for beta in args.betas
                   for seed in args.seeds]
    # Shard the sweep across Slurm array tasks.
    experiments = experiments[args.shard::args.num_shards]

    print(f"Running {len(experiments)} experiments (shard {args.shard}/{args.num_shards})")
    print(f"Arms: {args.arms}, alphas: {alphas}, betas: {args.betas}, seeds: {args.seeds}, stop: {args.stop_mode}")

    if args.one_per_gpu:
        # One single-process run per GPU, queued until all are done.
        parallel_fn = get_parallel_launch_wrapper(run_experiment)
        launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
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
    parser.add_argument('--max_steps', type=int, default=None, help='early mode only: cap on optimizer steps')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments in the sweep')
    parser.add_argument('--start_alpha', type=float, default=None, help='Alpha value to start from (inclusive)')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=['A'], help='Sweep: arms')
    parser.add_argument('--alphas', nargs='+', type=float, default=SWEEP_ALPHAS, help='Sweep: alphas')
    parser.add_argument('--betas', nargs='+', type=float, default=SWEEP_BETAS, help='Sweep: betas')
    parser.add_argument('--seeds', nargs='+', type=int, default=SWEEP_SEEDS, help='Sweep: seeds')
    parser.add_argument('--one_per_gpu', action='store_true', help='Sweep: one single-GPU run per visible GPU')
    parser.add_argument('--shard', type=int, default=0, help='Sweep: run every num_shards-th experiment from here')
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()

    # If specific experiment parameters are provided, run that single experiment
    if args.alpha is not None and args.beta is not None:
        print(f"Running single experiment: arm {args.arm}, alpha {args.alpha}, beta {args.beta}, seed {args.seed}, {args.stop_mode}")
        run_experiment(args.setup, args.alpha, args.beta, args.seed, args.arm, args.kor_targets,
                       args.stop_mode, args.fixed_steps, args.max_steps)

    # If run_all flag is set, run all experiments
    elif args.run_all or args.start_alpha is not None:
        run_all_experiments(args)

    # If no valid arguments provided, show help
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run a single experiment directly:")
        print("  accelerate launch run_partial_distill_language.py --arm A --alpha 0.3 --beta 0.1 --seed 111")
        print("  # Run a sweep, one run per GPU:")
        print("  python run_partial_distill_language.py --run_all --one_per_gpu --arms A B C D")
