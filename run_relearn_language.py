"""
Configured for a (any)xH200 GPU server.

Launch Command: python run_relearn_language.py
AGI-2569:       python run_relearn_language.py --groups refs partial --arms A C --lrs 1e-4 1e-5 [--teacher_lr 8e-5]
                    [--eval_every 25] [--shard i --num_shards n] [--list]
"""
import argparse
import json
import os
import sys

from src.tools.relearn_langarith import load_tokenizer, load_train_dataset, relearn
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR, USE_WANDB, WANDB_API_KEY
from accelerate import Accelerator
from utils.loss_functions import custom_login
from utils.loss_functions import print_acc
from utils.validation_functions import get_korean_and_english_evalaution_fn
from utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper
from src.utils.run_names import (
    REFERENCE_MODELS, ARMS, EPOCHS, RANDOM_TARGET_ARMS, KOR_TARGETS, NOISE_TARGETS, SALIENCY_SEQS, STOP_MODES, STOP_RULES, SWEEP_ALPHAS, SWEEP_BETAS, SWEEP_SEEDS,
    FIXED_STEPS, RELEARN_EVAL_EVERY, TEACHER_LR, partial_distill_name, reference_models, relearn_run_name,
)

SETUPS_TO_RUN = [
    "gemma-2-0.1B_train_only_forget"
]

MODELS_TO_RUN = [
    #'pretrained_models/gemma-2-0.1B_eng', # Pretrain Pure
    #'pretrained_models/gemma-2-0.1B_eng+kor', # Pretrained Base
    #'unlearned_models/GradDiff/gemma-2-0.1B_eng+kor_lr_6.0e-05', # Unlearned GA
    #'unlearned_models/RMU/gemma-2-0.1B_eng+kor_lr_3.0e-05', # Unlearned RMU
    #'unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05', # Unlearned UF

    #'distilled_models/pure/gemma-2-0.1B_pure', # Distilled Pure
    #'distilled_models/base/gemma-2-0.1B_impure', # Distilled Base
    #'distilled_models/GradDiff/gemma-2-0.1B_eng+kor', # Distilled GA
    #'distilled_models/RMU/gemma-2-0.1B_eng+kor', # Distilled RMU
    #'distilled_models/MaxEnt/gemma-2-0.1B_eng+kor', # Distilled UF

    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_0.1',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_0.5',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_1.0',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_1.0',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_1.0',  # partial_distill
    #'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_1.0',  # partial_distill

    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.1-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.2-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.3-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.4-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.5-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.6-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.7-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
    'partial_distill_models/gemma-2-0.1B_MaxEnt-language-partial_distill-alpha_0.8-beta_0.1-seed_111-fixed_steps_500',  # partial_distill
]
# 1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6
# , 4e-6, 1e-6

'''
for pareto plot
'''
#LR_RANGES = {
#    "GradDiff": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
#    "MaxEnt": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5],
#    "RMU": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5]
#}
#
#for method, lr_values in LR_RANGES.items():
#    for lr in lr_values:
#        # Format the learning rate with scientific notation
#        formatted_lr = f"{lr:.1e}"
#        # Add the model path to the list
#        model_path = f"unlearned_models/{method}/gemma-2-0.1B_eng+kor_lr_{formatted_lr}"
#        MODELS_TO_RUN.append(model_path)
#
#for model in MODELS_TO_RUN:
#    print(model)

# relearning learning rate (search adversary)
LRS_TO_RUN =[1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6]

api_key = WANDB_API_KEY

# Tokenizer for packing the Korean train set. Every model here has the same Gemma tokenizer, but the packing cache
# fingerprint includes the tokenizer's path, so loading it from one model (the base, whose relearns built this cache
# in calibration) gives every relearned model the same packed data from one cache instead of building its own.
DATASET_TOKENIZER = f"{MODEL_DIR}/{REFERENCE_MODELS['base']}/final_model"

custom_login()

setups = {
    "gemma-2-0.1B_train_only_forget": {
        'model_name'       : f"{MODEL_DIR}/model_path/final_student",
        'first_train_file'   : f"{DATASET_DIR}/pretrain/train_kor.jsonl",
        'second_train_file': "",
        'interleave_probs': [1],
        'eng_valid_file'   : f"{DATASET_DIR}/pretrain/valid_eng.jsonl",
        'kor_valid_file'   : f"{DATASET_DIR}/pretrain/valid_kor.jsonl",
        'output_dir'       : f"{MODEL_DIR}/relearned_models/model_name_only_forget_data",
        'cache_dir'        : CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence'         : True,
        'seed'                        : 42,
        'device'                      : "cuda",
        'batch_size'                  : 8,
        'gradient_accumulation_steps' : 1,
        'epochs'                      : 1,
        'learning_rate'               : None,       
        'max_steps'                   : 500,             
        'num_warmup_steps'            : 1,
        'validation_steps'            : 5,
        'save_checkpoint_steps'       : 999,
        'scheduler_type'              : "cosine", 
        
        'min_lr'                      : None,              
        'weight_decay'                : 0.0,         
        'gradient_clipping_threshold' : 1.0, 
        'max_length'                  : 2048,

        'use_wandb'        : USE_WANDB,
        'wandb_project'    : "gemma-2-0.1B_eng+kor",
        'wandb_run_name'   : None,
        'use_local_record' : True,
        'path_local_record': f"{MODEL_DIR}/local_records/relearned_models/model_name_only_forget_data.txt",

        'save_models'      : False
    }
}

def relearn_setup(setup_id, lr, model, eval_every=None):
    """The setup of one (model, lr) run: model path, record path and output dir filled in. eval_every (default:
    the setup's validation_steps, upstream's 5) is in the record path unless it is 5."""
    current_setup = setups[setup_id].copy()
    current_setup['learning_rate'] = lr
    current_setup['min_lr'] = lr
    if eval_every is not None:
        current_setup['validation_steps'] = eval_every
    current_setup['model_name'] = current_setup['model_name'].replace('model_path', model)
    if "distilled" not in current_setup['model_name']:
        current_setup['model_name'] = current_setup['model_name'].replace('final_student', "final_model")
    model_name = relearn_run_name(model, lr, current_setup['validation_steps'])
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('model_name', model_name)
    current_setup['output_dir'] = current_setup['output_dir'].replace('model_name', model_name)
    return current_setup, model_name

def launch_relearn(setup_id, lr, model, eval_every=None):
    current_setup, model_name = relearn_setup(setup_id, lr, model, eval_every)

    if relearn_finished(current_setup['path_local_record'], current_setup['max_steps']):
        print(f"[run_relearn_language.py] {model_name} already finished; skipping.")
        return
    # A rerun after a crash starts over, so drop the partial record rather than appending to it.
    if os.path.exists(current_setup['path_local_record']):
        os.remove(current_setup['path_local_record'])
    
    accelerator = Accelerator()

    language_eval_fn = get_korean_and_english_evalaution_fn(
        model_name          = current_setup['model_name'],
        max_length          = current_setup['max_length'],
        eng_valid_file      = current_setup['eng_valid_file'],
        kor_valid_file      = current_setup['kor_valid_file'],
        dataset_cache_dir   = current_setup['dataset_cache_dir'],
        cache_dir           = current_setup['cache_dir'],
        batch_size          = current_setup['batch_size'],
        accelerator         = accelerator
    )
    if not current_setup['second_train_file'] == "":
        train_files = [current_setup['first_train_file'], current_setup['second_train_file']]
    else:
        train_files = [current_setup['first_train_file']]
    relearn(
        model_name       = current_setup['model_name'],
        train_files      = train_files,
        interleave_probs = current_setup['interleave_probs'],
        output_dir       = current_setup['output_dir'],
        cache_dir        = current_setup['cache_dir'],
        dataset_cache_dir= current_setup['dataset_cache_dir'],
        eval_fn          = language_eval_fn,
        accelerator      = accelerator,
        join_or_subsequence   = current_setup['join_or_subsequence'],
        seed             = current_setup['seed'],
        device           = current_setup['device'],
        batch_size       = current_setup['batch_size'],
        gradient_accumulation_steps = current_setup['gradient_accumulation_steps'],
        epochs           = current_setup['epochs'],
        learning_rate    = current_setup['learning_rate'],
        max_steps        = current_setup['max_steps'],   
        num_warmup_steps = current_setup['num_warmup_steps'],
        validation_steps = current_setup['validation_steps'],
        save_checkpoint_steps = current_setup['save_checkpoint_steps'],
        scheduler_type   = current_setup['scheduler_type'],  
        min_lr           = current_setup['min_lr'],          
        weight_decay     = current_setup['weight_decay'],    
        gradient_clipping_threshold = current_setup['gradient_clipping_threshold'], 
        max_length       = current_setup['max_length'],
        use_wandb        = current_setup['use_wandb'],
        wandb_project    = current_setup['wandb_project'],
        wandb_run_name   = current_setup['wandb_run_name'],
        use_local_record = current_setup['use_local_record'],
        path_local_record= current_setup['path_local_record'],
        save_models      = current_setup['save_models'],
        overwrite_ok     = True,  # the record was removed above; output_dir may exist from a crashed run
        dataset_tokenizer_name = DATASET_TOKENIZER,
    )

def arrow_files():
    """The datasets cache's Arrow files, to report what --warm_cache built."""
    return {os.path.join(d, f) for d, _, files in os.walk(CACHE_DIR) for f in files if f.endswith(".arrow")}

def warm_cache(setup_id):
    """Build (or load) the datasets caches every relearn run reads, in one process, then return. Run it once
    before fanning out: file locks do not work across nodes, and concurrent first builds crashed runs."""
    before = arrow_files()
    current_setup = setups[setup_id]
    accelerator = Accelerator(cpu=True)  # only builds datasets; nothing runs on it
    get_korean_and_english_evalaution_fn(
        model_name=DATASET_TOKENIZER, max_length=current_setup['max_length'],
        eng_valid_file=current_setup['eng_valid_file'], kor_valid_file=current_setup['kor_valid_file'],
        dataset_cache_dir=current_setup['dataset_cache_dir'], cache_dir=current_setup['cache_dir'],
        batch_size=current_setup['batch_size'], accelerator=accelerator)
    train_files = [f for f in (current_setup['first_train_file'], current_setup['second_train_file']) if f]
    load_train_dataset(train_files, load_tokenizer(DATASET_TOKENIZER, current_setup['cache_dir']),
                       current_setup['max_length'], current_setup['join_or_subsequence'],
                       current_setup['interleave_probs'], current_setup['seed'], 'first_exhausted',
                       current_setup['dataset_cache_dir'])
    print(f"[run_relearn_language.py] datasets cache ready for {setup_id}")
    print(f"[warm_cache] {len(arrow_files() - before)} new cache files in {CACHE_DIR} (0: every cache existed)")

def relearn_finished(path_local_record, max_steps):
    """True if the record has the validation at the last step (written just before relearn() returns)."""
    if not os.path.exists(path_local_record):
        return False
    with open(path_local_record, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "val/kor_ce_loss" in record and record.get("train/step") == max_steps:
                return True
    return False

def agi2569_models(args):
    """Model paths (relative to MODEL_DIR) written by this fork's scripts, filtered by the CLI flags."""
    models = []
    if 'refs' in args.groups:
        models += [reference_models(args.teacher_lr)[role] for role in args.refs]
    if 'partial' in args.groups:
        for arm in args.arms:
            for kor_targets in (args.kor_targets if arm in RANDOM_TARGET_ARMS else ['uniform']):
                for alpha in args.alphas:
                    for beta in args.betas:
                        for seed in args.seeds:
                            for stop_mode in args.stop_modes:
                                models.append(partial_distill_name(arm, alpha, beta, seed, stop_mode, args.fixed_steps,
                                                                   kor_targets, args.teacher_lr, args.epochs,
                                                                   args.stop_rule, args.warmup_frac,
                                                                   args.noise_target, args.noise_frac,
                                                                   args.saliency_seqs))
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='*', choices=['refs', 'partial'], default=None,
                        help="AGI-2569 model list. Omit to use the upstream MODELS_TO_RUN above.")
    parser.add_argument('--refs', nargs='+', choices=list(REFERENCE_MODELS), default=list(REFERENCE_MODELS))
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--kor_targets', nargs='+', choices=KOR_TARGETS, default=['uniform'])
    parser.add_argument('--alphas', nargs='+', type=float, default=SWEEP_ALPHAS)
    parser.add_argument('--betas', nargs='+', type=float, default=SWEEP_BETAS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SWEEP_SEEDS)
    parser.add_argument('--stop_modes', nargs='+', choices=STOP_MODES, default=['fixed_steps'])
    parser.add_argument('--fixed_steps', type=int, default=FIXED_STEPS)
    parser.add_argument('--stop_rule', choices=STOP_RULES, default='pooled',
                        help="Partial models of --stop_modes early trained with this early-stop rule (any_shard: "
                             "the '-stop_any_shard' runs)")
    parser.add_argument('--warmup_frac', type=float, default=None,
                        help="Partial models of --stop_modes fixed_steps trained with this --warmup_frac (the "
                             "'-warmup_frac_<f>' runs)")
    parser.add_argument('--noise_target', choices=NOISE_TARGETS, default='all',
                        help="Partial models trained with this --noise_target (neuron_act/neuron_rand: the "
                             "'-nt_*' runs)")
    parser.add_argument('--noise_frac', type=float, default=None, help='... and this --noise_frac')
    parser.add_argument('--saliency_seqs', type=int, default=SALIENCY_SEQS, help='... and this --saliency_seqs')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Partial models of --stop_modes epoch (their epochs) or early (their cap on passes; default 1)')
    parser.add_argument('--teacher_lr', type=float, default=TEACHER_LR,
                        help='MaxEnt teacher LR of the unlearn/full_distill refs and partial models (default 8e-5)')
    parser.add_argument('--lrs', nargs='+', type=float, default=LRS_TO_RUN)
    parser.add_argument('--eval_every', type=int, default=RELEARN_EVAL_EVERY,
                        help='Validate every n steps (upstream 5); the last step always is. Not 5: in the record path')
    parser.add_argument('--shard', type=int, default=0, help='Run every num_shards-th (model, lr) pair from here')
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--list', action='store_true', help='Print the (model, lr) pairs and exit')
    parser.add_argument('--warm_cache', action='store_true',
                        help='Only build the datasets caches (shared by every model and LR), then exit')
    args = parser.parse_args()
    if args.warm_cache:
        for setup_id in SETUPS_TO_RUN:
            warm_cache(setup_id)
        raise SystemExit
    if args.groups is not None:
        MODELS_TO_RUN = agi2569_models(args)
    # ----------------------------------------------------------------- #
    # Run all experiments, if possible in parallel
    # ----------------------------------------------------------------- #
    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = [(setup_id, lr, model, args.eval_every)
                   for setup_id in SETUPS_TO_RUN for lr in args.lrs for model in MODELS_TO_RUN]
    experiments = experiments[args.shard::args.num_shards]
    if args.list:
        for experiment in experiments:
            print(experiment)
        print(f"{len(experiments)} runs")
        raise SystemExit
    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_relearn)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
    # The launcher ignores child failures; exit non-zero so Slurm afterok dependents wait.
    unfinished = [(lr, model) for setup_id, lr, model, eval_every in experiments
                  if not relearn_finished(relearn_setup(setup_id, lr, model, eval_every)[0]['path_local_record'],
                                          setups[setup_id]['max_steps'])]
    if unfinished:
        sys.exit(f"{len(unfinished)} of {len(experiments)} relearn runs have no record reaching step "
                 f"{setups[SETUPS_TO_RUN[0]]['max_steps']}: {unfinished}")
    