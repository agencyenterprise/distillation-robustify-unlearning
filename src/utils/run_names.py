"""
Model names shared by the AGI-2569 language scripts (partial distill, relearn) and the analysis.
Pure Python, no heavy imports, so analysis code can import it locally.

Paths are relative to MODEL_DIR. The relearn records of model `m` at learning rate `lr` live at
    {MODEL_DIR}/local_records/relearned_models/relearned_{m.replace('/', '_')}/{lr}_only_forget_data.txt
"""

# Reference models, by role (Fig 4 / normalisation endpoints).
REFERENCE_MODELS = {
    'base'        : 'pretrained_models/gemma-2-0.1B_eng+kor',                  # pretrained on English + Korean
    'filter'      : 'pretrained_models/gemma-2-0.1B_eng',                      # data-filtered oracle: English only
    'unlearn'     : 'unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05',  # MaxEnt teacher
    'full_distill': 'distilled_models/MaxEnt/gemma-2-0.1B_eng+kor',            # MaxEnt teacher -> random init
}

ARMS = ['A', 'B', 'C', 'D', 'E', 'F']
RANDOM_TARGET_ARMS = ['D', 'E']   # these take --kor_targets
KOR_TARGETS = ['uniform', 'unigram']
STOP_MODES = ['fixed_steps', 'early']

SWEEP_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SWEEP_BETAS = [0.1]
SWEEP_SEEDS = [111]   # the seed in the paper's relearn model names
FIXED_STEPS = 500


def arm_label(arm, kor_targets='uniform'):
    """'A'..'F', with the Korean target distribution appended for D and E (e.g. 'D_unigram')."""
    return f"{arm}_{kor_targets}" if arm in RANDOM_TARGET_ARMS else arm


def stop_label(stop_mode, fixed_steps=FIXED_STEPS):
    return f"fixed_steps_{fixed_steps}" if stop_mode == 'fixed_steps' else 'early_stop'


def fmt(x):
    """0.1 -> '0.1', 0.30000000000000004 -> '0.3', 0.0 -> '0.0' (matches upstream's str(float))."""
    return str(round(float(x), 6))


def partial_distill_name(arm, alpha, beta, seed, stop_mode='fixed_steps', fixed_steps=FIXED_STEPS, kor_targets='uniform'):
    return (f"partial_distill_models/gemma-2-0.1B-language-partial_distill-arm_{arm_label(arm, kor_targets)}"
            f"-alpha_{fmt(alpha)}-beta_{fmt(beta)}-seed_{seed}-{stop_label(stop_mode, fixed_steps)}")


def relearn_record_dir(model):
    return f"relearned_{model.replace('/', '_')}"
