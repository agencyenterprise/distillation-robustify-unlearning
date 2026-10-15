"""
Model names shared by the AGI-2569 language scripts (partial distill, relearn) and the analysis.
Pure Python, no heavy imports, so analysis code can import it locally.

Paths are relative to MODEL_DIR. The relearn records of model `m` at learning rate `lr` live at
    {MODEL_DIR}/local_records/relearned_models/relearned_{m.replace('/', '_')}/{lr}_only_forget_data.txt
"""

# MaxEnt teacher LR for the language setting: 8e-5, the paper's choice (App. B.1 bold, App. C; Fig 4b's unlearned
# model starts relearning at Korean 12.33, App. B.1's 8e-5 row gives 12.38). Upstream code hard-codes 3e-5.
# Every script that uses the teacher takes --teacher_lr (default this), and every path downstream of the teacher
# encodes it, so runs with different teachers never share a path.
TEACHER_LR = 8e-5


def lr_tag(lr):
    """8e-5 -> '8.0e-05', the format run_unlearn_language.py puts in its output paths."""
    return f"{float(lr):.1e}"


def unlearned_model(lr):
    """MaxEnt-unlearned model at this LR (run_unlearn_language.py's output path)."""
    return f"unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_{lr_tag(lr)}"


def full_distill_model(teacher_lr=TEACHER_LR):
    """MaxEnt teacher -> random-init student (Fig 4's full distillation)."""
    return f"distilled_models/MaxEnt/gemma-2-0.1B_eng+kor_teacher_lr_{lr_tag(teacher_lr)}"


def reference_models(teacher_lr=TEACHER_LR):
    """Reference models by role (Fig 4 / normalisation endpoints). base and filter do not depend on the teacher."""
    return {
        'base'        : 'pretrained_models/gemma-2-0.1B_eng+kor',   # pretrained on English + Korean
        'filter'      : 'pretrained_models/gemma-2-0.1B_eng',       # data-filtered oracle: English only
        'unlearn'     : unlearned_model(teacher_lr),                # MaxEnt teacher
        'full_distill': full_distill_model(teacher_lr),             # MaxEnt teacher -> random init
    }


REFERENCE_MODELS = reference_models()   # at the default teacher LR

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


def partial_distill_name(arm, alpha, beta, seed, stop_mode='fixed_steps', fixed_steps=FIXED_STEPS, kor_targets='uniform',
                         teacher_lr=TEACHER_LR):
    return (f"partial_distill_models/gemma-2-0.1B-language-partial_distill-teacher_lr_{lr_tag(teacher_lr)}"
            f"-arm_{arm_label(arm, kor_targets)}"
            f"-alpha_{fmt(alpha)}-beta_{fmt(beta)}-seed_{seed}-{stop_label(stop_mode, fixed_steps)}")


def relearn_record_dir(model):
    return f"relearned_{model.replace('/', '_')}"
