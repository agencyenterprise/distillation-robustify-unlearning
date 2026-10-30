"""
Model names shared by the AGI-2569 language scripts (partial distill, relearn) and the analysis.
Pure Python, no heavy imports, so analysis code can import it locally.

Paths are relative to MODEL_DIR. The relearn records of model `m` at learning rate `lr` live at
    {MODEL_DIR}/local_records/relearned_models/relearned_{m.replace('/', '_')}/{lr}_only_forget_data.txt
with `_eval_every_{n}` before `_only_forget_data` when relearning validates every n steps instead of upstream's 5
(relearn_run_name).
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

ARMS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
RANDOM_TARGET_ARMS = ['D', 'E']   # these take --kor_targets
KOR_TARGETS = ['uniform', 'unigram']
STOP_MODES = ['fixed_steps', 'early', 'epoch']
# Early-stop rule: 'pooled' (ours: English loss over the whole validation set) or 'any_shard' (upstream: stop when any
# rank's own validation shard is within the threshold). Only 'any_shard' adds to the name, so pooled names are unchanged.
STOP_RULES = ['pooled', 'any_shard']
# What the one-time noise hits (AGI-2635): 'all' (upstream: every 2-D parameter), or a fraction of the MLP neurons,
# chosen by forget/retain activation saliency ('neuron_act') or uniformly at random ('neuron_rand', its control), or
# every MLP neuron passing a firing-probability rule ('neuron_rule': P_kor >= q and P_eng <= r).
# Only the targeted modes add to the name, so 'all' names are unchanged.
NOISE_TARGETS = ['all', 'neuron_act', 'neuron_rand', 'neuron_rule']
SALIENCY_SEQS = 64        # neuron_act / neuron_rule: Korean and English train sequences each used to score neurons

SWEEP_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SWEEP_BETAS = [0.1]
SWEEP_SEEDS = [111]   # the seed in the paper's relearn model names
FIXED_STEPS = 500
EPOCHS = 1                # epoch mode: number of passes over the training data
RELEARN_EVAL_EVERY = 5    # upstream's relearn validation interval (steps)


def arm_label(arm, kor_targets='uniform'):
    """'A'..'G', with the Korean target distribution appended for D and E (e.g. 'D_unigram')."""
    return f"{arm}_{kor_targets}" if arm in RANDOM_TARGET_ARMS else arm


def stop_label(stop_mode, fixed_steps=FIXED_STEPS, epochs=EPOCHS):
    if stop_mode == 'fixed_steps':
        return f"fixed_steps_{fixed_steps}"
    if stop_mode == 'epoch':
        return f"epochs_{epochs}"
    # Early stop is capped at --epochs passes; the default 1 keeps calibration's names.
    return 'early_stop' if epochs == 1 else f"early_stop_epochs_{epochs}"


def stop_rule_suffix(stop_rule='pooled'):
    if stop_rule not in STOP_RULES:
        raise ValueError(f"unknown stop_rule {stop_rule!r}")
    return "" if stop_rule == 'pooled' else f"-stop_{stop_rule}"


def warmup_suffix(warmup_frac=None):
    """'' by default (the setup's fixed warmup); '-warmup_frac_<f>' when the warmup is scaled to the budget."""
    return "" if warmup_frac is None else f"-warmup_frac_{fmt(warmup_frac)}"


def noise_suffix(noise_target='all', noise_frac=None, saliency_seqs=SALIENCY_SEQS, noise_count=None, p_kor_min=None,
                 p_eng_max=None):
    """'' for 'all'. Targeted noise:
      neuron_act   '-nt_act_f<frac>'
      neuron_rand  '-nt_rand_f<frac>', or '-nt_rand_n<count>' with noise_count (exactly one of the two)
      neuron_rule  '-nt_rule_q<p_kor_min>_r<p_eng_max>'
    plus '_s<n>' when neuron_act / neuron_rule score neurons on a non-default number of sequences."""
    if noise_target not in NOISE_TARGETS:
        raise ValueError(f"unknown noise_target {noise_target!r}")
    if noise_target == 'all':
        if noise_frac is not None or noise_count is not None or p_kor_min is not None or p_eng_max is not None:
            raise ValueError("noise_frac / noise_count / p_kor_min / p_eng_max apply only to targeted noise")
        return ""
    if noise_target != 'neuron_rule' and (p_kor_min is not None or p_eng_max is not None):
        raise ValueError("p_kor_min / p_eng_max apply only to noise_target 'neuron_rule'")
    if noise_target != 'neuron_rand' and noise_count is not None:
        raise ValueError("noise_count applies only to noise_target 'neuron_rand'")
    seqs = (f"_s{saliency_seqs}" if noise_target in ('neuron_act', 'neuron_rule') and saliency_seqs != SALIENCY_SEQS
            else "")
    if noise_target == 'neuron_rule':
        if noise_frac is not None:
            raise ValueError("neuron_rule selects by threshold: no noise_frac")
        if p_kor_min is None or p_eng_max is None or not (0 <= p_kor_min <= 1 and 0 <= p_eng_max <= 1):
            raise ValueError(f"neuron_rule needs 0 <= p_kor_min, p_eng_max <= 1, got {p_kor_min!r}, {p_eng_max!r}")
        return f"-nt_rule_q{fmt(p_kor_min)}_r{fmt(p_eng_max)}{seqs}"
    if noise_count is not None:
        if noise_frac is not None:
            raise ValueError("neuron_rand takes noise_frac or noise_count, not both")
        if int(noise_count) != noise_count or noise_count < 1:
            raise ValueError(f"noise_count must be a positive integer, got {noise_count!r}")
        return f"-nt_rand_n{int(noise_count)}"
    if noise_frac is None or not 0 < noise_frac <= 1:
        raise ValueError(f"noise_target {noise_target!r} needs 0 < noise_frac <= 1, got {noise_frac!r}")
    return f"-nt_{noise_target.split('_')[1]}_f{fmt(noise_frac)}{seqs}"


def warmup_optimizer_steps(warmup_frac, fixed_steps):
    """--warmup_frac: warmup length in optimizer steps, max(1, round(warmup_frac * fixed_steps))."""
    return max(1, round(warmup_frac * fixed_steps))


def fmt(x):
    """0.1 -> '0.1', 0.30000000000000004 -> '0.3', 0.0 -> '0.0' (matches upstream's str(float))."""
    return str(round(float(x), 6))


def partial_distill_name(arm, alpha, beta, seed, stop_mode='fixed_steps', fixed_steps=FIXED_STEPS, kor_targets='uniform',
                         teacher_lr=TEACHER_LR, epochs=EPOCHS, stop_rule='pooled', warmup_frac=None,
                         noise_target='all', noise_frac=None, saliency_seqs=SALIENCY_SEQS, noise_count=None,
                         p_kor_min=None, p_eng_max=None):
    if stop_rule != 'pooled' and stop_mode != 'early':
        raise ValueError(f"stop_rule {stop_rule!r} applies only to stop_mode 'early'")
    if warmup_frac is not None and stop_mode != 'fixed_steps':
        raise ValueError("warmup_frac applies only to stop_mode 'fixed_steps'")
    return (f"partial_distill_models/gemma-2-0.1B-language-partial_distill-teacher_lr_{lr_tag(teacher_lr)}"
            f"-arm_{arm_label(arm, kor_targets)}"
            f"-alpha_{fmt(alpha)}-beta_{fmt(beta)}-seed_{seed}-{stop_label(stop_mode, fixed_steps, epochs)}"
            f"{stop_rule_suffix(stop_rule)}{warmup_suffix(warmup_frac)}"
            f"{noise_suffix(noise_target, noise_frac, saliency_seqs, noise_count, p_kor_min, p_eng_max)}")


def relearn_record_dir(model):
    return f"relearned_{model.replace('/', '_')}"


def relearn_run_name(model, lr, eval_every=RELEARN_EVAL_EVERY):
    """'relearned_<model>/<lr>', plus '_eval_every_<n>' unless n is upstream's 5 (so calibration paths are unchanged)."""
    suffix = "" if eval_every == RELEARN_EVAL_EVERY else f"_eval_every_{eval_every}"
    return f"{relearn_record_dir(model)}/{lr}{suffix}"
