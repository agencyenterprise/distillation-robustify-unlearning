"""
Aggregates AGI-2569 relearning records into robustness and compute tables.

Input: a copy of {MODEL_DIR}/local_records (e.g. `aws s3 sync s3://ae-gr/agi-2569/local_records <dir>`).
Only the small JSONL records are read; no models.

    python analysis/agi2569/aggregate.py <local_records dir> [--out <dir>] [--teacher_lr 8e-5]

Only models of one MaxEnt teacher LR are aggregated (--teacher_lr, default 8e-5, the paper's); base and the
data-filtered oracle do not depend on it.

Definitions
- Relearned Korean loss of a model: the attacker fine-tunes on Korean for 500 steps at each LR of a sweep;
  the worst case (for the defender) is the lowest Korean validation loss the attacker reaches. Two definitions,
  both reported, column `definition`:
    min_any_step  min over LRs and over every validation step (every `eval_every` steps: 5 upstream and in Phase 1,
                  25 in Phase 2, so Phase 2 values are slightly higher). The attacker may stop early.
    min_final     min over LRs of the loss at the last step. The attacker picks the LR, not the step.
- Normalised robustness: R = (P_x - P_unlearn) / (P_filter - P_unlearn), P = worst-case relearned Korean loss.
  0 = as easily relearned as the MaxEnt-unlearned teacher, 1 = as robust as data filtering.
- Compute ratio vs data filtering: tokens seen by the method's retraining over tokens seen pretraining the
  data-filtered oracle. `compute_ratio_tokens` counts tokens; `compute_ratio_flops` weights a KL token by 4
  (teacher forward 2N + student forward/backward 6N, in units of 2N) and a CE or pretraining token by 3.
  The unlearning step itself (100 MaxEnt steps) is not counted.
  `compute_ratio_steps` is the paper's Fig 5 axis: optimizer steps over the steps of full distillation, which ran
  the same number of steps as pretraining (~806 in the paper; see paper_targets.md). It needs both runs on the
  same GPU geometry (4 GPUs, 960 sequences per step).
- The paper's robustness uses the worst-case curve's value at the last step, which is `min_final`.
- Validation interval: a relearn record at an interval other than 5 has `_eval_every_<n>` in its file name
  (run_names.relearn_run_name). If a (model, LR) has records at several intervals, the finest is used. The envelope
  keeps only the steps every complete LR of the model validated at.
- Completeness: a relearn run counts only if its record reaches step 500 (RELEARN_STEPS, upstream's
  `relearn_finished`). Truncated or missing LRs (of the paper's 10, ATTACK_LRS) are dropped from the worst case and
  envelope, and the model is marked `provisional` (`lrs_incomplete` lists them). `robustness_provisional` is set
  when the model or either normalisation reference is provisional.
- MaxEnt sweep check (`maxent_sweep_check`, maxent_sweep.csv): final English / Korean validation loss of each
  unlearning LR against App. B.1's table (paper_targets.json, scalars.maxent_final).
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.utils.run_names import (  # noqa: E402
    RELEARN_EVAL_EVERY, TEACHER_LR, lr_tag, reference_models, relearn_record_dir, unlearned_model,
)

DEFINITIONS = ("min_any_step", "min_final")
KL_ARMS = ("A", "B")
PARTIAL_RE = re.compile(
    r"partial_distill-teacher_lr_(?P<teacher_lr>[\d.e+-]+)-arm_(?P<arm>[A-G](?:_uniform|_unigram)?)"
    r"-alpha_(?P<alpha>[\d.]+)-beta_(?P<beta>[\d.]+)-seed_(?P<seed>\d+)-(?P<stop>fixed_steps_\d+|early_stop_epochs_\d+|early_stop|epochs_\d+)"
    r"(?P<stop_rule>-stop_any_shard)?"   # --stop_rule any_shard runs (pooled runs have no suffix)
    r"(?:-warmup_frac_(?P<warmup_frac>[\d.]+))?$"   # --warmup_frac runs (the setup's warmup: no suffix)
)
RECORD_RE = re.compile(r"^(?P<lr>[^_]+)(?:_eval_every_(?P<eval_every>\d+))?_only_forget_data\.txt$")
ATTACK_LRS = (1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6, 4e-6, 1e-6)  # run_relearn_language.py LRS_TO_RUN
RELEARN_STEPS = 500   # run_relearn_language.py max_steps
PAPER_TARGETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_targets.json")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def describe_model(record_dir, teacher_lr=TEACHER_LR):
    """Metadata for a relearned_* directory name, or None if it is not one of ours at this teacher LR."""
    ref_dirs = {relearn_record_dir(path): role for role, path in reference_models(teacher_lr).items()}
    if record_dir in ref_dirs:
        return {"model": record_dir, "role": ref_dirs[record_dir], "arm": None, "alpha": None, "beta": None,
                "seed": None, "stop": None, "warmup_frac": None}
    match = PARTIAL_RE.search(record_dir)
    if match is None or match["teacher_lr"] != lr_tag(teacher_lr):
        return None
    return {"model": record_dir, "role": "partial", "arm": match["arm"], "alpha": float(match["alpha"]),
            "beta": float(match["beta"]), "seed": int(match["seed"]),
            # any_shard runs keep the suffix in `stop` ('early_stop-stop_any_shard'), a series of their own
            "stop": match["stop"] + (match["stop_rule"] or ""),
            # --warmup_frac runs: the fraction (stop stays 'fixed_steps_<n>'); None for the setup's fixed warmup
            "warmup_frac": None if match["warmup_frac"] is None else float(match["warmup_frac"])}


def load_relearn_curves(records_root, teacher_lr=TEACHER_LR):
    """Long table: one row per (model, lr, step) validation point, with `complete` (the LR's run reached
    RELEARN_STEPS). Incomplete runs are kept here; `worst_case` and `worst_case_curve` drop them."""
    rows = []
    base = os.path.join(records_root, "relearned_models")
    for record_dir in sorted(os.listdir(base)):
        meta = describe_model(record_dir, teacher_lr)
        if meta is None:
            print(f"[aggregate] skipping {record_dir} (not ours, or another teacher LR than {lr_tag(teacher_lr)})")
            continue
        for fname in os.listdir(os.path.join(base, record_dir)):
            match = RECORD_RE.match(fname)
            if match is None:
                continue
            lr = float(match["lr"])
            eval_every = int(match["eval_every"] or RELEARN_EVAL_EVERY)
            for rec in read_jsonl(os.path.join(base, record_dir, fname)):
                if "val/kor_ce_loss" in rec:
                    rows.append({**meta, "lr": lr, "eval_every": eval_every, "step": rec["train/step"],
                                 "kor_loss": rec["val/kor_ce_loss"], "eng_loss": rec["val/eng_ce_loss"]})
    curves = pd.DataFrame(rows)
    # One record per (model, lr): the finest validation interval that reached the end, else the finest.
    curves["complete"] = curves.groupby(["model", "lr", "eval_every"]).step.transform("max") >= RELEARN_STEPS
    pick = (curves.drop_duplicates(["model", "lr", "eval_every"])
            .sort_values(["complete", "eval_every"], ascending=[False, True])
            .drop_duplicates(["model", "lr"])[["model", "lr", "eval_every"]])
    curves = curves.merge(pick, on=["model", "lr", "eval_every"]).drop(columns="complete")
    # Duplicate final validations (end-of-epoch and max_steps) at the same step are the same model state.
    curves = curves.drop_duplicates(subset=["model", "lr", "step"], keep="last")
    curves["complete"] = curves.groupby(["model", "lr"]).step.transform("max") >= RELEARN_STEPS
    return curves


def completeness(curves):
    """Per model: the attacker LRs (of ATTACK_LRS) that are missing or did not reach RELEARN_STEPS.

    Returns a DataFrame indexed by model with `lrs_incomplete` (comma-separated LRs, '' if none) and `provisional`.
    """
    rows = []
    for model, group in curves.groupby("model"):
        done = set(group[group.complete].lr.round(12))
        bad = [lr for lr in ATTACK_LRS if round(lr, 12) not in done]
        rows.append({"model": model, "lrs_incomplete": ",".join(f"{lr:g}" for lr in bad), "provisional": bool(bad)})
    return pd.DataFrame(rows).set_index("model")


def _complete(curves):
    dropped = curves[~curves.complete].drop_duplicates(["model", "lr"])
    for _, r in dropped.iterrows():
        print(f"[aggregate] dropping incomplete relearn run: {r.model} lr {r.lr:g} (last step "
              f"{curves[(curves.model == r.model) & (curves.lr == r.lr)].step.max()})")
    return curves[curves.complete]


def worst_case(curves):
    """One row per (model, definition): worst-case relearned Korean loss and the LR that achieved it, over the
    complete LRs only; `lrs_incomplete` and `provisional` say which LRs were left out."""
    status = completeness(curves)
    curves = _complete(curves)
    rows = []
    for model, group in curves.groupby("model"):
        meta = group.iloc[0][["model", "role", "arm", "alpha", "beta", "seed", "stop", "warmup_frac"]].to_dict()
        meta["eval_every"] = group.eval_every.max()
        start = group[group.step == 0]
        before = {"kor_loss_before": start.kor_loss.mean(), "eng_loss_before": start.eng_loss.mean()}
        n_lrs = group.lr.nunique()
        best_any = group.loc[group.kor_loss.idxmin()]
        rows.append({**meta, **before, "definition": "min_any_step", "relearned_kor_loss": best_any.kor_loss,
                     "worst_lr": best_any.lr, "worst_step": best_any.step, "n_lrs": n_lrs})
        finals = group.loc[group.groupby("lr").step.idxmax()]
        best_final = finals.loc[finals.kor_loss.idxmin()]
        rows.append({**meta, **before, "definition": "min_final", "relearned_kor_loss": best_final.kor_loss,
                     "worst_lr": best_final.lr, "worst_step": best_final.step, "n_lrs": n_lrs})
    return pd.DataFrame(rows).join(status, on="model")


def worst_case_curve(curves):
    """Per model and step, the lowest Korean loss over complete LRs (the attacker's envelope; Fig 4 style), at the
    steps every complete LR validated at (so a model relearned at mixed intervals gets its coarsest grid)."""
    curves = curves[curves.complete]
    n_lrs = curves.groupby("model").lr.transform("nunique")
    curves = curves[curves.groupby(["model", "step"]).lr.transform("nunique") == n_lrs]
    return curves.groupby(["model", "role", "arm", "alpha", "stop", "warmup_frac", "step"], dropna=False).kor_loss.min().reset_index()


def add_robustness(summary):
    """Normalised robustness against the MaxEnt-unlearned (0) and data-filtered (1) references."""
    out = []
    for definition, group in summary.groupby("definition"):
        refs = group.set_index("role").relearned_kor_loss
        group = group.copy()
        if "unlearn" in refs and "filter" in refs:
            group["robustness"] = (group.relearned_kor_loss - refs["unlearn"]) / (refs["filter"] - refs["unlearn"])
            ref_prov = group[group.role.isin(["unlearn", "filter"])].provisional.any()
            group["robustness_provisional"] = group.provisional | bool(ref_prov)
        else:
            print(f"[aggregate] {definition}: unlearn or filter reference missing; no normalised robustness")
            group["robustness"] = float("nan")
            group["robustness_provisional"] = True
        out.append(group)
    return pd.concat(out, ignore_index=True)


def training_max(records_root, record_path, key):
    """Largest value of `key` (e.g. train/tokens_seen, train/step) in a training record; nan if missing."""
    path = os.path.join(records_root, record_path)
    if not os.path.exists(path):
        return float("nan")
    return max((rec.get(key, 0) for rec in read_jsonl(path)), default=float("nan"))


def training_tokens(records_root, record_path):
    return training_max(records_root, record_path, "train/tokens_seen")


def add_compute(summary, records_root, filter_tokens=None, filter_steps=None, teacher_lr=TEACHER_LR):
    """Tokens and steps each model's retraining took, and their ratios to data filtering."""
    refs = reference_models(teacher_lr)
    if filter_tokens is None:
        filter_tokens = training_tokens(records_root, refs["filter"] + ".txt")
    if filter_steps is None:
        filter_steps = training_max(records_root, refs["full_distill"] + ".txt", "train/step")
    tokens, steps, units = [], [], []
    for _, row in summary.iterrows():
        if row.role == "partial":
            # relearned_partial_distill_models_<name> -> partial_distill_models/<name>.txt
            name = row.model[len("relearned_partial_distill_models_"):]
            record = f"partial_distill_models/{name}.txt"
            units.append(4 if row.arm in KL_ARMS else 3)
        elif row.role == "full_distill":
            record = refs["full_distill"] + ".txt"
            units.append(4)
        else:
            record = None
            units.append(float("nan"))
        tokens.append(training_tokens(records_root, record) if record else float("nan"))
        steps.append(training_max(records_root, record, "train/step") if record else float("nan"))
    summary = summary.copy()
    summary["train_tokens"] = tokens
    summary["train_steps"] = steps
    summary["compute_ratio_steps"] = summary.train_steps / filter_steps
    summary["compute_ratio_tokens"] = summary.train_tokens / filter_tokens
    summary["compute_ratio_flops"] = summary.train_tokens * pd.Series(units, index=summary.index) / (3 * filter_tokens)
    summary.attrs["filter_tokens"] = filter_tokens
    summary.attrs["filter_steps"] = filter_steps
    return summary


def aggregate(records_root, filter_tokens=None, filter_steps=None, teacher_lr=TEACHER_LR):
    curves = load_relearn_curves(records_root, teacher_lr)
    summary = add_compute(add_robustness(worst_case(curves)), records_root, filter_tokens, filter_steps, teacher_lr)
    return curves, summary


def maxent_sweep_check(records_root, paper_path=PAPER_TARGETS):
    """Our final English / Korean loss per MaxEnt LR against App. B.1. One row per LR in the paper's table."""
    with open(paper_path, encoding="utf-8") as f:
        paper = json.load(f)["scalars"]["maxent_final"]
    rows = []
    for lr_key, (paper_eng, paper_kor) in paper.items():
        path = os.path.join(records_root, unlearned_model(float(lr_key)) + ".txt")
        vals = [r for r in read_jsonl(path) if "val/kor_ce_loss" in r] if os.path.exists(path) else []
        last = vals[-1] if vals else {}   # the final validation, written after training ends
        eng, kor = last.get("val/eng_ce_loss", float("nan")), last.get("val/kor_ce_loss", float("nan"))
        rows.append({"lr": float(lr_key), "eng_loss": eng, "paper_eng_loss": paper_eng, "eng_diff": eng - paper_eng,
                     "kor_loss": kor, "paper_kor_loss": paper_kor, "kor_diff": kor - paper_kor,
                     "step": last.get("train/step", float("nan"))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records_root", help="local copy of {MODEL_DIR}/local_records")
    parser.add_argument("--out", default="analysis/agi2569/out")
    parser.add_argument("--teacher_lr", type=float, default=TEACHER_LR,
                        help="aggregate the models of this MaxEnt teacher LR (default 8e-5, the paper's)")
    parser.add_argument("--filter_tokens", type=float, default=None,
                        help="override the oracle's pretraining tokens if its record is missing")
    parser.add_argument("--filter_steps", type=float, default=None,
                        help="override the steps-axis denominator (default: full distillation's steps; paper ~806)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sweep = maxent_sweep_check(args.records_root)
    sweep.to_csv(os.path.join(args.out, "maxent_sweep.csv"), index=False)
    print("== MaxEnt unlearning: final losses vs App. B.1 ==")
    print(sweep.to_string(index=False, formatters={"lr": "{:.0e}".format}, float_format=lambda x: f"{x:.4f}"))
    curves, summary = aggregate(args.records_root, args.filter_tokens, args.filter_steps, args.teacher_lr)
    curves.to_csv(os.path.join(args.out, "relearn_curves.csv"), index=False)
    worst_case_curve(curves).to_csv(os.path.join(args.out, "relearn_envelope.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    cols = ["role", "arm", "alpha", "stop", "eval_every", "eng_loss_before", "relearned_kor_loss", "worst_lr", "robustness",
            "compute_ratio_flops", "compute_ratio_steps", "provisional", "robustness_provisional"]
    for definition in DEFINITIONS:
        print(f"\n== worst case: {definition} ==")
        print(summary[summary.definition == definition][cols].sort_values(["role", "arm", "alpha"]).to_string(index=False))
    print(f"\nWrote {args.out}/summary.csv, relearn_curves.csv, relearn_envelope.csv, maxent_sweep.csv")


if __name__ == "__main__":
    main()
