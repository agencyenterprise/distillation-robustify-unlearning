"""
Aggregates AGI-2569 relearning records into robustness and compute tables.

Input: a copy of {MODEL_DIR}/local_records (e.g. `aws s3 sync s3://ae-gr/agi-2569/local_records <dir>`).
Only the small JSONL records are read; no models.

    python analysis/agi2569/aggregate.py <local_records dir> [--out <dir>]

Definitions
- Relearned Korean loss of a model: the attacker fine-tunes on Korean for 500 steps at each LR of a sweep;
  the worst case (for the defender) is the lowest Korean validation loss the attacker reaches. Two definitions,
  both reported, column `definition`:
    min_any_step  min over LRs and over every validation step (5-step granularity). The attacker may stop early.
    min_final     min over LRs of the loss at the last step. The attacker picks the LR, not the step.
- Normalised robustness: R = (P_x - P_unlearn) / (P_filter - P_unlearn), P = worst-case relearned Korean loss.
  0 = as easily relearned as the MaxEnt-unlearned teacher, 1 = as robust as data filtering.
- Compute ratio vs data filtering: tokens seen by the method's retraining over tokens seen pretraining the
  data-filtered oracle. `compute_ratio_tokens` counts tokens; `compute_ratio_flops` weights a KL token by 4
  (teacher forward 2N + student forward/backward 6N, in units of 2N) and a CE or pretraining token by 3.
  The unlearning step itself (100 MaxEnt steps) is not counted.
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.utils.run_names import REFERENCE_MODELS, relearn_record_dir  # noqa: E402

DEFINITIONS = ("min_any_step", "min_final")
KL_ARMS = ("A", "B")
PARTIAL_RE = re.compile(
    r"partial_distill-arm_(?P<arm>[A-F](?:_uniform|_unigram)?)-alpha_(?P<alpha>[\d.]+)-beta_(?P<beta>[\d.]+)"
    r"-seed_(?P<seed>\d+)-(?P<stop>fixed_steps_\d+|early_stop)$"
)
REF_DIRS = {relearn_record_dir(path): role for role, path in REFERENCE_MODELS.items()}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def describe_model(record_dir):
    """Metadata for a relearned_* directory name, or None if it is not one of ours."""
    if record_dir in REF_DIRS:
        return {"model": record_dir, "role": REF_DIRS[record_dir], "arm": None, "alpha": None, "beta": None,
                "seed": None, "stop": None}
    match = PARTIAL_RE.search(record_dir)
    if match is None:
        return None
    return {"model": record_dir, "role": "partial", "arm": match["arm"], "alpha": float(match["alpha"]),
            "beta": float(match["beta"]), "seed": int(match["seed"]), "stop": match["stop"]}


def load_relearn_curves(records_root):
    """Long table: one row per (model, lr, step) validation point."""
    rows = []
    base = os.path.join(records_root, "relearned_models")
    for record_dir in sorted(os.listdir(base)):
        meta = describe_model(record_dir)
        if meta is None:
            print(f"[aggregate] skipping unrecognised {record_dir}")
            continue
        for fname in os.listdir(os.path.join(base, record_dir)):
            if not fname.endswith("_only_forget_data.txt"):
                continue
            lr = float(fname[: -len("_only_forget_data.txt")])
            for rec in read_jsonl(os.path.join(base, record_dir, fname)):
                if "val/kor_ce_loss" in rec:
                    rows.append({**meta, "lr": lr, "step": rec["train/step"],
                                 "kor_loss": rec["val/kor_ce_loss"], "eng_loss": rec["val/eng_ce_loss"]})
    # Duplicate final validations (end-of-epoch and max_steps) at the same step are the same model state.
    return pd.DataFrame(rows).drop_duplicates(subset=["model", "lr", "step"], keep="last")


def worst_case(curves):
    """One row per (model, definition): worst-case relearned Korean loss and the LR that achieved it."""
    rows = []
    for model, group in curves.groupby("model"):
        meta = group.iloc[0][["model", "role", "arm", "alpha", "beta", "seed", "stop"]].to_dict()
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
    return pd.DataFrame(rows)


def worst_case_curve(curves):
    """Per model and step, the lowest Korean loss over LRs (the attacker's envelope; Fig 4 style)."""
    return curves.groupby(["model", "role", "arm", "alpha", "step"], dropna=False).kor_loss.min().reset_index()


def add_robustness(summary):
    """Normalised robustness against the MaxEnt-unlearned (0) and data-filtered (1) references."""
    out = []
    for definition, group in summary.groupby("definition"):
        refs = group.set_index("role").relearned_kor_loss
        group = group.copy()
        if "unlearn" in refs and "filter" in refs:
            group["robustness"] = (group.relearned_kor_loss - refs["unlearn"]) / (refs["filter"] - refs["unlearn"])
        else:
            print(f"[aggregate] {definition}: unlearn or filter reference missing; no normalised robustness")
            group["robustness"] = float("nan")
        out.append(group)
    return pd.concat(out, ignore_index=True)


def training_tokens(records_root, record_path):
    path = os.path.join(records_root, record_path)
    if not os.path.exists(path):
        return float("nan")
    return max((rec.get("train/tokens_seen", 0) for rec in read_jsonl(path)), default=float("nan"))


def add_compute(summary, records_root, filter_tokens=None):
    """Tokens each model's retraining saw, and the ratio to data filtering's pretraining."""
    if filter_tokens is None:
        filter_tokens = training_tokens(records_root, REFERENCE_MODELS["filter"] + ".txt")
    tokens, units = [], []
    for _, row in summary.iterrows():
        if row.role == "partial":
            # relearned_partial_distill_models_<name> -> partial_distill_models/<name>.txt
            name = row.model[len("relearned_partial_distill_models_"):]
            tokens.append(training_tokens(records_root, f"partial_distill_models/{name}.txt"))
            units.append(4 if row.arm in KL_ARMS else 3)
        elif row.role == "full_distill":
            tokens.append(training_tokens(records_root, REFERENCE_MODELS["full_distill"] + ".txt"))
            units.append(4)
        else:
            tokens.append(float("nan"))
            units.append(float("nan"))
    summary = summary.copy()
    summary["train_tokens"] = tokens
    summary["compute_ratio_tokens"] = summary.train_tokens / filter_tokens
    summary["compute_ratio_flops"] = summary.train_tokens * pd.Series(units, index=summary.index) / (3 * filter_tokens)
    summary.attrs["filter_tokens"] = filter_tokens
    return summary


def aggregate(records_root, filter_tokens=None):
    curves = load_relearn_curves(records_root)
    summary = add_compute(add_robustness(worst_case(curves)), records_root, filter_tokens)
    return curves, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records_root", help="local copy of {MODEL_DIR}/local_records")
    parser.add_argument("--out", default="analysis/agi2569/out")
    parser.add_argument("--filter_tokens", type=float, default=None,
                        help="override the oracle's pretraining tokens if its record is missing")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    curves, summary = aggregate(args.records_root, args.filter_tokens)
    curves.to_csv(os.path.join(args.out, "relearn_curves.csv"), index=False)
    worst_case_curve(curves).to_csv(os.path.join(args.out, "relearn_envelope.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    cols = ["role", "arm", "alpha", "stop", "eng_loss_before", "relearned_kor_loss", "worst_lr", "robustness",
            "compute_ratio_flops"]
    for definition in DEFINITIONS:
        print(f"\n== worst case: {definition} ==")
        print(summary[summary.definition == definition][cols].sort_values(["role", "arm", "alpha"]).to_string(index=False))
    print(f"\nWrote {args.out}/summary.csv, relearn_curves.csv, relearn_envelope.csv")


if __name__ == "__main__":
    main()
