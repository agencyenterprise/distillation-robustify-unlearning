"""
AGI-2635 diagnostic: per-MLP-neuron activation statistics of a model on Korean (forget) and English (retain) train text.

For every MLP neuron (layer i, index j) and each language, over the non-pad tokens of the first --n_seqs packed train
sequences (the same packing, tokenizer and max_length as --noise_target neuron_act's saliency):
  mean_abs   mean |h|, h = input to layers[i].mlp.down_proj = act_fn(gate_proj x) * up_proj x  (as neuron_saliency)
  mean       mean h (signed)
  p_fire     fraction of tokens with act_fn(gate_proj x) > 0 (LAPE's notion of firing; Gemma-2's act_fn is
             gelu_pytorch_tanh, which is > 0 exactly where gate_proj x > 0)
Each is stored for all n_seqs ('all') and for the two disjoint halves (first / second n_seqs // 2 sequences: 'h0', 'h1')
for split-half stability, plus mean_abs over the first --ratio_seqs sequences ('first<r>'), which is what
select_mlp_neurons scored in the 9f170d7 runs (saliency_seqs 64). Arrays are [num_layers, intermediate_size].

Output: one .npz with keys <stat>_<lang>_<split> (stat in mean_abs, mean, p_fire; lang in kor, eng), n_tokens_<lang>_<split>,
and 'meta' (a JSON string: model path, tokenizer, files, n_seqs, token counts, fork commit, precision).

Run (cluster): sbatch --time=00:45:00 cluster/agi2635/neuron_stats.sbatch
Local:         python run_neuron_stats_language.py --model <path> --out stats.npz
"""
import argparse
import json
import os
import subprocess

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DataCollatorWithPadding

from src.tools.partial_distill_langarith import load_tokenizer, load_train_dataset, mlp_modules
from src.utils.paths import CACHE_DIR, DATASET_DIR, MODEL_DIR
from src.utils.run_names import TEACHER_LR, reference_models

# As run_partial_distill_language.py: MATMUL_PRECISION (env) sets torch's fp32 matmul precision (the v4 runs: "high").
if os.environ.get("MATMUL_PRECISION"):
    torch.set_float32_matmul_precision(os.environ["MATMUL_PRECISION"])

STATS = ("mean_abs", "mean", "p_fire")
MAX_LENGTH = 2048  # setups['gemma-2-0.1B_MaxEnt']['max_length'] in run_partial_distill_language.py


@torch.no_grad()
def neuron_stats(model, ds, n_seqs, batch_size, tokenizer, max_length, device, ratio_seqs=64):
    """Per-neuron stats on the first n_seqs rows of ds, for splits 'all', 'h0', 'h1' and f'first{ratio_seqs}'.
    Returns {split: {stat: [num_layers, width] float64 ndarray}} and {split: n_tokens}."""
    assert len(ds) >= n_seqs >= 2, f"need {n_seqs} sequences, the dataset has {len(ds)}"
    half = n_seqs // 2
    mlps = mlp_modules(model)
    layers = getattr(model, "module", model).model.layers
    num_layers, width = len(mlps), mlps[0][2].in_features
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    model.eval()
    model.to(device)
    # acc[split_index][stat]; split 0 = h0, 1 = h1, 2 = first ratio_seqs (a subset of h0/h1, accumulated alongside)
    acc = [{s: torch.zeros(num_layers, width, dtype=torch.float64, device=device) for s in STATS} for _ in range(3)]
    tokens = [0.0, 0.0, 0.0]
    state = {}

    def down_hook(i):
        def fn(module, args):
            h, m = args[0].double(), state["mask"][..., None]
            for k in state["targets"]:
                acc[k]["mean_abs"][i] += (h.abs() * m).sum(dim=(0, 1))
                acc[k]["mean"][i] += (h * m).sum(dim=(0, 1))
        return fn

    def gate_hook(i):
        act_fn = layers[i].mlp.act_fn
        def fn(module, args, output):
            fire = (act_fn(output) > 0).double() * state["mask"][..., None]
            for k in state["targets"]:
                acc[k]["p_fire"][i] += fire.sum(dim=(0, 1))
        return fn

    handles = [down.register_forward_pre_hook(down_hook(i)) for i, (_, _, down) in enumerate(mlps)]
    handles += [gate.register_forward_hook(gate_hook(i)) for i, (gate, _, _) in enumerate(mlps)]
    try:
        # Batches never straddle a split boundary (half, ratio_seqs), so each batch has one set of targets.
        bounds = sorted({0, half, n_seqs, *( [ratio_seqs] if 0 < ratio_seqs < n_seqs else [] )})
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            for start in range(lo, hi, batch_size):
                end = min(start + batch_size, hi)
                rows = ds.select(range(start, end))
                batch = collator([{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows])
                mask = batch["attention_mask"].to(device)
                state["mask"] = mask.double()
                state["targets"] = [0 if start < half else 1] + ([2] if start < ratio_seqs else [])
                n = mask.sum().item()
                for k in state["targets"]:
                    tokens[k] += n
                model.model(input_ids=batch["input_ids"].to(device), attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    out, n_tokens = {}, {}
    names = ["h0", "h1", f"first{ratio_seqs}"]
    for k, name in enumerate(names):
        if tokens[k]:
            out[name] = {s: (acc[k][s] / tokens[k]).cpu().numpy() for s in STATS}
            n_tokens[name] = int(tokens[k])
    tot = tokens[0] + tokens[1]
    out["all"] = {s: ((acc[0][s] + acc[1][s]) / tot).cpu().numpy() for s in STATS}
    n_tokens["all"] = int(tot)
    return out, n_tokens


def flatten(per_lang, n_tokens_per_lang):
    """{lang: {split: {stat: arr}}} -> npz-ready {f'{stat}_{lang}_{split}': arr, f'n_tokens_{lang}_{split}': int}."""
    arrays = {}
    for lang, splits in per_lang.items():
        for split, stats in splits.items():
            for stat, arr in stats.items():
                arrays[f"{stat}_{lang}_{split}"] = arr.astype(np.float32 if stat != "p_fire" else np.float64)
            arrays[f"n_tokens_{lang}_{split}"] = np.int64(n_tokens_per_lang[lang][split])
    return arrays


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="model dir (default: the base model, model_path('base') under MODEL_DIR)")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir (default: the MaxEnt teacher's, as partial_distill "
                                                     "packs with it; falls back to --model if that does not exist)")
    p.add_argument("--n_seqs", type=int, default=256, help="sequences per language (split into two halves)")
    p.add_argument("--ratio_seqs", type=int, default=64, help="also mean_abs over the first this many (9f170d7 saliency)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    refs = reference_models(TEACHER_LR)
    model_dir = args.model or f"{MODEL_DIR}/{refs['base']}/final_model"
    tok_dir = args.tokenizer or f"{MODEL_DIR}/{refs['unlearn']}/final_model"
    if not os.path.exists(tok_dir):
        print(f"[neuron_stats] tokenizer {tok_dir} not found; using the model's")
        tok_dir = model_dir
    files = {"kor": f"{DATASET_DIR}/pretrain/train_kor.jsonl", "eng": f"{DATASET_DIR}/pretrain/train_eng.jsonl"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    tokenizer = load_tokenizer(tok_dir, CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    print(f"[neuron_stats] model {model_dir} ({args.dtype}, {device}), tokenizer {tok_dir}, act_fn "
          f"{type(model.model.layers[0].mlp.act_fn).__name__}, matmul {torch.get_float32_matmul_precision()}")
    per_lang, n_tokens = {}, {}
    for lang, f in files.items():
        # As partial_distill loads its saliency sets: join_or_subsequence True, one file, seed unused.
        ds = load_train_dataset([f], tokenizer, args.max_length, True, [1.0], 111, "first_exhausted", CACHE_DIR)
        per_lang[lang], n_tokens[lang] = neuron_stats(model, ds, args.n_seqs, args.batch_size, tokenizer,
                                                      args.max_length, device, args.ratio_seqs)
        print(f"[neuron_stats] {lang}: {n_tokens[lang]} tokens; mean p_fire per layer "
              f"{np.round(per_lang[lang]['all']['p_fire'].mean(axis=1), 3).tolist()}")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    meta = {"model": model_dir, "tokenizer": tok_dir, "files": files, "n_seqs": args.n_seqs,
            "half": args.n_seqs // 2, "ratio_seqs": args.ratio_seqs, "max_length": args.max_length,
            "dtype": args.dtype, "matmul_precision": torch.get_float32_matmul_precision(), "n_tokens": n_tokens,
            "fork_commit": commit, "agi2569_v": os.environ.get("AGI2569_V"),
            "num_layers": int(per_lang["kor"]["all"]["p_fire"].shape[0]),
            "intermediate_size": int(per_lang["kor"]["all"]["p_fire"].shape[1]),
            "act_fn": type(model.model.layers[0].mlp.act_fn).__name__,
            "h": "input to mlp.down_proj = act_fn(gate_proj x) * up_proj x", "p_fire": "act_fn(gate_proj x) > 0"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, meta=np.array(json.dumps(meta)), **flatten(per_lang, n_tokens))
    print(f"[neuron_stats] wrote {args.out}")


if __name__ == "__main__":
    main()
