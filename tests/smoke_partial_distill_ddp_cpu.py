"""
CPU check of partial distillation under 2-process DDP (gloo), the geometry the cluster runs with 4 GPUs.
Checks: the noised student is identical on every rank before the DDP wrap; records are written once; tokens_seen
counts every rank; the LR schedule advances num_processes ticks per optimizer step (accelerate's prepared scheduler,
so upstream's total_steps, computed from the unsharded loader, is in ticks); fixed-step mode's cosine spans exactly
its optimizer steps (min_lr at the last, no cycling); fixed-step and early-stop runs end together on all ranks; epoch
mode runs every epoch with the cosine over all of them; a rerun is skipped.

    python tests/smoke_partial_distill_ddp_cpu.py      # from the repo root; launches itself with torchrun
"""
import json
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPROC, BATCH, ACCUM, STEPS, WARMUP = 2, 2, 2, 6, 3

if "LOCAL_RANK" not in os.environ:
    work = tempfile.mkdtemp(prefix="undo_ddp_smoke_")
    env = {**os.environ, "SMOKE_WORK": work}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(NPROC), "--master_port", "29561",
           os.path.abspath(__file__)]
    sys.exit(subprocess.run(cmd, env=env, cwd=ROOT).returncode)

WORK = os.environ["SMOKE_WORK"]
for name in ("CACHE_DIR", "DATASET_DIR", "MODEL_DIR"):
    os.environ[name] = os.path.join(WORK, name.lower())
os.environ.update(WANDB_MODE="disabled", NUM_PROC="1", ACCELERATE_USE_CPU="true")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import random  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from accelerate import PartialState  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from src.utils.paths import DATASET_DIR, MODEL_DIR  # noqa: E402
from src.utils.run_names import REFERENCE_MODELS, partial_distill_name  # noqa: E402

VOCAB, SEQ = 300, 64


def make_model_dir(path, seed):  # as in smoke_language_cpu.py
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "\n": 3, **{f"w{i}": i for i in range(4, VOCAB)}}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<eos>")),
                                        pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")
    torch.manual_seed(seed)
    config = Gemma2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=SEQ,
                          pad_token_id=0, eos_token_id=1, bos_token_id=2)
    Gemma2ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


def make_jsonl(path, n_docs, low, high):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _ in range(n_docs):
            ids = [random.randrange(low, high) for _ in range(random.randint(20, 50))]
            f.write(json.dumps({"text": "x", "input_ids": ids, "attention_mask": [1] * len(ids),
                                "loss_mask": [1] * len(ids)}) + "\n")


state = PartialState()
assert state.num_processes == NPROC, state
if state.is_main_process:
    random.seed(0)
    for split, n in (("train", 400), ("valid", 40)):
        make_jsonl(f"{DATASET_DIR}/pretrain/{split}_eng.jsonl", n, 4, 150)
        make_jsonl(f"{DATASET_DIR}/pretrain/{split}_kor.jsonl", n, 150, VOCAB)
    make_model_dir(f"{MODEL_DIR}/{REFERENCE_MODELS['base']}/final_model", seed=1)
    make_model_dir(f"{MODEL_DIR}/{REFERENCE_MODELS['unlearn']}/final_model", seed=2)
state.wait_for_everyone()

import run_partial_distill_language as pd  # noqa: E402
import src.tools.partial_distill_langarith as tool  # noqa: E402

# Record a checksum of the student right after noising, on every rank.
noised = []
_do_corruption = tool.do_corruption
def recording_corruption(model, *args, **kwargs):
    _do_corruption(model, *args, **kwargs)
    noised.append(sum(p.double().sum().item() for p in model.parameters()))
tool.do_corruption = recording_corruption

pd.setups["gemma-2-0.1B_MaxEnt"].update(batch_size=BATCH, gradient_accumulation_steps=ACCUM, validation_steps=2,
                                         max_length=SEQ, num_warmup_steps=WARMUP, device="cpu")


def upstream_lr(tick, total_ticks, lr=9e-4, min_lr=7e-4):
    """partial_distill's cosine lambda (warmup, then cosine to min_lr) at a scheduler tick."""
    if tick < WARMUP:
        return lr * tick / WARMUP
    progress = (tick - WARMUP) / max(1, total_ticks - WARMUP)
    return lr * (0.5 * (1 + math.cos(math.pi * progress)) * (1 - min_lr / lr) + min_lr / lr)


EPOCHS = 2
for arm, stop_mode in (("A", "fixed_steps"), ("D", "fixed_steps"), ("C", "early"), ("C", "epoch")):
    pd.run_experiment("gemma-2-0.1B_MaxEnt", 0.3, 0.1, 111, arm, "uniform", stop_mode, fixed_steps=STEPS,
                      max_steps=None if stop_mode == "epoch" else 4, epochs=EPOCHS)
    state.wait_for_everyone()
    sums = [None] * NPROC
    dist.all_gather_object(sums, noised[-1])
    assert len(set(sums)) == 1, f"noised student differs across ranks: {sums}"
    if state.is_main_process:
        name = partial_distill_name(arm, 0.3, 0.1, 111, stop_mode, STEPS, epochs=EPOCHS)
        with open(f"{MODEL_DIR}/local_records/{name}.txt") as f:
            records = [json.loads(line) for line in f]
        train = [r for r in records if "train/lr" in r]
        steps = [r["train/step"] for r in train]
        assert steps == list(range(1, len(steps) + 1)), f"records duplicated or missing: {steps}"
        if stop_mode == "fixed_steps":
            assert len(steps) == STEPS, steps
            # Every row is a full max_length sequence (join_or_subsequence): count all ranks' tokens.
            assert train[-1]["train/tokens_seen"] == STEPS * ACCUM * BATCH * NPROC * SEQ, train[-1]
            # The cosine spans STEPS optimizer steps = STEPS * NPROC ticks; optimizer step k is tick k * NPROC.
            expected = [upstream_lr(k * NPROC, STEPS * NPROC) for k in steps]
            assert all(math.isclose(r["train/lr"], e, rel_tol=1e-6) for r, e in zip(train, expected)), \
                ([r["train/lr"] for r in train], expected)
            # First (in warmup), middle and last optimizer step, by hand: WARMUP = 3 ticks, 12 ticks in all.
            lrs = [r["train/lr"] for r in train]
            mid = STEPS // 2   # step 3, tick 6: cosine progress (6 - 3) / (12 - 3) = 1/3
            by_hand = {1: 9e-4 * 2 / 3,
                       mid: 7e-4 + 2e-4 * 0.5 * (1 + math.cos(math.pi / 3)),
                       STEPS: 7e-4}
            for k, lr in by_hand.items():
                assert math.isclose(lrs[k - 1], lr, rel_tol=1e-6), (k, lrs[k - 1], lr)
            assert all(a >= b for a, b in zip(lrs[1:], lrs[2:])), f"LR rose after warmup: {lrs}"
        if stop_mode == "epoch":
            # Every epoch in full, no early stop, and the cosine over all of them: total_steps is the unsharded
            # loader's batches // ACCUM x EPOCHS ticks, as in early mode (not fixed mode's step cap).
            teacher_tokenizer = tool.load_tokenizer(pd.model_path("unlearn"), None)
            rows = len(tool.load_train_dataset([pd.TRAIN_FILES["eng"]], teacher_tokenizer, SEQ, True, [1.0], 111,
                                               "first_exhausted", os.environ["CACHE_DIR"], False))
            batches = math.ceil(rows / BATCH)
            assert len(steps) == EPOCHS * (math.ceil(batches / NPROC) // ACCUM), (len(steps), rows)
            total_ticks = batches // ACCUM * EPOCHS
            expected = [upstream_lr(k * NPROC, total_ticks) for k in steps]
            assert all(math.isclose(r["train/lr"], e, rel_tol=1e-6) for r, e in zip(train, expected)), \
                ([r["train/lr"] for r in train], expected)
        assert sum(r.get("validation_stage") == "final_student" for r in records) == 1
        assert os.path.exists(f"{MODEL_DIR}/{name}/final_model/config.json")
        print(f"[ddp smoke] arm {arm} {stop_mode} OK: {len(steps)} steps, {train[-1]['train/tokens_seen']} tokens, "
              f"noised checksum {sums[0]:.6f} on all ranks", flush=True)

# Targeted noise (AGI-2635): the same neurons selected and the same noised student on every rank.
for kw in (dict(noise_target="neuron_act", noise_frac=0.05), dict(noise_target="neuron_rand", noise_frac=0.05),
           dict(noise_target="neuron_rand", noise_count=5), dict(noise_target="neuron_rule", p_kor_min=0.3, p_eng_max=0.7)):
    pd.run_experiment("gemma-2-0.1B_MaxEnt", 1.0, 1.0, 111, "F", "uniform", "early", max_steps=2, saliency_seqs=4, **kw)
    state.wait_for_everyone()
    sums = [None] * NPROC
    dist.all_gather_object(sums, noised[-1])
    assert len(set(sums)) == 1, f"{kw}: noised student differs across ranks: {sums}"
    if state.is_main_process:
        name = partial_distill_name("F", 1.0, 1.0, 111, "early", saliency_seqs=4, **kw)
        with open(f"{MODEL_DIR}/local_records/{name}_noise_selection.json") as f:
            k = json.load(f)["k"]
        assert k == {"neuron_rule": k}.get(kw["noise_target"], kw.get("noise_count", round(0.05 * 2 * 64))) > 0
        print(f"[ddp smoke] arm F {name.rsplit('-', 1)[-1]} OK: k={k}, noised checksum {sums[0]:.6f} on all ranks",
              flush=True)

# A rerun is skipped on every rank (no hang, no new noising).
before = len(noised)
pd.run_experiment("gemma-2-0.1B_MaxEnt", 0.3, 0.1, 111, "A", "uniform", "fixed_steps", fixed_steps=STEPS)
assert len(noised) == before
state.wait_for_everyone()
if state.is_main_process:
    print("[ddp smoke] all OK", flush=True)
dist.destroy_process_group()
