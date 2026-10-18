"""
CPU smoke test of the language pipeline on a tiny random Gemma-2 and fake token data (no downloads).
Runs: oracle pretraining (English only), partial distillation for every arm, and one relearn run.

    python tests/smoke_language_cpu.py      # from the repo root; takes a few minutes
"""
import json
import os
import random
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = tempfile.mkdtemp(prefix="undo_smoke_")
for name in ("CACHE_DIR", "DATASET_DIR", "MODEL_DIR"):
    os.environ[name] = os.path.join(WORK, name.lower())
os.environ["WANDB_MODE"] = "disabled"
os.environ["NUM_PROC"] = "1"  # macOS cannot fork 100 dataset workers
os.environ["ACCELERATE_USE_CPU"] = "true"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from src.utils.paths import DATASET_DIR, MODEL_DIR  # noqa: E402
from src.utils.run_names import REFERENCE_MODELS, partial_distill_name  # noqa: E402

VOCAB, SEQ = 300, 64


def make_model_dir(path, seed):
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "\n": 3, **{f"w{i}": i for i in range(4, VOCAB)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<eos>")),
        pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
    )
    torch.manual_seed(seed)
    config = Gemma2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=SEQ,
                          pad_token_id=0, eos_token_id=1, bos_token_id=2)
    Gemma2ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


def make_jsonl(path, n_docs, low, high):
    """Fake 'languages': English uses token ids [low, high), Korean a disjoint range."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _ in range(n_docs):
            n = random.randint(20, 50)
            ids = [random.randrange(low, high) for _ in range(n)]
            f.write(json.dumps({"text": "x", "input_ids": ids, "attention_mask": [1] * n, "loss_mask": [1] * n}) + "\n")


def last_record(path, key):
    with open(path) as f:
        return [json.loads(line) for line in f if key in line][-1]


def cache_files():
    """The datasets cache's Arrow files (map/filter outputs and indices)."""
    return {os.path.join(d, f) for d, _, files in os.walk(os.environ["CACHE_DIR"]) for f in files if f.endswith(".arrow")}


def main():
    random.seed(0)
    for split, n in (("train", 400), ("valid", 40)):
        make_jsonl(f"{DATASET_DIR}/pretrain/{split}_eng.jsonl", n, 4, 150)
        make_jsonl(f"{DATASET_DIR}/pretrain/{split}_kor.jsonl", n, 150, VOCAB)
    counts = torch.zeros(VOCAB, dtype=torch.long)
    counts[150:] = torch.randint(1, 100, (VOCAB - 150,))
    torch.save(counts, f"{DATASET_DIR}/pretrain/kor_unigram_counts.pt")
    make_model_dir(f"{MODEL_DIR}/random_init_models/gemma-2-0.1B", seed=0)
    make_model_dir(f"{MODEL_DIR}/{REFERENCE_MODELS['base']}/final_model", seed=1)
    make_model_dir(f"{MODEL_DIR}/{REFERENCE_MODELS['unlearn']}/final_model", seed=2)

    # 1) English-only oracle pretraining (was broken upstream: truncate kwarg, nested file list).
    import run_pretrain_language as pretrain
    setup = pretrain.setups["gemma-2-0.1B_eng"]
    setup.update(batch_size=2, gradient_accumulation_steps=2, max_steps=3, validation_steps=2, max_length=SEQ,
                 save_checkpoint_steps=-1, num_warmup_steps=1, device="cpu")
    pretrain.launch_pretrain("gemma-2-0.1B_eng")
    assert os.path.exists(f"{setup['output_dir']}/final_model/config.json")
    print("[smoke] oracle pretraining OK:", last_record(setup["path_local_record"], "val/eng_ce_loss"))

    # 2) Partial distillation, every arm, fixed_steps and one early-stop run.
    import run_partial_distill_language as pd
    pd.setups["gemma-2-0.1B_MaxEnt"].update(batch_size=2, gradient_accumulation_steps=2, validation_steps=2,
                                             max_length=SEQ, num_warmup_steps=1, device="cpu")
    runs = [(arm, "uniform", "fixed_steps") for arm in "ABCDEF"] + [("D", "unigram", "fixed_steps"), ("C", "uniform", "early")]
    for arm, kor_targets, stop_mode in runs:
        pd.run_experiment("gemma-2-0.1B_MaxEnt", 0.3, 0.1, 111, arm, kor_targets, stop_mode, fixed_steps=3, max_steps=4)
        name = partial_distill_name(arm, 0.3, 0.1, 111, stop_mode, 3, kor_targets)
        assert os.path.exists(f"{MODEL_DIR}/{name}/final_model/config.json"), name
        record = last_record(f"{MODEL_DIR}/local_records/{name}.txt", "train/loss_")
        expected = {"A": {"eng", "kor"}, "B": {"eng"}, "C": {"eng"}, "D": {"eng", "kor"}, "E": {"eng", "kor"}, "F": {"eng", "kor"}}[arm]
        assert {k.split("_")[-1] for k in record if k.startswith("train/loss_tokens_")} == expected, record
        print(f"[smoke] arm {arm} {kor_targets} {stop_mode} OK: {record}")
    # Rerun is a no-op.
    pd.run_experiment("gemma-2-0.1B_MaxEnt", 0.3, 0.1, 111, "A", "uniform", "fixed_steps", fixed_steps=3)

    # 3) Relearn one partial-distill model.
    import run_relearn_language as rl
    rl.setups["gemma-2-0.1B_train_only_forget"].update(batch_size=2, max_steps=3, validation_steps=1, max_length=SEQ, device="cpu")
    model = partial_distill_name("C", 0.3, 0.1, 111, "fixed_steps", 3)
    # --warm_cache builds every cache a relearn run reads: the runs that follow (two models) build none.
    rl.warm_cache("gemma-2-0.1B_train_only_forget")
    warm = cache_files()
    rl.launch_relearn("gemma-2-0.1B_train_only_forget", 1e-4, model)
    rl.launch_relearn("gemma-2-0.1B_train_only_forget", 1e-4, partial_distill_name("A", 0.3, 0.1, 111, "fixed_steps", 3))
    assert cache_files() == warm, sorted(cache_files() - warm)
    rl.launch_relearn("gemma-2-0.1B_train_only_forget", 1e-4, model)  # skipped as finished
    record_path = f"{MODEL_DIR}/local_records/relearned_models/relearned_{model.replace('/', '_')}/0.0001_eval_every_1_only_forget_data.txt"
    print("[smoke] relearn OK:", last_record(record_path, "val/kor_ce_loss"))
    # --eval_every 2 over 3 steps: validated at steps 0 and 2, and at the last step; its own record path.
    rl.launch_relearn("gemma-2-0.1B_train_only_forget", 1e-4, model, eval_every=2)
    record_path = record_path.replace("_eval_every_1_", "_eval_every_2_")
    with open(record_path) as f:
        val_steps = sorted({r["train/step"] for r in map(json.loads, f) if "val/kor_ce_loss" in r})
    assert val_steps == [0, 2, 3], val_steps
    assert rl.relearn_finished(record_path, 3)
    print("[smoke] all OK")
    shutil.rmtree(WORK)


if __name__ == "__main__":
    main()
