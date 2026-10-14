"""
Counts Korean token frequencies in train_kor.jsonl, for arm D/E's unigram random targets
(run_partial_distill_language.py --kor_targets unigram). Writes a 1-D int64 tensor of counts,
indexed by token id, to {DATASET_DIR}/pretrain/kor_unigram_counts.pt.
"""
import os
import sys

import numpy as np
import orjson
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import DATASET_DIR, MODEL_DIR

KOR_TRAIN = DATASET_DIR + "/pretrain/train_kor.jsonl"
OUT_PATH = DATASET_DIR + "/pretrain/kor_unigram_counts.pt"
TOKENIZER = MODEL_DIR + "/random_init_models/gemma-2-0.1B"


def count_tokens(path, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.int64)
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            counts += np.bincount(orjson.loads(line)["input_ids"], minlength=vocab_size)
            if (i + 1) % 100_000 == 0:
                print(f"[kor_unigram.py] {i + 1} documents, {counts.sum()} tokens", flush=True)
    return counts


if __name__ == "__main__":
    vocab_size = len(AutoTokenizer.from_pretrained(TOKENIZER))
    counts = count_tokens(KOR_TRAIN, vocab_size)
    torch.save(torch.from_numpy(counts), OUT_PATH)
    print(f"[kor_unigram.py] {counts.sum()} tokens, {(counts > 0).sum()} distinct ids of {vocab_size} => {OUT_PATH}")
