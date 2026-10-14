#!/bin/bash
#SBATCH --job-name=agi2569-data
#SBATCH --wckey=gradient-routing
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# Needs internet (Hugging Face Hub) and HF_TOKEN with the Gemma licence. CPU only. Every step skips
# outputs that already exist, so a rerun resumes.
ONLINE=1 source cluster/agi2569/env.sh

# 1) Random-init Gemma-2-0.1B (seeded) + Gemma tokenizer, from google/gemma-2-2b's config (gated).
$PY src/prepare_models/reduce_gemma.py --sizes 0.1B
# 2) FineWeb-Edu (English) and FineWeb-2 Korean: 3M documents each, ~2x what step 3 needs.
$PY src/prepare_data/download_datasets.py --settings language --max_rows 3000000
# 3) Tokenize: 1B train + 0.5M valid tokens per language. Check the log for "[WARNING] ... ran out".
$PY src/prepare_data/prepare.py
# 4) Korean token counts for arm D/E --kor_targets unigram.
[ -f "$DATASET_DIR/pretrain/kor_unigram_counts.pt" ] || $PY src/prepare_data/kor_unigram.py
# 5) Build the datasets cache for the four files once, so parallel jobs on different nodes only read it
#    (file locks do not work across nodes on /mnt/polished-lake).
$PY - <<'PYEOF'
import os
from datasets import load_dataset
for name in ("train_eng", "train_kor", "valid_eng", "valid_kor"):
    ds = load_dataset("json", data_files=f"{os.environ['DATASET_DIR']}/pretrain/{name}.jsonl", split="train",
                      cache_dir=os.environ["CACHE_DIR"])
    print(name, len(ds), "documents")
PYEOF
du -sh "$DATASET_DIR"/* "$CACHE_DIR"
