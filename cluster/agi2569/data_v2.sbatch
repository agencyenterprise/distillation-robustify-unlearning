#!/bin/bash
#SBATCH --job-name=agi2569-data-v2
#SBATCH --wckey=gradient-routing
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/polished-lake/home/allegralaro/agi-2569/logs/%x-%j.out
# AGI-2569 data v2: as data.sbatch, but fineweb-2 kor_Hang pinned to v2.0.1 (the paper's likely revision; v2.1.0
# re-filtered Korean on 2025-06-26). English (fineweb-edu sample-10BT) is unchanged across revisions but is re-prepared
# too: prepare.py draws one random.seed(42) stream for the train/valid split, Korean first, so the English split
# depends on the Korean documents. Writes to $WORK/datasets_v2; the v1 data is untouched.
ONLINE=1 source cluster/agi2569/env.sh
export DATASET_DIR=$WORK/datasets_v2
KOR_REVISION=4c6643774aad7f5970955b43fa357008a92471d8
$PY src/prepare_data/download_datasets.py --settings language --max_rows 3000000 --kor_revision $KOR_REVISION
$PY src/prepare_data/prepare.py
[ -f "$DATASET_DIR/pretrain/kor_unigram_counts.pt" ] || $PY src/prepare_data/kor_unigram.py
$PY - <<'PYEOF'
import os
from datasets import load_dataset
for name in ("train_eng", "train_kor", "valid_eng", "valid_kor"):
    ds = load_dataset("json", data_files=f"{os.environ['DATASET_DIR']}/pretrain/{name}.jsonl", split="train",
                      cache_dir=os.environ["CACHE_DIR"])
    print(name, len(ds), "documents")
PYEOF
du -sh "$DATASET_DIR"/* "$CACHE_DIR"
