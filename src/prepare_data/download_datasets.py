import os
import orjson 
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import CACHE_DIR, DATASET_DIR

OUTPUT_DIR = DATASET_DIR + "/fineweb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_dataset(dataset_id, subset_name, output_filename, split="train", cache_dir=CACHE_DIR, batch_size=1_000_000, max_rows=None,
                     revision=None):
    print(f"Downloading subset '{subset_name}' from '{dataset_id}' (revision {revision or 'default branch'})...")
    if subset_name is None:
        ds = load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir, revision=revision)
    else:
        ds = load_dataset(dataset_id, name=subset_name, split=split, streaming=True, cache_dir=cache_dir, revision=revision)
    total = ds.num_rows if hasattr(ds, "num_rows") and ds.num_rows is not None else None

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"{output_path} already exists; skipping. Delete it to re-download.")
        return
    partial_path = output_path + ".partial"  # renamed on success, so a crash never looks finished
    buffer = []
    count = 0

    with open(partial_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, total=total, desc=f"Processing {subset_name}", mininterval=5):
            buffer.append(orjson.dumps(sample).decode("utf-8"))
            count += 1
            if count % batch_size == 0:
                f.write("\n".join(buffer) + "\n")
                # Optionally print a summary less frequently
                print(f"Processed {count} lines so far...")
                buffer.clear()

            if max_rows is not None and count >= max_rows:
                break

        if buffer:
            f.write("\n".join(buffer) + "\n")
            print(f"Processed {count} lines in total.")
    os.replace(partial_path, output_path)
    print("Finished streaming dataset to:", output_path)

# Upstream downloaded 10M rows of each corpus. prepare.py keeps 1B train tokens per language, which
# needs roughly 1.5M FineWeb-Edu or FineWeb-2 Korean documents, so --max_rows 3000000 leaves ~2x headroom.
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', nargs='+', default=['wmdp', 'language', 'arithmetic'], choices=['wmdp', 'language', 'arithmetic'])
    parser.add_argument('--max_rows', type=int, default=10_000_000)
    # Upstream pins no revision. The paper's data scripts date from 2025-05-30, when fineweb-2's default branch was
    # v2.0.1; v2.1.0 (2025-06-26) re-filtered kor_Hang. AGI-2569 pins v2.0.1 for an exact reproduction.
    parser.add_argument('--kor_revision', default=None, help='fineweb-2 revision for kor_Hang (default: the default branch)')
    args = parser.parse_args()
    max_rows = args.max_rows

    # ====== REQUIRED FOR WMDP SETTING ======
    if 'wmdp' in args.settings:
        download_dataset(
            dataset_id="Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1",
            output_filename="magpie.jsonl",
            max_rows=max_rows,
            subset_name=None
        )

        download_dataset(
            dataset_id="Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1",
            output_filename="magpie3-1.jsonl",
            max_rows=max_rows,
            subset_name=None
        )

        download_dataset(
            dataset_id="Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1",
            output_filename="magpie-qwen.jsonl",
            max_rows=max_rows,
            subset_name=None
        ) 

        download_dataset(
            dataset_id="Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1",
            output_filename="magpie-3.jsonl",
            max_rows=max_rows,
            subset_name=None
        ) 
        download_dataset(
            dataset_id="Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1",
            output_filename="magpie-qwen2.jsonl",
            max_rows=max_rows,
            subset_name=None
        ) 
        download_dataset(
            dataset_id="Magpie-Align/Magpie-Phi3-Pro-1M-v0.1",
            output_filename="magpie-phi3.jsonl",
            max_rows=max_rows,
            subset_name=None
        ) 
        download_dataset(
            dataset_id="Magpie-Align/Magpie-Gemma2-Pro-534K-v0.1",
            output_filename="magpie-gemma2.jsonl",
            max_rows=max_rows,
            subset_name=None
        ) 

        download_dataset(
            dataset_id="Magpie-Align/Magpie-Pro-300K-Filtered",
            output_filename="magpie-filtered.jsonl",
            subset_name=None,
            max_rows=max_rows
        )

        download_dataset(
            dataset_id="Salesforce/wikitext",
            subset_name="wikitext-2-v1",
            output_filename="wikitext.jsonl",
            max_rows=max_rows
        )

        download_dataset(
            dataset_id="legacy-datasets/wikipedia",
            subset_name="20220301.en",
            output_filename="wikipedia.jsonl",
            max_rows=max_rows
        )

    # ====== REQUIRED FOR LANGUAGE SETTING ======
    if 'language' in args.settings:
        download_dataset(
            dataset_id="HuggingFaceFW/fineweb-2",
            subset_name="kor_Hang",
            output_filename="fineweb2_kor.jsonl",
            max_rows=max_rows,
            revision=args.kor_revision
        )

    # ====== REQUIRED FOR LANGUAGE, ARITHMETIC, AND WMDP SETTING ======
    download_dataset(
        dataset_id="HuggingFaceFW/fineweb-edu",
        subset_name="sample-10BT",
        output_filename="fineweb_eng_sample-10BT.jsonl",
        max_rows=max_rows
    )
