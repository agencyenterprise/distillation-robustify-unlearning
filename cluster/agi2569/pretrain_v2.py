"""AGI-2569 exact Fig 4b reproduction: base pretraining as the released code and README literally run it.
Joined (packed) documents, one process (README: `python run_pretrain_language.py`), no mixed precision (upstream loads the
model in fp32 and sets TF32 matmuls), on data v2 (Korean pinned to fineweb-2 v2.0.1). Setup id from argv (default base)."""
import os, sys
sys.path.insert(0, os.getcwd())
import run_pretrain_language as m
setup = sys.argv[1] if len(sys.argv) > 1 else "gemma-2-0.1B_eng+kor"
m.setups[setup]["join_or_subsequence"] = True
m.launch_pretrain(setup)
