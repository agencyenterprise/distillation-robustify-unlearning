"""AGI-2569: base pretraining on data v2 (packed, fp32 + TF32) with an explicit global batch geometry, to test which
geometry gives the paper's base (English 4.94 / Korean 3.47). Env: BATCH, GA (per process), TAG (output suffix); the
process count comes from `accelerate launch --num_processes`. Paper App. B.2 says full distillation used "the same number
of update steps as its pretrained counterpart" (807 steps at 4 x 60 x 4 = 960 sequences); App. A lists 5 x 80 per GPU."""
import os, sys
sys.path.insert(0, os.getcwd())
import run_pretrain_language as m
setup = sys.argv[1] if len(sys.argv) > 1 else "gemma-2-0.1B_eng+kor"
tag = os.environ["TAG"]
s = m.setups[setup]
s.update(join_or_subsequence=True, batch_size=int(os.environ["BATCH"]), gradient_accumulation_steps=int(os.environ["GA"]),
         output_dir=s["output_dir"] + f"_{tag}", path_local_record=s["path_local_record"].replace(".txt", f"_{tag}.txt"))
m.launch_pretrain(setup)
