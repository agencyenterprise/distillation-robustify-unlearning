# 🔬 Distillation Robustifies Unlearning
<p align="center">
    | 📄 <a href="https://arxiv.org/pdf/2506.06278">arXiv</a> | 🎮 <a href="https://addiefoote.com/distillation-robustifies-demo/">Demo</a> |
</p>

Code used for Distillation Robustifies Unlearning. `/src` directory and `run-*.py` host all runnable scripts.

## 📋 Abstract
Large language models can acquire undesirable capabilities during pretraining that complicate model deployment.
Machine unlearning offers one approach to this challenge by attempting to remove these capabilities, but current methods only offer surface-level suppression that can be easily reversed through finetuning.
We show that distilling unlearned models into randomly initialized students enables robust capability removal.
However, full distillation is computationally expensive for large models.
We address this with Unlearn-Noise-Distill-on-Outputs (UNDO), which approximates full distillation by copying and noising the weights of an unlearned teacher model.
Using this approach, we demonstrate robust unlearning across synthetic language and arithmetic tasks: UNDO achieves Pareto-optimal performance while matching gold-standard data filtering robustness at a fraction of the compute cost, and successfully robustifies unlearning on the more realistic WMDP benchmark.
Given that distillation is already widely used, adding an unlearning step beforehand enables robust capability removal at little extra cost.

## ⚡ Quick Start
For users who want to run a minimal example on language tasks:

1. Complete the [Environment Setup](#-setting-up-environment)
2. Follow [Set up only language](#-set-up-only-language) instructions
3. Run scripts in the following sequence: pretrain, unlearn, partial-distill, relearn

This will get you up and running with the core functionality on a single task type.

## 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU(s) recommended. Params set for H200s. For GPU's with less GPU memory, try reducing batch size and increasing gradient accumulation by the same factor.

## 📝 General Notes
- All scripts are meant to be run from distillation-robustifies-unlearning directory.
- Most run_* scripts will automatically run on all available GPUs, running several processes in parallel or sequentially as available until all specified settings have been run. To restrict the GPU's, precede the command with `CUDA_VISIBLE_DEVICES={desired devices}`.

## 🛠️ Setting Up Environment
1. `git clone https://github.com/AddieFoote/distillation-robustify-unlearning`
2. `pip install uv`
3. `cd distillation-robustifies-unlearning`
4. `uv sync`
5. `source .venv/bin/activate`

## 🚀 Initial Dataset + Model Processing
### ⚙️ Set up for all settings
1. Add a huggingface token to `tokens/hf_token.txt` and a wandb token to `tokens/wandb_token.txt`
2. `python src/prepare_models/reduce_gemma.py`
3. `python src/prepare_data/download_datasets.py`
4. `python src/prepare_data/download_arithmetic.py`
5. Contact us for the WMDP question-answer datasets that were generated via `wmdp_question_extraction.py`
6. `python src/prepare_data/prepare.py`

### 🗣️ Set up only language
Run all steps above, but run step 2 as `python src/prepare_models/reduce_gemma.py --sizes 0.1B` and step 3 as `python src/prepare_data/download_datasets.py --settings language --max_rows 3000000` (only the two language corpora, and enough rows for 1B tokens each). This will substantially speed up steps 3 and 6. Skip steps 4 and 5.

### 🔢 Set up only arithmetic
Run all steps above, but before running step 3, open the file `src/prepare_data/download_datasets.py` and comment out all calls to `download_dataset` except those that are indicated as required for arithmetic setting in the comments (the second to last). This will substantially speed up steps 3 and 6. Skip step 5.

### 🧪 Set up only WMDP
Run all steps above, but before running step 3, open the file `src/prepare_data/download_datasets.py` and comment out all calls to `download_dataset` except those that are indicated as required for WMDP setting in the comments (only the last is not required). This will speed up steps 3 and 6. Skip step 4.

## ▶️ Running Scripts
All scripts can be run using
```
python run_...py
```

## Citation

```bibtex
@misc{
    lee2025distillationrobustifiesunlearning,
    title={Distillation Robustifies Unlearning}, 
    author={Bruce W. Lee and Addie Foote and Alex Infanger and Leni Shor and Harish Kamath and Jacob Goldman-Wetzler and Bryce Woodworth and Alex Cloud and Alexander Matt Turner},
    year={2025},
    eprint={2506.06278},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2506.06278}, 
}
```
