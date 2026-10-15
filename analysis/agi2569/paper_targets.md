# Paper targets: language setting (Figs 4 and 5)

What the Phase 1 calibration should reproduce, from Lee et al. 2025, *Distillation Robustifies Unlearning*,
[arXiv 2506.06278v3](https://arxiv.org/html/2506.06278v3). The HTML version ships the figures as SVG, so curve
values were read from the path coordinates, not by eye. They are accurate to about ±0.01 in loss and ±0.5 points
in percent. The full curves are in `paper_targets.json`, which `plot.py --paper` overlays.

## Definitions

- **Metric.** Korean (forget) validation cross-entropy, on held-out Korean of 500K tokens (App. A). English (retain)
  is the same on 500K English tokens.
- **Relearning attack.** 500 steps of fine-tuning on Korean, batch 8 x 2048 tokens (about 8M tokens), 1 warmup step,
  validation every 5 steps, on one GPU. Attacker learning rates: 1e-3, 7e-4, 4e-4, 1e-4, 7e-5, 5e-5, 1e-5, 7e-6,
  4e-6, 1e-6 (App. A, Table 4).
- **Worst-case adversary.** At each validation step, the lowest Korean loss over the 10 learning rates: "the best
  performance the adversary could obtain by varying the learning rate, at each validation step" (App. A). Checked on
  Fig 4b: every bold curve is the pointwise minimum of its 10 thin curves. In `aggregate.py` this is
  `worst_case_curve`.
- **Robustness (Fig 5a).** (P_UNDO − P_unlearn) / (P_filter − P_unlearn), with P the forget performance. P is the
  worst-case curve's value at step 500. Check: the Fig 5b endpoints give 18.7%, 34.0%, 51.3% and 93.3% for
  alpha 0.2, 0.4, 0.6 and 0.8, against Fig 5a's medians of 18.9%, 35.5%, 56.8% and 93.1%. In `aggregate.py` this is
  the `min_final` definition. `min_any_step` agrees whenever the worst-case curve is still falling at step 500,
  as it is in every paper curve.
- **Compute (Fig 5a).** S_UNDO / S_DataFiltering, with S counted in training steps. The steps are UNDO optimizer
  steps of 960 sequences: 4 GPUs x batch 4 x accumulation 60.
  - **S_DataFiltering is about 806 steps** (inferred). Fig 5a's compute values sit on a 6.2% grid: 12.4, 18.6, 24.8,
    31.0, 37.2, 55.8, 62.0, 68.2, 80.5. Early stopping is checked every 50 steps, and 50 / 0.062 = 806.
  - The Fig 4 full distillation also ran 807 steps (Fig 9, language panels). That is one epoch of the English + Korean
    stream at 960 sequences per step, which ends before the 1000-step cap. App. B.2 says each distilled model
    "undergoes the same number of update steps as its pretrained counterpart".
  - With our records, compute is (partial-distill `train/step`) / (full-distill final `train/step`). This needs the
    full distillation to run on 4 GPUs. It is `--compute steps` in `plot.py`.
- **UNDO stopping rule (Fig 5).** Distil until English recovers to within 5% of the teacher, or one epoch, whichever
  comes first (Sec. 5). This is our `--stop_mode early`, without `--max_steps`. The paper's relearn script names
  (`fixed_steps_500`) belong to an unreleased variant, not to Fig 5. Fig 5a is the median over five seeds; Fig 5b
  does not say whether it shows one seed or the median.

## Scalars

| Quantity | Paper value | Source |
|---|---|---|
| Base model (English + Korean), English / Korean loss | 4.9359 / 3.4672 | App. B.1, App. C |
| MaxEnt lr 3e-5, after unlearning: English / Korean | 4.942 / 11.984 | App. B.1 |
| MaxEnt lr 8e-5, after unlearning: English / Korean | 4.9895 / 12.3808 | App. B.1 (bold), App. C |
| Data-filtered oracle, Korean loss before relearning | 10.96 | Fig 4b, step 0 |
| Full distillation (MaxEnt -> random init): steps, final English / Korean | 807, ~4.98 / ~12.6 | Fig 9b (read off plot, ±0.05) |
| Random-init loss | ~12.5 | Fig 9 |

The App. B.1 sweep gives final English / Korean losses for all eight MaxEnt LRs (1e-5 to 8e-5). They are in
`paper_targets.json` under `scalars.maxent_final`, and they check pretraining and unlearning together. Phase 1 runs
the same sweep; `aggregate.py` compares it row by row (`maxent_sweep.csv`).

## Fig 4b: MaxEnt, language. Worst-case Korean loss during relearning

| Step | 0 | 5 | 10 | 20 | 40 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| Unlearn only (MaxEnt) | 12.33* | 5.01 | 4.08 | 3.71 | 3.59 | 3.51 | 3.49 | 3.49 | 3.48 |
| Unlearn + full distill | >7 | 6.70 | 6.21 | 5.84 | 5.50 | 4.91 | 4.49 | 4.28 | 3.89 |
| Data filtering | 10.96* | 9.52* | 7.51* | 6.51 | 6.30 | 5.82 | 5.28 | 4.76 | 4.28 |

\* Off the drawn axis, which spans 3.3 to 7.0. These values come from path coordinates beyond the clip, so they are
approximate. ">7" means the path was clamped and the value is not recoverable.

## Fig 5b: UNDO (MaxEnt teacher, beta 0.1, early stop), language

| Step | 5 | 10 | 20 | 40 | 100 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|---|
| Unlearn only | 5.48 | 4.38 | 3.76 | 3.60 | 3.53 | 3.51 | 3.50 | 3.50 |
| alpha 0.2 | 6.29 | 5.27 | 4.59 | 4.22 | 3.96 | 3.85 | 3.80 | 3.64 |
| alpha 0.4 | 6.31 | 5.77 | 5.38 | 4.92 | 4.43 | 4.18 | 4.05 | 3.76 |
| alpha 0.6 | 6.55 | 6.14 | 5.68 | 5.30 | 4.78 | 4.43 | 4.27 | 3.90 |
| alpha 0.8 | 6.95 | 6.32 | 5.98 | 5.67 | 5.15 | 4.73 | 4.49 | 4.23 |
| Data filtering | – | 7.51 | 6.51 | 6.30 | 5.82 | 5.28 | 4.76 | 4.28 |

## Fig 5a: robustness vs compute, language

The values are medians over 5 seeds. The ranges are the error bars as drawn; the caption calls them "variation
across five random seeds". The step counts are compute x 806.

| alpha | Compute (% of DF) | ≈ steps | Compute bar | Robustness | Robustness bar |
|---|---|---|---|---|---|
| 0.1 | 0.1% | 1 | – | 7.7% | 5.7–9.7 |
| 0.2 | 12.4% | 100 | – | 18.9% | 16.9–20.9 |
| 0.3 | 13.6% | 110 | 12.4–18.6 | 26.8% | 24.2–31.2 |
| 0.4 | 18.6% | 150 | – | 35.5% | 32.0–37.5 |
| 0.5 | 24.8% | 200 | – | 45.4% | 42.4–50.1 |
| 0.6 | 35.9% | 290 | 31.0–37.2 | 56.8% | 48.1–68.8 |
| 0.7 | 58.2% | 470 | 55.8–62.0 | 84.3% | 77.9–86.3 |
| 0.8 | 76.8% | 620 | 68.2–80.5 | 93.1% | 83.4–98.8 |

Data filtering is the star at (100%, 100%). The paper's dashed trend line runs roughly from (−10%, 0%) to
(94%, 121%). The limitations appendix says the same thing in prose: at alpha 0.6, language UNDO needs about 35% of the pretraining
compute.

## Discrepancies to keep in mind

- **Teacher LR.** App. C and the bold entry in App. B.1 give MaxEnt lr **8e-5** for language. Upstream code hard-codes
  3e-5 (`run_partial_distill_language.py`, `run_distill_language.py`, `run_relearn_language.py`). Fig 4b's unlearned
  model starts relearning at Korean 12.33, which matches 8e-5 (12.38) and not 3e-5 (11.98). We use 8e-5
  (`TEACHER_LR` in `src/utils/run_names.py`); `--teacher_lr 3e-5` reproduces upstream, under separate paths.
- **Unlearn-only endpoint.** Fig 4b's worst case ends at 3.48 and Fig 5b's at 3.496, so the two figures likely come
  from different runs. Both are close to the base model's Korean loss of 3.467.
- **Fixed 500 steps vs early stop.** Fig 5 used early stopping. Fixed 500-step runs on 4 GPUs (0.98B tokens) train
  longer than the paper's early stop for alpha ≤ 0.7 (about 1 to 470 steps) and shorter for alpha 0.8 (about 620
  steps). Only the early-stop runs are a like-for-like comparison with Fig 5, so Phase 1 runs only those; fixed
  mode stays in the code.
- **Pretraining cost.** App. E reports two to three days on 4xH200 for pretraining. Our FLOP estimate is a few hours.
  Check the first pretraining log's throughput before trusting the estimate.
