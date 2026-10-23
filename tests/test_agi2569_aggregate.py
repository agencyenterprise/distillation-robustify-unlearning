"""Synthetic-data test of analysis/agi2569. Run: python -m pytest tests/"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "agi2569"))
import aggregate  # noqa: E402
import plot  # noqa: E402
from src.utils.run_names import (  # noqa: E402
    REFERENCE_MODELS, TEACHER_LR, arm_label, partial_distill_name, reference_models, relearn_record_dir, relearn_run_name,
    unlearned_model,
)


def write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def relearn(root, model, lr, kor_by_step, eval_every=5):
    path = f"{root}/relearned_models/{relearn_run_name(model, lr, eval_every)}_only_forget_data.txt"
    records = []
    for step, kor in kor_by_step.items():
        records.append({"val/eng_ce_loss": 3.0, "val/kor_ce_loss": kor, "train/step": step, "train/tokens_seen": 0})
        records.append({"train/ce_loss": kor, "train/step": step})  # train lines are ignored
    write(path, records)


@pytest.fixture
def records(tmp_path):
    root = str(tmp_path)
    # References: unlearned relearns to 2.0, data filter only to 6.0.
    relearn(root, REFERENCE_MODELS["unlearn"], 1e-4, {0: 12.0, 5: 2.0, 500: 2.5})
    relearn(root, REFERENCE_MODELS["filter"], 1e-4, {0: 9.0, 5: 6.0, 500: 6.0})
    relearn(root, REFERENCE_MODELS["base"], 1e-4, {0: 1.5, 5: 1.4, 500: 1.4})
    write(f"{root}/{REFERENCE_MODELS['filter']}.txt", [{"train/tokens_seen": 1000}, {"train/tokens_seen": 2000}])
    # Full distillation: 800 steps, the steps-axis denominator.
    write(f"{root}/{REFERENCE_MODELS['full_distill']}.txt", [{"train/step": 799}, {"train/step": 800}])
    # Arm C, alpha 0.3: LR 1e-4 dips to 3.0 mid-run then ends at 5.0; LR 1e-5 ends at 4.0.
    c = partial_distill_name("C", 0.3, 0.1, 111)
    relearn(root, c, 1e-4, {0: 10.0, 5: 3.0, 500: 5.0})
    relearn(root, c, 1e-5, {0: 10.0, 5: 4.5, 500: 4.0})
    write(f"{root}/{c}.txt", [{"train/tokens_seen": 100, "train/step": 100}, {"train/tokens_seen": 200, "train/step": 200}])
    # Arm A (KL), alpha 0.3: ends at 6.0 with 200 tokens.
    a = partial_distill_name("A", 0.3, 0.1, 111)
    relearn(root, a, 1e-4, {0: 10.0, 5: 6.0, 500: 6.0})
    write(f"{root}/{a}.txt", [{"train/tokens_seen": 200}])
    return root


def row(summary, arm, definition):
    return summary[(summary.arm == arm) & (summary.definition == definition)].iloc[0]


def test_worst_case_definitions(records):
    _, summary = aggregate.aggregate(records)
    assert row(summary, "C", "min_any_step").relearned_kor_loss == 3.0
    assert row(summary, "C", "min_any_step").worst_lr == 1e-4
    assert row(summary, "C", "min_final").relearned_kor_loss == 4.0
    assert row(summary, "C", "min_final").worst_lr == 1e-5
    assert row(summary, "C", "min_final").kor_loss_before == 10.0


def test_normalised_robustness(records):
    _, summary = aggregate.aggregate(records)
    # (3 - 2) / (6 - 2) with min_any_step; (4 - 2.5) / (6 - 2.5) with min_final.
    assert row(summary, "C", "min_any_step").robustness == pytest.approx(0.25)
    assert row(summary, "C", "min_final").robustness == pytest.approx(1.5 / 3.5)
    assert row(summary, "A", "min_any_step").robustness == pytest.approx(1.0)


def test_compute_ratio(records):
    _, summary = aggregate.aggregate(records)
    c, a = row(summary, "C", "min_final"), row(summary, "A", "min_final")
    assert c.compute_ratio_tokens == pytest.approx(0.1)
    assert c.compute_ratio_flops == pytest.approx(0.1)        # CE: 3 units per token, like pretraining
    assert a.compute_ratio_flops == pytest.approx(0.1 * 4 / 3)  # KL: plus a teacher forward
    assert c.compute_ratio_steps == pytest.approx(200 / 800)    # steps over full distillation's steps


def test_envelope_and_plots(records, tmp_path):
    curves, summary = aggregate.aggregate(records)
    env = aggregate.worst_case_curve(curves)
    c_env = env[env.arm == "C"].set_index("step").kor_loss
    assert c_env.to_dict() == {0: 10.0, 5: 3.0, 500: 4.0}
    out = str(tmp_path / "out")
    os.makedirs(out)
    paper = plot.load_paper()
    plot.fig4(curves, out, paper)
    plot.fig5(summary, out, "min_any_step", "fixed_steps_500", "flops", paper)
    plot.fig5(summary, out, "min_final", "early_stop", "steps", paper)
    plot.fig5b(curves, out, "C", "fixed_steps_500", paper)
    for name in ("fig4_relearning", "fig5_robustness", "fig5b_relearning"):
        assert os.path.getsize(f"{out}/{name}.png") > 0


def test_incomplete_lrs_dropped_and_flagged(records):
    # A truncated run (killed at step 5) with a lower loss than any complete run must not set the worst case.
    c = partial_distill_name("C", 0.3, 0.1, 111)
    relearn(records, c, 1e-3, {0: 10.0, 5: 0.5})
    curves, summary = aggregate.aggregate(records)
    assert row(summary, "C", "min_final").relearned_kor_loss == 4.0
    assert row(summary, "C", "min_any_step").relearned_kor_loss == 3.0
    assert row(summary, "C", "min_final").provisional
    assert "0.001" in row(summary, "C", "min_final").lrs_incomplete
    env = aggregate.worst_case_curve(curves)
    assert env[(env.arm == "C") & (env.step == 5)].kor_loss.iloc[0] == 3.0
    # Every reference has 9 of 10 LRs missing in the fixture, so every robustness is provisional.
    assert summary.robustness_provisional.all()


def test_eval_every_in_path_and_coarser_curves(records):
    # Upstream's 5 keeps the calibration path; 25 is in the file name.
    b = partial_distill_name("B", 0.3, 0.1, 111, "early")
    assert relearn_run_name(b, 1e-4) == f"{relearn_record_dir(b)}/0.0001"
    assert relearn_run_name(b, 1e-4, 25) == f"{relearn_record_dir(b)}/0.0001_eval_every_25"
    relearn(records, b, 1e-4, {0: 10.0, 25: 5.0, 50: 4.0, 475: 3.5, 500: 4.5}, eval_every=25)
    relearn(records, b, 1e-5, {0: 10.0, 25: 6.0, 50: 5.0, 475: 5.0, 500: 5.0}, eval_every=25)
    # A second, finer record of one LR (a rerun at the upstream interval) is preferred to the coarse one.
    relearn(records, b, 1e-5, {0: 10.0, 5: 3.0, 25: 6.0, 50: 5.0, 475: 5.0, 500: 4.0})
    curves, summary = aggregate.aggregate(records)
    assert set(curves[curves.arm == "B"].groupby("lr").eval_every.first().to_dict().items()) == {(1e-4, 25), (1e-5, 5)}
    assert row(summary, "B", "min_final").relearned_kor_loss == 4.0
    assert row(summary, "B", "min_final").eval_every == 25
    assert row(summary, "C", "min_final").eval_every == 5
    # The envelope keeps only the steps both LRs validated at.
    env = aggregate.worst_case_curve(curves)
    assert env[env.arm == "B"].set_index("step").kor_loss.to_dict() == {0: 10.0, 25: 5.0, 50: 4.0, 475: 3.5, 500: 4.0}
    assert env[env.arm == "C"].set_index("step").kor_loss.to_dict() == {0: 10.0, 5: 3.0, 500: 4.0}


def test_epoch_mode_run(records, tmp_path):
    # Arm C at alpha 0 trained 2 epochs: its own name, recognised, a curve in fig 4 and a point in fig 5.
    c0 = partial_distill_name("C", 0.0, 0.1, 111, "epoch", epochs=2)
    assert c0.endswith("-seed_111-epochs_2")
    assert partial_distill_name("C", 0.0, 0.1, 111, "epoch").endswith("-epochs_1")
    relearn(records, c0, 1e-4, {0: 10.0, 25: 5.5, 500: 5.0}, eval_every=25)
    write(f"{records}/{c0}.txt", [{"train/tokens_seen": 1600, "train/step": 808}])
    curves, summary = aggregate.aggregate(records)
    r = row(summary[summary.alpha == 0.0], "C", "min_final")
    assert (r.stop, r.relearned_kor_loss, r.compute_ratio_steps) == ("epochs_2", 5.0, 808 / 800)
    out = str(tmp_path / "out")
    os.makedirs(out)
    plot.fig4(curves, out)
    plot.fig5(summary, out, "min_final", "early_stop", "steps")
    assert plot.alpha0_label("C", "epochs_2").startswith("unlearn + CE fine-tune on English, 2 epochs")


def test_teacher_lr_in_every_downstream_path():
    assert TEACHER_LR == 8e-5
    refs3, refs8 = reference_models(3e-5), reference_models(8e-5)
    assert refs8 == REFERENCE_MODELS
    assert refs8["unlearn"] == "unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_8.0e-05"   # run_unlearn_language.py's path
    assert refs3["unlearn"] == "unlearned_models/MaxEnt/gemma-2-0.1B_eng+kor_lr_3.0e-05"
    for role in ("unlearn", "full_distill"):
        assert refs3[role] != refs8[role]
    for role in ("base", "filter"):
        assert refs3[role] == refs8[role]
    assert partial_distill_name("A", 0.3, 0.1, 111, teacher_lr=3e-5) != partial_distill_name("A", 0.3, 0.1, 111)
    assert "teacher_lr_8.0e-05" in partial_distill_name("A", 0.3, 0.1, 111)


def test_other_teacher_lr_kept_apart(records):
    # A 3e-5 teacher's unlearned ref and arm A run: ignored by default, the only teacher-dependent rows at 3e-5.
    relearn(records, reference_models(3e-5)["unlearn"], 1e-4, {0: 11.0, 5: 1.0, 500: 1.0})
    a3 = partial_distill_name("A", 0.3, 0.1, 111, teacher_lr=3e-5)
    relearn(records, a3, 1e-4, {0: 10.0, 5: 5.0, 500: 5.0})
    _, summary = aggregate.aggregate(records)
    assert len(summary[summary.arm == "A"]) == 2   # one row per definition: only the 8e-5 run
    assert row(summary, "A", "min_any_step").robustness == pytest.approx(1.0)
    _, summary3 = aggregate.aggregate(records, teacher_lr=3e-5)
    assert set(summary3.arm.dropna()) == {"A"}
    # unlearn ref 1.0 (min_any_step), filter 6.0: (5 - 1) / (6 - 1)
    assert row(summary3, "A", "min_any_step").robustness == pytest.approx(0.8)


def test_maxent_sweep_check(tmp_path):
    root = str(tmp_path)
    write(f"{root}/{unlearned_model(8e-5)}.txt", [
        {"val/eng_ce_loss": 4.9, "val/kor_ce_loss": 3.5, "train/step": 0},
        {"train/loss": 1.0, "train/step": 100},
        {"val/eng_ce_loss": 5.1, "val/kor_ce_loss": 12.0, "train/step": 100},
        {"val/eng_ce_loss": 5.0, "val/kor_ce_loss": 12.5, "train/step": 100},   # final validation
    ])
    sweep = aggregate.maxent_sweep_check(root).set_index("lr")
    assert len(sweep) == 8
    assert sweep.loc[8e-5, "eng_loss"] == 5.0
    assert sweep.loc[8e-5, "kor_diff"] == pytest.approx(12.5 - 12.3808)
    assert sweep.loc[3e-5, "paper_kor_loss"] == 11.9844
    assert sweep.loc[3e-5, "eng_loss"] != sweep.loc[3e-5, "eng_loss"]   # missing record -> nan


def test_fixed_compute_names(records):
    """Fixed-compute check: KL arms at 302 steps, CE arms at 403, relearn validating every 20 steps."""
    for arm, steps, kor in (("A", 302, "uniform"), ("D", 403, "uniform"), ("F", 403, "uniform")):
        name = partial_distill_name(arm, 0.6, 0.1, 111, "fixed_steps", steps, kor)
        assert name.endswith(f"-fixed_steps_{steps}")
        meta = aggregate.describe_model(relearn_record_dir(name))
        assert (meta["arm"], meta["alpha"], meta["stop"]) == (arm_label(arm, kor), 0.6, f"fixed_steps_{steps}"), meta
        relearn(records, name, 1e-4, {s: 5.0 for s in range(0, 501, 20)}, eval_every=20)
    curves = aggregate.load_relearn_curves(records)
    fixed = curves[curves.stop.isin(["fixed_steps_302", "fixed_steps_403"])]
    assert set(fixed.eval_every) == {20} and fixed.step.max() == 500 and 40 in set(fixed.step)


def test_stop_rule_names(records):
    """--stop_rule any_shard: its own name suffix, parsed into its own stop series; pooled names are unchanged."""
    pooled = partial_distill_name("A", 0.7, 0.1, 111, "early")
    shard = partial_distill_name("A", 0.7, 0.1, 111, "early", stop_rule="any_shard")
    assert shard == pooled + "-stop_any_shard"
    assert partial_distill_name("A", 0.7, 0.1, 111, "early", stop_rule="pooled") == pooled
    with pytest.raises(ValueError):
        partial_distill_name("A", 0.7, 0.1, 111, "fixed_steps", stop_rule="any_shard")
    assert aggregate.describe_model(relearn_record_dir(pooled))["stop"] == "early_stop"
    meta = aggregate.describe_model(relearn_record_dir(shard))
    assert (meta["arm"], meta["alpha"], meta["stop"]) == ("A", 0.7, "early_stop-stop_any_shard")
    epochs2 = partial_distill_name("A", 0.7, 0.1, 111, "early", epochs=2, stop_rule="any_shard")
    assert aggregate.describe_model(relearn_record_dir(epochs2))["stop"] == "early_stop_epochs_2-stop_any_shard"
