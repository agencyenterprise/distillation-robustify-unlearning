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
from src.utils.run_names import REFERENCE_MODELS, partial_distill_name, relearn_record_dir  # noqa: E402


def write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def relearn(root, model, lr, kor_by_step):
    path = f"{root}/relearned_models/{relearn_record_dir(model)}/{lr}_only_forget_data.txt"
    records = []
    for step, kor in kor_by_step.items():
        records.append({"val/eng_ce_loss": 3.0, "val/kor_ce_loss": kor, "train/step": step, "train/tokens_seen": 0})
        records.append({"train/ce_loss": kor, "train/step": step})  # train lines are ignored
    write(path, records)


@pytest.fixture
def records(tmp_path):
    root = str(tmp_path)
    # References: unlearned relearns to 2.0, data filter only to 6.0.
    relearn(root, REFERENCE_MODELS["unlearn"], 1e-4, {0: 12.0, 5: 2.0, 10: 2.5})
    relearn(root, REFERENCE_MODELS["filter"], 1e-4, {0: 9.0, 5: 6.0, 10: 6.0})
    relearn(root, REFERENCE_MODELS["base"], 1e-4, {0: 1.5, 5: 1.4, 10: 1.4})
    write(f"{root}/{REFERENCE_MODELS['filter']}.txt", [{"train/tokens_seen": 1000}, {"train/tokens_seen": 2000}])
    # Arm C, alpha 0.3: LR 1e-4 dips to 3.0 mid-run then ends at 5.0; LR 1e-5 ends at 4.0.
    c = partial_distill_name("C", 0.3, 0.1, 111)
    relearn(root, c, 1e-4, {0: 10.0, 5: 3.0, 10: 5.0})
    relearn(root, c, 1e-5, {0: 10.0, 5: 4.5, 10: 4.0})
    write(f"{root}/{c}.txt", [{"train/tokens_seen": 100}, {"train/tokens_seen": 200}])
    # Arm A (KL), alpha 0.3: ends at 6.0 with 200 tokens.
    a = partial_distill_name("A", 0.3, 0.1, 111)
    relearn(root, a, 1e-4, {0: 10.0, 5: 6.0, 10: 6.0})
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


def test_envelope_and_plots(records, tmp_path):
    curves, summary = aggregate.aggregate(records)
    env = aggregate.worst_case_curve(curves)
    c_env = env[env.arm == "C"].set_index("step").kor_loss
    assert c_env.to_dict() == {0: 10.0, 5: 3.0, 10: 4.0}
    out = str(tmp_path / "out")
    os.makedirs(out)
    plot.fig4(curves, out)
    plot.fig5(summary, out, "min_any_step", "fixed_steps_500", "flops")
    assert os.path.getsize(f"{out}/fig4_relearning.png") > 0 and os.path.getsize(f"{out}/fig5_robustness.png") > 0
