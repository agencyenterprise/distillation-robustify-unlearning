"""AGI-2635 targeted noise on a tiny random Gemma-2 (CPU, no downloads). Run: python -m pytest tests/"""
import copy
import os
import sys

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from src.tools.partial_distill_langarith import do_corruption, mlp_modules, neuron_stats, select_mlp_neurons  # noqa: E402
from src.utils.run_names import partial_distill_name, relearn_record_dir  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "analysis", "agi2569"))
import aggregate  # noqa: E402

LAYERS, HIDDEN, WIDTH, SEQ = 3, 32, 48, 16


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return Gemma2ForCausalLM(Gemma2Config(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=WIDTH, num_hidden_layers=LAYERS, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=SEQ, pad_token_id=0))


def old_do_corruption(model, noise_alpha, noise_beta=0.1, seed=42):
    """do_corruption as of 77c04a8 (before AGI-2635), verbatim."""
    for param in model.parameters():
        if param.requires_grad:
            if len(param.data.shape) == 2:
                noise = torch.nn.init.xavier_uniform_(torch.empty_like(param.data))
            elif len(param.data.shape) == 1:
                noise = torch.zeros_like(param.data)
            corruption = noise_beta * noise
            param.data = (1 - noise_alpha) * param.data + noise_alpha * corruption


def fake_ds(low, high, n=8):
    g = torch.Generator().manual_seed(low)
    ids = torch.randint(low, high, (n, SEQ), generator=g)
    mask = torch.ones(n, SEQ, dtype=torch.long)
    mask[0, SEQ // 2:] = 0
    return Dataset.from_dict({"input_ids": ids.tolist(), "attention_mask": mask.tolist()})


def select(model, target, frac=0.1, seed=111, **kw):
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "x": 1}, unk_token="x")),
                                  pad_token="<pad>")
    if target == "neuron_rule" or kw.get("noise_count") is not None:
        frac = None
    return select_mlp_neurons(model, target, frac, seed, saliency_ds=[fake_ds(50, 100), fake_ds(1, 50)],
                              saliency_seqs=6, batch_size=4, tokenizer=tok, max_length=SEQ, **kw)


def test_all_mode_unchanged():
    a, b = tiny_model(), tiny_model()
    torch.manual_seed(7)
    old_do_corruption(a, 0.6, 0.1)
    rng_old = torch.get_rng_state()
    torch.manual_seed(7)
    do_corruption(b, 0.6, 0.1)
    assert torch.equal(rng_old, torch.get_rng_state())
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(p, q), n


@pytest.mark.parametrize("target,kw", [("neuron_act", {}), ("neuron_rand", {}), ("neuron_rand", {"noise_count": 9}),
                                       ("neuron_rule", {"p_kor_min": 0.5, "p_eng_max": 0.5}),
                                       ("neuron_engrank", {"p_kor_min": 0.0})])
def test_selects_exactly_k_and_only_those_change(target, kw):
    model = tiny_model()
    rng_before = torch.get_rng_state()
    masks, record = select(model, target, **kw)
    assert torch.equal(rng_before, torch.get_rng_state())  # selection leaves the global RNG alone
    k = {"neuron_rule": record["k"]}.get(target, kw.get("noise_count", round(0.1 * LAYERS * WIDTH)))
    assert 0 < k < LAYERS * WIDTH
    assert record["k"] == k == sum(int(m.sum()) for m in masks) == sum(record["per_layer_counts"])
    assert sum(len(s) for s in record["selected"]) == k
    if target == "neuron_act":
        assert record["score_summary"]["kth_selected_over_median"] >= 1
    before = copy.deepcopy(model)
    torch.manual_seed(7)
    do_corruption(model, 1.0, 1.0, neuron_masks=masks)
    rng_targeted = torch.get_rng_state()
    torch.manual_seed(7)
    do_corruption(copy.deepcopy(before), 1.0, 1.0)
    assert torch.equal(rng_targeted, torch.get_rng_state())  # same RNG draws as 'all' mode: same data order after
    mlp_ids = {}
    for i, (gate, up, down) in enumerate(mlp_modules(model)):
        mlp_ids.update({id(gate.weight): (i, 0), id(up.weight): (i, 0), id(down.weight): (i, 1)})
    for (name, p), q in zip(model.named_parameters(), before.parameters()):
        if id(p) not in mlp_ids:
            assert torch.equal(p, q), name
            continue
        i, dim = mlp_ids[id(p)]
        changed = (p != q).any(dim=1 - dim)
        assert torch.equal(changed, masks[i]), name
        # alpha=1: the selected slice is beta * xavier of the FULL matrix, bound sqrt(6 / (fan_in + fan_out))
        bound = (6 / sum(p.shape)) ** 0.5
        sel = p[masks[i]] if dim == 0 else p[:, masks[i]]
        assert sel.abs().max() <= bound and sel.abs().max() > 0.5 * bound, name


def test_rand_is_seeded():
    m = tiny_model()
    a, b, c = (select(m, "neuron_rand", seed=s)[1]["selected"] for s in (1, 1, 2))
    assert a == b != c


def test_act_prefers_forget_neurons():
    """Make neuron 5 of layer 1 fire only on 'Korean' tokens: it must be picked."""
    model = tiny_model()
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        emb[50:] = 0
        emb[50:, 0] = 1.0
        up, gate = model.model.layers[1].mlp.up_proj.weight, model.model.layers[1].mlp.gate_proj.weight
        up[5] = 0; gate[5] = 0
        up[5, 0] = 50.0; gate[5, 0] = 50.0
        emb[1:50, 0] = 0
    masks, _ = select(model, "neuron_act", frac=1 / (LAYERS * WIDTH))
    assert masks[1][5]


def test_names_round_trip():
    plain = partial_distill_name("F", 1.0, 1.0, 111, "early")
    assert partial_distill_name("F", 1.0, 1.0, 111, "early", noise_target="all") == plain
    assert not plain.endswith("_f0.05")
    act = partial_distill_name("F", 1.0, 1.0, 111, "early", noise_target="neuron_act", noise_frac=0.05)
    rand = partial_distill_name("F", 1.0, 1.0, 111, "early", noise_target="neuron_rand", noise_frac=0.05)
    s32 = partial_distill_name("F", 1.0, 1.0, 111, "early", noise_target="neuron_act", noise_frac=0.05, saliency_seqs=32)
    assert act == plain + "-nt_act_f0.05" and rand == plain + "-nt_rand_f0.05" and s32 == act + "_s32"
    for name, noise in ((plain, None), (act, "nt_act_f0.05"), (rand, "nt_rand_f0.05"), (s32, "nt_act_f0.05_s32")):
        meta = aggregate.describe_model(relearn_record_dir(name))
        assert (meta["arm"], meta["alpha"], meta["beta"], meta["stop"], meta["noise"]) == ("F", 1.0, 1.0, "early_stop", noise)
    wu = partial_distill_name("F", 0.5, 0.1, 111, "fixed_steps", warmup_frac=0.1, noise_target="neuron_rand", noise_frac=0.02)
    meta = aggregate.describe_model(relearn_record_dir(wu))
    assert (meta["warmup_frac"], meta["noise"]) == (0.1, "nt_rand_f0.02")
    with pytest.raises(ValueError):
        partial_distill_name("F", 1.0, 1.0, 111, "early", noise_target="neuron_act")
    with pytest.raises(ValueError):
        partial_distill_name("F", 1.0, 1.0, 111, "early", noise_frac=0.05)


def test_rand_count_is_exact_and_seeded():
    m = tiny_model()
    for n in (1, 9, 82 % (LAYERS * WIDTH), LAYERS * WIDTH):
        masks, record = select(m, "neuron_rand", noise_count=n)
        assert record["k"] == n == sum(int(x.sum()) for x in masks) == sum(record["per_layer_counts"])
    a, b, c = (select(m, "neuron_rand", seed=s, noise_count=9)[1]["selected"] for s in (1, 1, 2))
    assert a == b != c
    # the count draws from the same seeded permutation as the fraction: n = round(frac x total) gives the same set
    frac = select(m, "neuron_rand", frac=0.1)[1]["selected"]
    assert select(m, "neuron_rand", noise_count=round(0.1 * LAYERS * WIDTH))[1]["selected"] == frac
    with pytest.raises(AssertionError):
        select_mlp_neurons(m, "neuron_rand", 0.1, 1, noise_count=9)
    with pytest.raises(AssertionError):
        select_mlp_neurons(m, "neuron_act", 0.1, 1, noise_count=9)


def test_names_round_trip_rule_and_count():
    plain = partial_distill_name("F", 1.0, 1.0, 112, "fixed_steps", 81)
    rule = partial_distill_name("F", 1.0, 1.0, 112, "fixed_steps", 81, noise_target="neuron_rule", p_kor_min=0.2,
                                p_eng_max=0.02, saliency_seqs=256)
    rule64 = partial_distill_name("F", 1.0, 1.0, 112, "fixed_steps", 81, noise_target="neuron_rule", p_kor_min=0.1,
                                  p_eng_max=0.1)
    rand = partial_distill_name("F", 1.0, 1.0, 112, "fixed_steps", 81, noise_target="neuron_rand", noise_count=9)
    # saliency_seqs does not enter rand names (as for '-nt_rand_f')
    assert rand == partial_distill_name("F", 1.0, 1.0, 112, "fixed_steps", 81, noise_target="neuron_rand",
                                        noise_count=9, saliency_seqs=256)
    assert rule == plain + "-nt_rule_q0.2_r0.02_s256" and rule64 == plain + "-nt_rule_q0.1_r0.1"
    assert rand == plain + "-nt_rand_n9"
    for name, noise in ((rule, "nt_rule_q0.2_r0.02_s256"), (rule64, "nt_rule_q0.1_r0.1"), (rand, "nt_rand_n9"),
                        (plain, None)):
        meta = aggregate.describe_model(relearn_record_dir(name))
        assert (meta["arm"], meta["alpha"], meta["seed"], meta["stop"], meta["noise"]) == \
            ("F", 1.0, 112, "fixed_steps_81", noise), meta
        m = aggregate.PARTIAL_RE.search(relearn_record_dir(name))
        if noise == "nt_rule_q0.2_r0.02_s256":
            assert (m["noise_target"], m["p_kor_min"], m["p_eng_max"], m["saliency_seqs"]) == ("rule", "0.2", "0.02", "256")
        if noise == "nt_rand_n9":
            assert (m["noise_target"], m["noise_count"], m["noise_frac"]) == ("rand", "9", None)
    bad = [dict(noise_target="neuron_rand", noise_frac=0.05, noise_count=9),
           dict(noise_target="neuron_rand"),
           dict(noise_target="neuron_act", noise_count=9),
           dict(noise_target="neuron_rule", p_kor_min=0.2),
           dict(noise_target="neuron_rule", p_kor_min=0.2, p_eng_max=0.02, noise_frac=0.05),
           dict(noise_target="neuron_act", noise_frac=0.05, p_kor_min=0.2, p_eng_max=0.02),
           dict(noise_target="all", noise_count=9),
           dict(noise_target="neuron_rand", noise_count=0)]
    for kw in bad:
        with pytest.raises(ValueError):
            partial_distill_name("F", 1.0, 1.0, 112, "fixed_steps", 81, **kw)


def brute_force_engrank(model, frac, q):
    """p_fire as neuron_rule computes it, then eligibility, ascending P_eng, ties by flat index, in plain Python."""
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "x": 1}, unk_token="x")),
                                  pad_token="<pad>")
    p = [neuron_stats(model, ds, 6, 4, tok, SEQ, torch.device("cpu"), ratio_seqs=0)[0]["all"]["p_fire"].flatten()
         for ds in (fake_ds(50, 100), fake_ds(1, 50))]
    eligible = [i for i in range(LAYERS * WIDTH) if p[0][i] >= q and p[0][i] > p[1][i]]
    k = round(frac * LAYERS * WIDTH)
    return sorted(sorted(eligible, key=lambda i: (p[1][i], i))[:k]), eligible, p


@pytest.mark.parametrize("frac,q", [(0.1, 0.0), (0.05, 0.2), (1 / (LAYERS * WIDTH), 0.0)])
def test_engrank_matches_brute_force(frac, q):
    model = tiny_model()
    masks, record = select(model, "neuron_engrank", frac=frac, p_kor_min=q)
    want, eligible, p = brute_force_engrank(model, frac, q)
    got = [i * WIDTH + j for i, sel in enumerate(record["selected"]) for j in sel]
    assert got == want and record["k"] == len(want) == round(frac * LAYERS * WIDTH)
    assert record["n_eligible"] == len(eligible) and record["p_kor_min"] == q and record["p_eng_max"] is None
    assert record["effective_r"] == max(p[1][i] for i in want)
    assert record["median_p_eng_selected"] == torch.tensor(p[1][want]).median().item()
    assert record["median_p_kor_selected"] == torch.tensor(p[0][want]).median().item()
    assert record["per_layer_counts"] == [int(m.sum()) for m in masks]


def test_engrank_tie_break_and_too_few_eligible(monkeypatch):
    """Constant p_fire tables: every eligible neuron ties on P_eng, so the lowest flat indices win; k > eligible fails."""
    import src.tools.partial_distill_langarith as pdl
    total = LAYERS * WIDTH
    p_kor = torch.zeros(total, dtype=torch.float64)
    p_kor[::2] = 0.5   # 72 eligible (even indices) at q 0.3
    p_eng = torch.full((total,), 0.1, dtype=torch.float64)
    p_eng[10] = 0.05   # strictly lowest: first
    tables = iter([p_kor, p_eng] * 2)
    monkeypatch.setattr(pdl, "neuron_stats", lambda *a, **kw: (
        {"all": {"p_fire": next(tables).view(LAYERS, WIDTH).numpy()}}, {"all": 1}))
    _, record = select(tiny_model(), "neuron_engrank", frac=0.05, p_kor_min=0.3)   # k = 7
    got = [i * WIDTH + j for i, sel in enumerate(record["selected"]) for j in sel]
    assert got == [0, 2, 4, 6, 8, 10, 12] and record["n_eligible"] == 72 and record["effective_r"] == 0.1
    with pytest.raises(AssertionError, match="neuron_engrank"):
        select(tiny_model(), "neuron_engrank", frac=0.6, p_kor_min=0.3)   # k = 86 > 72


def test_engrank_flags():
    m = tiny_model()
    for kw in (dict(p_kor_min=0.2, p_eng_max=0.02), dict(p_kor_min=0.2, noise_count=9)):
        with pytest.raises(AssertionError):
            select_mlp_neurons(m, "neuron_engrank", 0.1, 1, **kw)
    with pytest.raises(AssertionError):
        select_mlp_neurons(m, "neuron_engrank", 0.1, 1)


def test_names_round_trip_engrank():
    plain = partial_distill_name("F", 1.0, 1.0, 111, "fixed_steps", 81)
    er = partial_distill_name("F", 1.0, 1.0, 111, "fixed_steps", 81, noise_target="neuron_engrank", p_kor_min=0.2,
                              noise_frac=0.005, saliency_seqs=256)
    er64 = partial_distill_name("F", 1.0, 1.0, 111, "fixed_steps", 81, noise_target="neuron_engrank", p_kor_min=0.2,
                                noise_frac=0.02)
    assert er == plain + "-nt_engrank_q0.2_f0.005_s256" and er64 == plain + "-nt_engrank_q0.2_f0.02"
    for name, noise, frac, seqs in ((er, "nt_engrank_q0.2_f0.005_s256", "0.005", "256"),
                                    (er64, "nt_engrank_q0.2_f0.02", "0.02", None)):
        meta = aggregate.describe_model(relearn_record_dir(name))
        assert (meta["arm"], meta["seed"], meta["stop"], meta["noise"]) == ("F", 111, "fixed_steps_81", noise), meta
        m = aggregate.PARTIAL_RE.search(relearn_record_dir(name))
        assert (m["noise_target"], m["p_kor_min"], m["engrank_frac"], m["p_eng_max"], m["noise_frac"],
                m["saliency_seqs"]) == ("engrank", "0.2", frac, None, None, seqs)
    for kw in (dict(p_kor_min=0.2), dict(noise_frac=0.005), dict(p_kor_min=0.2, noise_frac=0.005, p_eng_max=0.02),
               dict(p_kor_min=0.2, noise_frac=0.005, noise_count=9), dict(p_kor_min=1.5, noise_frac=0.005)):
        with pytest.raises(ValueError):
            partial_distill_name("F", 1.0, 1.0, 111, "fixed_steps", 81, noise_target="neuron_engrank", **kw)
