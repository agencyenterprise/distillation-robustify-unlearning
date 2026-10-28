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
from src.tools.partial_distill_langarith import do_corruption, mlp_modules, select_mlp_neurons  # noqa: E402
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


def select(model, target, frac=0.1, seed=111):
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "x": 1}, unk_token="x")),
                                  pad_token="<pad>")
    return select_mlp_neurons(model, target, frac, seed, saliency_ds=[fake_ds(50, 100), fake_ds(1, 50)],
                              saliency_seqs=6, batch_size=4, tokenizer=tok, max_length=SEQ)


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


@pytest.mark.parametrize("target", ["neuron_act", "neuron_rand"])
def test_selects_exactly_k_and_only_those_change(target):
    model = tiny_model()
    rng_before = torch.get_rng_state()
    masks, record = select(model, target)
    assert torch.equal(rng_before, torch.get_rng_state())  # selection leaves the global RNG alone
    k = round(0.1 * LAYERS * WIDTH)
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
