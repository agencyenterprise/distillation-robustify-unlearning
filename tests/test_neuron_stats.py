"""AGI-2635 neuron stats diagnostic on a tiny random Gemma-2 (CPU, no downloads). Run: python -m pytest tests/"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_neuron_stats_language import flatten, neuron_stats  # noqa: E402
from src.tools.partial_distill_langarith import neuron_saliency  # noqa: E402
from test_targeted_noise import LAYERS, SEQ, WIDTH, fake_ds, select, tiny_model  # noqa: E402,F401
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402


def tok():
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, "x": 1}, unk_token="x")),
                                   pad_token="<pad>")


@torch.no_grad()
def reference(model, ds, rows):
    """Brute force: run the MLPs by hand (output_hidden_states can't see inside them), via a capture of each mlp input."""
    caps = []
    hs = [m.mlp.register_forward_hook(lambda mod, a, o: caps.append(a[0])) for m in model.model.layers]
    sub = ds.select(rows)
    ids = torch.tensor(sub["input_ids"]); mask = torch.tensor(sub["attention_mask"])
    with torch.no_grad():
        model.model(input_ids=ids, attention_mask=mask)
    for h in hs:
        h.remove()
    keep = mask.bool()
    out = {"mean_abs": [], "mean": [], "p_fire": []}
    for layer, x in zip(model.model.layers, caps):
        g = layer.mlp.act_fn(layer.mlp.gate_proj(x))
        h = (g * layer.mlp.up_proj(x))[keep].double()
        out["mean_abs"].append(h.abs().mean(0)); out["mean"].append(h.mean(0))
        out["p_fire"].append((g[keep] > 0).double().mean(0))
    return {k: torch.stack(v).numpy() for k, v in out.items()}, int(keep.sum())


def test_stats_match_brute_force_and_saliency():
    model, ds = tiny_model(), fake_ds(50, 100, n=10)   # row 0 is half padding
    stats, n_tok = neuron_stats(model, ds, 8, 3, tok(), SEQ, torch.device("cpu"), ratio_seqs=2)
    assert set(stats) == {"all", "h0", "h1", "first2"}
    for split, rows in {"all": range(8), "h0": range(4), "h1": range(4, 8), "first2": range(2)}.items():
        ref, n = reference(model, ds, rows)
        assert n_tok[split] == n, split
        for s in ref:
            assert stats[split][s].shape == (LAYERS, WIDTH)
            np.testing.assert_allclose(stats[split][s], ref[s], rtol=1e-5, atol=1e-7, err_msg=f"{split} {s}")
    assert n_tok["h0"] + n_tok["h1"] == n_tok["all"] == 8 * SEQ - SEQ // 2
    assert 0 < stats["all"]["p_fire"].mean() < 1
    # mean_abs on the first r sequences is exactly what neuron_saliency (the 9f170d7 score's input) computes
    (sal,) = neuron_saliency(model, [ds], 2, 4, tok(), SEQ, torch.device("cpu"))
    np.testing.assert_allclose(stats["first2"]["mean_abs"], sal.numpy(), rtol=1e-6)


def test_flatten_keys():
    model = tiny_model()
    per_lang, n = {}, {}
    for lang, (lo, hi) in {"kor": (50, 100), "eng": (1, 50)}.items():
        per_lang[lang], n[lang] = neuron_stats(model, fake_ds(lo, hi), 6, 4, tok(), SEQ, torch.device("cpu"), 64)
    arrays = flatten(per_lang, n)
    for lang in ("kor", "eng"):
        for split in ("all", "h0", "h1", "first64"):
            assert arrays[f"p_fire_{lang}_{split}"].shape == (LAYERS, WIDTH)
            assert int(arrays[f"n_tokens_{lang}_{split}"]) > 0
    # ratio_seqs >= n_seqs: 'first64' covers every sequence
    np.testing.assert_allclose(arrays["mean_abs_kor_first64"], arrays["mean_abs_kor_all"], rtol=1e-6)
