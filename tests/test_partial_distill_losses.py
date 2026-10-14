"""Unit tests for the AGI-2569 per-token losses. Run: python -m pytest tests/"""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.loss_functions import (  # noqa: E402
    cross_entropy_loss_fn, forward_kl_loss_fn, masked_mean, per_token_cross_entropy, per_token_forward_kl,
    sample_random_targets,
)

PAD = 0
torch.manual_seed(0)


def test_kl_matches_upstream():
    teacher, student = torch.randn(3, 9, 17), torch.randn(3, 9, 17)
    input_ids = torch.randint(1, 17, (3, 9))
    input_ids[1, 6:] = PAD  # right padding
    labels = input_ids[..., 1:]
    mask = (labels != PAD) & (labels != -100)
    ours = masked_mean(per_token_forward_kl(teacher, student), mask)
    upstream = forward_kl_loss_fn(teacher, student, input_ids, PAD)
    assert torch.allclose(ours, upstream, atol=1e-6)


def test_kl_is_zero_for_identical_logits():
    logits = torch.randn(2, 5, 11)
    assert per_token_forward_kl(logits, logits).abs().max() < 1e-6


def test_ce_matches_upstream_on_real_labels():
    logits = torch.randn(2, 8, 13)
    input_ids = torch.randint(1, 13, (2, 8))  # no pad
    ours = masked_mean(per_token_cross_entropy(logits, input_ids), torch.ones(2, 7, dtype=torch.bool))
    assert torch.allclose(ours, cross_entropy_loss_fn(logits, input_ids, PAD), atol=1e-6)


def test_ce_keeps_targets_equal_to_pad_id():
    """The bug this guards against: upstream CE ignores label == pad id (0 for Gemma), so random target 0 would vanish."""
    logits = torch.randn(1, 4, 5)
    targets = torch.tensor([[3, 0, 0, 0]])  # every next-token target is 0
    attention = torch.ones(1, 4, dtype=torch.long)
    ours = masked_mean(per_token_cross_entropy(logits, targets), attention[..., 1:].bool())
    expected = -torch.log_softmax(logits[0, :3], dim=-1)[:, 0].mean()
    assert torch.allclose(ours, expected, atol=1e-6)
    # Upstream's loss divides 0 by the count and silently returns 0 here.
    assert cross_entropy_loss_fn(logits, targets, PAD).item() == 0.0


def test_masked_mean_ignores_masked_positions():
    loss = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
    mask = torch.tensor([[True, False], [True, True]])
    assert masked_mean(loss, mask).item() == 3.0


def test_random_targets_uniform_and_unigram():
    g = torch.Generator().manual_seed(1)
    uniform = sample_random_targets((4, 1000), 7, g, "cpu")
    assert uniform.shape == (4, 1000) and uniform.min() >= 0 and uniform.max() == 6
    probs = torch.tensor([0.0, 0.0, 0.25, 0.75])
    unigram = sample_random_targets((2, 5000), 4, g, "cpu", probs=probs)
    assert set(unigram.unique().tolist()) <= {2, 3}
    assert abs((unigram == 3).float().mean().item() - 0.75) < 0.02
