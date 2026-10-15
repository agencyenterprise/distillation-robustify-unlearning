"""GPU grouping for multi-GPU runs packed on one node. Run: python -m pytest tests/"""
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from utils.parallel_launch import gpu_groups  # noqa: E402


def test_two_groups_of_four_on_a_node():
    assert gpu_groups(4, [str(i) for i in range(8)]) == [["0", "1", "2", "3"], ["4", "5", "6", "7"]]


def test_slurm_ids_kept_and_leftovers_unused():
    assert gpu_groups(4, ["2", "3", "5", "6", "7"]) == [["2", "3", "5", "6"]]


def test_too_few_gpus():
    with pytest.raises(RuntimeError):
        gpu_groups(4, ["0", "1"])
