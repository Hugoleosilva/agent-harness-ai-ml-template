import importlib.util
import random
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[1] / ".agent-harness" / "templates"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, T / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_is_reproducible():
    seed = _load("seed_everything").seed_everything
    seed(7)
    a = (random.random(), np.random.rand())
    seed(7)
    b = (random.random(), np.random.rand())
    assert a == b


def test_leakage_detected():
    find = _load("check_leakage").find_overlaps
    assert find({"train": [1, 2], "test": [2, 3]}) == {"train&test": 1}
    assert find({"train": [1], "val": [2], "test": [3]}) == {}
