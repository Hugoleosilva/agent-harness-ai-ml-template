import numpy as np

from src.data import make_splits
from src.evaluate import f1_binary
from src.features import Standardizer
from src.model import MajorityBaseline


def test_splits_disjoint_and_deterministic():
    ids = list(range(100))
    a, b = make_splits(ids, seed=1), make_splits(ids, seed=1)
    assert a == b
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["val"]) & set(a["test"]))
    assert sum(len(v) for v in a.values()) == 100


def test_group_split_keeps_groups_together():
    ids = list(range(100))
    groups = [i // 5 for i in ids]
    s = make_splits(ids, groups=groups)
    seen = {name: {g for i, g in zip(ids, groups) if i in set(v)} for name, v in s.items()}
    assert not (seen["train"] & seen["test"]) and not (seen["train"] & seen["val"])


def test_standardizer_uses_train_stats_only():
    tr, te = np.array([[0.0], [2.0]]), np.array([[100.0]])
    s = Standardizer().fit(tr)
    assert s.mean_[0] == 1.0
    assert s.transform(te)[0, 0] > 90


def test_baseline_and_f1():
    m = MajorityBaseline().fit(np.array([0, 0, 1]))
    assert (m.predict(3) == 0).all()
    assert f1_binary(np.array([1, 0, 1]), np.array([1, 0, 0])) == 2 / 3
