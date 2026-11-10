"""Data layer: deterministic splits persisted to disk (never re-split ad hoc)."""
import json
from pathlib import Path

import numpy as np


def make_splits(ids, seed=42, val=0.15, test=0.15, groups=None):
    """Split `ids` into train/val/test. If `groups` is given, whole groups stay together."""
    ids = np.asarray(ids)
    keys = np.asarray(groups) if groups is not None else ids
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test, n_val = int(len(uniq) * test), int(len(uniq) * val)
    part = {
        "test": set(uniq[:n_test].tolist()),
        "val": set(uniq[n_test:n_test + n_val].tolist()),
    }
    out = {"train": [], "val": [], "test": []}
    for i, k in zip(ids.tolist(), keys.tolist()):
        name = "test" if k in part["test"] else "val" if k in part["val"] else "train"
        out[name].append(i)
    return out


def save_splits(splits, path="data/splits.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(splits), encoding="utf-8")
