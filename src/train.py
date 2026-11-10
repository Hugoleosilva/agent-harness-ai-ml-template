"""Training entry point skeleton. Always seeds first, always reports the baseline."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "seed_everything",
    Path(__file__).resolve().parents[1] / ".agent-harness" / "templates" / "seed_everything.py",
)
_seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seed)
seed_everything = _seed.seed_everything


def main(seed=42):
    seed_everything(seed)
    # 1. load data  2. make_splits + save_splits  3. fit features on train only
    # 4. fit baseline, record metric in ctx_memory.md  5. fit real model, compare
    raise NotImplementedError("Fill in after the Setup Wizard defines the task.")


if __name__ == "__main__":
    main()
