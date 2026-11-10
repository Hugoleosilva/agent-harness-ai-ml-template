"""Claude Code Stop hook: the agent cannot finish while checks fail.

Runs checks itself (independent of what the model claims). Exit code 2 blocks the stop
and feeds stderr back to the model. Skips checks whose tool/file is not present.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(name, cmd):
    r = subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return f"[{name}] FAILED\n{(r.stdout + r.stderr)[-1500:]}"
    return None


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        payload = {}
    if payload.get("stop_hook_active"):  # avoid infinite loops
        return 0

    failures = []
    if shutil.which("ruff"):
        failures.append(run("ruff", ["ruff", "check", "."]))
    splits = ROOT / "data" / "splits.json"
    if splits.exists():
        failures.append(
            run("leakage", [sys.executable, ".agent-harness/templates/check_leakage.py", str(splits)])
        )
    if (ROOT / "tests").exists():
        probe = subprocess.run([sys.executable, "-m", "pytest", "--version"], check=False, capture_output=True)
        if probe.returncode == 0:
            failures.append(run("pytest", [sys.executable, "-m", "pytest", "tests", "-q"]))

    failures = [f for f in failures if f]
    if failures:
        print("Harness gate: fix these before finishing.\n" + "\n".join(failures), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
