"""Claude Code PostToolUse hook: append-only audit trail of what the agent actually did.

Compare this file with the agent's claims ("I ran the tests") to catch unfaithful reports.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "audit.jsonl"


def main():
    try:
        p = json.load(sys.stdin)
    except ValueError:
        return 0
    inp = p.get("tool_input", {})
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": p.get("tool_name"),
        "target": inp.get("command") or inp.get("file_path"),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
