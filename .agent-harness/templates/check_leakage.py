"""Fail (exit 1) if any id appears in more than one split.

Usage: python check_leakage.py data/splits.json
splits.json format: {"train": [ids...], "val": [ids...], "test": [ids...]}
Use entity ids (user/patient/session) instead of row ids for grouped data.
"""
import itertools
import json
import sys


def find_overlaps(splits: dict) -> dict:
    overlaps = {}
    for a, b in itertools.combinations(splits, 2):
        common = set(splits[a]) & set(splits[b])
        if common:
            overlaps[f"{a}&{b}"] = len(common)
    return overlaps


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        splits = json.load(f)
    overlaps = find_overlaps(splits)
    if overlaps:
        print(f"LEAKAGE DETECTED: {overlaps}")
        return 1
    print("OK: no overlap between splits.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_leakage.py <splits.json>")
    sys.exit(main(sys.argv[1]))
