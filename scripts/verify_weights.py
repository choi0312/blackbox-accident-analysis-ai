#!/usr/bin/env python3
"""Verify that omitted model checkpoints match the submitted artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights.manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    """Hash a large file without loading the whole checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = 0
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            print(f"MISSING  {record['path']}")
            failures += 1
            continue
        if path.stat().st_size != record["bytes"]:
            print(f"SIZE     {record['path']}")
            failures += 1
            continue
        actual = sha256(path)
        if actual != record["sha256"]:
            print(f"SHA256   {record['path']}")
            failures += 1
            continue
        print(f"OK       {record['path']}")
    if failures:
        print(f"\n{failures} checkpoint(s) missing or invalid.")
        return 1
    print("\nAll checkpoints match weights.manifest.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
