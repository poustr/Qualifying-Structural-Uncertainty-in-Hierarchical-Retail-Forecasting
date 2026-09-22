"""Verify hashes recorded in artifacts/v1/manifests/ARTIFACT_MANIFEST.csv."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "v1" / "manifests" / "ARTIFACT_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures = []
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["repository_path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                failures.append(row["repository_path"])
    if failures:
        raise SystemExit("artifact verification failed: " + ", ".join(failures))
    print("v1 artifact verification passed")


if __name__ == "__main__":
    main()
