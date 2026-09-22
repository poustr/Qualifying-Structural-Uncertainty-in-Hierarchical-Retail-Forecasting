from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def test_all_released_artifacts_match_manifest() -> None:
    manifest = ROOT / "artifacts/v1/manifests/ARTIFACT_MANIFEST.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all((ROOT / row["repository_path"]).is_file() for row in rows)
    assert all(_sha256(ROOT / row["repository_path"]) == row["sha256"] for row in rows)
