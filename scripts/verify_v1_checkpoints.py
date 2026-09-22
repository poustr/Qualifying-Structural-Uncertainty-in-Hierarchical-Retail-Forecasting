"""Verify downloaded unified_rerun_v1 final checkpoints without loading models."""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v1/checkpoint_manifests/CHECKPOINT_MANIFEST.csv"
def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint-root", required=True, type=Path); p.add_argument("--sample", type=int, default=0)
    args = p.parse_args(); rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    if args.sample: rows = rows[:args.sample]
    missing=[]; mismatched=[]
    for row in rows:
        path=args.checkpoint_root / row["expected_release_asset"] / row["task_id"] / row["checkpoint_name"]
        if not path.is_file(): missing.append(row["task_id"]); continue
        if digest(path) != row["checkpoint_sha256"]: mismatched.append(row["task_id"])
    print({"checked":len(rows),"missing":len(missing),"mismatched":len(mismatched)})
    if missing or mismatched: raise SystemExit("checkpoint verification failed")
if __name__ == "__main__": main()
