"""Validate the published checkpoint manifest without accessing checkpoint binaries."""
from __future__ import annotations
import csv, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REQUIRED={"experiment_id","task_id","dataset","model_id","forecast_origin","seed","checkpoint_name","archived_relative_path","expected_release_asset","checkpoint_sha256","bytes","resolved_config_sha256","run_status_sha256","prediction_sha256","model_code_hash","data_protocol_hash","checkpoint_role","release_status","replay_status"}
def main() -> None:
    rows=list(csv.DictReader((ROOT/"artifacts/v1/checkpoint_manifests/CHECKPOINT_MANIFEST.csv").open(encoding="utf-8-sig")))
    assert rows and REQUIRED.issubset(rows[0]), "manifest columns incomplete"
    assert len({r["task_id"] for r in rows}) == len(rows), "duplicate task id"
    assert all(r["checkpoint_name"]=="checkpoint_final.pt" for r in rows)
    assert all(r["release_status"]=="planned_not_uploaded" for r in rows)
    assert all(re.fullmatch(r"[0-9a-f]{64}",r["checkpoint_sha256"]) for r in rows)
    assert all(not re.search(r"^[A-Za-z]:\\|/Users/|/home/",r["archived_relative_path"]) for r in rows)
    print({"records":len(rows),"status":"format_valid","binary_hash_sampling":"deferred_until_assets_exist"})
if __name__=="__main__": main()
