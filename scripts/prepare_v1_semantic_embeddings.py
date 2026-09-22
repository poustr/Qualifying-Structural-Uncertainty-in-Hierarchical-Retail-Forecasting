"""Rebuild the locked M5/Favorita semantic input from official metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adapters.semantic_graph_adapter import encode_metadata_texts, metadata_text_sha256


M5_TEMPLATE = "Retail series. Category: {category}. Department: {department}. Store: {store}. State: {state}."
FAVORITA_TEMPLATE = (
    "Retail series. Family: {family}. Class: {class_name}. Perishable: {perishable}. "
    "Store city: {city}. Store state: {state}. Store type: {store_type}. Store cluster: {cluster}."
)
EXPECTED_TEXT_SHA256 = {
    "m5": "ff07c9cceb19547d5f1b449edec7f0b80e09558f851ee0dfe12914ff9645ade1",
    "favorita": "700ca68eab4c3daf7e957aa9b67cba8b2c57cab8c7c4547b0b69741811f37540",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_node_ids(prepared_root: Path) -> list[str]:
    frame = pd.read_csv(prepared_root / "node_order.csv").sort_values("node_index")
    result = frame.node_id.astype(str).tolist()
    if len(result) != 90 or len(set(result)) != 90:
        raise ValueError("expected 90 unique locked node IDs")
    return result


def m5_texts(raw_root: Path, node_ids: list[str]) -> list[str]:
    columns = ["id", "cat_id", "dept_id", "store_id", "state_id"]
    frame = pd.read_csv(raw_root / "sales_train_evaluation.csv", usecols=columns)
    frame = frame.set_index("id").reindex(node_ids)
    if frame.isna().any().any():
        raise ValueError("M5 metadata does not cover the locked node order")
    return [M5_TEMPLATE.format(category=r.cat_id, department=r.dept_id,
                               store=r.store_id, state=r.state_id)
            for r in frame.itertuples()]


def favorita_texts(raw_root: Path, node_ids: list[str], repo_root: Path) -> list[str]:
    pairs = pd.read_csv(repo_root / "configs/v1/locked_nodes/favorita_90.csv").sort_values("node_index")
    if pairs.node_index.tolist() != list(range(90)) or node_ids != [str(i) for i in range(90)]:
        raise ValueError("Favorita node order differs from the locked 90-pair map")
    items = pd.read_csv(raw_root / "items.csv", usecols=["item_nbr", "family", "class", "perishable"])
    stores = pd.read_csv(raw_root / "stores.csv", usecols=["store_nbr", "city", "state", "type", "cluster"])
    frame = pairs.merge(items, left_on="item_id", right_on="item_nbr", how="left", validate="many_to_one")
    frame = frame.merge(stores, left_on="store_id", right_on="store_nbr", how="left", validate="many_to_one")
    if frame[["family", "class", "perishable", "city", "state", "type", "cluster"]].isna().any().any():
        raise ValueError("Favorita metadata does not cover the locked 90 pairs")
    return [FAVORITA_TEMPLATE.format(
        family=row["family"], class_name=row["class"], perishable=row["perishable"],
        city=row["city"], state=row["state"], store_type=row["type"], cluster=row["cluster"])
        for _, row in frame.iterrows()]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["m5", "favorita"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoder-cache-dir", type=Path)
    parser.add_argument("--text-check-only", action="store_true")
    args = parser.parse_args()
    nodes = locked_node_ids(args.prepared_root)
    texts = (m5_texts(args.raw_root, nodes) if args.dataset == "m5" else
             favorita_texts(args.raw_root, nodes, root))
    text_hash = metadata_text_sha256(texts)
    if text_hash != EXPECTED_TEXT_SHA256[args.dataset]:
        raise SystemExit(f"Locked semantic text hash mismatch for {args.dataset}: {text_hash}")
    if args.text_check_only:
        print(json.dumps({"dataset": args.dataset, "nodes": len(nodes),
                          "text_sha256": text_hash, "status": "locked_text_match"}))
        return 0
    encoder_cache = args.encoder_cache_dir or args.output_root / "huggingface"
    embeddings, encoder_manifest = encode_metadata_texts(texts, encoder_cache, device="cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_root / f"{args.dataset}_embeddings.pt"
    temporary = tensor_path.with_suffix(".pt.tmp")
    with temporary.open("wb") as handle:
        torch.save({"node_ids": nodes, "embeddings": embeddings}, handle)
    os.replace(temporary, tensor_path)
    manifest = dict(encoder_manifest)
    manifest.update({"dataset": args.dataset, "template": M5_TEMPLATE if args.dataset == "m5" else FAVORITA_TEMPLATE,
                     "semantic_k": 5, "node_ids": nodes, "texts": texts,
                     "tensor_file": tensor_path.name, "tensor_sha256": file_sha256(tensor_path),
                     "formal_sample_lock": True})
    manifest_path = args.output_root / f"{args.dataset}_embedding_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({"dataset": args.dataset, "nodes": len(nodes), "text_sha256": text_hash,
                      "tensor_sha256": manifest["tensor_sha256"], "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
