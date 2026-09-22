# Single-task reproduction

Install the training environment only for this path:

```bash
python -m pip install -r requirements-training.txt
```

## Deterministic reference

After downloading M5:

```bash
python scripts/prepare_v1_data.py --dataset m5 --raw-root data/raw/m5 --output-root data/derived/m5
python scripts/run_single_task.py --task-id v1_0001_m5_20151207_seasonal_naive_s13 --data-root data/derived --output-root outputs/single_task --resume
```

The task writes its resolved configuration, node hash, empty non-training
history, deterministic forecast, RMSSE/WAPE/MAE, and `completed.json`. A second
call with `--resume` skips the completed task.

The same data-preparation entry point supports `favorita` and `store_item`
using the locked item-store maps in `configs/v1/locked_nodes/`. These commands
prepare validation-period demand. Separate neural-tensor commands for all
three data sets are documented in `docs/neural_tensor_reconstruction_audit.md`.

## Controlled TCN and optional graph task

The formal representative neural task is
`v1_0019_m5_20151207_tcn_no_graph_s13`. The exact author-owned
`MVPModel` and direct dependencies are in `src/author_core_v1/`; their
hashes match the historical validation lock. The original training loop has
completed this real M5 task, including checkpoint reload and resume checks.
Prepare its exact historical tensors first:

```bash
python scripts/prepare_m5_neural_tensors.py --raw-root data/raw/m5 --output-root data/derived/m5/neural_tensors
python scripts/run_single_task.py --task-id v1_0019_m5_20151207_tcn_no_graph_s13 --data-root data/derived --output-root outputs/single_task --device cuda --resume
```

The same Controlled TCN no-graph wrapper accepts the corresponding Favorita
and Store Item tasks after their exact tensors have been prepared. Their
end-to-end neural training has not yet been checked here. The first-party
business, statistical, target-only, and semantic task IDs are routed to the
original formal trainer, but only the M5 no-graph task has passed real neural
training. Statistical tasks additionally require the original per-origin
structure inputs; third-party model tasks require their pinned sources.
No substitute model or tensor is used.

For M5 and Favorita semantic tasks, generate the locked MiniLM input from
official metadata after preparing the demand and neural tensors:

```bash
python scripts/prepare_v1_semantic_embeddings.py --dataset m5 --raw-root data/raw/m5 --prepared-root data/derived/m5 --output-root data/derived/semantic_embeddings
python scripts/prepare_v1_semantic_embeddings.py --dataset favorita --raw-root data/raw/favorita --prepared-root data/derived/favorita --output-root data/derived/semantic_embeddings
```

This step downloads the pinned encoder on first use unless it is already in
the supplied Hugging Face cache. The metadata text hashes and resulting
embedding file hashes matched the historical M5 and Favorita inputs in local
checks. No encoder weights or metadata tensor is redistributed here. Store
Item has no locked semantic embedding task in the formal manifest.

The original Stage 2 Global LightGBM task is also connected and has completed
one real M5 run with checkpoint reload, successful `--resume`, the same selected
tree count, and published RMSSE/MAE exactly matching:

```bash
python scripts/run_single_task.py --task-id v1_0010_m5_20151207_global_lightgbm_s13 --data-root data/derived --output-root outputs/single_task --device cpu --resume
```
