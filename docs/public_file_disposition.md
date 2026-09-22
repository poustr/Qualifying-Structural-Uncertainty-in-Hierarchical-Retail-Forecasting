# Public-file disposition

| Decision | Files or class | Reason |
|---|---|---|
| Keep | `scripts/reproduce_paper_results.py`, original-formula aggregator, comparison tool, task-level tables, graph diagnostics, node metrics, identity graph matrix | Necessary for path A to recompute results and qualification, not just check hashes |
| Keep | `scripts/prepare_v1_data.py`, `scripts/run_single_task.py`, `configs/v1/locked_nodes/` | Path B: original 90-node identity and verified deterministic tasks |
| Keep | Formal 639-task manifest, `scripts/run_unified_rerun_v1.py`, tests | Path C selection, dry-run, and eventual dispatch |
| Keep | `scripts/validate_v1_checkpoint_manifest.py`, `scripts/verify_v1_checkpoints.py` | Checksum/provenance utilities; neither is presented as neural replay |
| Delete | `docs/data_access.md`, `docs/reproducibility.md` | Duplicated the maintained data-download and three-path guidance, including obsolete blocker claims |
| Delete | `docs/checkpoint_replay.md`, `scripts/download_v1_checkpoints.py`, `scripts/replay_v1_predictions.py` | Described unuploaded assets and a nonfunctional replay placeholder outside the three primary paths |
| Already removed | `aggregation_failures.csv`, internal release audits, caches, local logs | Historical mismatch record or local-only material, not public reproduction input |

The released artifact manifest remains at 28 entries: no published table or
graph diagnostic was removed in this pass. The public node maps are protocol
configuration, not released prediction or training data. The identity graph
matrix is explicitly exempted from the general `.npz` Git ignore rule.
