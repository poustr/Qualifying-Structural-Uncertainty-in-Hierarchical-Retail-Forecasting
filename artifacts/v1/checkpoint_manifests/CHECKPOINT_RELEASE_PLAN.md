# Checkpoint release plan

## Scope

The planned release contains exactly the 639 `checkpoint_final.pt` files in `CHECKPOINT_MANIFEST.csv`. It excludes `checkpoint_best_inner_validation.pt`, `checkpoint_last.pt`, the short-refit/preflight checkpoint, prediction files, graph arrays, caches, and raw data.

## Planned assets

| Dataset group | Planned asset | Formal checkpoints | Size (GiB) |
|---|---|---:|---:|
| M5 | `unified_rerun_v1_checkpoints_m5.zip` | see `CHECKPOINT_RELEASE_GROUPS.csv` | see manifest |
| Favorita | `unified_rerun_v1_checkpoints_favorita.zip` | see `CHECKPOINT_RELEASE_GROUPS.csv` | see manifest |
| Store Item | `unified_rerun_v1_checkpoints_store_item.zip` | see `CHECKPOINT_RELEASE_GROUPS.csv` | see manifest |

No planned asset has been uploaded. No DOI or Release URL exists.

## Distribution decision

Zenodo is recommended for the archival checkpoint record because it provides versioned research artifacts and stable citation support. GitHub Release may be used as a secondary distribution channel if the final verified archive sizes meet the platform's current limits. Do not put binary weights in Git history.

## Packaging rules

Each archive must preserve `asset/task_id/checkpoint_final.pt`, include the matching manifest copy, and be checksum-verified before publication. Publish only after checking licenses for each model dependency and filling author/license metadata.
