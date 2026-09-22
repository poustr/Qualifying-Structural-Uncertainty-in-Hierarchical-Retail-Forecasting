# Repository scope

## Included

- The frozen 639-task manifest and experiment configuration.
- Task-level and node-level metric inputs needed to recompute paper tables.
- Frozen graph matrices needed to recompute graph reliability.
- A NumPy/pandas-only aggregation and qualification path.
- Locked 90-node validation-demand and historical neural-tensor preparers for
  all three data sets and verified seasonal-naive tasks.
- A hash-verified, unchanged 13-file snapshot of the author-owned Controlled
  TCN core and its direct dependencies under `src/author_core_v1/`.
- A real M5 Controlled TCN no-graph task with checkpoint and resume checks.
- A real M5 Global LightGBM task with checkpoint, resume, and published-metric checks.
- M5 and Favorita semantic-input generation from official metadata, with
  historical text and embedding-file hash matches.
- A filterable, resumable 639-task dry-run coordinator.
- Adapter interfaces and provenance records for optional model families.

## Not included

- Raw competition data, processed panels, credentials, caches, logs, and model checkpoints.
- Third-party model source trees.
- MAGE source, because the recorded upstream repository has no license file.
- Raw or processed feature tensors. All three tensor-generation commands
  are included and their generated files match historical hashes.

The original per-origin statistical-structure generator has not yet been
ported into a reader-side data-preparation command. Semantic model training
and optional third-party model families have not all passed real single-task
checks; MixLinear and MAGE have unresolved redistribution licenses. The
repository therefore must not be described as fully retrainable. Quick result
recomputation is independent of them and is fully executable.
