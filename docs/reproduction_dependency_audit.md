# Original-source dependency and validation audit

This audit distinguishes a completed historical experiment from a portable
public retraining package. It does not authorize replacement models or new
samples. Path labels below identify original local research components without
embedding author-machine absolute paths in the public repository.

| Input | Original SHA-256 | Public role | Current disposition |
|---|---|---|---|
| M5 audit final_sample.csv | `a3b14c9a7ff19128d2ab3a07c5ef635e42c16278d049e0f15c0bc3f12c4e651e` | Original 90 item-store series | IDs recovered from released node metrics; raw data not copied |
| Favorita pre-experiment final_sample_reselected.csv | `c7eb250ef9279e808bd53bcf9cac7d92603fc74c2948d3f0f764437f578d07a6` | Original 90 pairs and order | Minimal public map `configs/v1/locked_nodes/favorita_90.csv`; SHA-256 `d752f818b9609cffd00b8a5ea838a4cda19fb40be9984e9b3d6afd8f266a14ad` |
| Store Item pre-experiment final_sample_screen.csv | `9347846ecb32d2af04156f95a8699a554db26f5a09656c592a425fe0645fe4da` | Original 90 pairs and order | Minimal public map `configs/v1/locked_nodes/store_item_90.csv`; SHA-256 `cae27d7854599dd1cb8c1be0a60088e084a74c449272d7abe29b15e2ed84f6f2` |
| Original Favorita prepare_favorita_aligned.py | `af9e43a6c42038543937c757c258b22c8e62ce4073606c64703d029473a44eef` | Selected panel and feature rules | Read-only; public script reconstructs demand only |
| Original Store Item prepare_store_item_scheme2.py | `d10a036af2d776d4576d2f46f02b4d30c19350b5758e9244697e5197fc3a0858` | Selected panel and feature rules | Read-only; public script reconstructs demand only |
| Controlled TCN mvp_model.py | `799ed0ad58a323c0a48c3a3887c69c2e77a1ac4036c7276af98a891c82358abd` | Original MVPModel imported by the unified-run adapter | Exact first-party copy in `src/author_core_v1/models/` |
| Controlled TCN losses.py | `0a6d0a7fc8cc251768adaaa2480b3286d6de231780f2198b006b8d42a5fcef9d` | Original confidence regularizer | Exact first-party copy in `src/author_core_v1/models/` |

The 13-file first-party snapshot is enumerated in
`docs/author_source_manifest.csv`. Every copied file has the same SHA-256
as the historical validation protocol lock. The unchanged MVPModel passes
a 90-node synthetic forward check with 17 historical and nine future features.
The archived runner still depends on per-origin tensors and relation inputs;
the snapshot alone does not make 639 tasks runnable.

The former author-source path referenced by the historical runner is absent on
this machine. The author explicitly authorized read-only copying from the
adjacent, historically hash-matched source directory. No third-party source,
processed panel, training result, or checkpoint was copied into this repository.
Public licensing remains undecided and must be set by the author before release.

## Real-data checks performed

- M5: 90-node demand preparation succeeded; one formal seasonal-naive task
  reproduced released RMSSE/WAPE/MAE within 2.7e-8.
- Favorita: 1,604 validation-period days by 90 nodes reconstructed from the
  official raw training file and locked ID map. All demand cells equal the
  historical panel. One formal seasonal-naive task matched published metrics
  within 4.8e-7.
- Store Item: 1,686 days by 90 nodes reconstructed from train.csv; all cells
  equal the historical panel. test.csv and sample_submission.csv were not
  opened. One formal seasonal-naive task matched within 1.9e-7.
- A repeated M5 task invocation with --resume skipped the completed task.

The base preparation command emits validation-demand matrices and manifests.
Separate M5, Favorita, and Store Item commands now regenerate all 42 historical
17/9-channel rolling tensors and three business graphs per data set with exact
file-hash matches; see `docs/neural_tensor_reconstruction_audit.md`.
The original formal runner completed one real M5 Controlled TCN task in 30.3
minutes with selection/refit history, checkpoint reload, and resumable task
marker. Its published and rerun RMSSE and MAE are exactly equal, WAPE differs
only by 1.1e-16, and the selected epoch is 12 in both records. All three data
sets passed no-gradient tensor/model shape checks.
M5 and Favorita semantic embeddings are now regenerated from official
metadata with historical text and tensor SHA-256 matches. The semantic
adapters passed synthetic no-gradient forward checks, not real training.
Third-party model installation and all-family runner validation remain open.
The 639-task manifest dry-run is verified; the 639
physical training tasks have not been run in this public copy.
