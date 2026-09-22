# Retail Structure Qualification

This repository accompanies the `unified_rerun_v1` study of when candidate
retail relations are sufficiently reliable and useful to enter a forecasting
model. The frozen experiment contains 639 tasks on M5, Corporación Favorita,
and Store Item Demand, with 90 series and three 28-day forecast origins per
data set.

## Choose a reproduction path

| Path | What it does | Raw data | Training | Typical use |
|---|---|---:|---:|---|
| A. Quick results | Recomputes published summaries, paired bootstrap intervals, graph reliability, and qualification decisions from released task-level inputs | No | No | Reviewers and most readers |
| B. One task | Prepares one public data set and runs one formal task with resumable outputs | Yes | Optional | Validate the execution chain |
| C. Full experiment | Filters and resumes the frozen 639-task manifest, then invokes the same task runner | Yes | Yes | Full replication |

### A. Quick results

```bash
python -m pip install -r requirements-minimal.txt
python scripts/reproduce_paper_results.py
```

Results are written to `results/reproduced/`. This command performs the
statistical aggregation again; it does not merely compare file hashes. It
requires neither PyTorch nor third-party forecasting repositories.

### B. One task

Download official data as described in [docs/data_download.md](docs/data_download.md), then run:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_v1_data.py --dataset m5 --raw-root data/raw/m5 --output-root data/derived/m5
python scripts/run_single_task.py --task-id v1_0001_m5_20151207_seasonal_naive_s13 --data-root data/derived --output-root outputs/single_task --resume
```

The deterministic M5 example is available without neural training. Locked
90-node validation-demand preparation also works for Favorita and Store Item.
The exact Controlled TCN source and neural feature tensors are not yet
portable; neural tasks fail explicitly rather than substituting a model or
feature set. See [docs/single_task_reproduction.md](docs/single_task_reproduction.md)
and [docs/repository_scope.md](docs/repository_scope.md).

### C. Full experiment

Inspect the entire frozen plan without starting training:

```bash
python scripts/run_unified_rerun_v1.py --config configs/v1/full_unified_training_v1.yaml --data-root data/derived --output-root outputs/full_unified_run_v1 --resume --dry-run --summary-only
```

The dry run resolves 639 unique task keys and supports `--dataset`, `--stage`,
`--task-id`, and `--max-tasks`. Full training was not run while preparing this
repository. Current model-source and neural-feature blockers are stated in
[docs/full_reproduction.md](docs/full_reproduction.md); the verified dependency
boundary is recorded in [docs/reproduction_dependency_audit.md](docs/reproduction_dependency_audit.md).

## Data, optional models, and scope

- Official download locations and prohibited Store Item files are documented in [docs/data_download.md](docs/data_download.md).
- Third-party revisions and licenses are recorded in `third_party/sources.csv`; install only the model needed for a selected task.
- MAGE is not redistributed or auto-installed because its recorded source has no license file.
- Raw data, credentials, caches, checkpoints, and third-party source trees are not committed.
- Candidate directional structures are predictive constructs, not verified causal relations or physical supply-chain links.

See [docs/quickstart.md](docs/quickstart.md), [docs/environment.md](docs/environment.md),
[docs/result_aggregation.md](docs/result_aggregation.md), and
[docs/troubleshooting.md](docs/troubleshooting.md) for details.

## Citation and license

`CITATION.cff`, the repository URL, and the code license still require author
completion before public release. Do not infer a license from the presence of
source files.
