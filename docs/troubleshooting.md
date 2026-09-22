# Troubleshooting

## Quick reproduction imports PyTorch

It should not. Install `requirements-minimal.txt` in a clean environment and
run only `scripts/reproduce_paper_results.py`. Report an issue if `torch`,
`transformers`, or a third-party model is imported.

## NumPy and pandas binary errors

Use the version bounds in `requirements-minimal.txt`. In particular, the
validated quick path uses NumPy 1.x with pandas 2.x.

## Raw file is missing

Follow [data_download.md](data_download.md) and pass the directory containing
the extracted CSV files, not the download archive itself.

## Favorita or Store Item neural training is blocked

The original 90-node item-store mappings have been recovered, and validation
demand preparation has been checked against the archived panels. The original
neural feature tensors and graph inputs are not yet portable. The runner stops
rather than fabricating them or choosing a different sample.

## Controlled TCN task is blocked

The archived adapter imports author-owned model and loss files from a separate
local source tree. Recover those exact files and verify their recorded hashes;
do not replace them with a newly designed TCN.

## CUDA is unavailable

The quick path and deterministic task use CPU. Neural tasks can accept
`--device cpu`, but full reproduction is expected to be substantially slower.
For GPU execution, install a PyTorch build matching the local CUDA driver.

## Resume behavior

Use `--resume`. A task is skipped only if its task directory contains
`completed.json`. Failure records are retained, and a missing task is run
again. The coordinator never advances through a failed task by creating an
empty forecast.

## Windows and Linux paths

All public commands accept normal `pathlib` paths. Quote a path containing
spaces. The scripts contain no author-machine absolute path.

## Optional model installation

Inspect `third_party/sources.csv`, then run, for example,
`python scripts/install_third_party.py --model gts`. MAGE is not auto-installed
because its recorded repository has no license file.
