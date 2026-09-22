# Reproduction environment

## Minimal result reproduction

- Python 3.10 or 3.11 is recommended; continuous integration uses Python 3.11.
- Only NumPy and pandas are required.
- A CPU-only Windows, Linux, or macOS environment is sufficient.
- PyTorch, CUDA, transformers, and optional graph-model repositories are not
  imported by the quick path.

## Training environment

The archived experiment records PyTorch 2.1.0 with CUDA 12.1. The release was
also checked locally with Python 3.10.18, PyTorch 2.1.0+cu121, and an NVIDIA
GeForce RTX 3060 Laptop GPU. These values document the validated environment;
they are not a claim that every compatible hardware and driver combination has
been tested.

- CPU execution is supported for data preparation, seasonal naive tasks, and
  quick aggregation.
- A CUDA-capable GPU is recommended for neural and full-manifest training.
- Install only the optional third-party model required by the selected task.
- MAGE and MixLinear are not installed automatically because the pinned source
  records do not provide a redistributable license file.

## Paths and outputs

All public entry points accept paths from command-line arguments and use
`pathlib`; no author-machine path is required. Quote paths containing spaces.
Task outputs are written beneath the selected output root, and `--resume`
skips only task directories containing a valid `completed.json` marker.

The quick path is fully validated. The original Controlled TCN source is
hash-verified; all three data sets' feature tensors have been reconstructed
exactly from official raw inputs. One M5 no-graph neural task completed through
the original training loop. The remaining model families still require
third-party source, graph/embedding input, and runner audits before a full-run
claim.
