# Quick result reproduction

This path is intended for readers who want to verify the numerical results
without downloading retail data or installing forecasting models.

```bash
python -m pip install -r requirements-minimal.txt
python scripts/reproduce_paper_results.py
```

The second command reads the released 639-task table, 57,510 node-level metric
rows, the formal task manifest, and 153 frozen graph matrices. It recomputes
model, origin, seed, group, graph-reliability, paired-bootstrap, secondary
contrast, and structural-qualification tables. It then compares every field,
row count, numeric value, and qualification label with the released tables.

Expected output: `results/reproduced/reproduction_comparison.json` contains
`passed: true`. On the validated Windows environment the run takes roughly one
to two minutes. Runtime varies by CPU and storage.

Neither PyTorch nor raw data is used. Delete `results/reproduced/` and rerun to
test a clean output directory.
