# Result aggregation and qualification

The public aggregator is a path-parameterized version of the frozen v1
analysis. It preserves the following definitions.

- Node RMSSE is the square root of mean squared forecast error divided by the
  mean squared first difference of up to 1,092 pre-origin observations. Nodes
  with a non-positive scale are missing under the original rule.
- WAPE is absolute error divided by absolute actual demand on the stated
  aggregation unit. MAE is mean absolute error.
- Every paired RMSSE contrast is `control - candidate`, so a positive value
  favors the candidate.
- The 2,000-repetition bootstrap resamples paired node-seed observations within
  each of the three fixed origins and averages the three origin effects. Its
  deterministic seed is 20260901 plus the original contrast counter.
- Graph reliability uses node coverage and weighted Jaccard similarity after
  removing diagonal weights.
- Qualification evaluates, in order, task evaluability, graph reliability,
  propagation utility, and topology specificity. MAGE retains its approximate
  control label; Store Item semantic structure is not applicable.

Run `python scripts/reproduce_paper_results.py`. The command creates all
recomputed tables under `results/reproduced/` and then checks schemas, row
counts, text labels, and numeric values at tolerance `1e-10` against the
released reference tables.
