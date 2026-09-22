# Locked neural tensor reconstruction

The portable M5, Favorita, and Store Item preparers call the same author-owned
`make_origin_tensors`, feature definitions, and business-graph functions
used in the archived validation experiment. They do not select new nodes,
change model inputs, or read independent-test targets.

| Data set | Rolling-origin tensors | Historical NPZ SHA-256 matches | Validation business-graph SHA-256 matches | Last demand date read |
|---|---:|---:|---:|---|
| M5 | 42 | 42/42 | 3/3 | 2016-02-28 |
| Favorita | 42 | 42/42 | 3/3 | 2017-05-23 |
| Store Item | 42 | 42/42 | 3/3 | 2017-08-13 |

Each tensor contains `x_hist` of shape `[1,90,84,17]`,
`x_future` of shape `[1,90,28,9]`, the 28-day target, and
90 three-field static IDs. The file hashes match the existing historical
cache, not only floating-point values. The comparison was read-only with
respect to the archived cache; the generated tensors were kept in a local
temporary directory and are not released with this repository.

The M5 public entry point needs official `calendar.csv`,
`sales_train_evaluation.csv`, and `sell_prices.csv`:

```bash
python scripts/prepare_m5_neural_tensors.py --raw-root data/raw/m5 --output-root data/derived/m5/neural_tensors
```

The Store Item entry point consumes the validation-demand panel generated
by `prepare_v1_data.py` from official `train.csv`; it does not open
`test.csv` or `sample_submission.csv`:

```bash
python scripts/prepare_store_item_neural_tensors.py --prepared-root data/derived/store_item --output-root data/derived/store_item/neural_tensors
```

The Favorita preparer reconstructs the archived promotion, holiday, return,
and generated feature fields from the official raw tables and locked 90-pair
map. Its 42 tensor files and three graphs match the historical file hashes:

```bash
python scripts/prepare_favorita_neural_tensors.py --raw-root data/raw/favorita --output-root data/derived/favorita/neural_tensors
```

Exact tensor reconstruction does not by itself validate all model families.
One real M5 Controlled TCN task has completed with checkpoint reload and
resume checks; Favorita and Store Item have passed no-gradient model-interface
checks but not a real neural training task in this public copy.
