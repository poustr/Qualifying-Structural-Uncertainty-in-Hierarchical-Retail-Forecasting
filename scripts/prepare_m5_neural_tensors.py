'''Rebuild the locked M5 v1 tensors using the unchanged author feature functions.

Only official validation-period sales columns are read. The output directory
must be derived/local and is never part of the released task-level artifacts.
'''
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/author_core_v1'))
from data.dataset import make_origin_tensors
from data.features import engineer_features
from graphs.business_graph import build_business_graph

VALIDATION = ['2015-12-07', '2016-01-04', '2016-02-01']
LAST_VALIDATION_DATE = pd.Timestamp('2016-02-28')
FIRST_TEST_DATE = pd.Timestamp('2016-02-29')
META = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locked_node_ids() -> list[str]:
    rows = pd.read_csv(
        ROOT / 'artifacts/v1/tables/node_metrics.csv.gz',
        usecols=['dataset', 'node_index', 'node_id'],
    )
    selected = (rows[rows.dataset.eq('M5')][['node_index', 'node_id']]
                .drop_duplicates().sort_values('node_index'))
    if len(selected) != 90 or selected.node_index.tolist() != list(range(90)):
        raise RuntimeError('Released M5 90-node order is incomplete')
    return selected.node_id.astype(str).tolist()


def load_panel(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [raw_root / name for name in
                ['calendar.csv', 'sales_train_evaluation.csv', 'sell_prices.csv']]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit('Missing official M5 raw files: ' + ', '.join(missing))
    node_ids = locked_node_ids()
    calendar = pd.read_csv(raw_root / 'calendar.csv', parse_dates=['date'])
    calendar = calendar[calendar.date.le(LAST_VALIDATION_DATE)].copy()
    dcols = calendar.d.astype(str).tolist()
    raw = pd.read_csv(raw_root / 'sales_train_evaluation.csv',
                      usecols=META + dcols)
    raw = raw[raw.id.astype(str).isin(node_ids)].set_index('id').reindex(node_ids)
    if len(raw) != 90 or raw[META[1:]].isna().any().any():
        raise RuntimeError('Official M5 sales cannot be aligned to released nodes')
    sample = raw[META[1:]].reset_index(drop=True)
    sample['node_id'] = np.arange(90)
    expected_ids = (sample.item_id.astype(str) + '_' +
                    sample.store_id.astype(str) + '_evaluation').tolist()
    if expected_ids != node_ids:
        raise RuntimeError('Released M5 node order differs from original item/store IDs')
    long = raw.reset_index(drop=True).melt(
        id_vars=META[1:], value_vars=dcols, var_name='d', value_name='sales')
    cal_cols = ['date', 'd', 'wm_yr_wk', 'event_type_1',
                'event_type_2', 'snap_CA']
    panel = long.merge(calendar[cal_cols], on='d', validate='many_to_one')
    weeks = set(calendar.wm_yr_wk)
    pairs = set(zip(sample.item_id, sample.store_id))
    parts = []
    for chunk in pd.read_csv(raw_root / 'sell_prices.csv', chunksize=500000):
        mask = (chunk.wm_yr_wk.isin(weeks) &
                pd.Series([(item, store) in pairs for item, store in
                           zip(chunk.item_id, chunk.store_id)], index=chunk.index))
        if mask.any():
            parts.append(chunk.loc[mask])
    if not parts:
        raise RuntimeError('No official sell prices for locked M5 nodes')
    prices = pd.concat(parts, ignore_index=True)
    panel = panel.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'],
                        how='left', validate='many_to_one')
    panel = (panel.merge(sample[['item_id', 'store_id', 'node_id']],
                         on=['item_id', 'store_id'], validate='many_to_one')
             .sort_values(['node_id', 'date']).reset_index(drop=True))
    panel = engineer_features(
        panel, {'origin': VALIDATION[0], 'discount_window_days': 91,
                'discount_ratio': 0.95})
    if panel.date.max() >= FIRST_TEST_DATE:
        raise RuntimeError('Independent-test sales entered M5 feature panel')
    return panel, sample


def requested_origins(panel: pd.DataFrame, requested: str | None) -> list[str]:
    if requested:
        if requested not in VALIDATION and requested not in all_origins(panel):
            raise SystemExit('Requested origin is outside the locked rolling schedule')
        return [requested]
    return all_origins(panel)


def all_origins(panel: pd.DataFrame) -> list[str]:
    origins = set(VALIDATION)
    for formal in VALIDATION:
        point = pd.Timestamp(formal)
        for step in range(39, 0, -1):
            origin = point - pd.Timedelta(days=28 * step)
            if origin - pd.Timedelta(days=84) >= panel.date.min():
                origins.add(str(origin.date()))
    return sorted(origins)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--origin', help='One locked origin for an interface audit; omit for all')
    args = parser.parse_args()
    panel, sample = load_panel(args.raw_root)
    origins = requested_origins(panel, args.origin)
    cfg = {'history_length': 84, 'forecast_horizon': 28,
           'model_train_days': 1092, 'origin': VALIDATION[0]}
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for origin in origins:
        batch = make_origin_tensors(panel, sample, cfg, origin)
        path = output / f'{origin.replace("-", "")}.npz'
        np.savez_compressed(
            path, x_hist=batch['x_hist'], x_future=batch['x_future'],
            y=batch['y'], static=batch['static'])
        rows.append({
            'origin': origin, 'tensor_file': path.name,
            'sha256': sha256(path), 'history_end': str(batch['hist_end'].date()),
            'future_end': str(batch['future_end'].date()),
            'is_validation': origin in VALIDATION,
        })
        if origin in VALIDATION:
            adjacency, edges = build_business_graph(panel, sample, origin, 3)
            np.save(output / f'adjacency_{origin.replace("-", "")}_k3.npy',
                    adjacency)
            edges.to_csv(output / f'business_edges_{origin.replace("-", "")}_k3.csv',
                         index=False)
    pd.DataFrame(rows).to_csv(output / 'tensor_manifest.csv', index=False)
    sample.to_csv(output / 'node_order.csv', index=False)
    (output / 'M5_NEURAL_TENSOR_MANIFEST.json').write_text(json.dumps({
        'source': 'unchanged_author_core_v1_features_and_dataset',
        'origins_written': len(rows),
        'validation_origins': VALIDATION,
        'last_target_date_loaded': str(panel.date.max().date()),
        'independent_test_accessed': False,
        'node_count': 90,
        'history_features': 17,
        'future_features': 9,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'origins_written': len(rows), 'node_count': 90,
                      'last_date': str(panel.date.max().date())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
