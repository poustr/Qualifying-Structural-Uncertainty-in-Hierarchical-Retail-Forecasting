'''Prepare a locked public-data panel for a v1 reproduction task.'''
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ORIGINS = {
    'm5': ['2015-12-07', '2016-01-04', '2016-02-01'],
    'favorita': ['2017-03-01', '2017-03-29', '2017-04-26'],
    'store_item': ['2017-05-22', '2017-06-19', '2017-07-17'],
}
RAW_FILES = {
    'm5': ['calendar.csv', 'sales_train_evaluation.csv'],
    'favorita': ['train.csv', 'items.csv', 'stores.csv', 'holidays_events.csv'],
    'store_item': ['train.csv'],
}
VALIDATION_END = {'m5': '2016-02-28', 'favorita': '2017-05-23', 'store_item': '2017-08-13'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def released_node_order(root: Path, dataset: str) -> list[str]:
    display = {'m5': 'M5', 'favorita': 'Favorita', 'store_item': 'Store Item'}[dataset]
    metrics = pd.read_csv(root / 'artifacts/v1/tables/node_metrics.csv.gz', usecols=['dataset', 'node_index', 'node_id'])
    nodes = metrics[metrics.dataset == display][['node_index', 'node_id']].drop_duplicates().sort_values('node_index')
    if len(nodes) != 90 or nodes.node_index.nunique() != 90:
        raise RuntimeError(f'Released node order is incomplete for {dataset}')
    return nodes.node_id.astype(str).tolist()


def require_raw_files(dataset: str, raw_root: Path) -> None:
    missing = [str(raw_root / name) for name in RAW_FILES[dataset] if not (raw_root / name).is_file()]
    if missing:
        raise SystemExit('Missing official raw files:\n  ' + '\n  '.join(missing))
    if dataset == 'store_item':
        prohibited = [raw_root / 'test.csv', raw_root / 'sample_submission.csv']
        if any(path.is_file() for path in prohibited):
            print('Notice: prohibited Store Item files are present but will not be opened.')


def prepare_m5(root: Path, raw_root: Path) -> pd.DataFrame:
    node_ids = released_node_order(root, 'm5')
    calendar = pd.read_csv(raw_root / 'calendar.csv', usecols=['d', 'date'], parse_dates=['date'])
    header = pd.read_csv(raw_root / 'sales_train_evaluation.csv', nrows=0).columns
    day_columns = [str(day) for day, date in zip(calendar.d, calendar.date)
                   if str(day) in header and date <= pd.Timestamp(VALIDATION_END['m5'])]
    sales = pd.read_csv(raw_root / 'sales_train_evaluation.csv', usecols=['id'] + day_columns)
    sales = sales[sales.id.astype(str).isin(node_ids)].set_index('id').reindex(node_ids)
    if len(sales) != 90 or sales.isna().all(axis=1).any():
        raise RuntimeError('The official M5 file cannot be aligned to the released 90-node order')
    dates = calendar.set_index('d').loc[day_columns, 'date']
    panel = sales[day_columns].T
    panel.index = pd.DatetimeIndex(dates.to_numpy(), name='date')
    panel.columns = node_ids
    return panel


def locked_pairs(root: Path, dataset: str) -> pd.DataFrame:
    path = root / 'configs' / 'v1' / 'locked_nodes' / f'{dataset}_90.csv'
    if not path.is_file():
        raise SystemExit(f'Locked 90-node mapping is missing: {path}')
    pairs = pd.read_csv(path, dtype={'item_id': str, 'store_id': str})
    if list(pairs.columns) != ['node_index', 'item_id', 'store_id']:
        raise RuntimeError(f'Unexpected node-map columns in {path}')
    pairs = pairs.sort_values('node_index').reset_index(drop=True)
    if len(pairs) != 90 or pairs.node_index.tolist() != list(range(90)):
        raise RuntimeError(f'Node map does not contain exactly indices 0..89: {path}')
    if pairs[['item_id', 'store_id']].duplicated().any():
        raise RuntimeError(f'Duplicate item-store pair in {path}')
    return pairs


def prepare_external(root: Path, dataset: str, raw_root: Path) -> pd.DataFrame:
    pairs = locked_pairs(root, dataset)
    columns = (['date', 'store_nbr', 'item_nbr', 'unit_sales'] if dataset == 'favorita'
               else ['date', 'store', 'item', 'sales'])
    store_col, item_col, value_col = (
        ('store_nbr', 'item_nbr', 'unit_sales') if dataset == 'favorita'
        else ('store', 'item', 'sales')
    )
    cutoff = pd.Timestamp(VALIDATION_END[dataset])
    candidate_items = set(pairs.item_id)
    candidate_stores = set(pairs.store_id)
    chunks = []
    for block in pd.read_csv(raw_root / 'train.csv', usecols=columns,
                             dtype={'date': str, store_col: str, item_col: str},
                             chunksize=2_000_000):
        keep = (block.date.le(VALIDATION_END[dataset])
                & block[item_col].isin(candidate_items)
                & block[store_col].isin(candidate_stores))
        block = block.loc[keep, ['date', item_col, store_col, value_col]]
        if not block.empty:
            chunks.append(block)
    if not chunks:
        raise RuntimeError(f'No locked {dataset} item-store observations in official train.csv')
    observed = pd.concat(chunks, ignore_index=True)
    observed['date'] = pd.to_datetime(observed['date'], errors='raise')
    if observed.duplicated(['date', item_col, store_col]).any():
        raise RuntimeError('Official train.csv contains duplicate locked item-store dates')
    observed[value_col] = pd.to_numeric(observed[value_col], errors='raise').clip(lower=0)
    observed = observed.merge(pairs, left_on=[item_col, store_col],
                              right_on=['item_id', 'store_id'], validate='many_to_one')
    start = pd.Timestamp('2013-01-01')
    dates = pd.date_range(start, cutoff, name='date')
    panel = observed.pivot(index='date', columns='node_index', values=value_col)
    panel = panel.reindex(index=dates, columns=range(90)).fillna(0)
    panel.columns = pairs.node_index.astype(str).tolist()
    if not panel.notna().all().all() or (panel.to_numpy() < 0).any():
        raise RuntimeError('Prepared demand panel contains invalid values')
    return panel


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=sorted(ORIGINS))
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    require_raw_files(args.dataset, args.raw_root)
    panel = (prepare_m5(root, args.raw_root) if args.dataset == 'm5'
             else prepare_external(root, args.dataset, args.raw_root))
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / 'demand_matrix.csv.gz'
    panel.to_csv(matrix_path, compression={'method': 'gzip', 'mtime': 0})
    node_order = pd.DataFrame({'node_index': range(90), 'node_id': panel.columns})
    node_order_path = output / 'node_order.csv'
    node_order.to_csv(node_order_path, index=False)
    eligibility = []
    for origin in ORIGINS[args.dataset]:
        point = pd.Timestamp(origin)
        target_end = point + pd.Timedelta(days=27)
        eligible = panel.index.min() <= point - pd.Timedelta(days=84) and panel.index.max() >= target_end
        eligibility.append({'forecast_origin': origin, 'information_cutoff': str((point - pd.Timedelta(days=1)).date()), 'target_end': str(target_end.date()), 'eligible': bool(eligible)})
    eligibility_path = output / 'forecast_origin_eligibility.csv'
    pd.DataFrame(eligibility).to_csv(eligibility_path, index=False)
    manifest = {
        'dataset': args.dataset, 'nodes': 90, 'history_days': 84, 'horizon_days': 28,
        'forecast_origins': ORIGINS[args.dataset],
        'raw_files': {name: sha256_file(args.raw_root / name) for name in RAW_FILES[args.dataset]},
        'demand_matrix_sha256': sha256_file(matrix_path),
        'node_order_sha256': sha256_file(node_order_path),
        'forecast_origin_eligibility_sha256': sha256_file(eligibility_path),
        'locked_node_map_sha256': (sha256_file(root / 'configs' / 'v1' / 'locked_nodes' /
                                               f'{args.dataset}_90.csv')
                                   if args.dataset != 'm5' else None),
        'prepared_scope': 'validation_targets_only',
        'model_feature_tensors_prepared': False,
        'future_information_used': False,
    }
    (output / 'DATASET_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
