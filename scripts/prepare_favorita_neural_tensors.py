'''Rebuild archived Favorita neural tensors from official validation-period rows.'''
from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORES = [44, 3, 47]
VALIDATION = ['2017-03-01', '2017-03-29', '2017-04-26']
CUTOFF = '2017-05-23'
HIST = [
    'sales', 'store_excl', 'dept_excl', 'store_dept_excl',
    'onpromotion', 'promotion_known', 'record_present',
    'dow_sin', 'dow_cos', 'year_sin', 'year_cos',
    'event_Holiday', 'event_Event', 'event_Additional', 'event_Other',
    'return_flag', 'zero_flag',
]
FUT = [
    'dow_sin', 'dow_cos', 'year_sin', 'year_cos',
    'event_Holiday', 'event_Event', 'event_Additional', 'event_Other',
    'onpromotion',
]


def locked_sample(raw_root: Path) -> pd.DataFrame:
    pairs = pd.read_csv(ROOT / 'configs/v1/locked_nodes/favorita_90.csv')
    pairs = pairs.sort_values('node_index')
    if pairs.node_index.tolist() != list(range(90)):
        raise RuntimeError('Locked Favorita node indices are incomplete')
    sample = pairs.rename(columns={
        'node_index': 'node_id', 'item_id': 'item_nbr',
        'store_id': 'store_nbr',
    }).copy()
    sample['item_nbr'] = sample.item_nbr.astype(int)
    sample['store_nbr'] = sample.store_nbr.astype(int)
    sample = sample.merge(pd.read_csv(raw_root / 'items.csv'), on='item_nbr')
    sample = sample.merge(pd.read_csv(raw_root / 'stores.csv'), on='store_nbr')
    sample = sample.sort_values('node_id').reset_index(drop=True)
    if len(sample) != 90 or sample.node_id.tolist() != list(range(90)):
        raise RuntimeError('Official Favorita metadata does not match locked IDs')
    if sample.sort_values(['family', 'item_nbr', 'store_nbr']).node_id.tolist() != list(range(90)):
        raise RuntimeError('Locked Favorita order differs from original order')
    return sample


def pretest_row_count(train_path: Path) -> int:
    count = 0
    for dates in pd.read_csv(train_path, usecols=['date'], chunksize=2_000_000):
        within = dates.date.astype(str).le(CUTOFF)
        count += int(within.sum())
        if not bool(within.all()):
            first_later = int(np.flatnonzero(~within.to_numpy())[0])
            if bool(within.iloc[first_later:].any()):
                raise RuntimeError('Favorita raw train is not sorted by date at cutoff')
            break
    return count


def load_panel(raw_root: Path, sample: pd.DataFrame) -> pd.DataFrame:
    train_path = raw_root / 'train.csv'
    count = pretest_row_count(train_path)
    selected = set(sample.item_nbr)
    parts = []
    for block in pd.read_csv(
        train_path, usecols=['date', 'store_nbr', 'item_nbr',
                             'unit_sales', 'onpromotion'],
        parse_dates=['date'], chunksize=2_000_000, nrows=count,
    ):
        frame = block[block.store_nbr.isin(STORES) &
                      block.item_nbr.isin(selected)]
        if not frame.empty:
            parts.append(frame.copy())
    if not parts:
        raise RuntimeError('No selected Favorita rows before cutoff')
    observed = pd.concat(parts, ignore_index=True)
    observed['return_flag'] = observed.unit_sales < 0
    observed['observed_sales'] = observed.unit_sales.clip(lower=0)
    observed['promotion_known'] = observed.onpromotion.notna()
    observed['onpromotion'] = observed.onpromotion.fillna(False).astype(bool)
    dates = pd.date_range('2013-01-01', CUTOFF)
    grid = pd.MultiIndex.from_product(
        [sorted(selected), STORES, dates],
        names=['item_nbr', 'store_nbr', 'date'],
    ).to_frame(index=False)
    panel = grid.merge(
        observed[['item_nbr', 'store_nbr', 'date', 'unit_sales',
                  'observed_sales', 'onpromotion', 'promotion_known',
                  'return_flag']],
        how='left', validate='one_to_one',
    )
    panel['record_present'] = panel.unit_sales.notna()
    panel['observed_sales'] = panel.observed_sales.fillna(0).astype('float32')
    for column in ['onpromotion', 'promotion_known', 'return_flag']:
        panel[column] = panel[column].fillna(False).astype(bool)
    panel = panel.merge(
        sample[['item_nbr', 'store_nbr', 'node_id', 'family',
                'state', 'city', 'type', 'cluster']],
        on=['item_nbr', 'store_nbr'], how='left', validate='many_to_one',
    )
    panel['sales'] = panel.observed_sales
    panel['item_id'] = panel.item_nbr.astype(str)
    panel['store_id'] = panel.store_nbr.astype(str)
    panel['dept_id'] = panel.family
    store_total = panel.groupby(['date', 'store_nbr']).sales.transform('sum')
    family_total = panel.groupby(['date', 'family']).sales.transform('sum')
    store_family_total = panel.groupby(
        ['date', 'store_nbr', 'family']).sales.transform('sum')
    panel['store_excl'] = store_total - panel.sales
    panel['dept_excl'] = family_total - panel.sales
    panel['store_dept_excl'] = store_family_total - panel.sales
    calendar = panel.date.dt
    panel['dow_sin'] = np.sin(2 * np.pi * calendar.dayofweek / 7)
    panel['dow_cos'] = np.cos(2 * np.pi * calendar.dayofweek / 7)
    panel['year_sin'] = np.sin(2 * np.pi * calendar.dayofyear / 365.25)
    panel['year_cos'] = np.cos(2 * np.pi * calendar.dayofyear / 365.25)
    holidays = pd.read_csv(raw_root / 'holidays_events.csv',
                           parse_dates=['date'])
    holidays = holidays[~holidays.transferred.fillna(False)].copy()
    holidays['kind'] = holidays.type.where(
        holidays.type.isin(['Holiday', 'Event', 'Additional']), 'Other')
    holiday_flags = holidays.assign(v=1).pivot_table(
        index='date', columns='kind', values='v',
        aggfunc='max', fill_value=0).reset_index()
    for kind in ['Holiday', 'Event', 'Additional', 'Other']:
        if kind not in holiday_flags:
            holiday_flags[kind] = 0
    panel = panel.merge(
        holiday_flags[['date', 'Holiday', 'Event', 'Additional', 'Other']],
        on='date', how='left',
    )
    for kind in ['Holiday', 'Event', 'Additional', 'Other']:
        panel[f'event_{kind}'] = panel[kind].fillna(0).astype('float32')
    # The archived tensor runner reloaded the prepared CSV before fitting
    # per-origin robust scalers. Preserve that serialization boundary.
    panel = pd.read_csv(StringIO(panel.to_csv(index=False)), parse_dates=['date'])
    panel['zero_flag'] = (panel.sales == 0).astype(float)
    if panel.date.max() > pd.Timestamp(CUTOFF):
        raise RuntimeError('Favorita independent-test demand was loaded')
    return panel


def origins(panel: pd.DataFrame) -> list[str]:
    selected = set(VALIDATION)
    for validation in VALIDATION:
        point = pd.Timestamp(validation)
        for k in range(39, 0, -1):
            origin = point - pd.Timedelta(days=28 * k)
            if origin - pd.Timedelta(days=84) >= panel.date.min():
                selected.add(str(origin.date()))
    return sorted(selected)


def business_adj(panel: pd.DataFrame, sample: pd.DataFrame,
                 origin: str, top_k: int = 3) -> np.ndarray:
    adjacency = np.eye(90, dtype=np.float32)
    for _, group in sample.groupby('item_nbr'):
        nodes = group.node_id.tolist()
        for target in nodes:
            for source in nodes:
                adjacency[target, source] = 1
    history = panel[
        (panel.date < pd.Timestamp(origin)) &
        (panel.date >= pd.Timestamp(origin) - pd.Timedelta(days=364))
    ].pivot(index='date', columns='node_id', values='sales')
    for _, group in sample.groupby('family'):
        nodes = group.node_id.tolist()
        corr = np.log1p(history[nodes]).corr().fillna(0)
        for target in nodes:
            for source in corr[target].drop(target).abs().nlargest(top_k).index:
                adjacency[target, int(source)] = 1
                adjacency[int(source), target] = 1
    return adjacency


def make_tensors(panel: pd.DataFrame, sample: pd.DataFrame,
                 origin: str) -> dict:
    point = pd.Timestamp(origin)
    fit = panel[
        (panel.date >= point - pd.Timedelta(days=1092)) &
        (panel.date < point)
    ].copy()
    scaled_fit = fit[HIST].copy()
    for column in HIST[:4]:
        scaled_fit[column] = np.log1p(scaled_fit[column].clip(lower=0))
    median = scaled_fit[HIST[:4]].median()
    iqr = (scaled_fit[HIST[:4]].quantile(0.75) -
           scaled_fit[HIST[:4]].quantile(0.25)).replace(0, 1)
    history = panel[
        (panel.date >= point - pd.Timedelta(days=84)) &
        (panel.date < point)
    ].sort_values(['node_id', 'date']).copy()
    future = panel[
        (panel.date >= point) &
        (panel.date < point + pd.Timedelta(days=28))
    ].sort_values(['node_id', 'date']).copy()
    if len(history) != 90 * 84 or len(future) != 90 * 28:
        raise RuntimeError(f'Incomplete Favorita origin {origin}')
    for column in HIST[:4]:
        history[column] = np.log1p(history[column].clip(lower=0))
    history[HIST[:4]] = (history[HIST[:4]] -
                         median[HIST[:4]]) / iqr[HIST[:4]]
    static = np.stack([
        pd.Categorical(sample.item_nbr).codes,
        pd.Categorical(sample.family,
                       categories=['GROCERY I', 'BEVERAGES']).codes,
        pd.Categorical(sample.store_nbr, categories=STORES).codes,
    ], axis=1).astype(np.int64)
    return {
        'x_hist': history[HIST].to_numpy(np.float32).reshape(90, 84, 17)[None],
        'x_future': future[FUT].to_numpy(np.float32).reshape(90, 28, 9)[None],
        'y': future.sales.to_numpy(np.float32).reshape(90, 28)[None],
        'static': static,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--origin', help='One origin for historical array audit')
    args = parser.parse_args()
    for filename in ['train.csv', 'items.csv', 'stores.csv', 'holidays_events.csv']:
        if not (args.raw_root / filename).is_file():
            raise SystemExit(f'Missing official Favorita file: {filename}')
    sample = locked_sample(args.raw_root)
    panel = load_panel(args.raw_root, sample)
    schedule = origins(panel)
    if args.origin and args.origin not in schedule:
        raise SystemExit('Origin outside locked rolling schedule')
    selected = [args.origin] if args.origin else schedule
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for origin in selected:
        result = make_tensors(panel, sample, origin)
        name = f'{origin.replace("-", "")}.npz'
        np.savez_compressed(output / name, **result)
        rows.append({'origin': origin, 'tensor_file': name,
                     'is_validation': origin in VALIDATION})
        if origin in VALIDATION:
            np.save(output / f'adjacency_{origin.replace("-", "")}_k3.npy',
                    business_adj(panel, sample, origin))
    pd.DataFrame(rows).to_csv(output / 'tensor_manifest.csv', index=False)
    sample.to_csv(output / 'node_order.csv', index=False)
    (output / 'FAVORITA_NEURAL_TENSOR_MANIFEST.json').write_text(json.dumps({
        'source': 'archived_favorita_panel_and_tensor_rules',
        'origins_written': len(rows), 'node_count': 90,
        'history_features': 17, 'future_features': 9,
        'last_target_date_loaded': str(panel.date.max().date()),
        'independent_test_accessed': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'origins_written': len(rows),
                      'last_date': str(panel.date.max().date())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
