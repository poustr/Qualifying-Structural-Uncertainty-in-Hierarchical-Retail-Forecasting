'''Rebuild the archived Store Item 17/9 tensors from locked public node IDs.'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/author_core_v1'))
from data.dataset import make_origin_tensors
from graphs.business_graph import build_business_graph

VALIDATION = ['2017-05-22', '2017-06-19', '2017-07-17']
LAST_VALIDATION_DATE = pd.Timestamp('2017-08-13')


def load_panel(prepared_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(ROOT / 'configs/v1/locked_nodes/store_item_90.csv')
    pairs = pairs.sort_values('node_index')
    if pairs.node_index.tolist() != list(range(90)):
        raise RuntimeError('Locked Store Item node indices are incomplete')
    node_order = pd.read_csv(prepared_root / 'node_order.csv')
    if node_order.sort_values('node_index').node_id.astype(str).tolist() != [str(i) for i in range(90)]:
        raise RuntimeError('Prepared node order differs from locked Store Item order')
    matrix = pd.read_csv(prepared_root / 'demand_matrix.csv.gz',
                         parse_dates=[0], index_col=0)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.reindex(columns=[str(i) for i in range(90)])
    if matrix.index.max() > LAST_VALIDATION_DATE or matrix.isna().any().any():
        raise RuntimeError('Store Item panel includes test dates or missing cells')
    panel = matrix.rename_axis('date').reset_index().melt(
        id_vars='date', var_name='node_id', value_name='sales')
    panel['node_id'] = panel.node_id.astype(int)
    sample = pairs.rename(columns={'node_index': 'node_id'}).copy()
    sample['item_id'] = sample.item_id.astype(int)
    sample['store_id'] = sample.store_id.astype(int)
    selected_items = sorted(sample.item_id.unique())
    tier = {
        item: ('screen_tier_low_mid' if index < 15 else 'screen_tier_high')
        for index, item in enumerate(selected_items)
    }
    sample['dept_id'] = sample.item_id.map(tier)
    panel = panel.merge(sample, on='node_id', validate='many_to_one')
    panel['item'] = panel.item_id
    panel['store'] = panel.store_id
    panel['sales'] = panel.sales.astype(float)
    if (panel.sales < 0).any():
        raise RuntimeError('Store Item sales must be nonnegative')
    item_total = panel.groupby(['date', 'item']).sales.transform('sum')
    store_total = panel.groupby(['date', 'store']).sales.transform('sum')
    overall_total = panel.groupby('date').sales.transform('sum')
    panel['store_excl'] = (store_total - panel.sales).clip(lower=0)
    panel['item_excl'] = (item_total - panel.sales).clip(lower=0)
    panel['overall_excl'] = (overall_total - panel.sales).clip(lower=0)
    panel['dept_excl'] = panel.item_excl
    panel['store_dept_excl'] = panel.overall_excl
    panel['dow_sin'] = np.sin(2 * np.pi * panel.date.dt.dayofweek / 7)
    panel['dow_cos'] = np.cos(2 * np.pi * panel.date.dt.dayofweek / 7)
    panel['year_sin'] = np.sin(2 * np.pi * panel.date.dt.dayofyear / 365.25)
    panel['year_cos'] = np.cos(2 * np.pi * panel.date.dt.dayofyear / 365.25)
    for column in [
        'discount_proxy', 'price_available', 'event_Cultural',
        'event_National', 'event_Religious', 'event_Sporting',
        'snap_CA', 'onpromotion', 'sell_price',
    ]:
        panel[column] = 0.0
    panel['relative_price'] = 1.0
    panel['record_present'] = 1.0
    panel['return_flag'] = 0.0
    panel['zero_flag'] = (panel.sales == 0).astype(float)
    return panel.sort_values(['date', 'node_id']), sample


def origins(panel: pd.DataFrame) -> list[str]:
    dates = set(VALIDATION)
    for validation in VALIDATION:
        point = pd.Timestamp(validation)
        for k in range(39, 0, -1):
            value = point - pd.Timedelta(days=28 * k)
            if value - pd.Timedelta(days=84) >= panel.date.min():
                dates.add(str(value.date()))
    return sorted(dates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--origin', help='One origin for historical array audit')
    args = parser.parse_args()
    panel, sample = load_panel(args.prepared_root)
    schedule = origins(panel)
    if args.origin and args.origin not in schedule:
        raise SystemExit('Origin outside locked rolling schedule')
    selected = [args.origin] if args.origin else schedule
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    cfg = {'history_length': 84, 'forecast_horizon': 28,
           'model_train_days': 1092, 'origin': VALIDATION[0]}
    records = []
    for origin in selected:
        result = make_origin_tensors(panel, sample, cfg, origin)
        name = f'{origin.replace("-", "")}.npz'
        np.savez_compressed(
            output / name, x_hist=result['x_hist'],
            x_future=result['x_future'], y=result['y'],
            static=result['static'])
        records.append({
            'origin': origin, 'tensor_file': name,
            'history_end': str(result['hist_end'].date()),
            'future_end': str(result['future_end'].date()),
            'is_validation': origin in VALIDATION,
        })
        if origin in VALIDATION:
            adjacency, edges = build_business_graph(panel, sample, origin, 3)
            np.save(output / f'adjacency_{origin.replace("-", "")}_k3.npy',
                    adjacency)
            edges.to_csv(output / f'business_edges_{origin.replace("-", "")}_k3.csv',
                         index=False)
    pd.DataFrame(records).to_csv(output / 'tensor_manifest.csv', index=False)
    sample.to_csv(output / 'node_order.csv', index=False)
    (output / 'STORE_ITEM_NEURAL_TENSOR_MANIFEST.json').write_text(json.dumps({
        'source': 'archived_scheme2_panel_rules_plus_unchanged_author_core',
        'origins_written': len(records),
        'node_count': 90, 'history_features': 17, 'future_features': 9,
        'last_target_date_loaded': str(panel.date.max().date()),
        'independent_test_accessed': False,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'origins_written': len(records),
                      'last_date': str(panel.date.max().date())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
