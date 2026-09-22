'''Recompute published v1 summaries from released task-level inputs.'''
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260901
DATASETS = ['M5', 'Favorita', 'Store Item']
GRAPH_MODELS = ['tcn_business', 'tcn_stat_confidence', 'gts_native', 'mtgnn_native', 'mage_native', 'semantic_knn_tcn']
FIXED_GRAPH_MODELS = {'tcn_business', 'semantic_knn_tcn'}
CONTRASTS = {
    'tcn_business': {'graph': 'tcn_no_graph'},
    'tcn_stat_confidence': {'graph': 'tcn_no_graph', 'weight': 'tcn_stat_equal', 'topology': 'tcn_stat_placebo'},
    'gts_native': {'graph': 'gts_no_graph', 'topology': 'gts_placebo'},
    'mtgnn_native': {'graph': 'mtgnn_no_graph', 'topology': 'mtgnn_placebo'},
    'mage_native': {'graph': 'mage_no_graph', 'topology': 'mage_placebo'},
    'semantic_knn_tcn': {'graph': 'metadata_embedding_tcn', 'topology': 'semantic_placebo_tcn'},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def weighted_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a).copy()
    b = np.abs(b).copy()
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)
    denominator = np.maximum(a, b).sum()
    return 1.0 if denominator == 0 else float(np.minimum(a, b).sum() / denominator)


def stratified_origin_bootstrap(frame: pd.DataFrame, metric: str, seed: int) -> tuple[float, float, float]:
    origins = list(frame['forecast_origin'].drop_duplicates())
    if not origins:
        return np.nan, np.nan, np.nan
    observed = float(frame[metric].mean())
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_REPS):
        origin_means = []
        for origin in origins:
            values = frame.loc[frame['forecast_origin'] == origin, metric].dropna().to_numpy(float)
            idx = rng.integers(0, len(values), len(values))
            origin_means.append(float(np.nanmean(values[idx])))
        estimates.append(float(np.nanmean(origin_means)))
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return observed, float(low), float(high)


def load_inputs(tables: Path, manifest: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [tables / 'full_task_results.csv', tables / 'node_metrics.csv.gz', tables / 'identity_graph_matrices.npz', manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing released inputs:\n  ' + '\n  '.join(missing))
    tasks = pd.read_csv(tables / 'full_task_results.csv')
    nodes = pd.read_csv(tables / 'node_metrics.csv.gz')
    official = pd.read_csv(manifest)
    if len(tasks) != 639 or tasks['full_run_task_id'].nunique() != 639:
        raise ValueError('Released task input must contain 639 unique tasks')
    needed = {'dataset', 'forecast_origin', 'seed', 'model_id', 'node_id', 'node_index', 'MAE', 'WAPE', 'RMSSE'}
    if not needed.issubset(nodes.columns):
        raise ValueError(f'node_metrics is missing {sorted(needed - set(nodes.columns))}')
    if set(tasks['full_run_task_id']) != set(official['full_run_task_id']):
        raise ValueError('Task results and formal manifest task keys differ')
    return tasks, nodes


def graph_tables(tasks: pd.DataFrame, archive_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_index = tasks.set_index('full_run_task_id')
    records = []
    matrices = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for task_id in archive.files:
            if task_id not in task_index.index:
                raise ValueError(f'Unknown graph task: {task_id}')
            row = task_index.loc[task_id]
            matrix = np.asarray(archive[task_id], dtype=np.float32)
            if matrix.shape != (90, 90):
                raise ValueError(f'Unexpected graph shape for {task_id}: {matrix.shape}')
            matrices[task_id] = matrix
            offdiag = matrix.copy()
            np.fill_diagonal(offdiag, 0.0)
            records.append({
                'task_id': task_id, 'dataset': row['dataset'],
                'forecast_origin': row['forecast_origin'], 'seed': int(row['seed']),
                'model_id': row['model_id'],
                'coverage': float((np.abs(offdiag).sum(axis=1) > 0).mean()),
                'offdiag_edges': int(np.count_nonzero(offdiag)),
                'matrix_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
            })
    diagnostics = pd.DataFrame(records)
    reliability = []
    for (dataset, model_id), group in diagnostics.groupby(['dataset', 'model_id']):
        keys = group['task_id'].tolist()
        similarities = [weighted_jaccard(matrices[a], matrices[b]) for a, b in combinations(keys, 2)]
        reliability.append({
            'dataset': dataset, 'model_id': model_id,
            'coverage_median': float(group.coverage.median()),
            'coverage_min': float(group.coverage.min()),
            'weighted_jaccard_median': float(np.median(similarities)) if similarities else np.nan,
            'weighted_jaccard_min': float(np.min(similarities)) if similarities else np.nan,
            'graphs': len(keys),
            'integrity_passed': True,
        })
    return diagnostics, pd.DataFrame(reliability)


def contrast_tables(nodes: pd.DataFrame, tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ['dataset', 'forecast_origin', 'seed', 'node_id']
    rmsse_rows, secondary_rows = [], []
    counter = 0
    for candidate, controls in CONTRASTS.items():
        for contrast_type, control in controls.items():
            left = nodes[nodes.model_id == candidate][keys + ['RMSSE']].rename(columns={'RMSSE': 'candidate_RMSSE'})
            right = nodes[nodes.model_id == control][keys + ['RMSSE']].rename(columns={'RMSSE': 'control_RMSSE'})
            merged = left.merge(right, on=keys, how='inner')
            merged['delta'] = merged.control_RMSSE - merged.candidate_RMSSE
            for dataset in DATASETS:
                subset = merged[merged.dataset == dataset]
                if subset.empty:
                    continue
                observed = float(subset.groupby('forecast_origin').delta.mean().mean())
                low, high = stratified_origin_bootstrap(subset, 'delta', BOOTSTRAP_SEED + counter)[1:]
                counter += 1
                origin_means = subset.groupby('forecast_origin').delta.mean()
                rmsse_rows.append({
                    'dataset': dataset, 'candidate': candidate, 'contrast_type': contrast_type,
                    'control': control, 'delta_RMSSE': observed, 'ci_low': low, 'ci_high': high,
                    'favorable_origins': int((origin_means > 0).sum()),
                    'origins': int(origin_means.size), 'paired_node_seed_units': int(len(subset)),
                })
    task_keys = ['dataset', 'forecast_origin', 'seed']
    for candidate, controls in CONTRASTS.items():
        for contrast_type, control in controls.items():
            left = tasks[tasks.model_id == candidate][task_keys + ['WAPE', 'MAE']]
            right = tasks[tasks.model_id == control][task_keys + ['WAPE', 'MAE']]
            paired = left.merge(right, on=task_keys, suffixes=('_candidate', '_control'))
            for dataset, group in paired.groupby('dataset'):
                for metric in ['WAPE', 'MAE']:
                    delta = group[f'{metric}_control'] - group[f'{metric}_candidate']
                    secondary_rows.append({
                        'dataset': dataset, 'candidate': candidate, 'contrast_type': contrast_type,
                        'control': control, 'metric': metric, 'delta': float(delta.mean()),
                        'paired_origin_seed_units': int(delta.size),
                    })
    return pd.DataFrame(rmsse_rows), pd.DataFrame(secondary_rows)


def contrast_direction(row: pd.Series | None) -> str:
    if row is None:
        return 'not_computed'
    if row.ci_low > 0 and row.favorable_origins >= 2:
        return 'pass'
    if row.ci_high < 0:
        return 'fail'
    return 'uncertain'


def qualification_table(tasks: pd.DataFrame, reliability: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for candidate in GRAPH_MODELS:
            if dataset == 'Store Item' and candidate == 'semantic_knn_tcn':
                rows.append({'dataset': dataset, 'candidate': candidate, 'qualification': 'not applicable'})
                continue
            expected = 9
            actual = int(((tasks.dataset == dataset) & (tasks.model_id == candidate)).sum())
            eval_pass = actual == expected
            rel_rows = reliability[(reliability.dataset == dataset) & (reliability.model_id == candidate)]
            rel = rel_rows.iloc[0] if len(rel_rows) else None
            if candidate in FIXED_GRAPH_MODELS:
                rel_pass = rel is not None and bool(rel.integrity_passed)
                rel_label = 'fixed-map integrity' if rel_pass else 'integrity failed'
            else:
                rel_pass = rel is not None and rel.coverage_median >= 0.30 and rel.weighted_jaccard_median >= 0.40
                rel_label = f'c={rel.coverage_median:.2f}, J={rel.weighted_jaccard_median:.2f}' if rel is not None else 'not recovered'
            contrast_rows = {}
            for contrast_type in ['graph', 'weight', 'topology']:
                found = contrasts[(contrasts.dataset == dataset) & (contrasts.candidate == candidate) & (contrasts.contrast_type == contrast_type)]
                contrast_rows[contrast_type] = found.iloc[0] if len(found) else None
            utility = contrast_direction(contrast_rows['graph'])
            specificity = contrast_direction(contrast_rows['topology'])
            weight = contrast_direction(contrast_rows['weight'])
            if not eval_pass:
                status = 'evidence incomplete: evaluability'
            elif not rel_pass:
                status = 'failed: reliability'
            elif utility == 'fail':
                status = 'failed: utility'
            elif utility == 'uncertain':
                status = 'evidence incomplete: utility'
            elif contrast_rows['topology'] is None:
                status = 'evidence incomplete: topology'
            elif specificity == 'fail':
                status = 'failed: specificity'
            elif specificity == 'uncertain':
                status = 'evidence incomplete: specificity'
            else:
                status = 'qualified (approximate controls)' if candidate == 'mage_native' else 'qualified'
            record = {
                'dataset': dataset, 'candidate': candidate, 'evaluability': actual / expected,
                'reliability': rel_label, 'reliability_pass': rel_pass,
                'graph_outcome': utility, 'weight_outcome': weight,
                'topology_outcome': specificity, 'qualification': status,
            }
            for name, row in contrast_rows.items():
                if row is not None:
                    record[f'{name}_delta'] = row.delta_RMSSE
                    record[f'{name}_ci_low'] = row.ci_low
                    record[f'{name}_ci_high'] = row.ci_high
                    record[f'{name}_favorable_origins'] = int(row.favorable_origins)
            rows.append(record)
    return pd.DataFrame(rows)


def aggregate(tables: Path, manifest: Path, output: Path) -> dict:
    tasks, nodes = load_inputs(tables, manifest)
    output.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(output / 'full_task_results.csv', index=False)
    nodes.to_csv(output / 'node_metrics.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summary = tasks.groupby(['dataset', 'model_id'], as_index=False).agg(
        tasks=('full_run_task_id', 'count'),
        RMSSE_mean=('RMSSE', 'mean'), RMSSE_sd=('RMSSE', 'std'),
        WAPE_mean=('WAPE', 'mean'), WAPE_sd=('WAPE', 'std'),
        MAE_mean=('MAE', 'mean'), MAE_sd=('MAE', 'std'),
    )
    origin = tasks.groupby(['dataset', 'model_id', 'forecast_origin'], as_index=False).agg(
        RMSSE=('RMSSE', 'mean'), WAPE=('WAPE', 'mean'), MAE=('MAE', 'mean'),
        seeds=('seed', 'nunique'),
    )
    seed = tasks.groupby(['dataset', 'model_id', 'seed'], as_index=False).agg(
        RMSSE=('RMSSE', 'mean'), WAPE=('WAPE', 'mean'), MAE=('MAE', 'mean'),
        origins=('forecast_origin', 'nunique'),
    )
    origin_stability = origin.groupby(['dataset', 'model_id'], as_index=False).agg(
        RMSSE_origin_mean=('RMSSE', 'mean'), RMSSE_origin_sd=('RMSSE', 'std'),
        RMSSE_origin_min=('RMSSE', 'min'), RMSSE_origin_max=('RMSSE', 'max'),
        WAPE_origin_sd=('WAPE', 'std'), MAE_origin_sd=('MAE', 'std'),
    )
    seed_stability = seed.groupby(['dataset', 'model_id'], as_index=False).agg(
        RMSSE_seed_sd=('RMSSE', 'std'), WAPE_seed_sd=('WAPE', 'std'), MAE_seed_sd=('MAE', 'std'),
    )
    group = nodes.groupby(['dataset', 'model_id', 'item_block'], as_index=False).agg(
        RMSSE=('RMSSE', 'mean'), WAPE=('WAPE', 'mean'), MAE=('MAE', 'mean'),
        nodes=('node_id', 'nunique'),
    )
    epoch_rows = tasks[tasks.selected_epoch.notna()].copy()
    epoch_rows['selected_epoch'] = pd.to_numeric(epoch_rows.selected_epoch, errors='coerce')
    early = epoch_rows.groupby(['dataset', 'model_id'], as_index=False).agg(
        tasks=('full_run_task_id', 'count'), selected_epoch_mean=('selected_epoch', 'mean'),
        selected_epoch_median=('selected_epoch', 'median'), selected_epoch_min=('selected_epoch', 'min'),
        selected_epoch_max=('selected_epoch', 'max'),
    )
    diagnostics, reliability = graph_tables(tasks, tables / 'identity_graph_matrices.npz')
    contrasts, secondary = contrast_tables(nodes, tasks)
    qualification = qualification_table(tasks, reliability, contrasts)
    outputs = {
        'full_model_summary.csv': summary, 'origin_metrics.csv': origin,
        'origin_stability.csv': origin_stability, 'seed_metrics.csv': seed,
        'seed_stability.csv': seed_stability, 'group_metrics.csv': group,
        'early_stopping_summary.csv': early,
        'table_rmsse_mean.csv': summary.pivot(index='model_id', columns='dataset', values='RMSSE_mean').reset_index(),
        'table_wape_mean.csv': summary.pivot(index='model_id', columns='dataset', values='WAPE_mean').reset_index(),
        'table_mae_mean.csv': summary.pivot(index='model_id', columns='dataset', values='MAE_mean').reset_index(),
        'graph_task_diagnostics.csv': diagnostics, 'graph_reliability.csv': reliability,
        'paired_rmsse_contrasts.csv': contrasts,
        'appendix_secondary_metric_contrasts.csv': secondary,
        'structural_qualification.csv': qualification,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    analysis = {
        'status': 'FULL_UNIFIED_RUN_V1_ANALYSIS_COMPLETE', 'manifest_sha256': sha256_file(manifest),
        'valid_tasks': len(tasks),
        'stage1_tasks': int((tasks.execution_stage == 'stage_1').sum()),
        'stage2_tasks': int((tasks.execution_stage == 'stage_2').sum()),
        'model_configurations': int(tasks.model_id.nunique()),
        'datasets': sorted(tasks.dataset.unique().tolist()),
        'bootstrap_reps': BOOTSTRAP_REPS, 'bootstrap_seed': BOOTSTRAP_SEED,
        'graph_matrices_recovered': int(reliability.graphs.sum()),
        'qualification_records': len(qualification),
        'store_item_prohibited_files_accessed': False,
    }
    with (output / 'analysis_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
    return analysis


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=root / 'artifacts/v1/tables')
    parser.add_argument('--manifest', type=Path, default=root / 'artifacts/v1/manifests/full_unified_task_manifest_v1.csv')
    parser.add_argument('--output-root', type=Path, default=root / 'results/reproduced')
    args = parser.parse_args()
    result = aggregate(args.input_root.resolve(), args.manifest.resolve(), args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
