'''Run one formal v1 task from the released manifest.'''
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def valid_completion(task_dir: Path, task_id: str) -> bool:
    marker = task_dir / 'completed.json'
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record.get('status') != 'complete' or record.get('task_id') != task_id:
            return False
        for filename, key in [
            ('resolved_config.json', 'resolved_config_sha256'),
            (record.get('forecast_file', 'forecast.csv.gz'), 'forecast_sha256'),
            (record.get('training_history_file', 'training_history.csv'), 'training_history_sha256'),
        ]:
            file = task_dir / filename
            if not file.is_file() or sha256_file(file) != record.get(key):
                return False
        if record.get('checkpoint_status') == 'not_applicable':
            return (task_dir / 'checkpoint_not_applicable.json').is_file()
        if record.get('refit_history_file'):
            refit = task_dir / record['refit_history_file']
            if not refit.is_file() or sha256_file(refit) != record.get('refit_history_sha256'):
                return False
        checkpoint = task_dir / 'checkpoint_final.pt'
        return checkpoint.is_file() and sha256_file(checkpoint) == record.get('checkpoint_sha256')
    except (OSError, ValueError, TypeError):
        return False


def load_panel(path: Path, node_ids: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f'Prepared demand matrix not found: {path}')
    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    frame.columns = frame.columns.astype(str)
    missing = [node for node in node_ids if node not in frame.columns]
    if missing:
        raise SystemExit(f'Prepared panel is missing {len(missing)} locked nodes')
    return frame.reindex(columns=node_ids).sort_index()


def seasonal_naive(panel: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_end = origin - pd.Timedelta(days=1)
    last_week = panel.loc[origin - pd.Timedelta(days=7):history_end]
    target_dates = pd.date_range(origin, periods=28, freq='D')
    if len(last_week) != 7 or not set(target_dates).issubset(panel.index):
        raise SystemExit('Prepared panel does not cover the required history and 28-day target')
    prediction = np.tile(last_week.to_numpy(float), (4, 1))
    actual = panel.loc[target_dates].to_numpy(float)
    return pd.DataFrame(prediction, index=target_dates, columns=panel.columns), pd.DataFrame(actual, index=target_dates, columns=panel.columns)


def compute_metrics(panel: pd.DataFrame, origin: pd.Timestamp, prediction: pd.DataFrame, actual: pd.DataFrame) -> dict:
    history = panel.loc[:origin - pd.Timedelta(days=1)].tail(1092).to_numpy(float)
    scale = np.mean(np.diff(history, axis=0) ** 2, axis=0)
    squared = np.mean((actual.to_numpy() - prediction.to_numpy()) ** 2, axis=0)
    eligible = scale > 0
    rmsse = np.sqrt(squared[eligible] / scale[eligible])
    error = np.abs(actual.to_numpy() - prediction.to_numpy())
    denominator = float(np.abs(actual.to_numpy()).sum())
    return {
        'RMSSE': float(np.mean(rmsse)), 'RMSSE_eligible_nodes': int(eligible.sum()),
        'WAPE': float(error.sum() / denominator) if denominator > 0 else None,
        'MAE': float(error.mean()),
    }


def run_original_formal_task(task: dict, args, root: Path) -> int:
    dataset_key = str(task['dataset']).lower().replace(' ', '_')
    prepared = args.data_root / dataset_key
    model_id = str(task['model_id'])
    required = [
        prepared / 'demand_matrix.csv.gz',
        prepared / 'node_order.csv',
        prepared / 'neural_tensors/tensor_manifest.csv',
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit('Prepared neural inputs missing: ' + ', '.join(missing))
    if model_id in {'tcn_stat_confidence', 'tcn_stat_equal', 'tcn_stat_placebo'}:
        structure = {
            'm5': 'outputs/structures/four_confidence_definitions_all_edges.csv',
            'favorita': 'outputs/favorita/structures/four_confidence_definitions.csv',
            'store_item': 'outputs/store_item/structures/four_confidence_definitions.csv',
        }[dataset_key]
        root_path = Path(os.environ.get('RETAIL_STRUCTURE_ROOT', args.data_root / 'structure_inputs'))
        if not (root_path / structure).is_file():
            raise SystemExit(f'Locked statistical structure input missing: {root_path / structure}')
    if model_id in {'semantic_knn_tcn', 'metadata_embedding_tcn', 'semantic_placebo_tcn'}:
        cache = args.data_root / 'semantic_embeddings'
        needed = [cache / f'{dataset_key}_embeddings.pt', cache / f'{dataset_key}_embedding_manifest.json']
        if any(not path.is_file() for path in needed):
            raise SystemExit('Locked semantic embeddings missing; run prepare_v1_semantic_embeddings.py first')
    task_dir = args.output_root / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    if not (task_dir / 'completion.marker').is_file():
        environment = os.environ.copy()
        environment['RETAIL_DATA_ROOT'] = str(args.data_root.resolve())
        environment['RETAIL_DEVICE'] = args.device
        environment['RETAIL_SEMANTIC_ROOT'] = str((args.data_root / 'semantic_embeddings').resolve())
        entry = ('run_full_v1_stage2_task.py' if task['execution_stage'] == 'stage_2'
                 else 'run_full_v1_task.py')
        command = [
            sys.executable, str(root / 'scripts' / entry),
            '--task-id', args.task_id,
            '--output-root', str(args.output_root.resolve()),
            '--device', args.device,
        ]
        result = subprocess.run(command, cwd=root, env=environment, check=False)
        if result.returncode:
            atomic_json(task_dir / 'failure.json', {
                'task_id': args.task_id, 'exit_code': result.returncode,
                'source': 'original_full_v1_formal_task_loop',
                'run_status_file': 'run_status.json',
            })
            return result.returncode
    run_status = json.loads((task_dir / 'run_status.json').read_text(encoding='utf-8'))
    reload_check = json.loads((task_dir / 'checkpoint_reload_check.json').read_text(encoding='utf-8'))
    if run_status.get('status') != 'passed' or not reload_check.get('passed'):
        raise RuntimeError('Original training output or checkpoint reload audit failed')
    required_outputs = [
        'resolved_config.json', 'predictions.csv.gz',
        'epoch_selection_log.csv', 'final_refit_log.csv',
        'checkpoint_final.pt',
    ]
    if any(not (task_dir / name).is_file() for name in required_outputs):
        raise RuntimeError('Original training output is incomplete')
    marker = {
        'status': 'complete', 'task_id': args.task_id,
        'model_id': task['model_id'],
        'source': 'original_full_v1_formal_task_loop',
        'forecast_origin': task['forecast_origin'],
        'seed': int(task['seed']),
        'resolved_config_sha256': sha256_file(task_dir / 'resolved_config.json'),
        'node_order_sha256': sha256_file(prepared / 'node_order.csv'),
        'forecast_file': 'predictions.csv.gz',
        'forecast_sha256': sha256_file(task_dir / 'predictions.csv.gz'),
        'training_history_file': 'epoch_selection_log.csv',
        'training_history_sha256': sha256_file(task_dir / 'epoch_selection_log.csv'),
        'refit_history_file': 'final_refit_log.csv',
        'refit_history_sha256': sha256_file(task_dir / 'final_refit_log.csv'),
        'checkpoint_sha256': sha256_file(task_dir / 'checkpoint_final.pt'),
        'checkpoint_reload_passed': True,
        'RMSSE': run_status['RMSSE'],
        'WAPE': run_status['WAPE'],
        'MAE': run_status['MAE'],
    }
    atomic_json(task_dir / 'completed.json', marker)
    print(json.dumps(marker, indent=2))
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest_path = root / 'artifacts/v1/manifests/full_unified_task_manifest_v1.csv'
    manifest = pd.read_csv(manifest_path)
    matched = manifest[manifest.full_run_task_id == args.task_id]
    if len(matched) != 1:
        raise SystemExit(f'Unknown formal task ID: {args.task_id}')
    task = matched.iloc[0].to_dict()
    task_dir = args.output_root / args.task_id
    completed = task_dir / 'completed.json'
    if args.resume and valid_completion(task_dir, args.task_id):
        print(f'Skipping completed task: {args.task_id}')
        return 0
    if task['model_id'] in {
        'tcn_no_graph', 'tcn_business', 'tcn_stat_confidence', 'tcn_stat_equal',
        'tcn_stat_placebo', 'tcn_target_only', 'semantic_knn_tcn',
        'metadata_embedding_tcn', 'semantic_placebo_tcn', 'global_lightgbm',
    }:
        return run_original_formal_task(task, args, root)
    if task['model_id'] != 'seasonal_naive':
        raise SystemExit(
            'Blocked before training: this model or data set has not passed the '
            + 'portable original-feature and original-runner audit. No substitute '
            + 'model or tensor will be used. See docs/repository_scope.md.'
        )
    dataset_key = str(task['dataset']).lower().replace(' ', '_')
    prepared = args.data_root / dataset_key
    node_order_path = prepared / 'node_order.csv'
    if not node_order_path.is_file():
        raise SystemExit(f'Prepared node order not found: {node_order_path}')
    node_ids = pd.read_csv(node_order_path).sort_values('node_index').node_id.astype(str).tolist()
    if len(node_ids) != 90:
        raise SystemExit('Prepared node order must contain exactly 90 nodes')
    panel = load_panel(prepared / 'demand_matrix.csv.gz', node_ids)
    origin = pd.Timestamp(task['forecast_origin'])
    prediction, actual = seasonal_naive(panel, origin)
    metrics = compute_metrics(panel, origin, prediction, actual)
    task_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = task_dir / 'resolved_config.json'
    atomic_json(resolved_path, task)
    long_prediction = prediction.rename_axis('date').reset_index().melt(id_vars='date', var_name='node_id', value_name='prediction')
    forecast_path = task_dir / 'forecast.csv.gz'
    long_prediction.to_csv(forecast_path, index=False, compression={'method': 'gzip', 'mtime': 0})
    history_path = task_dir / 'training_history.csv'
    pd.DataFrame(columns=['epoch', 'training_loss', 'validation_loss']).to_csv(history_path, index=False)
    checkpoint_note = task_dir / 'checkpoint_not_applicable.json'
    atomic_json(checkpoint_note, {'model_id': 'seasonal_naive', 'reason': 'deterministic non-training method'})
    result = {
        'status': 'complete', 'task_id': args.task_id, 'model_id': task['model_id'],
        'forecast_origin': task['forecast_origin'], 'seed': int(task['seed']),
        'prediction_rows': len(long_prediction), **metrics,
        'resolved_config_sha256': sha256_file(resolved_path),
        'node_order_sha256': sha256_file(node_order_path),
        'forecast_sha256': sha256_file(forecast_path),
        'training_history_sha256': sha256_file(history_path),
        'checkpoint_status': 'not_applicable', 'resumable_completed_marker': True,
    }
    atomic_json(completed, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
