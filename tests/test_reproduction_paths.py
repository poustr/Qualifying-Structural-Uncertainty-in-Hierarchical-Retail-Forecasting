from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_quick_path_has_no_training_imports() -> None:
    for name in ['aggregate_full_v1_results.py', 'compare_reproduced_results.py', 'reproduce_paper_results.py']:
        text = (ROOT / 'scripts' / name).read_text(encoding='utf-8').lower()
        assert 'import torch' not in text
        assert 'import transformers' not in text


def test_full_dry_run_selects_639_tasks() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/run_unified_rerun_v1.py'), '--dry-run', '--summary-only'],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    summary = json.loads(result.stdout)
    assert summary['formal_manifest_tasks'] == 639
    assert summary['selected_tasks'] == 639
    assert summary['dry_run'] is True


def test_dataset_and_stage_filters_are_exact() -> None:
    spec = importlib.util.spec_from_file_location('full_runner', ROOT / 'scripts/run_unified_rerun_v1.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = pd.read_csv(ROOT / 'artifacts/v1/manifests/full_unified_task_manifest_v1.csv')
    selected = module.select_tasks(manifest, 'M5', '1', None)
    assert len(selected) == 126
    assert set(selected.dataset) == {'M5'}
    assert set(selected.execution_stage) == {'stage_1'}


def test_historical_aggregation_failure_file_is_absent() -> None:
    assert not (ROOT / 'artifacts/v1/tables/aggregation_failures.csv').exists()


def test_quick_reproduction_inputs_exist() -> None:
    tables = ROOT / 'artifacts/v1/tables'
    for name in ['full_task_results.csv', 'node_metrics.csv.gz', 'identity_graph_matrices.npz']:
        assert (tables / name).is_file()


def test_unresolved_release_metadata_remains_explicit() -> None:
    citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    assert 'TODO' in citation
    assert (ROOT / 'LICENSE-TO-CHOOSE.md').is_file()


def test_external_locked_node_maps_are_complete() -> None:
    for dataset in ['favorita', 'store_item']:
        frame = pd.read_csv(ROOT / 'configs/v1/locked_nodes' / f'{dataset}_90.csv')
        assert list(frame.columns) == ['node_index', 'item_id', 'store_id']
        assert frame.node_index.tolist() == list(range(90))
        assert not frame[['item_id', 'store_id']].duplicated().any()


def test_public_identity_graph_is_not_gitignored() -> None:
    rules = (ROOT / '.gitignore').read_text(encoding='utf-8')
    assert '!artifacts/v1/tables/identity_graph_matrices.npz' in rules
    assert (ROOT / 'artifacts/v1/tables/identity_graph_matrices.npz').is_file()


def test_external_preparation_uses_locked_pairs_and_validation_cutoff(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location('prepare_v1', ROOT / 'scripts/prepare_v1_data.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    pairs = module.locked_pairs(ROOT, 'store_item')
    first = pairs.iloc[0]
    raw = tmp_path / 'train.csv'
    pd.DataFrame([
        {'date': '2013-01-01', 'store': first.store_id, 'item': first.item_id, 'sales': 7},
        {'date': '2017-08-14', 'store': first.store_id, 'item': first.item_id, 'sales': 999},
    ]).to_csv(raw, index=False)
    panel = module.prepare_external(ROOT, 'store_item', tmp_path)
    assert panel.shape[1] == 90
    assert str(panel.index.max().date()) == '2017-08-13'
    assert panel.loc['2013-01-01', '0'] == 7
    assert panel.to_numpy().max() == 7


def test_resume_rejects_corrupt_completion_marker(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location('single_task', ROOT / 'scripts/run_single_task.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    task_id = 'example_task'
    task_dir = tmp_path / task_id
    task_dir.mkdir()
    for name in ['resolved_config.json', 'forecast.csv.gz', 'training_history.csv']:
        (task_dir / name).write_bytes(b'valid')
    (task_dir / 'checkpoint_not_applicable.json').write_text('{}', encoding='utf-8')
    record = {
        'status': 'complete', 'task_id': task_id,
        'checkpoint_status': 'not_applicable',
        'resolved_config_sha256': module.sha256_file(task_dir / 'resolved_config.json'),
        'forecast_sha256': module.sha256_file(task_dir / 'forecast.csv.gz'),
        'training_history_sha256': module.sha256_file(task_dir / 'training_history.csv'),
    }
    (task_dir / 'completed.json').write_text(json.dumps(record), encoding='utf-8')
    assert module.valid_completion(task_dir, task_id)
    (task_dir / 'forecast.csv.gz').write_bytes(b'corrupt')
    assert not module.valid_completion(task_dir, task_id)


def test_neural_resume_checks_checkpoint_and_both_histories(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location('single_task', ROOT / 'scripts/run_single_task.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    task_id = 'neural_task'
    task_dir = tmp_path / task_id
    task_dir.mkdir()
    names = [
        'resolved_config.json', 'predictions.csv.gz',
        'epoch_selection_log.csv', 'final_refit_log.csv', 'checkpoint_final.pt',
    ]
    for name in names:
        (task_dir / name).write_bytes(name.encode())
    record = {
        'status': 'complete', 'task_id': task_id,
        'forecast_file': 'predictions.csv.gz',
        'training_history_file': 'epoch_selection_log.csv',
        'refit_history_file': 'final_refit_log.csv',
        'resolved_config_sha256': module.sha256_file(task_dir / 'resolved_config.json'),
        'forecast_sha256': module.sha256_file(task_dir / 'predictions.csv.gz'),
        'training_history_sha256': module.sha256_file(task_dir / 'epoch_selection_log.csv'),
        'refit_history_sha256': module.sha256_file(task_dir / 'final_refit_log.csv'),
        'checkpoint_sha256': module.sha256_file(task_dir / 'checkpoint_final.pt'),
    }
    (task_dir / 'completed.json').write_text(json.dumps(record), encoding='utf-8')
    assert module.valid_completion(task_dir, task_id)
    (task_dir / 'final_refit_log.csv').write_bytes(b'changed')
    assert not module.valid_completion(task_dir, task_id)
