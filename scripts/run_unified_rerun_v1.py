'''Filter, inspect, and resume the formal 639-task v1 manifest.'''
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_single_task import valid_completion


def select_tasks(frame: pd.DataFrame, dataset: str | None, stage: str | None, task_id: str | None) -> pd.DataFrame:
    selected = frame.copy()
    if dataset:
        selected = selected[selected.dataset == dataset]
    if stage:
        normalized = stage if stage.startswith('stage_') else f'stage_{stage}'
        selected = selected[selected.execution_stage == normalized]
    if task_id:
        selected = selected[selected.full_run_task_id == task_id]
    return selected.reset_index(drop=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'configs/v1/full_unified_training_v1.yaml')
    parser.add_argument('--manifest', type=Path, default=root / 'artifacts/v1/manifests/full_unified_task_manifest_v1.csv')
    parser.add_argument('--data-root', type=Path, default=root / 'data/derived')
    parser.add_argument('--output-root', type=Path, default=root / 'outputs/full_unified_run_v1')
    parser.add_argument('--dataset', choices=['m5', 'favorita', 'store_item'])
    parser.add_argument('--stage', choices=['1', '2', 'stage_1', 'stage_2'])
    parser.add_argument('--task-id')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    if not args.config.is_file() or not args.manifest.is_file():
        parser.error('The locked config or 639-task manifest is missing')
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 639 or manifest.full_run_task_id.nunique() != 639:
        raise SystemExit('Refusing to run: formal manifest is not 639 unique tasks')
    dataset_name = {'m5': 'M5', 'favorita': 'Favorita', 'store_item': 'Store Item'}.get(args.dataset)
    selected = select_tasks(manifest, dataset_name, args.stage, args.task_id)
    if args.max_tasks is not None:
        if args.max_tasks < 0:
            parser.error('--max-tasks must be nonnegative')
        selected = selected.head(args.max_tasks)
    plan = []
    for row in selected.itertuples(index=False):
        task_dir = args.output_root / row.full_run_task_id
        state = ('skip_completed' if args.resume and
                 valid_completion(task_dir, row.full_run_task_id) else 'pending')
        plan.append({'task_id': row.full_run_task_id, 'dataset': row.dataset, 'stage': row.execution_stage, 'model_id': row.model_id, 'state': state})
    summary = {
        'formal_manifest_tasks': len(manifest), 'selected_tasks': len(plan),
        'pending_tasks': sum(item['state'] == 'pending' for item in plan),
        'resume_skips': sum(item['state'] == 'skip_completed' for item in plan),
        'dry_run': args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        if not args.summary_only:
            for item in plan:
                print('{:14s} {}'.format(item['state'], item['task_id']))
        return 0
    runner = root / 'scripts/run_single_task.py'
    if not runner.is_file():
        raise SystemExit('run_single_task.py is missing')
    for item in plan:
        if item['state'] == 'skip_completed':
            continue
        command = [
            sys.executable, str(runner), '--task-id', item['task_id'],
            '--data-root', str(args.data_root), '--output-root', str(args.output_root),
            '--device', args.device,
        ]
        if args.resume:
            command.append('--resume')
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
