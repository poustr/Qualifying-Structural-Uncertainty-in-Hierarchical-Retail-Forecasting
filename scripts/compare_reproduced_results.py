'''Compare recomputed tables with the immutable released reference tables.'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_TABLES = [
    'full_task_results.csv', 'full_model_summary.csv', 'origin_metrics.csv',
    'origin_stability.csv', 'seed_metrics.csv', 'seed_stability.csv',
    'group_metrics.csv', 'early_stopping_summary.csv', 'table_rmsse_mean.csv',
    'table_wape_mean.csv', 'table_mae_mean.csv', 'graph_task_diagnostics.csv',
    'graph_reliability.csv', 'paired_rmsse_contrasts.csv',
    'appendix_secondary_metric_contrasts.csv', 'structural_qualification.csv',
]


def compare_csv(reference: Path, reproduced: Path, tolerance: float) -> dict:
    left = pd.read_csv(reference)
    right = pd.read_csv(reproduced)
    result = {'rows_reference': len(left), 'rows_reproduced': len(right)}
    if list(left.columns) != list(right.columns):
        result.update({'passed': False, 'reason': 'column mismatch'})
        return result
    if len(left) != len(right):
        result.update({'passed': False, 'reason': 'row-count mismatch'})
        return result
    maximum = 0.0
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            a = left[column].to_numpy(float)
            b = right[column].to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                result.update({'passed': False, 'reason': f'NaN mismatch in {column}'})
                return result
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(a[finite] - b[finite]))))
        else:
            a = left[column].fillna('<NA>').astype(str).to_numpy()
            b = right[column].fillna('<NA>').astype(str).to_numpy()
            if not np.array_equal(a, b):
                result.update({'passed': False, 'reason': f'value mismatch in {column}'})
                return result
    result.update({'passed': maximum <= tolerance, 'max_abs_numeric_difference': maximum})
    if not result['passed']:
        result['reason'] = 'numeric tolerance exceeded'
    return result


def compare(reference_root: Path, reproduced_root: Path, tolerance: float) -> dict:
    checks = {}
    for name in CSV_TABLES:
        reference = reference_root / name
        reproduced = reproduced_root / name
        if not reference.is_file() or not reproduced.is_file():
            checks[name] = {'passed': False, 'reason': 'missing file'}
        else:
            checks[name] = compare_csv(reference, reproduced, tolerance)
    reference_json = json.loads((reference_root / 'analysis_summary.json').read_text(encoding='utf-8'))
    reproduced_json = json.loads((reproduced_root / 'analysis_summary.json').read_text(encoding='utf-8'))
    json_passed = reference_json == reproduced_json
    checks['analysis_summary.json'] = {'passed': json_passed, 'reason': '' if json_passed else 'JSON mismatch'}
    report = {'passed': all(item['passed'] for item in checks.values()), 'tolerance': tolerance, 'checks': checks}
    (reproduced_root / 'reproduction_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=root / 'artifacts/v1/tables')
    parser.add_argument('--reproduced-root', type=Path, default=root / 'results/reproduced')
    parser.add_argument('--tolerance', type=float, default=1e-10)
    args = parser.parse_args()
    report = compare(args.reference_root.resolve(), args.reproduced_root.resolve(), args.tolerance)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
