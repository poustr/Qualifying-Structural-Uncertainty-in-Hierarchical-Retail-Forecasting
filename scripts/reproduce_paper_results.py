'''One-command public reproduction of the paper's aggregate results.'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_full_v1_results import aggregate
from compare_reproduced_results import compare
from verify_v1_artifacts import main as verify_artifact_hashes


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=root / 'results/reproduced')
    parser.add_argument('--tolerance', type=float, default=1e-10)
    args = parser.parse_args()
    output = args.output_root.resolve()
    verify_artifact_hashes()
    aggregate(root / 'artifacts/v1/tables', root / 'artifacts/v1/manifests/full_unified_task_manifest_v1.csv', output)
    report = compare(root / 'artifacts/v1/tables', output, args.tolerance)
    print(json.dumps({'output_root': str(output), 'artifact_hashes_passed': True, 'comparison_passed': report['passed']}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
