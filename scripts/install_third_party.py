'''Install one optional third-party model from its recorded official source.'''
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sources = pd.read_csv(root / 'third_party/sources.csv')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=sorted(sources.component.str.lower()))
    parser.add_argument('--destination-root', type=Path)
    args = parser.parse_args()
    row = sources[sources.component.str.lower() == args.model].iloc[0]
    if row.component == 'controlled_framework':
        print('The hash-matched author-owned source is already in src/author_core_v1; no clone is required.')
        return 0
    if row.license == 'NO_LICENSE_FILE_FOUND':
        raise SystemExit(f'{row.component} is not redistributed or automatically installed because the recorded source has no license file. Obtain it from the official source after reviewing its terms.')
    if row.component == 'semantic_encoder':
        raise SystemExit('Install requirements-training.txt; transformers downloads the pinned all-MiniLM-L6-v2 revision only when a semantic task is requested.')
    if str(row.official_source) == 'not_recovered':
        raise SystemExit(f'No public source is recorded for {row.component}')
    component = str(row.component)
    if args.destination_root is not None:
        destination = args.destination_root / component
    elif component in {'GTS', 'MTGNN', 'MAGE'}:
        destination = root / 'external' / component
    else:
        folder = 'PyPOTS' if component == 'PyPOTS_TimeMixerPP' else component
        destination = root / 'third_party/modern_baselines' / folder
    if destination.exists():
        print(f'Already present: {destination}')
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', str(row.official_source), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', str(row.v1_revision)], check=True)
    print(f'Installed {row.component} at revision {row.v1_revision}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
