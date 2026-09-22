from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_author_snapshot_matches_historical_lock() -> None:
    manifest = pd.read_csv(ROOT / 'docs/author_source_manifest.csv')
    assert len(manifest) == 13
    assert manifest.relative_path.is_unique
    for row in manifest.itertuples():
        path = ROOT / 'src/author_core_v1' / row.relative_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256


def test_author_model_synthetic_forward() -> None:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / 'src/author_core_v1'))
    from models.mvp_model import MVPModel

    summing = np.concatenate(
        [np.zeros((42, 90), dtype=np.float32), np.eye(90, dtype=np.float32)]
    )
    model = MVPModel(summing, hidden=32, dropout=0.1, horizon=28).eval()
    nodes = torch.arange(90)
    static_ids = torch.stack((nodes // 3, nodes // 45, nodes % 3), dim=1)
    with torch.no_grad():
        output = model(
            torch.rand(2, 90, 84, 17),
            torch.rand(2, 90, 28, 9),
            static_ids,
            torch.eye(90),
        )
    assert tuple(output['bottom_log'].shape) == (2, 90, 28)
    assert tuple(output['upper_log'].shape) == (2, 42, 28)
    assert tuple(output['attention'].shape) == (2, 90, 90)
    assert all(bool(torch.isfinite(value).all()) for value in output.values())
