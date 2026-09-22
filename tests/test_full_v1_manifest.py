from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FullV1ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "artifacts/v1/manifests/full_unified_task_manifest_v1.csv"
        with path.open(encoding="utf-8-sig") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_total_and_stages(self):
        self.assertEqual(len(self.rows), 639)
        self.assertEqual(Counter(row["execution_stage"] for row in self.rows), Counter({"stage_1": 351, "stage_2": 288}))

    def test_unique_ids_and_three_origins(self):
        self.assertEqual(len({row["full_run_task_id"] for row in self.rows}), 639)
        for dataset in ("M5", "Favorita", "Store Item"):
            self.assertEqual(len({row["forecast_origin"] for row in self.rows if row["dataset"] == dataset}), 3)

    def test_stage_one_roster(self):
        expected = {
            "gts_native", "gts_no_graph", "gts_placebo", "mtgnn_native", "mtgnn_no_graph", "mtgnn_placebo",
            "mage_native", "mage_no_graph", "mage_placebo", "metadata_embedding_tcn", "semantic_knn_tcn",
            "semantic_placebo_tcn", "tcn_no_graph", "tcn_stat_confidence",
        }
        self.assertEqual({row["model_id"] for row in self.rows if row["execution_stage"] == "stage_1"}, expected)

    def test_no_store_item_semantic(self):
        bad = [row for row in self.rows if row["dataset"] == "Store Item" and row["model_id"] in {
            "metadata_embedding_tcn", "semantic_knn_tcn", "semantic_placebo_tcn"}]
        self.assertFalse(bad)

    def test_training_rules_and_caveat(self):
        neural = [row for row in self.rows if row["model_id"] not in {"seasonal_naive", "global_lightgbm"}]
        self.assertTrue(all(row["max_epochs"] == "50" and row["min_epochs"] == "10" for row in neural))
        mage = [row for row in self.rows if row["model_id"] == "mage_placebo"]
        self.assertTrue(mage and all(row["control_faithfulness"] == "approximate_control" for row in mage))


if __name__ == "__main__":
    unittest.main()

