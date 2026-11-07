import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "two_way_cluster_bootstrap_audit.py"
spec = importlib.util.spec_from_file_location("two_way_cluster_bootstrap_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TwoWayClusterBootstrapTests(unittest.TestCase):
    def test_product_weights_use_both_cluster_multiplicities(self):
        date_counts = np.asarray([[2, 0, 1]], dtype=np.int64)
        team_counts = np.asarray([[0, 3, 1, 2]], dtype=np.int64)
        date_index = np.asarray([0, 0, 1, 2], dtype=np.int64)
        team_index = np.asarray([1, 2, 3, 0], dtype=np.int64)
        weights = audit.product_cluster_weights(date_counts, team_counts, date_index, team_index)
        self.assertEqual(weights.tolist(), [[6, 2, 0, 0]])

    def test_prepare_arrays_builds_two_team_side_rows_per_match(self):
        by_k = {
            f"{k:.1f}": {
                "home": [0.4, 0.3, 0.2, 0.1],
                "away": [0.3, 0.4, 0.2, 0.1],
            }
            for k in audit.K_VALUES
        }
        doc = {"allKPredictions": [{
            "date": "2026-09-19", "comp": "J1", "fold": 1,
            "home": "A", "away": "B", "hg": 1, "ag": 2,
            "byK": by_k,
        }]}
        data = audit.prepare_arrays(doc, enforce_expected_cohort=False)
        self.assertEqual(len(data["actual"]), 2)
        self.assertEqual(data["actual"].tolist(), [1, 2])
        self.assertEqual(data["probabilities"].shape, (2, 16, 4))

    def test_prepare_arrays_rejects_invalid_probability_simplex(self):
        by_k = {
            f"{k:.1f}": {
                "home": [0.4, 0.3, 0.2, 0.2],
                "away": [0.3, 0.4, 0.2, 0.1],
            }
            for k in audit.K_VALUES
        }
        doc = {"allKPredictions": [{
            "date": "2026-09-19", "comp": "J1", "home": "A", "away": "B",
            "hg": 1, "ag": 2, "byK": by_k,
        }]}
        with self.assertRaisesRegex(ValueError, "outside simplex"):
            audit.prepare_arrays(doc, enforce_expected_cohort=False)

    def test_prepare_arrays_rejects_duplicate_match(self):
        by_k = {
            f"{k:.1f}": {
                "home": [0.4, 0.3, 0.2, 0.1],
                "away": [0.3, 0.4, 0.2, 0.1],
            }
            for k in audit.K_VALUES
        }
        match = {
            "date": "2026-09-19", "comp": "J1", "home": "A", "away": "B",
            "hg": 1, "ag": 2, "byK": by_k,
        }
        with self.assertRaisesRegex(ValueError, "duplicate match key"):
            audit.prepare_arrays({"allKPredictions": [match, match]}, enforce_expected_cohort=False)


if __name__ == "__main__":
    unittest.main()
