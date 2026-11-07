import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "pair_preserving_date_team_bootstrap_audit.py"
spec = importlib.util.spec_from_file_location("pair_preserving_date_team_bootstrap_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PairPreservingBootstrapTests(unittest.TestCase):
    def test_home_and_away_targets_share_the_same_match_weight(self):
        date_counts = np.asarray([[2, 0]], dtype=np.int64)
        team_counts = np.asarray([[2, 3, 1]], dtype=np.int64)
        date_index = np.asarray([0, 0], dtype=np.int64)
        home_index = np.asarray([0, 1], dtype=np.int64)
        away_index = np.asarray([1, 2], dtype=np.int64)
        match_weights = audit.pair_preserving_match_weights(
            date_counts, team_counts, date_index, home_index, away_index
        )
        target_weights = np.repeat(match_weights, 2, axis=1)
        self.assertEqual(match_weights.tolist(), [[12, 6]])
        self.assertEqual(target_weights.tolist(), [[12, 12, 6, 6]])

    def test_prepare_arrays_preserves_two_linked_sides_per_match(self):
        by_k = {
            f"{k:.1f}": {
                "home": [0.4, 0.3, 0.2, 0.1],
                "away": [0.3, 0.4, 0.2, 0.1],
            }
            for k in audit.K_VALUES
        }
        document = {"allKPredictions": [{
            "date": "2026-09-19", "comp": "J1", "home": "A", "away": "B",
            "hg": 1, "ag": 2, "byK": by_k,
        }]}
        arrays = audit.prepare_arrays(document, enforce_expected_cohort=False)
        self.assertEqual(arrays["matches"], 1)
        self.assertEqual(arrays["actual"].tolist(), [1, 2])
        self.assertEqual(arrays["probabilities"].shape, (2, 16, 4))
        self.assertEqual(arrays["homeTeamIndex"].shape, (1,))
        self.assertEqual(arrays["awayTeamIndex"].shape, (1,))

    def test_prepare_arrays_rejects_invalid_score_and_duplicate_match(self):
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
        invalid = dict(match, hg=True)
        with self.assertRaisesRegex(ValueError, "invalid integer score"):
            audit.prepare_arrays({"allKPredictions": [invalid]}, enforce_expected_cohort=False)
        with self.assertRaisesRegex(ValueError, "duplicate match key"):
            audit.prepare_arrays({"allKPredictions": [match, match]}, enforce_expected_cohort=False)

    def test_prepare_arrays_rejects_probability_outside_simplex(self):
        by_k = {
            f"{k:.1f}": {
                "home": [0.4, 0.3, 0.2, 0.2],
                "away": [0.3, 0.4, 0.2, 0.1],
            }
            for k in audit.K_VALUES
        }
        document = {"allKPredictions": [{
            "date": "2026-09-19", "comp": "J1", "home": "A", "away": "B",
            "hg": 1, "ag": 2, "byK": by_k,
        }]}
        with self.assertRaisesRegex(ValueError, "outside simplex"):
            audit.prepare_arrays(document, enforce_expected_cohort=False)


if __name__ == "__main__":
    unittest.main()
