#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

from verification.goal3_independent.date_cluster_bootstrap import (
    flatten_matches, percentile_linear, score,
)

ROOT = Path(__file__).resolve().parents[1]

class DateClusterBootstrapTests(unittest.TestCase):
    def test_linear_percentile(self):
        self.assertAlmostEqual(percentile_linear([0.0, 10.0], 0.25), 2.5)

    def test_score_uses_four_category_brier_and_logloss(self):
        rows = [
            {"actual": 0, "k6": [1.0, 0.0, 0.0, 0.0], "k10": [0.5, 0.5, 0.0, 0.0]},
            {"actual": 3, "k6": [0.0, 0.0, 0.0, 1.0], "k10": [0.0, 0.0, 0.0, 1.0]},
        ]
        self.assertAlmostEqual(score(rows, "k6")["brier"], 0.0)
        self.assertAlmostEqual(score(rows, "k6")["logLoss"], 0.0)
        self.assertAlmostEqual(score(rows, "k10")["brier"], 0.25)
        self.assertAlmostEqual(score(rows, "k10")["logLoss"], -0.5 * __import__("math").log(0.5))

    def test_raw_artifact_flattens_to_paired_team_sides(self):
        sample = {
            "allKPredictions": [{
                "date": "2026-01-01", "comp": "J1", "home": "H", "away": "A", "hg": 3, "ag": 1,
                "byK": {"6.0": {"home": [0.1,0.2,0.3,0.4], "away": [0.4,0.3,0.2,0.1]},
                        "10.0": {"home": [0.2,0.2,0.2,0.4], "away": [0.3,0.3,0.2,0.2]}}
            }]
        }
        rows = flatten_matches(sample)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["side"] for row in rows}, {"home", "away"})
        self.assertEqual({row["actual"] for row in rows}, {1, 3})

    def test_detail_projection_integrity(self):
        path = ROOT / "verification" / "goal3_independent" / "goal3_outer_matches.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["targetRows"]
        self.assertEqual(len(rows), 472)
        self.assertEqual(data["cohort"]["matches"], 236)
        self.assertEqual(data["cohort"]["dates"], 17)
        self.assertEqual(len({row["matchId"] for row in rows}), 236)
        self.assertEqual(len({row["matchId"] + "|" + row["side"] for row in rows}), 472)
        for row in rows:
            self.assertIn(row["side"], {"home", "away"})
            for arm in ("k6", "k10"):
                self.assertAlmostEqual(sum(row["probabilities"][arm]), 1.0, delta=2e-6)

if __name__ == "__main__":
    unittest.main(verbosity=2)
