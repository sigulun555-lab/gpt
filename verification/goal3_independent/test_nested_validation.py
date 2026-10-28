import unittest

from verification.goal3_independent.nested_validation import (
    date_blocks, select_training, score, fit_shrinkage, predict_shrinkage,
)

class NestedValidationTests(unittest.TestCase):
    def setUp(self):
        self.h2025 = [
            {"date": "2025-02-01", "home": "A", "away": "B", "hg": 1, "ag": 0, "year": 2025},
            {"date": "2025-02-08", "home": "B", "away": "A", "hg": 0, "ag": 2, "year": 2025},
        ]
        self.h2026 = [
            {"date": "2026-08-01", "home": "A", "away": "B", "hg": 1, "ag": 1, "year": 2026},
            {"date": "2026-08-01", "home": "B", "away": "A", "hg": 0, "ag": 1, "year": 2026},
            {"date": "2026-08-08", "home": "A", "away": "B", "hg": 2, "ag": 0, "year": 2026},
        ]

    def test_training_cutoff_excludes_same_day_and_future(self):
        prior = select_training(self.h2025, self.h2026, "2026-08-08", "2025_plus_2026")
        self.assertTrue(all(m["date"] < "2026-08-08" for m in prior))
        only26 = select_training(self.h2025, self.h2026, "2026-08-08", "2026_only")
        self.assertEqual(len(only26), 2)
        self.assertTrue(all(m["year"] == 2026 for m in only26))

    def test_date_blocks_are_disjoint_and_cover_range(self):
        dates = [f"2026-09-{day:02d}" for day in range(1, 13)]
        blocks = date_blocks(dates, 3, 12, 3)
        flattened = [d for block in blocks for d in block]
        self.assertEqual(flattened, dates[3:])
        self.assertEqual(len(set(flattened)), len(flattened))

    def test_score_metrics_and_target_count(self):
        result = score([([1.0, 0.0, 0.0, 0.0], 0), ([0.0, 1.0, 0.0, 0.0], 1)])
        self.assertEqual(result["targetCount"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["brier"], 0.0)
        self.assertEqual(result["logloss"], 0.0)

    def test_fit_returns_normalized_distributions(self):
        params, idx, info = fit_shrinkage(self.h2025 + self.h2026, 10.0)
        self.assertTrue(info["success"])
        ph, pa = predict_shrinkage(params, idx, "A", "B")
        self.assertAlmostEqual(sum(ph), 1.0, places=10)
        self.assertAlmostEqual(sum(pa), 1.0, places=10)

if __name__ == "__main__":
    unittest.main()
