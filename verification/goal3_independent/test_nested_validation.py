import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from verification.goal3_independent.nested_validation import (
    date_blocks, select_training, score, fit_shrinkage, predict_shrinkage,
    build_input_provenance, sha256_file,
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

    def test_input_provenance_records_input_code_and_execution_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "history": root / "history.json",
                "aliases": root / "aliases.json",
                "csv2025": root / "2025.csv",
            }
            payloads = {
                "history": b'{"history": []}',
                "aliases": b'{"aliases": {}}',
                "csv2025": b"date,homeTeam,awayTeam,score\\n",
            }
            for name, path in paths.items():
                path.write_bytes(payloads[name])
            with patch.dict(os.environ, {
                "GITHUB_SHA": "test-commit",
                "GITHUB_REF": "refs/pull/20/merge",
                "GITHUB_RUN_ID": "12345",
                "GITHUB_RUN_ATTEMPT": "2",
            }, clear=False):
                provenance = build_input_provenance(
                    paths["history"], paths["aliases"], paths["csv2025"]
                )
            expected = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            }
            self.assertEqual(provenance["inputs"]["history2026"]["sha256"], expected["history"])
            self.assertEqual(provenance["inputs"]["teamAliases"]["sha256"], expected["aliases"])
            self.assertEqual(provenance["inputs"]["history2025Csv"]["sha256"], expected["csv2025"])
            self.assertEqual(provenance["script"]["sha256"], sha256_file(Path(__file__).with_name("nested_validation.py")))
            self.assertEqual(provenance["execution"]["githubSha"], "test-commit")
            self.assertEqual(provenance["execution"]["githubRef"], "refs/pull/20/merge")
            self.assertEqual(provenance["execution"]["workflowRunId"], "12345")
            self.assertEqual(provenance["execution"]["workflowRunAttempt"], "2")

    def test_fit_returns_normalized_distributions(self):
        params, idx, info = fit_shrinkage(self.h2025 + self.h2026, 10.0)
        self.assertTrue(info["success"])
        self.assertIsInstance(info["maxAbsGradient"], float)
        self.assertIsInstance(info["gradientL2"], float)
        self.assertGreaterEqual(info["maxAbsGradient"], 0.0)
        self.assertGreaterEqual(info["gradientL2"], 0.0)
        ph, pa = predict_shrinkage(params, idx, "A", "B")
        self.assertAlmostEqual(sum(ph), 1.0, places=10)
        self.assertAlmostEqual(sum(pa), 1.0, places=10)

if __name__ == "__main__":
    unittest.main()
