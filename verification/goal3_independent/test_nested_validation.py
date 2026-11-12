import hashlib
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from verification.goal3_independent.nested_validation import (
    date_blocks, select_training, score, fit_shrinkage, predict_shrinkage,
    build_input_provenance, sha256_file, parse_lambda_grid, evaluate_dates,
)
from verification.goal3_independent.optimizer_gradient import (
    STRICT_LBFGSB_OPTIONS, fit_shrinkage_strict,
    gradient_diagnostics, objective_gradient,
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

    def test_lambda_grid_defaults_and_explicit_original_protocol(self):
        self.assertEqual(
            parse_lambda_grid(None),
            [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
             15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0],
        )
        self.assertEqual(
            parse_lambda_grid("1,3,10,30,100"),
            [1.0, 3.0, 10.0, 30.0, 100.0],
        )

    def test_lambda_grid_rejects_invalid_values_and_order(self):
        for raw in ("", "1,,3", "1,0,3", "1,-2,3", "1,1,3",
                    "3,1,10", "1,nan,3", "1,inf,3"):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_lambda_grid(raw)

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

    def test_analytic_gradient_matches_finite_difference(self):
        matches = self.h2025 + self.h2026
        params, idx, _ = fit_shrinkage(matches, 10.0)
        analytic = objective_gradient(matches, params, idx, 10.0)
        n = len(idx)
        homes = np.asarray([idx[m["home"]] for m in matches], dtype=int)
        aways = np.asarray([idx[m["away"]] for m in matches], dtype=int)
        hg = np.asarray([m["hg"] for m in matches], dtype=float)
        ag = np.asarray([m["ag"] for m in matches], dtype=float)

        def value(theta):
            mu, home_adv = theta[0], theta[1]
            attack = theta[2:2 + n]
            defense = theta[2 + n:2 + 2 * n]
            eh = np.clip(mu + home_adv + attack[homes] - defense[aways], -20.0, 20.0)
            ea = np.clip(mu + attack[aways] - defense[homes], -20.0, 20.0)
            return float(
                np.sum(np.exp(eh) - hg * eh)
                + np.sum(np.exp(ea) - ag * ea)
                + 10.0 * (np.sum(attack ** 2) + np.sum(defense ** 2))
            )

        eps = 1e-6
        numeric = np.empty_like(params)
        for i in range(len(params)):
            plus, minus = params.copy(), params.copy()
            plus[i] += eps
            minus[i] -= eps
            numeric[i] = (value(plus) - value(minus)) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)

    def test_strict_fit_uses_fixed_optimizer_and_meets_gradient_tolerance(self):
        params, idx, info = fit_shrinkage_strict(self.h2025 + self.h2026, 10.0)
        self.assertTrue(info["success"])
        self.assertEqual(info["optimizerName"], "L-BFGS-B")
        self.assertEqual(info["optimizerOptions"], STRICT_LBFGSB_OPTIONS)
        self.assertLessEqual(info["maxAbsGradient"], 1e-6)
        diagnostics = gradient_diagnostics(self.h2025 + self.h2026, params, idx, 10.0)
        self.assertAlmostEqual(info["maxAbsGradient"], diagnostics["maxAbsGradient"], places=12)

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

    def test_evaluate_dates_reports_gradient_gate_without_filtering_fits(self):
        result = evaluate_dates(
            ["2026-08-01"], self.h2025, self.h2026, "2025_plus_2026", 10.0
        )
        diagnostics = result["optimizerDiagnostics"]
        self.assertEqual(result["fitCount"], 1)
        self.assertEqual(result["fitFailureCount"], 0)
        self.assertEqual(diagnostics["optimizerFitCount"], 1)
        self.assertEqual(
            diagnostics["withinGradientGate1eMinus6"]
            + diagnostics["aboveGradientGate1eMinus6"],
            1,
        )
        self.assertIsInstance(diagnostics["maxAbsGradient"], float)

    def test_fit_returns_normalized_distributions(self):
        params, idx, info = fit_shrinkage(self.h2025 + self.h2026, 10.0)
        self.assertTrue(info["success"])
        self.assertTrue(info["optimizerFit"])
        self.assertEqual(info["gradientDimension"], len(params))
        self.assertGreaterEqual(info["maxAbsGradient"], 0.0)
        self.assertGreaterEqual(info["gradientL2"], 0.0)
        diagnostics = gradient_diagnostics(self.h2025 + self.h2026, params, idx, 10.0)
        self.assertIsInstance(diagnostics["maxAbsGradient"], float)
        self.assertIsInstance(diagnostics["gradientL2"], float)
        self.assertEqual(diagnostics["gradientDimension"], len(params))
        self.assertGreaterEqual(diagnostics["maxAbsGradient"], 0.0)
        self.assertGreaterEqual(diagnostics["gradientL2"], 0.0)
        ph, pa = predict_shrinkage(params, idx, "A", "B")
        self.assertAlmostEqual(sum(ph), 1.0, places=10)
        self.assertAlmostEqual(sum(pa), 1.0, places=10)

if __name__ == "__main__":
    unittest.main()
