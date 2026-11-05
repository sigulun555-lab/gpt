import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "independent_optimizer_audit.py"
spec = importlib.util.spec_from_file_location("independent_optimizer_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


class IndependentOptimizerAuditTests(unittest.TestCase):
    def synthetic_matches(self):
        teams = ["A", "B", "C", "D", "E"]
        scores = [
            (1, 0), (2, 1), (0, 0), (1, 2), (3, 1),
            (0, 1), (2, 2), (1, 1), (0, 2), (2, 0),
            (1, 0), (0, 0), (3, 2), (1, 1), (2, 1),
        ]
        rows = []
        for i, (hg, ag) in enumerate(scores):
            rows.append({
                "date": f"2026-08-{i + 1:02d}",
                "comp": "J1",
                "home": teams[i % len(teams)],
                "away": teams[(i + 1) % len(teams)],
                "hg": hg,
                "ag": ag,
            })
        return rows

    def test_dense_gradient_matches_central_finite_difference(self):
        prior = self.synthetic_matches()
        teams, idx, xh, xa, hg, ag = audit.build_design(prior)
        n = len(teams)
        rng = np.random.default_rng(12345)
        theta = rng.normal(0.0, 0.08, size=2 + 2 * n)
        value, grad = audit.independent_objective_gradient(theta, xh, xa, hg, ag, 6.0, n)
        self.assertTrue(math.isfinite(value))
        numeric = np.zeros_like(theta)
        eps = 1e-6
        for j in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += eps
            minus[j] -= eps
            f_plus = audit.independent_objective_gradient(plus, xh, xa, hg, ag, 6.0, n)[0]
            f_minus = audit.independent_objective_gradient(minus, xh, xa, hg, ag, 6.0, n)[0]
            numeric[j] = (f_plus - f_minus) / (2.0 * eps)
        self.assertLess(float(np.max(np.abs(grad - numeric))), 2e-5)

    def test_independent_fit_emits_valid_probabilities_for_both_k_values(self):
        prior = self.synthetic_matches()
        for k in (6.0, 10.0):
            theta, idx, info = audit.fit_independent(prior, k)
            self.assertTrue(info["success"])
            self.assertEqual(info["trainingMatches"], len(prior))
            ph, pa = audit.predict_independent(theta, idx, "A", "Z")
            for p in (ph, pa):
                self.assertEqual(len(p), 4)
                self.assertTrue(np.all(np.isfinite(p)))
                self.assertTrue(np.all(p >= 0.0))
                self.assertAlmostEqual(float(p.sum()), 1.0, places=10)

    def test_logloss_uses_protocol_probability_floor(self):
        probabilities = np.asarray([[1e-13, 0.2, 0.3, 0.5 - 1e-13]], dtype=float)
        result = audit.score(probabilities, np.asarray([0], dtype=int))
        self.assertAlmostEqual(result["logloss"], -math.log(audit.LOGLOSS_FLOOR), places=12)

    def test_poisson_probability_tail_is_nonnegative(self):
        for rate in (0.15, 0.5, 1.25, 3.5, 10.0):
            p = audit.poisson_categories(rate)
            self.assertTrue(np.all(p >= 0.0))
            self.assertAlmostEqual(float(p.sum()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
