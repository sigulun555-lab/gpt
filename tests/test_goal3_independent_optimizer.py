import importlib.util
import json
import math
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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

    def test_strict_optimizer_convergence_and_probability_parity_on_frozen_cohort(self):
        """Re-fit all 34 date/k combinations with tighter stopping controls.

        This is a research-only sensitivity check. It does not write predictions,
        change the frozen CSV, or alter any production path.
        """
        aliases_doc = json.loads((ROOT / "data" / "team_aliases.json").read_text(encoding="utf-8"))
        aliases = aliases_doc.get("aliases", {})
        history, history_audit = audit.load_history(ROOT / "data" / "history.json", aliases)
        rows, matches, dates, fold_by_date = audit.load_frozen(
            ROOT / "verification" / "goal3_independent" / "goal3_outer_matches.csv"
        )
        self.assertEqual(history_audit["rows"], 429)
        self.assertEqual(len(matches), 236)
        self.assertEqual(len(rows), 472)
        self.assertEqual(len(dates), 17)

        options = {
            "maxiter": 2000,
            "ftol": 1e-12,
            "gtol": 1e-5,
            "maxls": 50,
            "maxcor": 20,
        }
        predictions = {6: [None] * len(rows), 10: [None] * len(rows)}
        row_indices_by_date = defaultdict(list)
        for i, row in enumerate(rows):
            row_indices_by_date[row["date"]].append(i)
        diagnostics = []

        for date in dates:
            prior = [match for match in history if match["date"] < date]
            self.assertGreater(len(prior), 0, f"empty strict prior for {date}")
            teams, idx, x_home, x_away, hg, ag = audit.build_design(prior)
            n_teams = len(teams)
            theta0 = np.zeros(2 + 2 * n_teams, dtype=float)
            theta0[0] = math.log(1.3)

            for k in (6.0, 10.0):
                def fun_and_jac(theta):
                    return audit.independent_objective_gradient(
                        theta, x_home, x_away, hg, ag, k, n_teams
                    )

                result = minimize(
                    fun_and_jac,
                    theta0.copy(),
                    jac=True,
                    method="L-BFGS-B",
                    options=options,
                )
                self.assertTrue(
                    result.success,
                    f"strict fit failed date={date} k={k}: {result.message}",
                )
                grad_norm = float(np.max(np.abs(result.jac)))
                self.assertLessEqual(
                    grad_norm,
                    options["gtol"],
                    f"gradient tolerance not met date={date} k={k}: {grad_norm}",
                )
                diagnostics.append({
                    "date": date,
                    "fold": fold_by_date[date],
                    "k": int(k),
                    "iterations": int(result.nit),
                    "gradientInfinityNorm": grad_norm,
                    "objective": float(result.fun),
                })

                for match in matches:
                    if match["date"] != date:
                        continue
                    p_home, p_away = audit.predict_independent(
                        result.x, idx, match["home"], match["away"]
                    )
                    for row_index in row_indices_by_date[date]:
                        row = rows[row_index]
                        if (
                            row["comp"], row["home"], row["away"]
                        ) != (
                            match["comp"], match["home"], match["away"]
                        ):
                            continue
                        predictions[int(k)][row_index] = (
                            p_home if row["side"] == "home" else p_away
                        )

        self.assertEqual(len(diagnostics), 34)
        self.assertTrue(all(p is not None for k in (6, 10) for p in predictions[k]))

        max_errors = {}
        for k in (6, 10):
            strict = np.asarray(predictions[k], dtype=float)
            frozen = np.asarray(
                [[row[f"k{k}_p{category}"] for category in range(4)] for row in rows],
                dtype=float,
            )
            max_error = float(np.max(np.abs(strict - frozen)))
            max_errors[str(k)] = max_error
            self.assertLessEqual(max_error, 2e-6)

        gradients = [item["gradientInfinityNorm"] for item in diagnostics]
        iterations = [item["iterations"] for item in diagnostics]
        print({
            "strictOptions": options,
            "fitCount": len(diagnostics),
            "maxGradientInfinityNorm": max(gradients),
            "medianGradientInfinityNorm": float(np.median(gradients)),
            "maxIterations": max(iterations),
            "maxProbabilityAbsErrorVsFrozenCsv": max_errors,
        })


if __name__ == "__main__":
    unittest.main()
