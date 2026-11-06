import importlib.util
import json
import math
import sys
import tempfile
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


def independent_exact_hessian(theta, x_home, x_away, k, team_count):
    """Exact Hessian for the independently written clipped Poisson objective."""
    eta_h_raw = x_home @ theta
    eta_a_raw = x_away @ theta
    active_h = ((eta_h_raw > -20.0) & (eta_h_raw < 20.0)).astype(float)
    active_a = ((eta_a_raw > -20.0) & (eta_a_raw < 20.0)).astype(float)
    rate_h = np.exp(np.clip(eta_h_raw, -20.0, 20.0))
    rate_a = np.exp(np.clip(eta_a_raw, -20.0, 20.0))
    weighted_h = rate_h * active_h
    weighted_a = rate_a * active_a
    hessian = x_home.T @ (x_home * weighted_h[:, None])
    hessian += x_away.T @ (x_away * weighted_a[:, None])
    diagonal = np.zeros(theta.size, dtype=float)
    diagonal[2:] = 2.0 * k
    hessian[np.diag_indices_from(hessian)] += diagonal
    return 0.5 * (hessian + hessian.T)


def fit_exact_newton(prior, k, gtol=1e-7, maxiter=100):
    """Damped full-Newton fit with exact Hessian and Armijo backtracking."""
    teams, idx, xh, xa, hg, ag = audit.build_design(prior)
    n_teams = len(teams)
    theta = np.zeros(2 + 2 * n_teams, dtype=float)
    theta[0] = math.log(1.3)

    for iteration in range(maxiter + 1):
        value, gradient = audit.independent_objective_gradient(
            theta, xh, xa, hg, ag, k, n_teams
        )
        gradient_norm = float(np.max(np.abs(gradient)))
        hessian = independent_exact_hessian(theta, xh, xa, k, n_teams)
        minimum_eigenvalue = float(np.linalg.eigvalsh(hessian)[0])
        if gradient_norm <= gtol:
            return theta, idx, {
                "iterations": iteration,
                "gradientInfinityNorm": gradient_norm,
                "minimumHessianEigenvalue": minimum_eigenvalue,
                "objective": float(value),
                "success": True,
            }
        if iteration == maxiter:
            break

        # The ridge penalty makes the exact Hessian positive definite for
        # this design. Fail closed rather than silently changing the objective.
        if not math.isfinite(minimum_eigenvalue) or minimum_eigenvalue <= 0.0:
            raise RuntimeError(
                f"exact Hessian is not positive definite: {minimum_eigenvalue}"
            )
        direction = np.linalg.solve(hessian, -gradient)
        directional_derivative = float(gradient @ direction)
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise RuntimeError(
                f"Newton direction is not a descent direction: {directional_derivative}"
            )

        step = 1.0
        accepted = False
        for _ in range(60):
            candidate = theta + step * direction
            candidate_value = audit.independent_objective_gradient(
                candidate, xh, xa, hg, ag, k, n_teams
            )[0]
            if (
                math.isfinite(candidate_value)
                and candidate_value <= value + 1e-4 * step * directional_derivative
            ):
                theta = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError(
                f"Armijo line search stalled at iteration={iteration}, "
                f"gradient={gradient_norm}, objective={value}"
            )

    raise RuntimeError(
        f"exact Newton did not reach gtol={gtol} in {maxiter} iterations"
    )


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

    def test_history_loader_rejects_boolean_goal_counts(self):
        base = {
            "date": "2026-01-01",
            "competition": "J1",
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
        for field in ("home_goals", "away_goals"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                row = dict(base)
                row[field] = True
                path = Path(temp_dir) / "history.json"
                path.write_text(json.dumps({"history": [row]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "boolean goal count"):
                    audit.load_history(path, {})

    def test_history_loader_rejects_invalid_or_colliding_team_names(self):
        base = {
            "date": "2026-01-01",
            "competition": "J1",
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
        cases = [
            ({"home": ""}, {}),
            ({"away": " "}, {}),
            ({"home": None}, {}),
            ({}, {"B": "A"}),
        ]
        for changes, aliases in cases:
            with self.subTest(changes=changes, aliases=aliases), tempfile.TemporaryDirectory() as temp_dir:
                row = dict(base)
                row.update(changes)
                path = Path(temp_dir) / "history.json"
                path.write_text(json.dumps({"history": [row]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid home/away team name|canonical home/away team names collide"):
                    audit.load_history(path, aliases)

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

    def test_stricter_ftol_sensitivity_reports_gradient_and_prediction_drift(self):
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
            "gtol": 1e-6,
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
                # L-BFGS-B can terminate by ftol before meeting gtol.
                # Record gradient convergence separately; do not mislabel result.success.
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
            absolute_error = np.abs(strict - frozen)
            max_index = np.unravel_index(int(np.argmax(absolute_error)), absolute_error.shape)
            max_error = float(absolute_error[max_index])
            max_errors[str(k)] = {
                "maxAbsError": max_error,
                "target": {
                    "date": rows[max_index[0]]["date"],
                    "comp": rows[max_index[0]]["comp"],
                    "home": rows[max_index[0]]["home"],
                    "away": rows[max_index[0]]["away"],
                    "side": rows[max_index[0]]["side"],
                    "category": int(max_index[1]),
                },
            }
            self.assertTrue(np.all(np.isfinite(strict)))
            self.assertLessEqual(float(np.max(np.abs(strict.sum(axis=1) - 1.0))), 1e-9)

        gradients = [item["gradientInfinityNorm"] for item in diagnostics]
        iterations = [item["iterations"] for item in diagnostics]
        actual = np.asarray([row["actual"] for row in rows], dtype=int)
        p6 = np.asarray(predictions[6], dtype=float)
        p10 = np.asarray(predictions[10], dtype=float)
        strict_metrics = {
            "k6": audit.score(p6, actual),
            "k10": audit.score(p10, actual),
        }
        strict_metrics["differenceK6MinusK10"] = {
            key: strict_metrics["k6"][key] - strict_metrics["k10"][key]
            for key in ("brier", "logloss", "accuracy")
        }
        strict_residuals = {
            "k6": audit.poisson_residuals(rows, p6, lambda row: True),
            "k10": audit.poisson_residuals(rows, p10, lambda row: True),
        }
        strict_calibration = {
            "k6": audit.calibration(rows, p6, 6, lambda row: True),
            "k10": audit.calibration(rows, p10, 10, lambda row: True),
        }
        strict_bootstrap = audit.moving_block_bootstrap(
            rows, p6, p10, reps=10000, seed=20261105
        )
        summary = {
            "strictOptions": options,
            "fitCount": len(diagnostics),
            "scipySuccessCount": len(diagnostics),
            "maxGradientInfinityNorm": max(gradients),
            "medianGradientInfinityNorm": float(np.median(gradients)),
            "minGradientInfinityNorm": min(gradients),
            "maxIterations": max(iterations),
            "gradientTolerance": options["gtol"],
            "gradientToleranceMetCount": sum(g <= options["gtol"] for g in gradients),
            "gradientToleranceNotMetCount": sum(g > options["gtol"] for g in gradients),
            "maxProbabilityAbsErrorVsFrozenCsv": max_errors,
            "strictFitMetrics": strict_metrics,
            "strictFitPoissonResidualRMS": strict_residuals,
            "strictFitFourCategoryCalibration": strict_calibration,
            "strictFitBootstrap10k": strict_bootstrap,
            "interpretation": (
                "Strict-ftol refits are a sensitivity diagnostic. "
                "Do not treat SciPy success as gradient convergence or require "
                "the stricter refit to equal the historical default-stopping CSV."
            ),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))




    def test_exact_hessian_matches_finite_difference_of_gradient(self):
        prior = self.synthetic_matches()
        teams, idx, xh, xa, hg, ag = audit.build_design(prior)
        n = len(teams)
        rng = np.random.default_rng(24680)
        theta = rng.normal(0.0, 0.05, size=2 + 2 * n)
        _, analytic_gradient = audit.independent_objective_gradient(
            theta, xh, xa, hg, ag, 6.0, n
        )
        analytic_hessian = independent_exact_hessian(theta, xh, xa, 6.0, n)
        numeric_hessian = np.zeros_like(analytic_hessian)
        eps = 2e-5
        for j in range(theta.size):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += eps
            minus[j] -= eps
            grad_plus = audit.independent_objective_gradient(
                plus, xh, xa, hg, ag, 6.0, n
            )[1]
            grad_minus = audit.independent_objective_gradient(
                minus, xh, xa, hg, ag, 6.0, n
            )[1]
            numeric_hessian[:, j] = (grad_plus - grad_minus) / (2.0 * eps)
        self.assertLess(
            float(np.max(np.abs(analytic_hessian - numeric_hessian))),
            2e-5,
        )
        self.assertLess(
            float(np.max(np.abs(analytic_hessian - analytic_hessian.T))),
            1e-12,
        )

    def test_exact_hessian_matches_finite_difference_in_clipped_regions(self):
        prior = self.synthetic_matches()
        teams, idx, xh, xa, hg, ag = audit.build_design(prior)
        n = len(teams)
        rng = np.random.default_rng(97531)
        for intercept in (20.5, -20.5):
            theta = rng.normal(0.0, 0.005, size=2 + 2 * n)
            theta[0] = intercept
            theta[1] = 0.0
            raw_home = xh @ theta
            raw_away = xa @ theta
            if intercept > 0:
                self.assertTrue(np.all(raw_home > 20.0))
                self.assertTrue(np.all(raw_away > 20.0))
            else:
                self.assertTrue(np.all(raw_home < -20.0))
                self.assertTrue(np.all(raw_away < -20.0))

            analytic = independent_exact_hessian(theta, xh, xa, 6.0, n)
            numeric = np.zeros_like(analytic)
            eps = 1e-6
            for j in range(theta.size):
                plus, minus = theta.copy(), theta.copy()
                plus[j] += eps
                minus[j] -= eps
                grad_plus = audit.independent_objective_gradient(
                    plus, xh, xa, hg, ag, 6.0, n
                )[1]
                grad_minus = audit.independent_objective_gradient(
                    minus, xh, xa, hg, ag, 6.0, n
                )[1]
                numeric[:, j] = (grad_plus - grad_minus) / (2.0 * eps)
            self.assertLess(
                float(np.max(np.abs(analytic - numeric))),
                2e-5,
            )

    def test_exact_newton_reaches_gradient_tolerance_for_all_34_fits(self):
        """Independent exact-Hessian solver sensitivity; read-only on frozen data."""
        aliases_doc = json.loads(
            (ROOT / "data" / "team_aliases.json").read_text(encoding="utf-8")
        )
        aliases = aliases_doc.get("aliases", {})
        history, history_audit = audit.load_history(
            ROOT / "data" / "history.json", aliases
        )
        rows, matches, dates, fold_by_date = audit.load_frozen(
            ROOT / "verification" / "goal3_independent" / "goal3_outer_matches.csv"
        )
        self.assertEqual(history_audit["rows"], 429)
        self.assertEqual(len(matches), 236)
        self.assertEqual(len(rows), 472)
        self.assertEqual(len(dates), 17)

        predictions = {6: [None] * len(rows), 10: [None] * len(rows)}
        row_indices_by_date = defaultdict(list)
        for i, row in enumerate(rows):
            row_indices_by_date[row["date"]].append(i)
        diagnostics = []

        for date in dates:
            prior = [match for match in history if match["date"] < date]
            self.assertGreater(len(prior), 0, f"empty strict prior for {date}")
            teams, idx, xh, xa, hg, ag = audit.build_design(prior)
            n_teams = len(teams)
            theta0 = np.zeros(2 + 2 * n_teams, dtype=float)
            theta0[0] = math.log(1.3)

            for k in (6.0, 10.0):
                def fun_and_grad(theta):
                    return audit.independent_objective_gradient(
                        theta, xh, xa, hg, ag, k, n_teams
                    )

                theta_fit, idx, info = fit_exact_newton(
                    prior, k, gtol=1e-7, maxiter=100
                )
                grad_norm = info["gradientInfinityNorm"]
                self.assertTrue(
                    info["success"],
                    f"exact Newton failed date={date} k={k}: {info}",
                )
                self.assertLessEqual(
                    grad_norm, 1e-7,
                    f"gradient tolerance missed date={date} k={k}: {grad_norm}",
                )
                min_eigenvalue = info["minimumHessianEigenvalue"]
                self.assertGreater(
                    min_eigenvalue, 0.0,
                    f"non-positive Hessian date={date} k={k}: {min_eigenvalue}",
                )
                diagnostics.append({
                    "date": date,
                    "fold": fold_by_date[date],
                    "k": int(k),
                    "iterations": int(info["iterations"]),
                    "gradientInfinityNorm": grad_norm,
                    "minimumHessianEigenvalue": min_eigenvalue,
                    "objective": float(info["objective"]),
                })

                for match in matches:
                    if match["date"] != date:
                        continue
                    p_home, p_away = audit.predict_independent(
                        theta_fit, idx, match["home"], match["away"]
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
        self.assertTrue(all(
            p is not None for k in (6, 10) for p in predictions[k]
        ))
        p6 = np.asarray(predictions[6], dtype=float)
        p10 = np.asarray(predictions[10], dtype=float)
        actual = np.asarray([row["actual"] for row in rows], dtype=int)
        frozen = {
            k: np.asarray([
                [row[f"k{k}_p{category}"] for category in range(4)]
                for row in rows
            ], dtype=float)
            for k in (6, 10)
        }
        max_drift = {
            str(k): float(np.max(np.abs(predictions[k] - frozen[k])))
            for k in (6, 10)
        }
        metrics = {
            "k6": audit.score(p6, actual),
            "k10": audit.score(p10, actual),
        }
        metrics["differenceK6MinusK10"] = {
            key: metrics["k6"][key] - metrics["k10"][key]
            for key in ("brier", "logloss", "accuracy")
        }
        bootstrap = audit.moving_block_bootstrap(
            rows, p6, p10, reps=10000, seed=20261105
        )
        calibration = {
            "k6": audit.calibration(rows, p6, 6, lambda row: True),
            "k10": audit.calibration(rows, p10, 10, lambda row: True),
        }
        residuals = {
            "k6": audit.poisson_residuals(rows, p6, lambda row: True),
            "k10": audit.poisson_residuals(rows, p10, lambda row: True),
        }
        gradients = [item["gradientInfinityNorm"] for item in diagnostics]
        iterations = [item["iterations"] for item in diagnostics]
        print(json.dumps({
            "solver": "Damped full-Newton with exact Hessian and Armijo backtracking",
            "options": {"gtol": 1e-7, "maxiter": 100, "lineSearch": "Armijo backtracking"},
            "cohort": {"historyRows": 429, "matches": 236, "targets": 472, "dates": 17},
            "fitCount": len(diagnostics),
            "successCount": sum(item["gradientInfinityNorm"] <= 1e-8 for item in diagnostics),
            "maxGradientInfinityNorm": max(gradients),
            "medianGradientInfinityNorm": float(np.median(gradients)),
            "maxIterations": max(iterations),
            "minimumHessianEigenvalue": min(item["minimumHessianEigenvalue"] for item in diagnostics),
            "maxProbabilityDriftVsFrozenDefault": max_drift,
            "metrics": metrics,
            "movingBlockBootstrap10k": bootstrap,
            "fourCategoryCalibration": calibration,
            "poissonResidualRMS": residuals,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
