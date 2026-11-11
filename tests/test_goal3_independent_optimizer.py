import importlib.util
import json
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from collections import defaultdict
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "independent_optimizer_audit.py"
spec = importlib.util.spec_from_file_location("independent_optimizer_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

NESTED_SCRIPT = ROOT / "verification" / "goal3_independent" / "nested_validation.py"
nested_spec = importlib.util.spec_from_file_location("goal3_nested_validation", NESTED_SCRIPT)
nested_validation = importlib.util.module_from_spec(nested_spec)
sys.modules[nested_spec.name] = nested_validation
nested_spec.loader.exec_module(nested_validation)

REVIEW_SCRIPT = ROOT / "verification" / "goal3_independent" / "review_outer_matches.py"
review_spec = importlib.util.spec_from_file_location("review_outer_matches", REVIEW_SCRIPT)
review = importlib.util.module_from_spec(review_spec)
sys.modules[review_spec.name] = review
review_spec.loader.exec_module(review)

# Load the strict sensitivity audit and its adjacent research-only modules.
sys.path.insert(0, str(ROOT / "verification" / "goal3_independent"))
STRICT_SENSITIVITY_SCRIPT = ROOT / "verification" / "goal3_independent" / "strict_optimizer_sensitivity_audit.py"
strict_sensitivity_spec = importlib.util.spec_from_file_location(
    "strict_optimizer_sensitivity_audit", STRICT_SENSITIVITY_SCRIPT
)
strict_sensitivity = importlib.util.module_from_spec(strict_sensitivity_spec)
sys.modules[strict_sensitivity_spec.name] = strict_sensitivity
strict_sensitivity_spec.loader.exec_module(strict_sensitivity)


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

    def test_outer_source_audit_rejects_boolean_goal_counts(self):
        for field in ("home_goals", "away_goals"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                (root / "data").mkdir()
                (root / "data" / "team_aliases.json").write_text(
                    json.dumps({"aliases": {"A": "A", "B": "B"}}), encoding="utf-8"
                )
                row = {
                    "date": "2026-01-01",
                    "competition": "J1",
                    "home": "A",
                    "away": "B",
                    "home_goals": 1,
                    "away_goals": 0,
                    "fetchedAt": "2026-01-02T00:00:00Z",
                }
                row[field] = True
                (root / "data" / "history.json").write_text(
                    json.dumps({"history": [row]}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "Invalid history score"):
                    review.source_audit(root)

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

    def test_history_loader_rejects_partial_negative_and_noninteger_scores(self):
        base = {
            "date": "2026-01-01",
            "competition": "J1",
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
        cases = [
            {"home_goals": 1, "away_goals": None},
            {"home_goals": None, "away_goals": 0},
            {"home_goals": -1, "away_goals": 0},
            {"home_goals": 1, "away_goals": -1},
            {"home_goals": 1.0, "away_goals": 0},
            {"home_goals": 1, "away_goals": "0"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp_dir:
                row = dict(base)
                row.update(changes)
                path = Path(temp_dir) / "history.json"
                path.write_text(json.dumps({"history": [row]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid scored goal counts"):
                    audit.load_history(path, {})

    def test_history_loader_skips_only_fully_unscored_rows(self):
        scored = {
            "date": "2026-01-01",
            "competition": "J1",
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
        unscored = {
            "date": "2026-01-02",
            "competition": "J1",
            "home": "C",
            "away": "D",
            "home_goals": None,
            "away_goals": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(
                json.dumps({"history": [scored, unscored]}), encoding="utf-8"
            )
            loaded, info = audit.load_history(path, {})
        self.assertEqual(len(loaded), 1)
        self.assertEqual(info["rows"], 1)
        self.assertEqual((loaded[0]["hg"], loaded[0]["ag"]), (1, 0))

    def test_history_loader_rejects_noncanonical_or_invalid_dates(self):
        base = {
            "date": "2026-01-01",
            "competition": "J1",
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
        for bad_date in ("2026-02-30", "2026-2-03", "2025-12-31", "2026-W01-1"):
            with self.subTest(date=bad_date), tempfile.TemporaryDirectory() as temp_dir:
                row = dict(base, date=bad_date)
                path = Path(temp_dir) / "history.json"
                path.write_text(json.dumps({"history": [row]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit.load_history(path, {})

    def test_history_loader_rejects_non_object_rows_and_invalid_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(json.dumps({"history": [None]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                audit.load_history(path, {})
            path.write_text(json.dumps({"history": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain a list"):
                audit.load_history(path, {})

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


    def test_independent_objective_and_gradient_match_reference_for_both_k(self):
        prior = self.synthetic_matches()
        teams, _, xh, xa, hg, ag = audit.build_design(prior)
        n = len(teams)
        rng = np.random.default_rng(20261108)
        theta_values = [
            np.zeros(2 + 2 * n, dtype=float),
            rng.normal(0.0, 0.06, size=2 + 2 * n),
        ]
        original_minimize = nested_validation.minimize
        captured_objectives = []

        def capture_minimize(fun, x0, jac, method, options):
            captured_objectives.append(fun)
            return SimpleNamespace(
                x=np.asarray(x0, dtype=float).copy(),
                success=True,
                status=0,
                nit=0,
                message="objective captured for parity test",
            )

        nested_validation.minimize = capture_minimize
        try:
            for k in (6.0, 10.0):
                nested_validation.fit_shrinkage(prior, k)
                reference_objective = captured_objectives[-1]
                for theta in theta_values:
                    with self.subTest(k=k, theta_norm=float(np.linalg.norm(theta))):
                        reference_value, reference_gradient = reference_objective(theta)
                        independent_value, independent_gradient = (
                            audit.independent_objective_gradient(
                                theta, xh, xa, hg, ag, k, n
                            )
                        )
                        self.assertAlmostEqual(
                            independent_value, reference_value, delta=1e-10
                        )
                        self.assertLess(
                            float(np.max(np.abs(independent_gradient - reference_gradient))),
                            1e-10,
                        )
        finally:
            nested_validation.minimize = original_minimize

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

        options = dict(audit.STRICT_LBFGSB_OPTIONS)
        predictions = {6: [None] * len(rows), 10: [None] * len(rows)}
        row_indices_by_date = defaultdict(list)
        for i, row in enumerate(rows):
            row_indices_by_date[row["date"]].append(i)
        diagnostics = []
        parameter_diagnostics = []

        for date in dates:
            prior = [match for match in history if match["date"] < date]
            self.assertGreater(len(prior), 0, f"empty strict prior for {date}")
            for k in (6.0, 10.0):
                strict_theta, strict_idx, strict_info = audit.fit_independent(
                    prior, k, options=options
                )
                self.assertTrue(
                    strict_info["success"],
                    f"strict fit failed date={date} k={k}: {strict_info['message']}",
                )
                grad_norm = strict_info["gradientInfinityNorm"]
                # L-BFGS-B can terminate by ftol before meeting gtol.
                # Record gradient convergence separately; do not mislabel result.success.
                diagnostics.append({
                    "date": date,
                    "fold": fold_by_date[date],
                    "k": int(k),
                    "iterations": strict_info["iterations"],
                    "gradientInfinityNorm": grad_norm,
                    "objective": strict_info["objective"],
                })

                for match in matches:
                    if match["date"] != date:
                        continue
                    p_home, p_away = audit.predict_independent(
                        strict_theta, strict_idx, match["home"], match["away"]
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
        default_predictions = {6: [None] * len(rows), 10: [None] * len(rows)}
        row_indices_by_date = defaultdict(list)
        for i, row in enumerate(rows):
            row_indices_by_date[row["date"]].append(i)
        diagnostics = []
        parameter_diagnostics = []

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
                default_theta, default_idx, default_info = audit.fit_independent(prior, k)
                self.assertEqual(default_idx, idx)
                parameter_delta = theta_fit - default_theta
                team_count = len(idx)
                objective_gap = float(default_info["objective"] - info["objective"])
                self.assertGreaterEqual(
                    objective_gap, -1e-7,
                    f"default L-BFGS-B objective beats exact Newton beyond tolerance: date={date} k={k} gap={objective_gap}",
                )
                parameter_diagnostics.append({
                    "date": date,
                    "fold": fold_by_date[date],
                    "k": int(k),
                    "defaultLbfgsbGradientInfinityNorm": float(default_info["gradientInfinityNorm"]),
                    "exactNewtonGradientInfinityNorm": float(info["gradientInfinityNorm"]),
                    "maxAbsParameterDifference": float(np.max(np.abs(parameter_delta))),
                    "l2ParameterDifference": float(np.linalg.norm(parameter_delta)),
                    "maxAbsByBlock": {
                        "intercept": float(abs(parameter_delta[0])),
                        "homeAdvantage": float(abs(parameter_delta[1])),
                        "attack": float(np.max(np.abs(parameter_delta[2:2 + team_count]))),
                        "defense": float(np.max(np.abs(parameter_delta[2 + team_count:]))),
                    },
                    "defaultObjective": float(default_info["objective"]),
                    "exactNewtonObjective": float(info["objective"]),
                    "defaultMinusExactObjective": objective_gap,
                })
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
                        default_p_home, default_p_away = audit.predict_independent(
                            default_theta, default_idx, match["home"], match["away"]
                        )
                        default_predictions[int(k)][row_index] = (
                            default_p_home if row["side"] == "home" else default_p_away
                        )

        self.assertEqual(len(diagnostics), 34)
        self.assertEqual(len(parameter_diagnostics), 34)
        self.assertTrue(all(
            p is not None for k in (6, 10) for p in predictions[k]
        ))
        self.assertTrue(all(
            p is not None for k in (6, 10) for p in default_predictions[k]
        ))
        p6 = np.asarray(predictions[6], dtype=float)
        p10 = np.asarray(predictions[10], dtype=float)
        default_p6 = np.asarray(default_predictions[6], dtype=float)
        default_p10 = np.asarray(default_predictions[10], dtype=float)
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
        default_vs_exact_probability_drift = {
            "k6": float(np.max(np.abs(p6 - default_p6))),
            "k10": float(np.max(np.abs(p10 - default_p10))),
        }
        parameter_summary_by_k = {}
        for k in (6, 10):
            items = [item for item in parameter_diagnostics if item["k"] == k]
            parameter_summary_by_k[str(k)] = {
                "fitCount": len(items),
                "maxAbsParameterDifference": max(x["maxAbsParameterDifference"] for x in items),
                "medianAbsParameterDifference": float(np.median([x["maxAbsParameterDifference"] for x in items])),
                "p95AbsParameterDifference": float(np.quantile([x["maxAbsParameterDifference"] for x in items], 0.95, method="linear")),
                "maxL2ParameterDifference": max(x["l2ParameterDifference"] for x in items),
                "maxDefaultGradientInfinityNorm": max(x["defaultLbfgsbGradientInfinityNorm"] for x in items),
                "maxExactNewtonGradientInfinityNorm": max(x["exactNewtonGradientInfinityNorm"] for x in items),
                "minimumDefaultMinusExactObjective": min(x["defaultMinusExactObjective"] for x in items),
                "maximumDefaultMinusExactObjective": max(x["defaultMinusExactObjective"] for x in items),
                "maxAbsByBlock": {
                    block: max(x["maxAbsByBlock"][block] for x in items)
                    for block in ("intercept", "homeAdvantage", "attack", "defense")
                },
            }
        print(json.dumps({
            "solver": "Damped full-Newton with exact Hessian and Armijo backtracking",
            "options": {"gtol": 1e-7, "maxiter": 100, "lineSearch": "Armijo backtracking"},
            "cohort": {"historyRows": 429, "matches": 236, "targets": 472, "dates": 17},
            "fitCount": len(diagnostics),
            "successCount": len(diagnostics),
            "gradientTolerance": 1e-7,
            "gradientToleranceMetCount": sum(item["gradientInfinityNorm"] <= 1e-7 for item in diagnostics),
            "gradientToleranceNotMetCount": sum(item["gradientInfinityNorm"] > 1e-7 for item in diagnostics),
            "stricter1e8Count": sum(item["gradientInfinityNorm"] <= 1e-8 for item in diagnostics),
            "maxGradientInfinityNorm": max(gradients),
            "medianGradientInfinityNorm": float(np.median(gradients)),
            "maxIterations": max(iterations),
            "minimumHessianEigenvalue": min(item["minimumHessianEigenvalue"] for item in diagnostics),
            "maxProbabilityDriftVsFrozenDefault": max_drift,
            "maxProbabilityDriftExactNewtonVsDefaultLbfgsb": default_vs_exact_probability_drift,
            "defaultLbfgsbVsExactNewtonParameterSummary": parameter_summary_by_k,
            "defaultLbfgsbVsExactNewtonPerFit": parameter_diagnostics,
            "metrics": metrics,
            "movingBlockBootstrap10k": bootstrap,
            "fourCategoryCalibration": calibration,
            "poissonResidualRMS": residuals,
        }, ensure_ascii=False, indent=2))


class NestedValidationHistoryInputTests(unittest.TestCase):
    @staticmethod
    def _row(**overrides):
        row = {
            "date": "2026-09-01",
            "competition": "J1",
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
        row.update(overrides)
        return row

    def _load(self, history_rows):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_path = root / "history.json"
            aliases_path = root / "aliases.json"
            csv_path = root / "2025.csv"
            history_path.write_text(
                json.dumps({"history": history_rows}), encoding="utf-8"
            )
            aliases_path.write_text(
                json.dumps({"aliases": {}}), encoding="utf-8"
            )
            csv_path.write_text(
                "date,homeTeam,awayTeam,score\n2025/01/01,A,B,1-0\n",
                encoding="utf-8",
            )
            return nested_validation.load_data(
                str(history_path), str(aliases_path), str(csv_path)
            )

    def test_rejects_boolean_goal_counts(self):
        for field in ("home_goals", "away_goals"):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "invalid scored goal counts"
            ):
                self._load([self._row(**{field: True})])

    def test_rejects_negative_and_partial_goal_counts(self):
        for changes in (
            {"home_goals": -1},
            {"away_goals": -1},
            {"home_goals": 1, "away_goals": None},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "invalid scored goal counts"
            ):
                self._load([self._row(**changes)])

    def test_rejects_missing_or_colliding_team_names(self):
        for changes in (
            {"home": ""},
            {"away": " "},
            {"away": "A"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._load([self._row(**changes)])

    def test_skips_fully_unscored_rows(self):
        history = [
            self._row(),
            self._row(
                date="2026-09-02",
                home="C",
                away="D",
                home_goals=None,
                away_goals=None,
            ),
        ]
        _, scored = self._load(history)
        self.assertEqual(len(scored), 1)
        self.assertEqual((scored[0]["hg"], scored[0]["ag"]), (1, 0))



class StrictOptimizerModuleTests(unittest.TestCase):
    @staticmethod
    def _prior():
        teams = ["A", "B", "C", "D", "E", "F"]
        score_pairs = [
            (1, 0), (2, 1), (0, 0), (1, 2), (3, 1), (0, 1),
            (2, 2), (1, 1), (0, 2), (2, 0), (1, 0), (0, 0),
            (3, 2), (1, 1), (2, 1), (0, 2), (2, 0), (1, 3),
            (1, 0), (2, 2), (0, 1), (3, 1), (1, 2), (0, 0),
        ]
        rows = []
        for i, (hg, ag) in enumerate(score_pairs):
            home = teams[i % len(teams)]
            away = teams[(i + 1) % len(teams)]
            rows.append({
                "date": f"2026-08-{i + 1:02d}",
                "comp": "J1",
                "home": home,
                "away": away,
                "hg": hg,
                "ag": ag,
            })
        return rows

    def test_module_exact_newton_uses_same_strict_gradient_tolerance_for_both_arms(self):
        prior = self._prior()
        for k in (6.0, 10.0):
            with self.subTest(k=k):
                theta, index, info = audit.fit_exact_newton(prior, k, gtol=1e-7, maxiter=100)
                self.assertTrue(info["success"])
                self.assertEqual(info["status"], "converged_gradient_tolerance")
                self.assertEqual(info["gradientTolerance"], 1e-7)
                self.assertLessEqual(info["gradientInfinityNorm"], 1e-7)
                self.assertGreater(info["minimumHessianEigenvalueAtSolution"], 0.0)
                self.assertTrue(np.all(np.isfinite(theta)))
                self.assertEqual(len(index), 6)

    def test_module_exact_hessian_matches_finite_difference_of_gradient_for_both_arms(self):
        prior = self._prior()
        _, _, xh, xa, hg, ag = audit.build_design(prior)
        theta = np.linspace(-0.04, 0.05, xh.shape[1])
        team_count = (theta.size - 2) // 2
        step = 1e-5

        for k in (6.0, 10.0):
            with self.subTest(k=k):
                analytic = audit.exact_hessian(theta, xh, xa, k, team_count)
                numeric = np.zeros_like(analytic)
                for j in range(theta.size):
                    delta = np.zeros_like(theta)
                    delta[j] = step
                    grad_plus = audit.independent_objective_gradient(
                        theta + delta, xh, xa, hg, ag, k, team_count
                    )[1]
                    grad_minus = audit.independent_objective_gradient(
                        theta - delta, xh, xa, hg, ag, k, team_count
                    )[1]
                    numeric[:, j] = (grad_plus - grad_minus) / (2 * step)
                self.assertLess(float(np.max(np.abs(analytic - numeric))), 1e-6)


class StrictSensitivityGitBlobHeaderTests(unittest.TestCase):
    def test_git_blob_sha_uses_nul_terminated_git_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "known.txt"
            path.write_bytes(b"abc")
            self.assertEqual(
                strict_sensitivity.git_blob_sha(path),
                "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f",
            )



class StrictSensitivityRuntimeProvenanceTests(unittest.TestCase):
    def test_runtime_fingerprint_is_key_order_invariant(self):
        first = {
            "machine": "x86_64",
            "numpyRuntimeConfig": '[{"architecture":"Haswell","num_threads":4}]',
            "numpyVersion": "2.4.6",
        }
        second = {
            "numpyVersion": "2.4.6",
            "numpyRuntimeConfig": '[{"architecture":"Haswell","num_threads":4}]',
            "machine": "x86_64",
        }
        a = strict_sensitivity.execution_environment_fingerprint(
            {"executionEnvironment": first}
        )
        b = strict_sensitivity.execution_environment_fingerprint(
            {"executionEnvironment": second}
        )
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["details"], first)

    def test_runtime_fingerprint_fails_closed_when_missing(self):
        with self.assertRaisesRegex(ValueError, "missing executionEnvironment"):
            strict_sensitivity.execution_environment_fingerprint({})

class StrictSensitivityExecutionProvenanceTests(unittest.TestCase):
    def test_pull_request_records_merge_sha_and_source_head_separately(self):
        merge_sha = "a" * 40
        head_sha = "b" * 40
        base_sha = "c" * 40
        provenance = strict_sensitivity.github_execution_provenance(
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_SHA": merge_sha,
                "GITHUB_REF": "refs/pull/20/merge",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
            },
            {
                "pull_request": {
                    "head": {"sha": head_sha},
                    "base": {"sha": base_sha},
                }
            },
        )
        self.assertEqual(provenance["githubSha"], merge_sha)
        self.assertEqual(provenance["githubEventSha"], merge_sha)
        self.assertEqual(provenance["pullRequestHeadSha"], head_sha)
        self.assertEqual(provenance["pullRequestBaseSha"], base_sha)
        self.assertIsNone(provenance["pushAfterSha"])
        self.assertEqual(provenance["eventName"], "pull_request")

    def test_push_records_after_sha_without_pr_fields(self):
        after_sha = "d" * 40
        provenance = strict_sensitivity.github_execution_provenance(
            {
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_SHA": after_sha,
                "GITHUB_REF": "refs/heads/research/example",
            },
            {"after": after_sha},
        )
        self.assertEqual(provenance["githubEventSha"], after_sha)
        self.assertEqual(provenance["pushAfterSha"], after_sha)
        self.assertIsNone(provenance["pullRequestHeadSha"])
        self.assertIsNone(provenance["pullRequestBaseSha"])

    def test_workflow_dispatch_requires_valid_event_sha(self):
        event_sha = "f" * 40
        env = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_SHA": event_sha,
            "GITHUB_REF": "refs/heads/research/example",
        }
        provenance = strict_sensitivity.github_execution_provenance(env, {})
        self.assertEqual(provenance["eventName"], "workflow_dispatch")
        self.assertEqual(provenance["githubEventSha"], event_sha)
        self.assertIsNone(provenance["pushAfterSha"])

        invalid_env = dict(env, GITHUB_SHA="short")
        with self.assertRaisesRegex(ValueError, "GITHUB_SHA"):
            strict_sensitivity.github_execution_provenance(invalid_env, {})

    def test_invalid_push_after_sha_fails_closed(self):
        event_sha = "d" * 40
        env = {
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_SHA": event_sha,
            "GITHUB_REF": "refs/heads/research/example",
        }
        invalid_payloads = ({}, {"after": "not-a-git-sha"}, {"after": "e" * 40})
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                strict_sensitivity.github_execution_provenance(env, payload)

        invalid_env = dict(env, GITHUB_SHA="short")
        with self.assertRaisesRegex(ValueError, "GITHUB_SHA"):
            strict_sensitivity.github_execution_provenance(
                invalid_env, {"after": event_sha}
            )

    def test_invalid_pull_request_metadata_fails_closed(self):
        valid_env = {
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_REF": "refs/pull/20/merge",
        }
        invalid_payloads = (
            {},
            {"pull_request": {}},
            {"pull_request": {"head": {"sha": "b" * 40}, "base": {}}},
            {
                "pull_request": {
                    "head": {"sha": "not-a-git-sha"},
                    "base": {"sha": "c" * 40},
                }
            },
            {
                "pull_request": {
                    "head": {"sha": "b" * 40},
                    "base": {"sha": "not-a-git-sha"},
                }
            },
            {"pull_request": {"head": None, "base": {}}},
            [],
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                strict_sensitivity.github_execution_provenance(valid_env, payload)

        invalid_env = dict(valid_env, GITHUB_SHA="short")
        valid_payload = {
            "pull_request": {
                "head": {"sha": "b" * 40},
                "base": {"sha": "c" * 40},
            }
        }
        with self.assertRaisesRegex(ValueError, "GITHUB_SHA"):
            strict_sensitivity.github_execution_provenance(invalid_env, valid_payload)


if __name__ == "__main__":
    unittest.main()