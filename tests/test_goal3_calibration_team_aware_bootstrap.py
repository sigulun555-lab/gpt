#!/usr/bin/env python3
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "goal3_independent"))
from calibration_team_aware_bootstrap_audit import (
    LEAGUES,
    audit,
    draw_cluster_weights,
    fit_weighted_logistic_batch,
    flatten_matches,
    weighted_bin_ess,
    weighted_bin_match_cluster_ess,
    prepare,
)


def synthetic_report():
    matches = []
    start = date(2026, 9, 1)
    rng = np.random.Generator(np.random.PCG64(9127))
    for index in range(236):
        league = LEAGUES[index % 3]
        home_index = index % 20
        away_index = (index * 7 + 3) % 20
        home = f"{league}-team-{home_index:02d}"
        away = f"{league}-team-{away_index:02d}"
        home_base = rng.dirichlet([5.0, 7.0, 4.0, 2.5])
        away_base = rng.dirichlet([5.0, 7.0, 4.0, 2.5])
        shift = rng.normal(0.0, 0.0005, size=4)
        shift -= shift.mean()
        by_k = {}
        for k in (6.0, 10.0):
            k_shift = shift * (1.0 if k == 6.0 else -1.0)
            home_p = home_base + k_shift
            away_p = away_base - k_shift
            by_k[f"{k:.1f}"] = {
                "home": home_p.tolist(),
                "away": away_p.tolist(),
            }
        matches.append({
            "date": (start + timedelta(days=index % 17)).isoformat(),
            "comp": league,
            "home": home,
            "away": away,
            "hg": int(rng.integers(0, 4)),
            "ag": int(rng.integers(0, 4)),
            "byK": by_k,
        })
    return {"allKPredictions": matches}


class TeamAwareCalibrationBootstrapTests(unittest.TestCase):
    def test_cohort_and_team_counts(self):
        matches, rows, dates, teams = flatten_matches(synthetic_report())
        self.assertEqual(len(matches), 236)
        self.assertEqual(len(rows), 472)
        self.assertEqual(len(dates), 17)
        self.assertEqual({league: len(value) for league, value in teams.items()},
                         {"J1": 20, "J2": 20, "J3": 20})

    def test_pair_preserving_weights_and_determinism(self):
        matches, rows, dates, teams = prepare(synthetic_report())
        first = draw_cluster_weights(
            matches, dates, teams, 100, np.random.Generator(np.random.PCG64(77))
        )
        second = draw_cluster_weights(
            matches, dates, teams, 100, np.random.Generator(np.random.PCG64(77))
        )
        date_counts_a, team_counts_a, match_weights_a = first
        date_counts_b, team_counts_b, match_weights_b = second
        np.testing.assert_array_equal(date_counts_a, date_counts_b)
        np.testing.assert_array_equal(match_weights_a, match_weights_b)
        for league in LEAGUES:
            np.testing.assert_array_equal(team_counts_a[league], team_counts_b[league])
            np.testing.assert_array_equal(team_counts_a[league].sum(axis=1), np.full(100, 20))
        np.testing.assert_array_equal(date_counts_a.sum(axis=1), np.full(100, 17))
        team_index = {lg: {team: j for j, team in enumerate(sorted(
            {t for m in matches if m["league"] == lg for t in (m["home"], m["away"])}
        ))} for lg in LEAGUES}
        date_index = {d: j for j, d in enumerate(dates)}
        for match_index, match in enumerate(matches):
            lg = match["league"]
            expected = (
                date_counts_a[:, date_index[match["date"]]]
                * team_counts_a[lg][:, team_index[lg][match["home"]]]
                * team_counts_a[lg][:, team_index[lg][match["away"]]]
            )
            np.testing.assert_array_equal(match_weights_a[:, match_index], expected)
        target_match = np.repeat(np.arange(236), 2)
        target_weights = match_weights_a[:, target_match]
        for match_index in range(236):
            np.testing.assert_array_equal(
                target_weights[:, 2 * match_index],
                target_weights[:, 2 * match_index + 1],
            )

    def test_weighted_bin_ess_uses_target_row_multiplicity(self):
        weights = np.asarray([[1, 1, 0], [10, 0, 0], [1, 2, 0]], dtype=float)
        targets_per_match = np.asarray([2, 1, 0], dtype=float)
        actual = weighted_bin_ess(weights, targets_per_match)
        np.testing.assert_allclose(actual, [3.0, 2.0, 16.0 / 6.0], rtol=0, atol=1e-12)

    def test_match_cluster_ess_aggregates_paired_targets_before_squaring(self):
        weights = np.asarray([[1, 1, 0], [10, 0, 0], [1, 2, 0]], dtype=float)
        targets_per_match = np.asarray([2, 1, 0], dtype=float)
        actual = weighted_bin_match_cluster_ess(weights, targets_per_match)
        np.testing.assert_allclose(actual, [9.0 / 5.0, 1.0, 2.0], rtol=0, atol=1e-12)
        row_ess = weighted_bin_ess(weights, targets_per_match)
        self.assertTrue(np.all(actual <= row_ess + 1e-12))

    def test_near_singular_fit_is_rejected(self):
        x = np.ones(20, dtype=float)
        y = np.asarray([0, 1] * 10, dtype=float)
        weights = np.ones((4, 20), dtype=float)
        _, valid, diagnostics = fit_weighted_logistic_batch(x, y, weights)
        self.assertFalse(valid.any())
        self.assertTrue(np.all(diagnostics["eigen"] < 1e-8))

    def test_audit_reproducible_and_reports_validity(self):
        report = synthetic_report()
        first = audit(report, "a" * 64, "b" * 64, replicates=1000, seed=20261113, batch_size=250)
        second = audit(report, "a" * 64, "b" * 64, replicates=1000, seed=20261113, batch_size=250)
        self.assertEqual(first, second)
        self.assertEqual(first["cohort"]["matches"], 236)
        self.assertEqual(first["cohort"]["teamSideTargets"], 472)
        self.assertEqual(len(first["calibration"]["perArm"]), 16)
        self.assertEqual(len(first["calibration"]["pairedK6MinusK10"]), 8)
        self.assertGreaterEqual(first["bootstrapValidity"]["commonValidCalibrationFraction"], 0.95)
        self.assertEqual(first["method"]["minimumBootstrapBinKishEffectiveSampleSize"], 20.0)
        self.assertEqual(first["method"]["minimumBootstrapBinMatchClusterKishEffectiveSampleSize"], 20.0)
        for item in first["probabilityBinReliability"]:
            if item["n"] >= 20:
                self.assertIn("bootstrapTargetRowEffectiveSampleSize", item)
                self.assertIn("bootstrapMatchClusterEffectiveSampleSize", item)
                self.assertLessEqual(
                    item["bootstrapMatchClusterEffectiveSampleSize"]["median"],
                    item["bootstrapTargetRowEffectiveSampleSize"]["median"] + 1e-12,
                )
            if item["ciStatus"] == "computed":
                self.assertGreaterEqual(item["bootstrapValidFraction"], 0.95)
                self.assertIn("simultaneous95Band", item)
            elif item["ciStatus"] == "insufficient_common_bootstrap_effective_sample_size":
                self.assertIsNone(first["criticalValues"]["probabilityBinMaxT"])
        self.assertEqual(first["method"]["pairPreservation"],
                         "one match weight is assigned to both home and away targets")

    def test_rejects_duplicate_match_and_invalid_simplex(self):
        report = synthetic_report()
        report["allKPredictions"][1] = dict(report["allKPredictions"][0])
        with self.assertRaises(ValueError):
            flatten_matches(report)
        report = synthetic_report()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            flatten_matches(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
