#!/usr/bin/env python3
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification" / "goal3_independent"))
from selection_calibration_bootstrap_audit import K_VALUES, LOGLOSS_FLOOR, audit, flatten_targets, score_sums_by_date


def synthetic_report():
    matches = []
    start = date(2026, 9, 1)
    base = np.asarray([0.25, 0.35, 0.22, 0.18], dtype=float)
    for i in range(236):
        by_k = {}
        for ki, k in enumerate(K_VALUES):
            shift = (ki - 7.5) * 0.0004
            p_home = base + np.asarray([shift, -shift / 3, -shift / 3, -shift / 3])
            p_away = base + np.asarray([-shift, shift / 3, shift / 3, shift / 3])
            by_k[f"{k:.1f}"] = {
                "home": p_home.tolist(),
                "away": p_away.tolist(),
            }
        matches.append({
            "date": (start + timedelta(days=i % 17)).isoformat(),
            "comp": ("J1", "J2", "J3")[i % 3],
            "fold": 1,
            "home": f"home-{i}",
            "away": f"away-{i}",
            "hg": i % 4,
            "ag": (i + 1) % 4,
            "byK": by_k,
        })
    return {"allKPredictions": matches}


class SelectionCalibrationBootstrapAuditTests(unittest.TestCase):
    def test_target_expansion_and_all_k_vectors(self):
        rows, dates = flatten_targets(synthetic_report())
        self.assertEqual(len(rows), 472)
        self.assertEqual(len(dates), 17)
        self.assertEqual({row["side"] for row in rows}, {"home", "away"})
        self.assertEqual(set(rows[0]["probabilities"]), set(K_VALUES))

    def test_audit_is_reproducible_and_reports_all_families(self):
        report = synthetic_report()
        first = audit(report, replicates=1000, seed=77)
        second = audit(report, replicates=1000, seed=77)
        self.assertEqual(first, second)
        self.assertIn("python", first["auditEnvironment"])
        self.assertIn("numpy", first["auditEnvironment"])

        selection = first["selectionStability"]
        self.assertEqual(len(selection["selectionAdjusted"]["intervals"]), 15)
        self.assertAlmostEqual(sum(selection["bestCandidateFrequency"]["brier"].values()), 1.0)
        self.assertAlmostEqual(sum(selection["bestCandidateFrequency"]["logloss"].values()), 1.0)
        self.assertIn("6", selection["selectionAdjusted"]["intervals"])
        self.assertEqual(set(first["fourCategoryCalibration"]["families"]),
                         {"k6_by_league", "k10_by_league", "k6_all", "k10_all", "global_32"})
        self.assertEqual(first["fourCategoryCalibration"]["families"]["k6_by_league"]["comparisonCount"], 12)
        self.assertEqual(first["fourCategoryCalibration"]["families"]["global_32"]["comparisonCount"], 32)

    def test_rejects_noncanonical_invalid_or_non_2026_dates(self):
        for value in ("2026-02-30", "2026-2-03", "2025-09-01", "2026-W01-1"):
            report = synthetic_report()
            report["allKPredictions"][0]["date"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid target date"):
                flatten_targets(report)

    def test_rejects_boolean_goal_counts_and_fold(self):
        for field in ("hg", "ag"):
            report = synthetic_report()
            report["allKPredictions"][0][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "invalid observed goals"):
                flatten_targets(report)
        report = synthetic_report()
        report["allKPredictions"][0]["fold"] = True
        with self.assertRaisesRegex(ValueError, "invalid fold"):
            flatten_targets(report)

    def test_rejects_blank_or_colliding_team_names(self):
        for field, value in (("home", " "), ("away", ""), ("away", "home-0")):
            report = synthetic_report()
            report["allKPredictions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                flatten_targets(report)

    def test_logloss_uses_protocol_probability_floor(self):
        p = np.asarray([1e-13, 0.2, 0.3, 0.5 - 1e-13], dtype=float)
        row = {
            "date": "2026-09-01",
            "actual": 0,
            "probabilities": {k: p for k in K_VALUES},
        }
        counts, sums = score_sums_by_date([row], ["2026-09-01"])
        observed = sums["logloss"][0, K_VALUES.index(6.0)] / counts[0]
        self.assertAlmostEqual(observed, -math.log(LOGLOSS_FLOOR), places=12)

    def test_rejects_invalid_probability_vector(self):
        report = synthetic_report()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            flatten_targets(report)

    def test_rejects_duplicate_match(self):
        report = synthetic_report()
        report["allKPredictions"][1] = dict(report["allKPredictions"][0])
        with self.assertRaises(ValueError):
            flatten_targets(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
