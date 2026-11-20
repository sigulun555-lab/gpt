#!/usr/bin/env python3
"""Unit tests for the GOAL3 ISO-week influence audit."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "verification" / "goal3_independent" / "week_influence_audit.py"
SPEC = importlib.util.spec_from_file_location("goal3_week_influence_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class WeekInfluenceTests(unittest.TestCase):
    def test_scores_and_leave_one_week_out(self):
        rows = []
        for week, date, n, p6, p10 in [
            ("2026-W01", "2026-01-01", 4, (0.8, 0.1, 0.05, 0.05), (0.7, 0.2, 0.05, 0.05)),
            ("2026-W02", "2026-01-08", 2, (0.6, 0.2, 0.1, 0.1), (0.7, 0.1, 0.1, 0.1)),
        ]:
            for i in range(n):
                rows.append({
                    "week": week, "actual": 0, "p6": p6, "p10": p10,
                    "match": (date, "J1", f"H{i}", f"A{i}"),
                })
        overall = MODULE.metric_means(rows)
        self.assertLess(overall["brier"], 0.0)
        self.assertGreater(MODULE.metric_means([r for r in rows if r["week"] == "2026-W02"])["brier"], 0.0)
        result = MODULE.audit(rows, {"matches": 3, "teamSideTargets": 6, "isoWeeks": 2}, "a" * 64, "b" * 64)
        self.assertEqual(result["leaveOneIsoWeekOut"]["signReversalCounts"]["brier"], 1)
        self.assertEqual(len(result["leaveOneIsoWeekOut"]["results"]), 2)

    def test_score_uses_multiclass_sum_and_log_floor(self):
        p = (1.0, 0.0, 0.0, 0.0)
        brier, logloss, accuracy = MODULE.score(p, 0)
        self.assertAlmostEqual(brier, 0.0)
        self.assertAlmostEqual(logloss, 0.0)
        self.assertEqual(accuracy, 1.0)
        p2 = (0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(MODULE.score(p2, 0)[1], -__import__("math").log(1e-12))

    def test_fixed_cohort_rejects_incomplete_data(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "small.csv"
            path.write_text(
                "date,comp,fold,home,away,hg,ag,side,actual,k6_p0,k6_p1,k6_p2,k6_p3,k10_p0,k10_p1,k10_p2,k10_p3\n"
                "2026-09-19,J1,1,A,B,1,0,home,1,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25\n"
                "2026-09-19,J1,1,A,B,1,0,away,0,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fixed cohort mismatch"):
                MODULE.load_rows(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
