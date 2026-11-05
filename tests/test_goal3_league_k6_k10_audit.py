#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification" / "goal3_independent"))
from league_k6_k10_audit import audit, build_targets


def synthetic():
    matches = []
    for i in range(236):
        by_k = {f"{k}.0": {"home": [0.25, 0.35, 0.22, 0.18],
                           "away": [0.30, 0.34, 0.21, 0.15]} for k in (6, 10)}
        matches.append({
            "date": f"2026-10-{1 + (i % 17):02d}",
            "comp": ("J1", "J2", "J3")[i % 3],
            "home": f"home-{i}", "away": f"away-{i}",
            "hg": i % 4, "ag": (i + 1) % 4, "byK": by_k,
        })
    return {"allKPredictions": matches}


class LeagueK6K10AuditTests(unittest.TestCase):
    def test_target_expansion_preserves_both_sides(self):
        rows = build_targets(synthetic())
        self.assertEqual(len(rows), 472)
        self.assertEqual({r["side"] for r in rows}, {"home", "away"})

    def test_audit_is_reproducible_and_has_all_scopes(self):
        report = synthetic()
        first = audit(report, replicates=1000, seed=77)
        second = audit(report, replicates=1000, seed=77)
        self.assertEqual(first, second)
        self.assertEqual(first["cohort"]["teamSideTargets"], 472)
        self.assertEqual(set(first["byScope"]), {"ALL", "J1", "J2", "J3"})
        self.assertEqual(len(first["calibration"]["J1"]["6.0"]), 4)

    def test_rejects_invalid_probability_sum(self):
        report = synthetic()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            build_targets(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
