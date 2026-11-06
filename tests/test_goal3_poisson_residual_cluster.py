#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification" / "goal3_independent"))
from poisson_residual_cluster_audit import audit, flatten_targets


def synthetic_report():
    matches = []
    for i in range(236):
        by_k = {}
        for k in (6, 10):
            by_k[f"{k}.0"] = {
                "home": [0.25, 0.35, 0.22, 0.18],
                "away": [0.30, 0.34, 0.21, 0.15],
            }
        matches.append({
            "date": f"2026-10-{1 + (i % 17):02d}",
            "comp": ("J1", "J2", "J3")[i % 3],
            "home": f"Home-{i}",
            "away": f"Away-{i}",
            "hg": i % 4,
            "ag": (i + 1) % 4,
            "byK": by_k,
        })
    return {"allKPredictions": matches}


class PoissonResidualClusterAuditTests(unittest.TestCase):
    def test_flatten_preserves_both_sides(self):
        rows = flatten_targets(synthetic_report())
        self.assertEqual(len(rows), 472)
        self.assertEqual({row["side"] for row in rows}, {"home", "away"})

    def test_audit_is_reproducible_and_returns_eight_bands(self):
        report = synthetic_report()
        first = audit(report, replicates=1000, seed=12345)
        second = audit(report, replicates=1000, seed=12345)
        self.assertEqual(first["cohort"], {"matches": 236, "teamSideTargets": 472, "dates": 17})
        self.assertEqual(first["results"], second["results"])
        self.assertEqual(len(first["results"]), 8)
        self.assertEqual(len(first["pairedK6MinusK10"]), 4)

    def test_rejects_invalid_probability_vector(self):
        report = synthetic_report()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            flatten_targets(report)


    def test_rejects_duplicate_match_key(self):
        report = synthetic_report()
        duplicate = dict(report["allKPredictions"][0])
        report["allKPredictions"][1]["date"] = duplicate["date"]
        report["allKPredictions"][1]["comp"] = duplicate["comp"]
        report["allKPredictions"][1]["home"] = duplicate["home"]
        report["allKPredictions"][1]["away"] = duplicate["away"]
        with self.assertRaisesRegex(ValueError, "duplicate match key"):
            flatten_targets(report)

    def test_rejects_invalid_competition_and_missing_team(self):
        report = synthetic_report()
        report["allKPredictions"][0]["comp"] = "J4"
        with self.assertRaisesRegex(ValueError, "invalid competition"):
            flatten_targets(report)
        report = synthetic_report()
        report["allKPredictions"][0]["home"] = ""
        with self.assertRaisesRegex(ValueError, "missing home/away team"):
            flatten_targets(report)

    def test_rejects_boolean_goal_count(self):
        report = synthetic_report()
        report["allKPredictions"][0]["hg"] = True
        with self.assertRaisesRegex(ValueError, "invalid observed goals"):
            flatten_targets(report)

if __name__ == "__main__":
    unittest.main(verbosity=2)
