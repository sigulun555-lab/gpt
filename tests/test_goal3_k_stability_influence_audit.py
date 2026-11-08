#!/usr/bin/env python3
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification" / "goal3_independent"))
from k_stability_influence_audit import audit, build_rows


def synthetic():
    matches = []
    leagues = ("J1", "J2", "J3")
    for i in range(236):
        comp = leagues[i % 3]
        home = f"{comp}-team-{i % 20:02d}"
        away_idx = (i * 7 + 3) % 20
        if away_idx == i % 20:
            away_idx = (away_idx + 1) % 20
        away = f"{comp}-team-{away_idx:02d}"
        matches.append({
            "date": (date(2026, 9, 19) + timedelta(days=i % 17)).isoformat(),
            "comp": comp, "home": home, "away": away,
            "hg": i % 5, "ag": (i + 2) % 5,
            "byK": {
                "6.0": {"home": [0.30, 0.30, 0.20, 0.20], "away": [0.20, 0.30, 0.30, 0.20]},
                "10.0": {"home": [0.25, 0.30, 0.25, 0.20], "away": [0.25, 0.30, 0.25, 0.20]},
            },
        })
    return {"allKPredictions": matches}


class KStabilityInfluenceAuditTests(unittest.TestCase):
    def test_fixed_cohort_expands_to_472_targets(self):
        rows = build_rows(synthetic())
        self.assertEqual(len(rows), 472)

    def test_reports_date_team_and_match_clusters(self):
        result = audit(synthetic())
        self.assertEqual(result["leaveOneDateOut"]["clusterCount"], 17)
        self.assertEqual(result["leaveOneTeamOut"]["clusterCount"], 60)
        self.assertEqual(result["leaveOneMatchOut"]["clusterCount"], 236)

    def test_match_exclusion_keeps_pairing(self):
        result = audit(synthetic())
        records = result["leaveOneMatchOut"]["records"]
        self.assertTrue(all(r["removedTargets"] == 2 for r in records))
        self.assertTrue(all(r["remainingTargets"] == 470 for r in records))

    def test_rejects_invalid_probability_simplex(self):
        report = synthetic()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaisesRegex(ValueError, "simplex"):
            build_rows(report)

    def test_rejects_duplicate_match(self):
        report = synthetic()
        report["allKPredictions"][1] = dict(report["allKPredictions"][0])
        with self.assertRaisesRegex(ValueError, "duplicate match"):
            build_rows(report)

    def test_rejects_boolean_goal_count(self):
        report = synthetic()
        report["allKPredictions"][0]["hg"] = True
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            build_rows(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
