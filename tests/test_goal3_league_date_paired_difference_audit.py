import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "league_date_paired_difference_audit.py"
SPEC = importlib.util.spec_from_file_location("league_date_paired_difference_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def match(date, league, home, away, home_goals, away_goals, k6_home, k10_home):
    return {"date": date, "comp": league, "home": home, "away": away,
            "hg": home_goals, "ag": away_goals,
            "byK": {"6.0": {"home": k6_home, "away": [0.25]*4},
                    "10.0": {"home": k10_home, "away": [0.25]*4}}}


class LeagueDatePairedDifferenceTests(unittest.TestCase):
    def test_four_category_scoring_and_3plus_bucket(self):
        brier, logloss, _ = audit.score([0.1, 0.2, 0.3, 0.4], 5)
        self.assertAlmostEqual(brier, 0.5)
        self.assertAlmostEqual(logloss, -math.log(0.4))

    def test_probability_floor_is_applied(self):
        _, logloss, _ = audit.score([0.0, 1.0, 0.0, 0.0], 0)
        self.assertAlmostEqual(logloss, -math.log(1e-12))

    def test_paired_counts_and_date_aggregation(self):
        doc = {"allKPredictions": [
            match("2026-01-01", "J1", "H1", "A1", 0, 1,
                  [0.1,0.3,0.3,0.3], [0.9,0.03,0.03,0.04]),
            match("2026-01-02", "J1", "H2", "A2", 3, 0,
                  [0.1,0.2,0.3,0.4], [0.2,0.2,0.2,0.4]),
        ]}
        report = audit.build_report(doc, "abc")
        self.assertEqual(report["totals"], {"matches": 2, "teamSideTargets": 4, "dates": 2})
        self.assertEqual(report["byLeague"]["J1"]["evaluationDates"], 2)
        self.assertEqual(report["method"]["comparison"], "k=6 minus k=10; positive means k=6 is worse")

    def test_warning_triggers_are_strictly_descriptive(self):
        doc = {"allKPredictions": [match("2026-01-01", "J3", "H", "A", 0, 0,
              [0.1,0.3,0.3,0.3], [0.9,0.03,0.03,0.04])]}
        report = audit.build_report(doc, "abc")
        dl = report["byLeague"]["J3"]["dateLevel"]
        self.assertEqual(dl["brierWarningExceedCount"], 1)
        self.assertEqual(dl["logLossWarningExceedCount"], 1)
        self.assertIn("not adoption criteria", report["method"]["warningThresholds"]["status"])

    def test_invalid_probability_sum_rejected(self):
        doc = {"allKPredictions": [match("d", "J1", "H", "A", 0, 0,
              [0.5,0.5,0.1,0.0], [0.25]*4)]}
        with self.assertRaises(ValueError):
            audit.build_report(doc, "abc")

    def test_duplicate_match_rejected(self):
        row = match("d", "J1", "H", "A", 0, 0, [0.25]*4, [0.25]*4)
        with self.assertRaisesRegex(ValueError, "duplicate match"):
            audit.build_report({"allKPredictions": [row, row]}, "abc")


if __name__ == "__main__":
    unittest.main()
