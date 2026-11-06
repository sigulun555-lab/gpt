#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "verification" / "goal3_independent" / "crossed_date_team_bootstrap_sensitivity.py"
spec = importlib.util.spec_from_file_location("crossed_date_team_bootstrap_sensitivity", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def synthetic_report():
    from datetime import date, timedelta
    teams = [f"T{i}" for i in range(8)]
    matches = []
    first = date(2026, 1, 3)
    for day_i in range(17):
        date_text = (first + timedelta(days=3 * day_i)).isoformat()
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                hg = (i + day_i) % 4
                ag = (j + 2 * day_i) % 4
                by_k = {}
                for k in module.K_VALUES:
                    shift = k / 2000.0
                    home = [0.31 - shift, 0.29, 0.22, 0.18 + shift]
                    away = [0.28 - shift, 0.30, 0.24, 0.18 + shift]
                    by_k[f"{k:.1f}"] = {"home": home, "away": away}
                matches.append({"date": date_text, "comp": "J1", "fold": day_i // 6 + 1,
                                "home": teams[i], "away": teams[j],
                                "hg": hg, "ag": ag, "byK": by_k})
    return {"allKPredictions": matches}

class CrossedDateTeamBootstrapTests(unittest.TestCase):
    def test_expands_both_team_sides_and_checks_simplex(self):
        arrays = module.build_arrays(synthetic_report(), strict=False)
        self.assertEqual(len(arrays["y"]), 952)
        self.assertEqual(len(arrays["dates"]), 17)
        self.assertEqual(len(arrays["teams"]), 8)

    def test_rejects_invalid_probability_simplex(self):
        report = synthetic_report()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            module.build_arrays(report, strict=False)

    def test_rejects_duplicate_match(self):
        report = synthetic_report()
        report["allKPredictions"].append(report["allKPredictions"][0])
        with self.assertRaises(ValueError):
            module.build_arrays(report, strict=False)

    def test_both_crossed_bootstraps_are_reproducible(self):
        report = synthetic_report()
        first = module.audit(report, replicates=1000, seed=77, strict=False)
        second = module.audit(report, replicates=1000, seed=77, strict=False)
        self.assertEqual(first, second)
        self.assertEqual(first["cohort"]["teamSideTargets"], 952)
        self.assertIn("date_x_target_team", first["crossedBootstrap"])
        self.assertIn("date_x_home_away_team_product", first["crossedBootstrap"])
        self.assertGreater(first["crossedBootstrap"]["date_x_target_team"]["minimumWeightedTargetMass"], 0)
        self.assertEqual(first["dateTeamCalibrationAndPoisson"]["calibration"]["familySize"], 16)
        self.assertEqual(first["calendarMovingBlockBootstrap"]["7_day"]["blockLengthCalendarDays"], 7)
        self.assertEqual(first["calendarMovingBlockBootstrap"]["14_day"]["blockLengthCalendarDays"], 14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
