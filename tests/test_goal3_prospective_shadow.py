import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "verification" / "goal3_independent" / "prospective_shadow.py"


class ProspectiveShadowTests(unittest.TestCase):
    def test_paired_k_arms_use_same_fixture_and_emit_valid_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            as_of = now.isoformat()
            deadline = (now + timedelta(days=2)).isoformat()
            target_date = (now + timedelta(days=1)).date().isoformat()
            history = []
            teams = ["A", "B", "C", "D"]
            scores = [(1, 0), (2, 1), (0, 0), (1, 2), (3, 1), (0, 1), (2, 2), (1, 1)]
            for i, (home_goals, away_goals) in enumerate(scores):
                history.append({
                    "date": f"2026-08-{i + 1:02d}",
                    "competition": "J1",
                    "home": teams[i % 4],
                    "away": teams[(i + 1) % 4],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "fetchedAt": as_of,
                })
            predictions = {
                "generatedAt": as_of,
                "status": "predictions_generated",
                "totoTargetsConfirmed": True,
                "round": 9999,
                "deadline": deadline,
                "totoSource": {"lastFetchedAt": as_of},
                "predictions": [
                    {"number": 1, "marketType": "totoGOAL3", "totoRound": 9999,
                     "totoTargetConfirmed": True, "date": target_date, "home": "A", "away": "B"},
                    {"number": 2, "marketType": "totoGOAL3", "totoRound": 9999,
                     "totoTargetConfirmed": True, "date": target_date, "home": "B", "away": "C"},
                    {"number": 3, "marketType": "totoGOAL3", "totoRound": 9999,
                     "totoTargetConfirmed": True, "date": target_date, "home": "C", "away": "D"},
                ],
            }
            aliases = {"aliases": {team: team for team in teams}}
            files = {
                "history.json": {"history": history},
                "predictions.json": predictions,
                "aliases.json": aliases,
            }
            for name, data in files.items():
                (tmp / name).write_text(json.dumps(data), encoding="utf-8")
            out = tmp / "shadow.json"
            command = [
                sys.executable, str(RUNNER),
                "--history", str(tmp / "history.json"),
                "--predictions", str(tmp / "predictions.json"),
                "--aliases", str(tmp / "aliases.json"),
                "--out", str(out),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            record = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(record["round"], 9999)
            self.assertEqual(len(record["fixtures"]), 3)
            self.assertEqual(len(record["fitDiagnostics"]), 6)
            for fixture in record["fixtures"]:
                self.assertEqual(set(fixture["arms"]), {"6", "10"})
                for arm in fixture["arms"].values():
                    for key in ("homeGoalProbabilities", "awayGoalProbabilities"):
                        self.assertAlmostEqual(sum(arm[key]), 1.0, places=9)
            # Exclusive output creation guarantees prior forecast records cannot be overwritten.
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)

            # A mixed-year source must fail closed instead of silently using
            # non-2026 rows in the nominally 2026-only experiment.
            mixed_history = list(history) + [{**history[0], "date": "2025-12-31"}]
            mixed_path = tmp / "mixed_history.json"
            mixed_path.write_text(json.dumps({"history": mixed_history}), encoding="utf-8")
            mixed_out = tmp / "mixed_shadow.json"
            mixed_command = [
                sys.executable, str(RUNNER),
                "--history", str(mixed_path),
                "--predictions", str(tmp / "predictions.json"),
                "--aliases", str(tmp / "aliases.json"),
                "--out", str(mixed_out),
            ]
            mixed_rejected = subprocess.run(mixed_command, capture_output=True, text=True)
            self.assertNotEqual(mixed_rejected.returncode, 0)
            self.assertIn("2026-only history guard", mixed_rejected.stderr)


if __name__ == "__main__":
    unittest.main()
