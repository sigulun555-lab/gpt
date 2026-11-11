import json
import os
import hashlib
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
            env = os.environ.copy()
            event_path = tmp / "event.json"
            event_path.write_text(json.dumps({
                "head_commit": {"message": "[goal3-shadow] test round 1671"}
            }), encoding="utf-8")
            env.update({
                "GITHUB_RUN_ID": "123456789",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_SHA": "a" * 40,
                "GITHUB_REF_NAME": "research/goal3-prospective-shadow-20261104-v2",
                "GITHUB_EVENT_PATH": str(event_path),
            })
            command = [
                sys.executable, str(RUNNER),
                "--history", str(tmp / "history.json"),
                "--predictions", str(tmp / "predictions.json"),
                "--aliases", str(tmp / "aliases.json"),
                "--out", str(out),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            record = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(record["round"], 9999)
            evaluation = record["evaluation"]
            self.assertEqual(evaluation["protocolVersion"], "1.5")
            self.assertEqual(evaluation["minimumEvidence"]["interimDescriptive"], {"rounds": 10, "matches": 30})
            self.assertEqual(evaluation["minimumEvidence"]["adoptionReview"], {"rounds": 20, "matches": 60})
            self.assertEqual(evaluation["bootstrap"]["replicates"], 10000)
            self.assertEqual(evaluation["bootstrap"]["rng"], "NumPy PCG64")
            self.assertEqual(evaluation["bootstrap"]["seed"], 20261105)
            self.assertEqual(evaluation["leagueNonInferiority"]["brierMargin"], 0.01)
            self.assertEqual(evaluation["leagueNonInferiority"]["loglossMargin"], 0.02)
            self.assertIn("1e-12", evaluation["scoring"]["logloss"])
            self.assertEqual(len(record["fixtures"]), 3)
            self.assertEqual(len(record["fitDiagnostics"]), 6)
            self.assertEqual(evaluation["optimizerDiagnostics"]["optimizerName"], "L-BFGS-B")
            self.assertEqual(evaluation["optimizerDiagnostics"]["optimizerOptions"], {
                "maxiter": 10000, "maxfun": 200000, "ftol": 0.0,
                "gtol": 1e-8, "maxls": 100, "maxcor": 50,
            })
            self.assertEqual(evaluation["optimizerDiagnostics"]["maxAbsGradientTolerance"], 1e-6)
            self.assertEqual(
                evaluation["optimizerDiagnostics"]["gradientSource"],
                "independent analytic gradient of the pinned objective; finite-difference tested",
            )
            for diagnostic in record["fitDiagnostics"]:
                self.assertIn("maxAbsGradient", diagnostic)
                self.assertIn("gradientL2", diagnostic)
                self.assertIn("gradientDimension", diagnostic)
                self.assertIn("optimizerStatus", diagnostic)
                self.assertEqual(diagnostic["optimizerName"], "L-BFGS-B")
                self.assertEqual(diagnostic["optimizerOptions"], evaluation["optimizerDiagnostics"]["optimizerOptions"])
                self.assertGreater(diagnostic["gradientDimension"], 0)
            self.assertRegex(record["code"]["optimizerGradientSha256"], r"^[0-9a-f]{64}$")
            snapshot_available_by = datetime.fromisoformat(
                record["snapshotAvailableBy"].replace("Z", "+00:00")
            )
            forecast_as_of = datetime.fromisoformat(
                record["forecastAsOf"].replace("Z", "+00:00")
            )
            parsed_deadline = datetime.fromisoformat(
                deadline.replace("Z", "+00:00")
            )
            self.assertGreaterEqual(snapshot_available_by, now)
            self.assertGreaterEqual(forecast_as_of, snapshot_available_by)
            self.assertLess(forecast_as_of, parsed_deadline)
            recorded_at = datetime.fromisoformat(record["recordedAt"].replace("Z", "+00:00"))
            self.assertGreater(recorded_at, forecast_as_of)
            self.assertLess(recorded_at, parsed_deadline)
            self.assertEqual(record["workflowRun"], {
                "id": 123456789,
                "attempt": 1,
                "event": "push",
                "headSha": "a" * 40,
                "headBranch": "research/goal3-prospective-shadow-20261104-v2",
                "headCommitMessage": "[goal3-shadow] test round 1671",
            })
            snapshot_dir = tmp / "shadow-inputs"
            self.assertTrue(snapshot_dir.is_dir())
            snapshot_files = record["inputSnapshots"]["files"]
            expected_snapshot_bytes = {
                "history": (tmp / "history.json").read_bytes(),
                "officialPredictions": (tmp / "predictions.json").read_bytes(),
                "aliases": (tmp / "aliases.json").read_bytes(),
            }
            for label, raw in expected_snapshot_bytes.items():
                entry = snapshot_files[label]
                copied = tmp / entry["path"]
                self.assertEqual(copied.read_bytes(), raw)
                self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(entry["sizeBytes"], len(raw))
            self.assertEqual(record["pipelineGeneratedAt"], as_of)
            self.assertEqual(record["pipelineDeclaredLastFetchedAt"], as_of)
            for fixture in record["fixtures"]:
                self.assertEqual(set(fixture["arms"]), {"6", "10"})
                for arm in fixture["arms"].values():
                    for key in ("homeGoalProbabilities", "awayGoalProbabilities"):
                        self.assertAlmostEqual(sum(arm[key]), 1.0, places=9)
            # Exclusive output creation guarantees prior forecast records cannot be overwritten.
            rejected = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(rejected.returncode, 0)

            # A pre-existing snapshot directory is immutable.
            blocked_out = tmp / "blocked.json"
            blocked_dir = tmp / "blocked-inputs"
            blocked_dir.mkdir()
            sentinel = blocked_dir / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            blocked_command = list(command)
            blocked_command[blocked_command.index("--out") + 1] = str(blocked_out)
            blocked = subprocess.run(blocked_command, capture_output=True, text=True, env=env)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("Append-only input snapshot guard", blocked.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse(blocked_out.exists())

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

            # A pipeline-declared fetch timestamp in the future must fail
            # closed relative to the research runner's post-fetch snapshot read.
            future_predictions = dict(predictions)
            future_predictions["totoSource"] = {
                "lastFetchedAt": (now + timedelta(hours=1)).isoformat()
            }
            future_predictions_path = tmp / "future_fetch_predictions.json"
            future_predictions_path.write_text(
                json.dumps(future_predictions), encoding="utf-8"
            )
            future_out = tmp / "future_fetch_shadow.json"
            future_command = [
                sys.executable, str(RUNNER),
                "--history", str(tmp / "history.json"),
                "--predictions", str(future_predictions_path),
                "--aliases", str(tmp / "aliases.json"),
                "--out", str(future_out),
            ]
            future_rejected = subprocess.run(
                future_command, capture_output=True, text=True
            )
            self.assertNotEqual(future_rejected.returncode, 0)
            self.assertIn("Pipeline-declared fetchedAt is in the future",
                          future_rejected.stderr)

            # A snapshot read at/after the official deadline must be rejected.
            expired_predictions = dict(predictions)
            expired_predictions["deadline"] = (
                now - timedelta(seconds=1)
            ).isoformat()
            expired_predictions_path = tmp / "expired_predictions.json"
            expired_predictions_path.write_text(
                json.dumps(expired_predictions), encoding="utf-8"
            )
            expired_out = tmp / "expired_shadow.json"
            expired_command = [
                sys.executable, str(RUNNER),
                "--history", str(tmp / "history.json"),
                "--predictions", str(expired_predictions_path),
                "--aliases", str(tmp / "aliases.json"),
                "--out", str(expired_out),
            ]
            expired_rejected = subprocess.run(
                expired_command, capture_output=True, text=True
            )
            self.assertNotEqual(expired_rejected.returncode, 0)
            self.assertIn("snapshot available at or after the official deadline",
                          expired_rejected.stderr)


if __name__ == "__main__":
    unittest.main()
