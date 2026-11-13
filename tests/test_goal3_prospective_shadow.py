import json
import os
import hashlib
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.goal3_independent.optimizer_gradient import (
    objective_gradient,
    gradient_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "goal3_independent"))
from prospective_shadow import extract_goal3_deadline

RUNNER = ROOT / "verification" / "goal3_independent" / "prospective_shadow.py"


def sales_terms_fixture(round_no, deadline_iso):
    deadline_dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
    deadline_jst = deadline_dt.astimezone(timezone(timedelta(hours=9)))
    date_text = f"{deadline_jst.year}年{deadline_jst.month}月{deadline_jst.day}日"
    goal3_time = f"{deadline_jst.hour}:{deadline_jst.minute:02d}"
    return f"""<html><body>
      <section><h2>第{round_no}回 totoくじ情報</h2>
        <p>販売終了日 {date_text}（当サイト(ネット決済) 12:50）</p></section>
      <section><h2>第{round_no}回 mini toto-A組 くじ情報</h2>
        <p>販売終了日 {date_text}（当サイト(ネット決済) 12:50）</p></section>
      <section><h2>第{round_no}回 mini toto-B組 くじ情報</h2>
        <p>販売終了日 {date_text}（当サイト(ネット決済) 12:50）</p></section>
      <section><h2>第{round_no}回 totoGOAL3 くじ情報</h2>
        <p>販売終了日 {date_text}（当サイト(ネット決済) {goal3_time}）</p></section>
    </body></html>"""


class ProspectiveShadowTests(unittest.TestCase):
    def test_product_specific_deadline_uses_goal3_block_not_first_product(self):
        html = sales_terms_fixture(1670, "2026-11-07T13:50:00+09:00")
        self.assertEqual(
            extract_goal3_deadline(html, 1670),
            "2026-11-07T13:50:00+09:00",
        )

    def test_missing_goal3_sales_block_fails_closed(self):
        html = "<html><body><h2>第1670回 totoくじ情報</h2>販売終了日 2026年11月7日 当サイト(ネット決済) 12:50</body></html>"
        with self.assertRaisesRegex(ValueError, "Expected exactly one official totoGOAL3 sales block"):
            extract_goal3_deadline(html, 1670)

    def test_v15_optimizer_gradient_matches_central_finite_difference(self):
        matches = [
            {"home": "A", "away": "B", "hg": 1, "ag": 0},
            {"home": "B", "away": "C", "hg": 2, "ag": 1},
            {"home": "C", "away": "A", "hg": 0, "ag": 2},
            {"home": "A", "away": "C", "hg": 3, "ag": 1},
            {"home": "B", "away": "A", "hg": 1, "ag": 1},
        ]
        teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
        team_index = {team: i for i, team in enumerate(teams)}
        n = len(teams)
        theta = np.asarray([
            np.log(1.3), 0.12,
            0.10, -0.08, 0.02,
            -0.04, 0.06, -0.03,
        ], dtype=float)
        homes = np.asarray([team_index[m["home"]] for m in matches], dtype=np.int64)
        aways = np.asarray([team_index[m["away"]] for m in matches], dtype=np.int64)
        home_goals = np.asarray([m["hg"] for m in matches], dtype=float)
        away_goals = np.asarray([m["ag"] for m in matches], dtype=float)

        for lam_pen in (6.0, 10.0):
            analytic = objective_gradient(matches, theta, team_index, lam_pen)

            def independent_objective(x):
                mu, home_adv = x[0], x[1]
                attack = x[2:2 + n]
                defense = x[2 + n:2 + 2 * n]
                eta_h_raw = mu + home_adv + attack[homes] - defense[aways]
                eta_a_raw = mu + attack[aways] - defense[homes]
                eta_h = np.clip(eta_h_raw, -20.0, 20.0)
                eta_a = np.clip(eta_a_raw, -20.0, 20.0)
                return float(
                    np.sum(np.exp(eta_h) - home_goals * eta_h)
                    + np.sum(np.exp(eta_a) - away_goals * eta_a)
                    + lam_pen * (np.sum(attack ** 2) + np.sum(defense ** 2))
                )

            eps = 1e-6
            numeric = np.empty_like(theta)
            for j in range(theta.size):
                plus = theta.copy()
                minus = theta.copy()
                plus[j] += eps
                minus[j] -= eps
                numeric[j] = (
                    independent_objective(plus) - independent_objective(minus)
                ) / (2.0 * eps)

            self.assertEqual(analytic.shape, theta.shape)
            self.assertTrue(np.all(np.isfinite(analytic)))
            np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=2e-7)

            diagnostics = gradient_diagnostics(matches, theta, team_index, lam_pen)
            self.assertAlmostEqual(
                diagnostics["maxAbsGradient"],
                float(np.max(np.abs(analytic))),
                places=14,
            )
            self.assertEqual(diagnostics["gradientDimension"], theta.size)

    def test_paired_k_arms_use_same_fixture_and_emit_valid_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            as_of = now.isoformat()
            deadline = (now + timedelta(days=2)).replace(second=0, microsecond=0).isoformat()
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
            sales_terms_path = tmp / "official_sales_terms.html"
            sales_terms_path.write_text(sales_terms_fixture(9999, deadline), encoding="utf-8")
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
                "--sales-terms-html", str(sales_terms_path),
                "--out", str(out),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            record = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(record["round"], 9999)
            evaluation = record["evaluation"]
            self.assertEqual(evaluation["protocolVersion"], "1.6")
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
                self.assertTrue(diagnostic["optimizerMessage"].strip())
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
            self.assertEqual(record["officialDeadlineVerification"]["status"], "verified")
            self.assertEqual(record["officialDeadlineVerification"]["marketType"], "totoGOAL3")
            self.assertEqual(record["officialDeadlineVerification"]["round"], 9999)
            self.assertEqual(record["officialDeadlineVerification"]["deadline"], deadline)
            self.assertTrue(record["inputs"]["officialSalesTermsSha256"])
            snapshot_files = record["inputSnapshots"]["files"]
            expected_snapshot_bytes = {
                "history": (tmp / "history.json").read_bytes(),
                "officialPredictions": (tmp / "predictions.json").read_bytes(),
                "aliases": (tmp / "aliases.json").read_bytes(),
                "officialSalesTerms": sales_terms_path.read_bytes(),
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
                "--sales-terms-html", str(sales_terms_path),
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
                "--sales-terms-html", str(sales_terms_path),
                "--out", str(future_out),
            ]
            future_rejected = subprocess.run(
                future_command, capture_output=True, text=True
            )
            self.assertNotEqual(future_rejected.returncode, 0)
            self.assertIn("Pipeline-declared fetchedAt is in the future",
                          future_rejected.stderr)


            # A mismatched generic/GOAL3 deadline must fail closed before any shadow is written.
            mismatch_deadline = (
                datetime.fromisoformat(deadline) + timedelta(hours=1)
            ).isoformat()
            mismatch_sales_path = tmp / "mismatched_official_sales_terms.html"
            mismatch_sales_path.write_text(
                sales_terms_fixture(9999, mismatch_deadline), encoding="utf-8"
            )
            mismatch_out = tmp / "mismatched_deadline_shadow.json"
            mismatch_command = list(command)
            mismatch_command[mismatch_command.index("--sales-terms-html") + 1] = str(mismatch_sales_path)
            mismatch_command[mismatch_command.index("--out") + 1] = str(mismatch_out)
            mismatch_result = subprocess.run(
                mismatch_command, capture_output=True, text=True, env=env
            )
            self.assertNotEqual(mismatch_result.returncode, 0)
            self.assertIn(
                "does not match independently parsed official totoGOAL3 sales deadline",
                mismatch_result.stderr,
            )
            self.assertFalse(mismatch_out.exists())

            # A snapshot read at/after the official deadline must be rejected.
            expired_predictions = dict(predictions)
            expired_deadline = (now - timedelta(minutes=1)).replace(second=0, microsecond=0).isoformat()
            expired_predictions["deadline"] = expired_deadline
            expired_predictions_path = tmp / "expired_predictions.json"
            expired_predictions_path.write_text(
                json.dumps(expired_predictions), encoding="utf-8"
            )
            expired_sales_terms_path = tmp / "expired_official_sales_terms.html"
            expired_sales_terms_path.write_text(
                sales_terms_fixture(9999, expired_deadline), encoding="utf-8"
            )
            expired_out = tmp / "expired_shadow.json"
            expired_command = [
                sys.executable, str(RUNNER),
                "--history", str(tmp / "history.json"),
                "--predictions", str(expired_predictions_path),
                "--aliases", str(tmp / "aliases.json"),
                "--sales-terms-html", str(expired_sales_terms_path),
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
