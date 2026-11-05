#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "score_prospective_shadow.py"
spec = importlib.util.spec_from_file_location("score_prospective_shadow", SCRIPT)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved_code_hashes():
    return {
        "runnerSha256": sha(ROOT / "verification" / "goal3_independent" / "prospective_shadow.py"),
        "modelSha256": sha(ROOT / "verification" / "goal3_independent" / "nested_validation.py"),
        "snapshotPipelineSha256": sha(ROOT / "scripts" / "auto_pipeline.js"),
    }


def make_forecast(round_no, legacy=False, failed_k6=False, fixture_date="2026-11-07"):
    now = "2026-11-05T04:00:00Z"
    snapshot = "2026-11-05T03:59:00Z"
    fixtures = []
    diagnostics = []
    for number, (home, away) in enumerate(
        [("Home A", "Away A"), ("Home B", "Away B"), ("Home C", "Away C")], 1
    ):
        arms = {}
        for k in ("6", "10"):
            success = not (failed_k6 and number == 1 and k == "6")
            p_home = [0.30, 0.35, 0.20, 0.15] if k == "6" else [0.25, 0.35, 0.22, 0.18]
            p_away = [0.28, 0.34, 0.22, 0.16] if k == "6" else [0.27, 0.35, 0.22, 0.16]
            arms[k] = {
                "homeGoalProbabilities": p_home,
                "awayGoalProbabilities": p_away,
                "fit": {"success": success},
            }
            diagnostics.append({
                "fixtureNumber": number,
                "k": int(k),
                "trainingRows": 429,
                "trainingCutoffExclusive": fixture_date,
                "fitSuccess": success,
            })
        fixtures.append({
            "number": number,
            "date": fixture_date,
            "home": home,
            "away": away,
            "arms": arms,
        })
    doc = {
        "schemaVersion": 1,
        "experiment": scorer.EXPERIMENT,
        "round": round_no,
        "forecastAsOf": now,
        "recordedAt": now,
        "snapshotAvailableBy": snapshot,
        "pipelineGeneratedAt": "2026-11-04T00:00:00Z",
        "pipelineDeclaredLastFetchedAt": "2026-11-04T00:01:00Z",
        "deadline": "2026-11-07T03:50:00Z",
        "status": "forecast_created_before_deadline",
        "productionImpact": "none",
        "model": {
            "arms": [6, 10],
            "trainingWindow": "2026_only",
            "otherAdjustments": "none; no market blend, Dixon-Coles or draw boost",
        },
        "inputs": {
            "historySha256": "a" * 64,
            "officialPredictionsSha256": "b" * 64,
            "aliasesSha256": "c" * 64,
            "historyEligibleRows": 429,
            "historyMaxDeclaredFetchedAt": "2026-11-04T00:01:00Z",
            "snapshotAvailableBy": snapshot,
        },
        "code": approved_code_hashes(),
        "fitDiagnostics": diagnostics,
        "fixtures": fixtures,
        "evaluation": {"protocolVersion": "1.1"},
    }
    if legacy:
        doc.pop("snapshotAvailableBy")
        doc["inputs"].pop("snapshotAvailableBy")
        doc["inputs"].pop("historyMaxDeclaredFetchedAt")
        doc["evaluation"].pop("protocolVersion")
    return doc


def make_results(round_no, incomplete=False, fixture_date="2026-11-07"):
    results = []
    for number, (home, away, league) in enumerate(
        [("Home A", "Away A", "J1"), ("Home B", "Away B", "J2"), ("Home C", "Away C", "J3")], 1
    ):
        item = {
            "number": number,
            "date": fixture_date,
            "home": home,
            "away": away,
            "competition": league,
            "status": "postponed" if incomplete and number == 3 else "final",
            "fullTimeHomeGoals": 1,
            "fullTimeAwayGoals": 0,
            "sourceType": "official",
            "sourceUrl": f"https://www.jleague.jp/match/{round_no}/{number}/",
            "retrievedAt": "2026-11-08T12:00:00Z",
            "publishedAt": "2026-11-08T11:55:00Z",
        }
        if incomplete and number == 3:
            item.pop("fullTimeHomeGoals")
            item.pop("fullTimeAwayGoals")
        results.append(item)
    return {
        "round": round_no,
        "resultBasis": "full_time_90_plus_stoppage",
        "fixtures": results,
    }


class ProspectiveShadowScoringTests(unittest.TestCase):
    def build_manifest(self, root, specs):
        entries = []
        for round_no, forecast_doc, result_doc in specs:
            forecast_path = root / f"forecast-{round_no}.json"
            result_path = root / f"results-{round_no}.json"
            forecast_path.write_text(json.dumps(forecast_doc), encoding="utf-8")
            result_path.write_text(json.dumps(result_doc), encoding="utf-8")
            entries.append({
                "round": round_no,
                "forecastPath": forecast_path.name,
                "forecastSha256": sha(forecast_path),
                "resultsPath": result_path.name,
                "resultsSha256": sha(result_path),
            })
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schemaVersion": 1,
            "protocolVersion": "1.2",
            "approvedCodeHashes": approved_code_hashes(),
            "rounds": entries,
        }), encoding="utf-8")
        return manifest_path

    def test_round_cluster_scoring_and_whole_round_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build_manifest(root, [
                (1, make_forecast(1), make_results(1)),
                (2, make_forecast(2, fixture_date="2026-11-08"),
                 make_results(2, fixture_date="2026-11-08")),
                (3, make_forecast(3), make_results(3, incomplete=True)),
                (4, make_forecast(4, legacy=True, fixture_date="2026-11-08"),
                 make_results(4, fixture_date="2026-11-08")),
                (5, make_forecast(5), make_results(5)),
            ])
            report = scorer.score_manifest(manifest, replicates=1000, seed=77)
            self.assertEqual(report["cohort"]["manifestRounds"], 5)
            self.assertEqual(report["cohort"]["includedRounds"], 3)
            self.assertEqual(report["cohort"]["includedMatches"], 9)
            self.assertEqual(report["cohort"]["includedTeamSideTargets"], 18)
            self.assertEqual(report["cohort"]["excludedRounds"], 2)
            self.assertIn("fixture_3_not_final", report["roundAudit"][2]["exclusionReasons"])
            self.assertIn("missing_or_invalid_snapshotAvailableBy", report["roundAudit"][3]["exclusionReasons"])
            equal_time = make_forecast(5)
            equal_time["forecastAsOf"] = equal_time["snapshotAvailableBy"]
            self.assertIn("snapshot_not_before_forecast", scorer.forecast_errors(equal_time, 5))
            self.assertEqual(set(report["overall"]), {"brier", "logloss"})
            self.assertTrue(report["bootstrap"]["sameRoundDrawsForBothMetrics"])
            self.assertEqual(report["sensitivity"]["dateCluster"]["status"], "computed")
            self.assertEqual(report["sensitivity"]["dateCluster"]["dates"], 2)
            self.assertFalse(report["decision"]["adoptionReviewEvidenceFloorMet"])
            self.assertFalse(report["decision"]["automaticProductionAdoption"])
            self.assertEqual(report["decision"]["status"], "RESEARCH_ONLY_NO_ADOPTION")

    def test_fixture_numbers_and_fit_cutoffs_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_number = make_forecast(1)
            bad_number["fixtures"][2]["number"] = 4
            manifest = self.build_manifest(root, [
                (1, bad_number, make_results(1)),
            ])
            report = scorer.score_manifest(manifest, replicates=1000, seed=11)
            reasons = report["roundAudit"][0]["exclusionReasons"]
            self.assertIn("fixture_numbers_not_exactly_1_2_3", reasons)
            self.assertFalse(report["roundAudit"][0]["included"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_cutoff = make_forecast(2)
            bad_cutoff["fitDiagnostics"][0]["trainingCutoffExclusive"] = "2026-11-08"
            manifest = self.build_manifest(root, [
                (2, bad_cutoff, make_results(2)),
            ])
            report = scorer.score_manifest(manifest, replicates=1000, seed=11)
            reasons = report["roundAudit"][0]["exclusionReasons"]
            self.assertIn("training_cutoff_mismatch:(1, 6)", reasons)
            self.assertFalse(report["roundAudit"][0]["included"])

    def test_recorded_at_must_follow_forecast_and_precede_deadline(self):
        forecast = make_forecast(1)
        forecast["recordedAt"] = "2026-11-05T03:58:00Z"
        self.assertIn("recorded_before_forecast", scorer.forecast_errors(forecast, 1))
        forecast["recordedAt"] = "2026-11-07T03:50:00Z"
        self.assertIn("recorded_not_before_deadline", scorer.forecast_errors(forecast, 1))

    def test_fit_failure_is_counted_and_blocks_adoption_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build_manifest(root, [
                (1, make_forecast(1, failed_k6=True), make_results(1)),
                (2, make_forecast(2), make_results(2)),
            ])
            report = scorer.score_manifest(manifest, replicates=1000, seed=11)
            self.assertEqual(report["fitFailures"]["6"], 1)
            self.assertEqual(report["fitFailures"]["10"], 0)
            self.assertEqual(report["missingForecastTargets"]["6"], 2)
            self.assertFalse(report["decision"]["fitFailureAndMissingTargetGuardPass"])

    def test_twenty_round_synthetic_case_only_qualifies_for_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = [
                (round_no,
                 make_forecast(round_no, fixture_date=("2026-11-07" if round_no % 2 else "2026-11-08")),
                 make_results(round_no, fixture_date=("2026-11-07" if round_no % 2 else "2026-11-08")))
                for round_no in range(1, 21)
            ]
            manifest = self.build_manifest(root, specs)
            report = scorer.score_manifest(manifest, replicates=1000, seed=20261105)
            self.assertEqual(report["cohort"]["includedRounds"], 20)
            self.assertEqual(report["cohort"]["includedMatches"], 60)
            self.assertTrue(report["decision"]["adoptionReviewEvidenceFloorMet"])
            self.assertTrue(report["decision"]["overallBothUpper95BelowZero"])
            self.assertTrue(report["decision"]["allLeagueNonInferiorityGuardsPass"])
            self.assertTrue(report["decision"]["eligibleForHumanAdoptionReview"])
            self.assertFalse(report["decision"]["automaticProductionAdoption"])
            self.assertEqual(report["decision"]["status"], "REVIEW_CRITERIA_MET_REQUIRES_HUMAN_DECISION")

    def test_each_league_requires_twenty_distinct_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = []
            for round_no in range(1, 21):
                fixture_date = "2026-11-07" if round_no % 2 else "2026-11-08"
                forecast = make_forecast(round_no, fixture_date=fixture_date)
                results = make_results(round_no, fixture_date=fixture_date)
                if round_no == 1:
                    # Keep the round complete, but leave J3 absent from one round.
                    results["fixtures"][2]["competition"] = "J2"
                specs.append((round_no, forecast, results))
            manifest = self.build_manifest(root, specs)
            report = scorer.score_manifest(manifest, replicates=1000, seed=20261105)
            self.assertEqual(report["cohort"]["includedRounds"], 20)
            self.assertTrue(report["decision"]["adoptionReviewEvidenceFloorMet"])
            self.assertEqual(
                report["leagueNonInferiority"]["J3"]["brier"]["roundsWithLeagueTargets"], 19
            )
            self.assertFalse(report["leagueNonInferiority"]["J3"]["brier"]["passes"])
            self.assertFalse(report["leagueNonInferiority"]["J3"]["logloss"]["passes"])
            self.assertFalse(report["decision"]["allLeagueNonInferiorityGuardsPass"])
            self.assertFalse(report["decision"]["eligibleForHumanAdoptionReview"])


    def test_result_url_must_use_exact_allowlisted_official_host(self):
        bad_urls = [
            "https://jleague.jp/match/1/1/",
            "https://www.jleague.jp.attacker.example/match/1/1/",
            "https://user@www.jleague.jp/match/1/1/",
            "https://www.jleague.jp:444/match/1/1/",
            "http://www.jleague.jp/match/1/1/",
        ]
        for url in bad_urls:
            with self.subTest(url=url):
                results = make_results(1)
                results["fixtures"][0]["sourceUrl"] = url
                errors = scorer.result_errors(results, 1, make_forecast(1))
                self.assertIn("fixture_1_source_url_not_allowlisted_official_host", errors)

    def test_manifest_pins_code_hashes_and_rejects_forecast_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forecast = make_forecast(1)
            forecast["code"]["runnerSha256"] = "9" * 64
            manifest = self.build_manifest(root, [(1, forecast, make_results(1))])
            report = scorer.score_manifest(manifest, replicates=1000, seed=7)
            reasons = report["roundAudit"][0]["exclusionReasons"]
            self.assertIn("approved_code_hash_mismatch:runnerSha256", reasons)
            self.assertFalse(report["roundAudit"][0]["included"])


    def test_manifest_code_lock_must_match_checked_out_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build_manifest(root, [
                (1, make_forecast(1), make_results(1)),
            ])
            doc = json.loads(manifest.read_text())
            doc["approvedCodeHashes"]["modelSha256"] = "8" * 64
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "do not match checked-out research code"):
                scorer.score_manifest(manifest, replicates=1000, seed=7)

    def test_manifest_requires_all_approved_code_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build_manifest(root, [(1, make_forecast(1), make_results(1))])
            doc = json.loads(manifest.read_text())
            doc["approvedCodeHashes"].pop("modelSha256")
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "approvedCodeHashes"):
                scorer.score_manifest(manifest, replicates=1000, seed=7)

    def test_manifest_hash_mismatch_and_path_traversal_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build_manifest(root, [(1, make_forecast(1), make_results(1))])
            doc = json.loads(manifest.read_text())
            doc["rounds"][0]["forecastSha256"] = "0" * 64
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "forecast SHA-256 mismatch"):
                scorer.score_manifest(manifest, replicates=1000, seed=1)
            doc["rounds"][0]["forecastPath"] = "../outside.json"
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "escapes"):
                scorer.score_manifest(manifest, replicates=1000, seed=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
