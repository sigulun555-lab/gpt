#!/usr/bin/env python3
"""Tests for GOAL3 all-k runner and historical team-alias provenance."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "goal3_independent"))

import reproduce_allk_outer as audit


class ExecutionEnvironmentTests(unittest.TestCase):
    def test_numerical_runtime_and_thread_settings_are_recorded(self):
        result = audit.execution_environment()
        self.assertTrue(result["pythonVersion"])
        self.assertTrue(result["numpyVersion"])
        self.assertTrue(result["scipyVersion"])
        self.assertTrue(result["platform"])
        self.assertIsInstance(result["threadEnvironment"], dict)
        self.assertIn("OPENBLAS_NUM_THREADS", result["threadEnvironment"])
        self.assertIn("numpyRuntimeConfig", result)


class TeamAliasProvenanceTests(unittest.TestCase):
    def test_alias_usage_and_pre_update_target_dates_are_explicit(self):
        rows = [
            {"date": "2026-01-01", "competition": "J1", "home": "Short", "away": "Other"},
            {"date": "2026-01-03", "competition": "J1", "home": "Canonical", "away": "Other"},
            {"date": "2026-01-04", "competition": "J2", "home": "Short", "away": "Canonical"},
            {"date": "2026-01-02", "competition": "J4", "home": "Ignored", "away": "Ignored"},
        ]
        aliases_doc = {
            "source": "teams.json",
            "updatedAt": "2026-01-02",
            "aliases": {"Short": "Canonical", "Canonical": "Canonical"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            aliases_path = Path(tmp) / "aliases.json"
            aliases_path.write_text(json.dumps(aliases_doc), encoding="utf-8")
            result = audit.build_team_alias_provenance(
                rows, aliases_doc,
                ["2026-01-01", "2026-01-03", "2026-01-05"],
                aliases_path,
            )

        self.assertEqual(result["historyTeamNameOccurrences"], 6)
        self.assertEqual(result["aliasMappedHistoryTeamNameOccurrences"], 2)
        self.assertEqual(result["uniqueAliasKeysUsed"], 1)
        self.assertEqual(result["targetDatesBeforeAliasUpdatedAt"], ["2026-01-01"])
        self.assertEqual(result["priorHistoryByTargetDate"][0]["priorAliasMappedOccurrences"], 0)
        self.assertEqual(result["priorHistoryByTargetDate"][1]["priorAliasMappedOccurrences"], 1)
        self.assertEqual(result["priorHistoryByTargetDate"][2]["priorAliasMappedOccurrences"], 2)
        self.assertTrue(result["fileSha256"])
        self.assertTrue(result["gitBlobSha"])

    def test_missing_alias_map_fails_closed(self):
        with self.assertRaises(ValueError):
            audit.build_team_alias_provenance(
                [{"date": "2026-01-01", "competition": "J1", "home": "A", "away": "B"}],
                {"updatedAt": "2026-01-02", "aliases": []},
                ["2026-01-03"],
                Path(__file__),
            )


if __name__ == "__main__":
    unittest.main()
