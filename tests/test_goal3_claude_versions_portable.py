#!/usr/bin/env python3
"""Portable GPT-side tests for Claude's pinned GOAL3 version runner.

The pinned source is materialized by verify_claude_goal3_share.py before this
suite runs. All test inputs are created in temporary directories; no Claude
home path, scratch path, production file, or prediction log is required.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "verification" / "goal3_independent" / "results" / "_claude_goal3_versions_portable.py"
ORIGINAL_MODULE_PATH = ROOT / "verification" / "goal3_independent" / "results" / "_claude_goal3_versions.py"
REPORT_PATH = ROOT / "verification" / "goal3_independent" / "results" / "claude_goal3_share_audit.json"


def load_pinned_module():
    if not MODULE_PATH.is_file():
        raise RuntimeError(
            "Patched research copy of the pinned Claude module is missing; run verify_claude_goal3_share.py before this test."
        )
    spec = importlib.util.spec_from_file_location("claude_goal3_versions_portable_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load pinned Claude GOAL3 version module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClaudeGoal3VersionsPortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g3 = load_pinned_module()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        matches = [
            ("2026-01-01", "J1", "A", "B", 1, 0),
            ("2026-01-02", "J1", "B", "A", 0, 1),
            ("2026-01-03", "J1", "A", "C", 2, 1),
            ("2026-01-04", "J1", "C", "A", 1, 1),
            ("2026-01-05", "J2", "B", "D", 0, 2),
            ("2026-01-06", "J2", "D", "B", 1, 0),
            ("2026-01-07", "J2", "C", "D", 2, 0),
            ("2026-01-08", "J2", "D", "C", 1, 2),
            ("2026-01-09", "J3", "A", "D", 1, 1),
            ("2026-01-10", "J3", "D", "A", 0, 0),
            ("2026-01-11", "J3", "B", "C", 1, 2),
            ("2026-01-12", "J3", "C", "B", 2, 1),
        ]
        history = []
        for date, comp, home, away, hg, ag in matches:
            history.append({
                "date": date, "competition": comp, "home": home, "away": away,
                "home_goals": hg, "away_goals": ag,
            })
        history.extend([
            {"date": "2025-12-31", "competition": "J1", "home": "A", "away": "B",
             "home_goals": 1, "away_goals": 0},
            {"date": "2026-01-13", "competition": "OTHER", "home": "A", "away": "B",
             "home_goals": 1, "away_goals": 0},
        ])
        self.history_path = self.root / "history.json"
        self.history_path.write_text(json.dumps({"history": history}), encoding="utf-8")
        self.fixtures_path = self.root / "fixtures.json"
        self.fixtures_path.write_text(json.dumps([["A", "B"]]), encoding="utf-8")
        self.market_path = self.root / "market.json"
        self.market_path.write_text(json.dumps({
            "A|B": [
                [0.40, 0.30, 0.20, 0.10],
                [0.30, 0.30, 0.25, 0.15],
            ]
        }), encoding="utf-8")

    def test_load_rows_filters_leagues_and_sorts_using_temporary_path(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            rows = self.g3.load_rows(str(self.history_path))
            gc.collect()
        self.assertFalse(
            [item for item in caught if issubclass(item.category, ResourceWarning)],
            "portable load_rows must close its history file",
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["date"] for r in rows], sorted(r["date"] for r in rows))
        self.assertTrue(all(r["competition"] in ("J1", "J2", "J3") for r in rows))

    def test_prior_before_is_strict_and_excludes_same_day(self):
        rows = self.g3.load_rows(str(self.history_path))
        prior = self.g3.prior_before(rows, "2026-01-07")
        self.assertEqual(len(prior), 6)
        self.assertTrue(all(r["date"] < "2026-01-07" for r in prior))

    def test_v0_and_v1_are_exactly_the_same_specification(self):
        rows = self.g3.load_rows(str(self.history_path))
        fixtures = [("A", "B")]
        v0 = self.g3.run_version("v0", rows, "2026-01-13", fixtures)
        v1 = self.g3.run_version("v1", rows, "2026-01-13", fixtures)
        self.assertEqual(v0, v1)

    def test_v2_integration_requires_market_and_runs_with_synthetic_market(self):
        rows = self.g3.load_rows(str(self.history_path))
        fixtures = [("A", "B")]
        with self.assertRaises(ValueError):
            self.g3.run_version("v2", rows, "2026-01-13", fixtures, market=None)
        market = {("A", "B"): [
            [0.40, 0.30, 0.20, 0.10],
            [0.30, 0.30, 0.25, 0.15],
        ]}
        result = self.g3.run_version("v2", rows, "2026-01-13", fixtures, market=market)
        for side in ("home", "away"):
            probs = result[("A", "B")][side]
            self.assertEqual(len(probs), 4)
            self.assertAlmostEqual(sum(probs), 1.0, places=12)

    def test_cli_marks_v2_not_run_without_market_and_runs_with_market(self):
        no_market_out = self.root / "no_market.json"
        argv = [
            "goal3_versions.py", "--history", str(self.history_path),
            "--target-date", "2026-01-13", "--fixtures", str(self.fixtures_path),
            "--versions", "v0,v2", "--out", str(no_market_out),
        ]
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), *argv[1:]],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONWARNINGS": "always::ResourceWarning"},
        )
        self.assertNotIn("ResourceWarning", completed.stderr)
        no_market = json.loads(no_market_out.read_text(encoding="utf-8"))
        self.assertEqual(no_market["versions"]["v2"]["status"], "not_run")
        self.assertEqual(no_market["versions"]["v0"]["status"], "ok")

        with_market_out = self.root / "with_market.json"
        argv = [
            "goal3_versions.py", "--history", str(self.history_path),
            "--target-date", "2026-01-13", "--fixtures", str(self.fixtures_path),
            "--market", str(self.market_path), "--versions", "v2",
            "--out", str(with_market_out),
        ]
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), *argv[1:]],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONWARNINGS": "always::ResourceWarning"},
        )
        self.assertNotIn("ResourceWarning", completed.stderr)
        with_market = json.loads(with_market_out.read_text(encoding="utf-8"))
        self.assertEqual(with_market["versions"]["v2"]["status"], "ok")
        for fixture in with_market["versions"]["v2"]["fixtures"]:
            self.assertAlmostEqual(sum(fixture["home_probs"]), 1.0, places=12)
            self.assertAlmostEqual(sum(fixture["away_probs"]), 1.0, places=12)

    def test_audit_report_keeps_raw_vmr_separate_from_conditional_pearson(self):
        self.assertTrue(REPORT_PATH.is_file(), "run the pinned-source verifier before this test")
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        review = report["poisson_residual_review"]
        self.assertIn("conditional_pearson_dispersion_from_lambda_minus_log_p0", review)
        self.assertIn("unadjusted_raw_count_variance_to_mean_not_a_conditional_poisson_residual", review)
        self.assertIn("not a model-based Pearson residual", review["note"])

    def test_pinned_source_is_preserved_and_portable_copy_is_file_handle_safe(self):
        self.assertTrue(ORIGINAL_MODULE_PATH.is_file())
        self.assertTrue(MODULE_PATH.is_file())
        original = ORIGINAL_MODULE_PATH.read_text(encoding="utf-8")
        portable = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('json.load(open(history_path, encoding="utf-8"))', original)
        self.assertNotIn('json.load(open(history_path, encoding="utf-8"))', portable)
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        review = report["version_implementation_review"]
        patch = review["portable_runtime_patch"]
        warning = review["resource_warning_review"]
        self.assertFalse(patch["upstream_source_modified"])
        self.assertEqual(len(patch["file_handle_patches"]), 4)
        self.assertEqual(warning["patched_portable_load_rows_resource_warning_count"], 0)
        self.assertTrue(warning["original_and_patched_rows_exact_match"])

    def test_audit_report_records_missing_claude_manifest_and_independent_hashes(self):
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        provenance = report["generation_provenance"]
        original = provenance["claude_original_generation_manifest"]
        self.assertEqual(original["status"], "not_provided")
        self.assertFalse(original["execution_linkage_proven"])
        loader = provenance["documented_loader_reproduction"]
        self.assertEqual(loader["2025_rows"], 1140)
        self.assertEqual(loader["2026_rows"], 429)
        self.assertTrue(loader["2025_rows_exact_match"])
        self.assertTrue(loader["2026_rows_exact_match"])
        self.assertFalse(loader["nested_validation_full_file_identical"])
        self.assertTrue(loader["nested_validation_core_from_bin_goals_identical"])
        independent = provenance["gpt_independent_reproduction_manifest"]
        self.assertEqual(independent["status"], "verified")
        for key in ("history.json", "team_aliases.json", "2025.csv", "nested_validation.py"):
            self.assertEqual(len(independent["inputs"][key]["sha256"]), 64)
            self.assertEqual(len(independent["inputs"][key]["git_blob_sha1"]), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
