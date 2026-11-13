#!/usr/bin/env python3
"""Unit tests for the portable Claude GOAL3 k6/k10 artifact verifier."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "verify_claude_k6_k10_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("goal3_claude_k6_k10_verifier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

SHARE_SCRIPT = ROOT / "verification" / "goal3_independent" / "verify_claude_goal3_share.py"
SHARE_SPEC = importlib.util.spec_from_file_location("goal3_claude_share_audit", SHARE_SCRIPT)
SHARE_MODULE = importlib.util.module_from_spec(SHARE_SPEC)
assert SHARE_SPEC and SHARE_SPEC.loader
SHARE_SPEC.loader.exec_module(SHARE_MODULE)


def make_row(date, match, side, actual, p6, p10, cutoff):
    b6, ll6, _ = MODULE.score(p6, actual)
    b10, ll10, _ = MODULE.score(p10, actual)
    return {
        "fold": 1,
        "date": date,
        "league": "J1",
        "match": match,
        "side": side,
        "actual": actual,
        "p_k6": p6,
        "p_k10": p10,
        "cutoff_prior_max_date": cutoff,
        "brier_k6": b6,
        "brier_k10": b10,
        "ll_k6": ll6,
        "ll_k10": ll10,
        "D_brier": b6 - b10,
        "D_ll": ll6 - ll10,
    }


def fixture_rows():
    return [
        make_row("2026-01-02", "2026-01-02|A|B", "home", 1,
                 [0.10, 0.50, 0.25, 0.15], [0.15, 0.45, 0.25, 0.15], "2026-01-01"),
        make_row("2026-01-02", "2026-01-02|A|B", "away", 0,
                 [0.45, 0.30, 0.15, 0.10], [0.40, 0.30, 0.20, 0.10], "2026-01-01"),
        make_row("2026-01-03", "2026-01-03|C|D", "home", 2,
                 [0.10, 0.20, 0.45, 0.25], [0.10, 0.25, 0.40, 0.25], "2026-01-02"),
        make_row("2026-01-03", "2026-01-03|C|D", "away", 3,
                 [0.05, 0.10, 0.25, 0.60], [0.05, 0.15, 0.30, 0.50], "2026-01-02"),
    ]


class ClaudeGoal3K6K10PortableTests(unittest.TestCase):
    def test_bootstrap_indices_match_pinned_claude_analysis(self):
        self.assertEqual(MODULE.analysis_order_statistic_indices(10000), (250, 9749))
        self.assertEqual(MODULE.analysis_order_statistic_indices(200), (5, 194))
        report = MODULE.audit_rows(fixture_rows(), seed=1, replicates=200)
        self.assertEqual(report["bootstrap"]["lowerIndex"], 5)
        self.assertEqual(report["bootstrap"]["upperIndex"], 194)
        self.assertEqual(
            report["bootstrap"]["indexConvention"],
            "zero-based indices matching pinned analysis.py: int(0.025*n), int(0.975*n)-1",
        )
        self.assertEqual(
            SHARE_MODULE.percentile_pair(list(range(10000))),
            [250, 9749],
        )

    def test_valid_paired_rows_are_recomputed(self):
        report = MODULE.audit_rows(fixture_rows(), seed=1, replicates=200)
        self.assertEqual(report["cohort"]["teamSideTargets"], 4)
        self.assertEqual(report["cohort"]["matches"], 2)
        self.assertEqual(report["cohort"]["dates"], 2)
        self.assertEqual(report["cohort"]["duplicateDateMatchSideKeys"], 0)
        self.assertEqual(report["cohort"]["sameDayOrFutureCutoffViolations"], 0)
        self.assertEqual(set(report["bootstrap"]["intervals"]), {"D_brier", "D_ll"})
        self.assertEqual(
            set(report["bootstrap"]["intervals"]["D_brier"]),
            {"date", "match"},
        )

    def test_bootstrap_is_deterministic_for_fixed_seed(self):
        rows = fixture_rows()
        first = MODULE.audit_rows(rows, seed=17, replicates=200)
        second = MODULE.audit_rows(rows, seed=17, replicates=200)
        self.assertEqual(first["bootstrap"], second["bootstrap"])

    def test_duplicate_date_match_side_is_rejected(self):
        rows = fixture_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            MODULE.audit_rows(rows, replicates=200)

    def test_same_day_or_future_cutoff_is_rejected(self):
        rows = fixture_rows()
        rows[0]["cutoff_prior_max_date"] = rows[0]["date"]
        with self.assertRaisesRegex(AssertionError, "same-day/future"):
            MODULE.audit_rows(rows, replicates=200)

    def test_non_simplex_probability_is_rejected(self):
        rows = fixture_rows()
        rows[0]["p_k6"] = [0.4, 0.4, 0.4, 0.1]
        with self.assertRaisesRegex(AssertionError, "simplex"):
            MODULE.audit_rows(rows, replicates=200)

    def test_saved_score_mismatch_is_rejected(self):
        rows = fixture_rows()
        rows[0]["brier_k6"] += 0.01
        with self.assertRaisesRegex(AssertionError, "brier_k6 mismatch"):
            MODULE.audit_rows(rows, replicates=200)

    def test_upstream_absolute_paths_are_reported_not_executed(self):
        report = MODULE.audit_rows(
            fixture_rows(), seed=1, replicates=200,
            analysis_source='MOD_DIR = "/home/claude/work/pr19"\n',
        )
        self.assertEqual(report["portability"]["claudeAnalysisAbsolutePathOccurrences"], 1)
        self.assertFalse(report["portability"]["claudeAnalysisPortable"])
        self.assertFalse(report["portability"]["claudeAnalysisExecuted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
