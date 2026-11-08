#!/usr/bin/env python3
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification" / "goal3_independent"))
from calibration_reliability_audit import K_VALUES, audit, flatten


def synthetic_report():
    matches = []
    start = date(2026, 9, 1)
    base = np.asarray([0.25, 0.35, 0.22, 0.18], dtype=float)
    for i in range(236):
        by_k = {}
        for k in K_VALUES:
            shift = (k - 8.0) * 0.0002
            ph = base + np.asarray([shift, -shift / 3, -shift / 3, -shift / 3])
            pa = base + np.asarray([-shift, shift / 3, shift / 3, shift / 3])
            by_k[f"{k:.1f}"] = {"home": ph.tolist(), "away": pa.tolist()}
        if i == 0:
            for k in K_VALUES:
                by_k[f"{k:.1f}"]["home"] = [0.05, 0.35, 0.22, 0.38]
        matches.append({
            "date": (start + timedelta(days=i % 17)).isoformat(),
            "comp": ("J1", "J2", "J3")[i % 3],
            "home": f"home-{i}", "away": f"away-{i}",
            "hg": i % 4, "ag": (i + 1) % 4,
            "byK": by_k,
        })
    return {"allKPredictions": matches}


class CalibrationReliabilityAuditTests(unittest.TestCase):
    def test_fixed_cohort_expansion(self):
        rows, dates = flatten(synthetic_report())
        self.assertEqual(len(rows), 472)
        self.assertEqual(len(dates), 17)
        self.assertEqual(set(rows[0]["probabilities"]), set(K_VALUES))

    def test_audit_reproducible_and_families_complete(self):
        report = synthetic_report()
        first = audit(report, "a" * 64, "b" * 64, replicates=200, seed=77)
        second = audit(report, "a" * 64, "b" * 64, replicates=200, seed=77)
        self.assertEqual(first, second)
        self.assertEqual(len(first["calibrationInterceptSlope"]), 8)
        self.assertEqual(first["method"]["logisticSimultaneousFamily"]["comparisonCount"], 16)
        self.assertEqual(first["method"]["pairedK6MinusK10SimultaneousFamily"]["comparisonCount"], 8)
        self.assertEqual(first["method"]["binSimultaneousFamily"]["comparisonCount"],
                         sum(x["ciStatus"] == "computed" for x in first["probabilityBinReliability"]))
        self.assertTrue(all(x["validBootstrapReplicates"] == 200
                            for x in first["calibrationInterceptSlope"]))

    def test_sparse_and_empty_bins_are_not_given_confidence_intervals(self):
        result = audit(synthetic_report(), "a" * 64, "b" * 64, replicates=200, seed=99)
        sparse = [x for x in result["probabilityBinReliability"] if 0 < x["n"] < 20]
        empty = [x for x in result["probabilityBinReliability"] if x["n"] == 0]
        self.assertTrue(sparse)
        self.assertTrue(empty)
        self.assertTrue(all(x["ciStatus"] == "not_computed_sparse_bin" for x in sparse))
        self.assertTrue(all(x["ciStatus"] == "empty_bin" for x in empty))
        self.assertTrue(all("simultaneous95Band" not in x for x in sparse + empty))

    def test_rejects_duplicate_match_and_invalid_probability(self):
        report = synthetic_report()
        report["allKPredictions"][1] = dict(report["allKPredictions"][0])
        with self.assertRaises(ValueError):
            flatten(report)
        report = synthetic_report()
        report["allKPredictions"][0]["byK"]["6.0"]["home"] = [0.5, 0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            flatten(report)

    def test_rejects_noncanonical_date(self):
        report = synthetic_report()
        report["allKPredictions"][0]["date"] = "2026-9-01"
        with self.assertRaisesRegex(ValueError, "invalid 2026 target date"):
            flatten(report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
