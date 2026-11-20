import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "verification" / "goal3_independent" / "controlled_openblas_dispatch_audit.py"
SPEC = importlib.util.spec_from_file_location("controlled_openblas_dispatch_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ControlledOpenBLASDispatchTests(unittest.TestCase):
    def test_cpu_flags_are_intersected_across_processors(self):
        cpuinfo = "processor : 0\nflags : fma avx2 avx512f\n\nprocessor : 1\nflags : fma avx2\n"
        self.assertEqual(AUDIT.parse_cpu_flags(cpuinfo), {"fma", "avx2"})

    def test_coretype_match_requires_every_openblas_library(self):
        pools = [
            {"internal_api": "openblas", "architecture": "Haswell"},
            {"internal_api": "openblas", "architecture": "HASWELL"},
        ]
        self.assertTrue(AUDIT.architecture_matches(pools, "Haswell"))
        pools.append({"internal_api": "openblas", "architecture": "SkylakeX"})
        self.assertFalse(AUDIT.architecture_matches(pools, "Haswell"))
        self.assertFalse(AUDIT.architecture_matches([], "Haswell"))

    def test_pair_comparison_reports_gate_flips_and_probability_drift(self):
        runtime = {"python": "3.11", "numpy": "2.4", "scipy": "1.17",
                   "threadpoolctl": "3.8", "openblasVersions": ["0.3.30"],
                   "openblasThreadCounts": [1]}
        left = {
            "mode": "Haswell", "sourceSha256": {"allk": "same"}, "runtime": runtime,
            "fitDiagnostics": [
                {"date": "2026-01-01", "k": 6.0, "maxAbsGradient": 0.9e-6,
                 "parameters": [1.0, 2.0], "parameterSha256": "a", "status": 0,
                 "success": True, "message": "PGTOL", "iterations": 10},
                {"date": "2026-01-01", "k": 10.0, "maxAbsGradient": 1.2e-6,
                 "parameters": [3.0, 4.0], "parameterSha256": "b", "status": 0,
                 "success": True, "message": "REL", "iterations": 11},
            ],
            "predictions": [{"matchIndex": 0, "k": 6.0,
                             "home": [0.4, 0.3, 0.2, 0.1],
                             "away": [0.1, 0.2, 0.3, 0.4]}],
        }
        right = {
            "mode": "SkylakeX", "sourceSha256": {"allk": "same"}, "runtime": runtime,
            "fitDiagnostics": [
                {"date": "2026-01-01", "k": 6.0, "maxAbsGradient": 1.1e-6,
                 "parameters": [1.0, 2.1], "parameterSha256": "c", "status": 0,
                 "success": True, "message": "REL", "iterations": 12},
                {"date": "2026-01-01", "k": 10.0, "maxAbsGradient": 1.3e-6,
                 "parameters": [3.0, 4.0], "parameterSha256": "b", "status": 0,
                 "success": True, "message": "REL", "iterations": 11},
            ],
            "predictions": [{"matchIndex": 0, "k": 6.0,
                             "home": [0.4, 0.3, 0.2, 0.1],
                             "away": [0.1, 0.2, 0.3, 0.400000001]}],
        }
        result = AUDIT.summarize_pair(left, right)
        self.assertEqual(result["gradientGateFlipCount"], 1)
        self.assertEqual(result["optimizerMessageChangeCount"], 1)
        self.assertEqual(result["iterationCountChangeCount"], 1)
        self.assertAlmostEqual(result["maximumParameterAbsoluteDifference"], 0.1)
        self.assertAlmostEqual(result["maximumProbabilityAbsoluteDifference"], 1e-9)
        self.assertEqual(result["probabilityCellDifferenceCount"], 1)
        self.assertEqual(result["perK"]["6"]["gradientGateFlipCount"], 1)
        self.assertEqual(result["perK"]["10"]["gradientGateFlipCount"], 0)
        self.assertEqual(result["perK"]["6"]["probabilityCellDifferenceCount"], 1)
        self.assertTrue(result["inputSha256Match"])
        self.assertTrue(result["runtimeVersionAndLibraryMatch"])

    def test_comparison_rejects_different_input_hashes(self):
        left = {"sourceSha256": {"history": "a"}, "runtime": {}}
        right = {"sourceSha256": {"history": "b"}, "runtime": {}}
        with self.assertRaisesRegex(ValueError, "different input SHA-256"):
            AUDIT.summarize_pair(left, right)

    def test_comparison_rejects_different_openblas_versions(self):
        base = {"sourceSha256": {"history": "same"},
                "runtime": {"python": "3.11", "numpy": "2.4", "scipy": "1.17",
                            "threadpoolctl": "3.8", "openblasVersions": ["0.3.30"],
                            "openblasThreadCounts": [1]}}
        other = {"sourceSha256": {"history": "same"},
                 "runtime": {"python": "3.11", "numpy": "2.4", "scipy": "1.17",
                             "threadpoolctl": "3.8", "openblasVersions": ["0.3.31"],
                             "openblasThreadCounts": [1]}}
        with self.assertRaisesRegex(ValueError, "different OpenBLAS openblasVersions"):
            AUDIT.summarize_pair(base, other)

    def test_a_partial_cpu_matrix_is_not_called_complete(self):
        complete = {"status": "complete", "fitCount": 272}
        results = {"native": complete, "Haswell": complete,
                   "SkylakeX": {"status": "skipped_hardware_flags"}}
        self.assertFalse(all(results[x].get("status") == "complete" for x in ("native", "Haswell", "SkylakeX")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
