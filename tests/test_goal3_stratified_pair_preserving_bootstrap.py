import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "verification" / "goal3_independent"
sys.path.insert(0, str(SCRIPT_DIR))

BASE_SPEC = importlib.util.spec_from_file_location(
    "pair_preserving_date_team_bootstrap_audit",
    SCRIPT_DIR / "pair_preserving_date_team_bootstrap_audit.py",
)
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

SPEC = importlib.util.spec_from_file_location(
    "stratified_pair_preserving_date_team_bootstrap_audit",
    SCRIPT_DIR / "stratified_pair_preserving_date_team_bootstrap_audit.py",
)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _match(date, league, home, away):
    by_k = {
        f"{k:.1f}": {
            "home": [0.4, 0.3, 0.2, 0.1],
            "away": [0.3, 0.4, 0.2, 0.1],
        }
        for k in base.K_VALUES
    }
    return {
        "date": date,
        "comp": league,
        "home": home,
        "away": away,
        "hg": 1,
        "ag": 0,
        "byK": by_k,
    }


class StratifiedPairPreservingBootstrapTests(unittest.TestCase):
    def test_paired_residual_max_t_family_handles_zero_variance_contrasts(self):
        bootstrap = np.column_stack([
            np.linspace(-1.0, 1.0, 1000),
            np.zeros(1000),
        ])
        point = np.asarray([0.0, 0.0])
        labels = [
            {"scope": "ALL", "contrast": "k6_minus_k10"},
            {"scope": "J2", "contrast": "k6_minus_k10"},
        ]
        result = audit.paired_residual_max_t_family(bootstrap, point, labels)
        self.assertEqual(result["comparisonCount"], 2)
        self.assertEqual(result["intervalsContainingZero"], 2)
        self.assertEqual(result["intervals"][1]["bootstrapSE"], 0.0)
        self.assertEqual(result["intervals"][1]["simultaneous95Band"], [0.0, 0.0])

    def test_source_provenance_is_pinned_without_pinning_float_serialization(self):
        nested_script = SCRIPT_DIR / "nested_validation.py"
        nested_bytes = nested_script.read_bytes()
        expected_blob = hashlib.sha1(
            b"blob " + str(len(nested_bytes)).encode("ascii") + b"\\0" + nested_bytes
        ).hexdigest()
        self.assertEqual(
            audit.EXPECTED_SOURCE_PROVENANCE["nestedValidationPySha256"],
            hashlib.sha256(nested_bytes).hexdigest(),
        )
        self.assertEqual(
            audit.EXPECTED_SOURCE_PROVENANCE["nestedValidationPyGitBlobSha"],
            expected_blob,
        )
        document = {"provenance": dict(audit.EXPECTED_SOURCE_PROVENANCE)}
        audit.validate_source_provenance(document)
        document["provenance"]["history2026Sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source provenance mismatch"):
            audit.validate_source_provenance(document)

    def test_build_team_strata_preserves_each_league_team_partition(self):
        document = {
            "allKPredictions": [
                _match("2026-09-19", "J1", "A", "B"),
                _match("2026-09-20", "J2", "C", "D"),
                _match("2026-09-21", "J3", "E", "F"),
            ]
        }
        data = base.prepare_arrays(document, enforce_expected_cohort=False)
        strata = audit.build_team_strata(
            data, expected_team_counts={"J1": 2, "J2": 2, "J3": 2}
        )
        self.assertEqual([len(x) for x in strata["teamIdsByLeague"]], [2, 2, 2])
        self.assertEqual(strata["matchLeagueIndex"].tolist(), [0, 1, 2])
        self.assertEqual(strata["homeLocal"].tolist(), [0, 0, 0])
        self.assertEqual(strata["awayLocal"].tolist(), [1, 1, 1])

    def test_team_identity_cannot_cross_league_strata(self):
        document = {
            "allKPredictions": [
                _match("2026-09-19", "J1", "A", "B"),
                _match("2026-09-20", "J2", "A", "C"),
                _match("2026-09-21", "J3", "D", "E"),
            ]
        }
        data = base.prepare_arrays(document, enforce_expected_cohort=False)
        with self.assertRaisesRegex(ValueError, "team identity crosses league strata"):
            audit.build_team_strata(
                data, expected_team_counts={"J1": 2, "J2": 2, "J3": 2}
            )

    def test_match_weight_uses_league_local_team_counts_and_preserves_pair(self):
        date_counts = np.asarray([[2, 1]], dtype=np.int64)
        team_counts = [
            np.asarray([[3, 2]], dtype=np.int64),
            np.asarray([[4, 1]], dtype=np.int64),
            np.asarray([[1, 2]], dtype=np.int64),
        ]
        date_index = np.asarray([0, 1, 1], dtype=np.int64)
        home_local = np.asarray([0, 0, 1], dtype=np.int64)
        away_local = np.asarray([1, 1, 0], dtype=np.int64)
        league_index = np.asarray([0, 1, 2], dtype=np.int64)
        match_weights = audit.stratified_pair_preserving_match_weights(
            date_counts, team_counts, date_index,
            home_local, away_local, league_index
        )
        self.assertEqual(match_weights.tolist(), [[12.0, 4.0, 2.0]])
        target_weights = np.repeat(match_weights, 2, axis=1)
        self.assertEqual(target_weights.tolist(), [[12.0, 12.0, 4.0, 4.0, 2.0, 2.0]])

    def test_rejects_wrong_team_count_for_league(self):
        document = {
            "allKPredictions": [
                _match("2026-09-19", "J1", "A", "B"),
                _match("2026-09-20", "J2", "C", "D"),
                _match("2026-09-21", "J3", "E", "F"),
            ]
        }
        data = base.prepare_arrays(document, enforce_expected_cohort=False)
        with self.assertRaisesRegex(ValueError, "unexpected team count for J1"):
            audit.build_team_strata(data)


if __name__ == "__main__":
    unittest.main()
