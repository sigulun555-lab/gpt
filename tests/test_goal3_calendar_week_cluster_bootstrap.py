from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification" / "goal3_independent"))
import calendar_week_cluster_bootstrap_audit as audit


class CalendarWeekBootstrapTests(unittest.TestCase):
    def test_iid_draws_are_deterministic(self):
        a = audit.make_week_draws(8, 100, 1234, "iid_iso_week")
        b = audit.make_week_draws(8, 100, 1234, "iid_iso_week")
        self.assertEqual(a.shape, (100, 8))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a >= 0) & (a < 8)))

    def test_two_week_blocks_preserve_adjacency_and_wrap(self):
        draws = audit.make_week_draws(8, 200, 99, "circular_two_week_blocks")
        self.assertEqual(draws.shape, (200, 8))
        for row in draws:
            for start in (0, 2, 4, 6):
                self.assertEqual(int(row[start + 1]), (int(row[start]) + 1) % 8)

    def test_common_valid_gate_requires_every_scope(self):
        counts = np.array([[2, 0, 3], [0, 4, 0]], dtype=int)
        draws = np.array([[0, 1], [1, 1], [0, 2]], dtype=int)
        valid = audit.common_valid_mask(counts, draws)
        np.testing.assert_array_equal(valid, np.array([True, False, False]))

    def test_max_t_interval_rejects_zero_variance(self):
        with self.assertRaises(ValueError):
            audit.max_t_intervals(np.array([1.0]), np.array([[1.0], [1.0]]), ["x"])

    def test_invalid_designs_and_small_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            audit.make_week_draws(1, 10, 1, "iid_iso_week")
        with self.assertRaises(ValueError):
            audit.make_week_draws(8, 10, 1, "unknown")
        with self.assertRaises(ValueError):
            audit.audit({}, replicates=999)


if __name__ == "__main__":
    unittest.main()
