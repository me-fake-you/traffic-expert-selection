"""Synthetic edge cases test code behavior, not the paper's scientific claims."""
import unittest
import numpy as np
from traffic_selection.core import choose, decomposition, fixed_matrix, metrics, pattern_support


class CoreTests(unittest.TestCase):
    def test_macro_not_accuracy(self):
        m = metrics([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertAlmostEqual(m["macro_f1"], 3/7)
        self.assertNotEqual(m["macro_f1"], .75)

    def test_correction_identity(self):
        m = metrics([0, 1, 1], [0, 0, 1], [1, 1, 1])
        self.assertEqual((m["C"], m["D"], m["switches"]), (1, 1, 2))

    def test_equal_scores_are_not_equal_vectors(self):
        matrix = np.array([[0, 1], [1, 0], [1, 1]])
        self.assertEqual(pattern_support(matrix), {"m": 2, "K": 2})

    def test_ties_are_stable(self):
        best, feasible, optimal, _ = choose([0, 1], [0, 1], np.array([[0, 0], [1, 1]]))
        self.assertEqual((best, feasible, optimal), (0, [0, 1], [0, 1]))

    def test_no_feasible_candidate_raises(self):
        with self.assertRaises(ValueError):
            choose([0, 1], [0, 1], np.array([[0], [0]]))

    def test_no_switch_with_zero_margin(self):
        p = fixed_matrix([.5, .49], [.1, .9], [1, 1], 1., 1.)
        self.assertTrue(np.array_equal(p[:, 15], [1, 0]))

    def test_strict_fixed_comparison(self):
        p = fixed_matrix([.75], [.4375], [1], 1., 1.)
        self.assertEqual(p[0, 0], 1)  # equal weighted margins: no switch

    def test_hindsight_identity(self):
        sm = np.array([[0, 0, 1], [1, 1, 1]])
        em = np.array([[1, 0, 0], [1, 1, 1]])
        d = decomposition([0, 1], [0, 1], sm, [0, 1], [0, 1], em)
        self.assertEqual((d["within_class_error_space"], d["between_class_error_space"],
                          d["total_error_space"]), (1, 0, 1))
        self.assertEqual(d["selected_pattern_candidates"], 2)
        self.assertEqual(d["selected_pattern_outer_patterns"], 2)

    def test_bad_shapes_and_probabilities(self):
        with self.assertRaises(ValueError):
            metrics([0, 1], [1])
        with self.assertRaises(ValueError):
            fixed_matrix([1.2], [.4], [1], 1., 1.)


if __name__ == "__main__":
    unittest.main()
