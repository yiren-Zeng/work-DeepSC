import unittest

from utils.raq_rvq import validate_independent_rvq_k_lists


class RaggedRvqPrefixValidationTest(unittest.TestCase):
    def test_accepts_different_prefix_depth_per_scale(self):
        actual = validate_independent_rvq_k_lists(
            [[2], [8, 2]],
            num_scales=2,
            rvq_depth=2,
            min_k=[2, 2],
            max_k=[64, 64],
        )

        self.assertEqual(actual, [[2], [8, 2]])

    def test_rejects_empty_scale_prefix(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 2 prefix stages"):
            validate_independent_rvq_k_lists(
                [[], [8, 2]],
                num_scales=2,
                rvq_depth=2,
            )

    def test_rejects_prefix_deeper_than_model(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 2 prefix stages"):
            validate_independent_rvq_k_lists(
                [[2, 2, 2], [8, 2]],
                num_scales=2,
                rvq_depth=2,
            )


if __name__ == "__main__":
    unittest.main()
