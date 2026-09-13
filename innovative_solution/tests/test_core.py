from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from innovative_solution.pipeline import (
    conformal_quantile,
    optimize_blend_weights,
    parse_timestamp_series,
    recency_weights,
)
from innovative_solution.phase_b_optimization import validate_released_a_answer


class CorePipelineTests(unittest.TestCase):
    def test_timestamp_parser_supports_competition_formats(self) -> None:
        values = pd.Series([20170712, 20170712173508, 2017071217350864, np.nan])
        parsed = parse_timestamp_series(values)
        self.assertEqual(int(parsed.notna().sum()), 3)
        self.assertAlmostEqual(float(parsed.iloc[2] - parsed.iloc[1]), 0.64, places=3)

    def test_recency_weights_favor_later_rows(self) -> None:
        weights = recency_weights(pd.Series([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(np.all(np.diff(weights) > 0))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_blend_weights_are_convex(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        predictions = pd.DataFrame(
            {
                "good": [0.0, 1.0, 2.0, 3.0],
                "weak": [1.0, 1.0, 1.0, 1.0],
            }
        )
        weights = optimize_blend_weights(y, predictions)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue((weights >= 0).all())
        self.assertGreater(float(weights["good"]), 0.99)

    def test_conformal_quantile_handles_insufficient_calibration_size(self) -> None:
        radius = conformal_quantile(np.array([0.1, -0.2, 0.3, -0.4]), alpha=0.1)
        self.assertTrue(np.isinf(radius))
        self.assertEqual(conformal_quantile(np.array([0.1, -0.2, 0.3, -0.4]), alpha=0.2), 0.4)

    def test_released_a_answer_requires_exact_id_order(self) -> None:
        expected = pd.Series(["NH0001", "NH0001", "NH0002"])
        valid = pd.DataFrame(
            [["NH0001", 2.1], ["NH0001", 2.2], ["NH0002", 2.3]]
        )
        result = validate_released_a_answer(valid, expected)
        self.assertEqual(result.tolist(), [2.1, 2.2, 2.3])

        invalid = valid.iloc[[0, 2, 1]].reset_index(drop=True)
        with self.assertRaises(ValueError):
            validate_released_a_answer(invalid, expected)


if __name__ == "__main__":
    unittest.main()
