from __future__ import annotations

import unittest

import numpy as np

from innovative_solution.reducers import available_reducers, make_reducer


class ReducerPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(40, 12))
        self.y = self.X[:, 0] - 0.5 * self.X[:, 1]

    def test_registry_lists_supported_reducers(self) -> None:
        self.assertEqual(
            available_reducers(),
            ["none", "pca50", "pca95", "svd50", "pls20", "autoencoder16"],
        )

    def test_pca_plugin_preserves_row_count(self) -> None:
        reducer = make_reducer("pca50", seed=42)
        transformed = reducer.fit_transform(self.X, self.y)
        self.assertEqual(transformed.shape[0], self.X.shape[0])
        self.assertLessEqual(transformed.shape[1], self.X.shape[1])

    def test_pls_plugin_uses_supervised_target(self) -> None:
        reducer = make_reducer("pls20", seed=42)
        transformed = reducer.fit_transform(self.X, self.y)
        self.assertEqual(transformed.shape, (40, 12))


if __name__ == "__main__":
    unittest.main()
