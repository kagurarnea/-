from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from innovative_solution.config import PipelineConfig
from innovative_solution.pipeline import (
    PreparedData, RunTrace, build_fold_features, conformal_quantile,
    prepare_data, purge_training_overlap, validation_groups,
)
from innovative_solution.advanced_analysis import (
    build_inner_feature_cache, build_fold_feature_cache, run_nested_grid_search,
)


def synthetic_data(n: int = 40) -> PreparedData:
    rng = np.random.default_rng(137)
    x = rng.normal(size=n)
    numeric = pd.DataFrame({
        "100X1": x,
        "100X2": np.where(np.arange(n) < 24, x, x + 10),
        "100X3": np.where(np.arange(n) < 24, 4.0, np.arange(n)),
        "100X4": np.nan,
        "100X5": np.where(np.arange(n) % 2, 3.0, np.nan),
        "100X6": 20170701000000 + np.arange(n) * 100,
    })
    cat = pd.DataFrame({"TOOL": np.where(np.arange(n) % 2, "a", "b")})
    empty_time = pd.DataFrame(index=numeric.index)
    return PreparedData(
        train_numeric=numeric, test_a_numeric=numeric.iloc[:2].copy(),
        test_b_numeric=numeric.iloc[:2].copy(), train_categorical=cat,
        test_a_categorical=cat.iloc[:2].copy(), test_b_categorical=cat.iloc[:2].copy(),
        train_time=empty_time, test_a_time=empty_time.iloc[:2].copy(),
        test_b_time=empty_time.iloc[:2].copy(), y=pd.Series(x + rng.normal(0, .1, n)),
        train_ids=pd.Series([f"id{i}" for i in range(n)]),
        test_a_ids=pd.Series(["a0", "a1"]), test_b_ids=pd.Series(["b0", "b1"]),
        order_key=pd.Series(np.arange(n), dtype=float),
        feature_operation={c: "100" for c in numeric}, time_operation={},
        operation_tool={"100": "TOOL"}, feature_audit=pd.DataFrame(),
        operation_audit=pd.DataFrame(), primary_tool_column="TOOL",
    )


class StrictReviewRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig(selector_estimators=8, top_k_raw_features=4)
        self.data = synthetic_data()

    def test_schema_and_fitted_features_ignore_heldout_values(self) -> None:
        train, valid = np.arange(24), np.arange(24, 40)
        first, X_train, X_valid = build_fold_features(self.data, self.config, train, valid)
        status = first.structural_report_.set_index("feature")["status"]
        self.assertEqual(status["100X2"], "removed_duplicate")
        self.assertEqual(status["100X3"], "removed_constant")
        self.assertEqual(status["100X4"], "removed_all_nan")
        self.assertEqual(status["100X5"], "kept_numeric")
        self.assertEqual(status["100X6"], "converted_timestamp")
        changed = self.data.train_numeric.copy()
        changed.iloc[valid] = 9.9e8
        changed_data = replace(self.data, train_numeric=changed,
                               y=self.data.y.where(self.data.y.index < 24, 9999))
        second, altered_train, _ = build_fold_features(changed_data, self.config, train, valid)
        pd.testing.assert_frame_equal(first.structural_report_, second.structural_report_)
        pd.testing.assert_frame_equal(first.selection_report, second.selection_report)
        pd.testing.assert_frame_equal(X_train, altered_train)
        self.assertEqual(list(X_train.columns), list(X_valid.columns))
        self.assertTrue(np.isfinite(X_valid.to_numpy()).all())

    def test_prepare_retains_raw_columns_for_future_fold_fitting(self) -> None:
        train = self.data.train_numeric.assign(ID=self.data.train_ids, Y=self.data.y)
        train["100X7"] = train["100X1"]
        test = train.iloc[:2].drop(columns="Y")
        frames = {self.config.train_file: train, self.config.test_a_file: test,
                  self.config.test_b_file: test}
        with patch("innovative_solution.pipeline.read_excel_cached",
                   side_effect=lambda path, cache: frames[Path(path).name].copy()):
            result = prepare_data(self.config, RunTrace())
        self.assertIn("100X4", result.train_numeric)  # globally all missing
        self.assertIn("100X7", result.train_numeric)  # globally duplicated
        self.assertIn("100X6", result.train_numeric)  # timestamp candidates remain raw

    def test_entity_groups_cover_transitive_id_and_predictor_duplicates(self) -> None:
        numeric = self.data.train_numeric.copy()
        cat = self.data.train_categorical.copy()
        numeric.iloc[2] = numeric.iloc[1]
        cat.iloc[2] = cat.iloc[1]
        ids = self.data.train_ids.copy()
        ids.iloc[1] = ids.iloc[0]
        groups = validation_groups(replace(self.data, train_numeric=numeric,
                                           train_categorical=cat, train_ids=ids))
        self.assertEqual(groups[0], groups[2])
        retained = purge_training_overlap(np.array([0, 1, 3, 4]), np.array([2]), groups)
        self.assertEqual(retained.tolist(), [3, 4])

    def test_inner_validation_labels_do_not_enter_inner_feature_selection(self) -> None:
        item = {"data": self.data, "train_index": np.arange(40), "fold": 1}
        original = build_inner_feature_cache(item, self.config)[0]
        changed_y = self.data.y.copy()
        changed_y.iloc[original["valid_index"]] = 1e6
        revised = build_inner_feature_cache(
            {**item, "data": replace(self.data, y=changed_y)}, self.config
        )[0]
        pd.testing.assert_frame_equal(original["X_train"], revised["X_train"])
        pd.testing.assert_frame_equal(original["X_valid"], revised["X_valid"])
        self.assertLess(original["train_index"].max(), original["valid_index"].min())

    def test_inner_search_rejects_only_outer_preselected_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw PreparedData"):
            build_inner_feature_cache({"fold": 1, "X_train": pd.DataFrame()}, self.config)

    def test_nested_search_runs_with_fresh_inner_features_and_records_scope(self) -> None:
        with TemporaryDirectory() as directory:
            config = replace(self.config, output_dir=Path(directory))
            cache = build_fold_feature_cache(self.data, config,
                                             [(np.arange(24), np.arange(24, 40))])
            outer, candidates = run_nested_grid_search(
                cache, config, parameter_grid={"n_estimators": [3], "max_depth": [1, 2]}
            )
        self.assertEqual(len(outer), 1)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.isfinite(outer["mse"]).all())
        self.assertTrue(candidates["preprocessing_scope"].eq("inner_training_rows_only").all())

    def test_invalid_calibration_inputs_fail_explicitly(self) -> None:
        for residuals, alpha in [(np.array([]), .1), (np.array([np.nan]), .1),
                                 (np.array([1.0]), 0.0), (np.array([1.0]), 1.0)]:
            with self.assertRaises(ValueError):
                conformal_quantile(residuals, alpha)


if __name__ == "__main__":
    unittest.main()
