from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib

from innovative_solution import review_validation as review
from innovative_solution.config import PipelineConfig
from innovative_solution.tests.test_core_review import synthetic_data


class ReviewProtocolTests(unittest.TestCase):
    def test_split_helpers_reject_empty_or_nonfinite_boundaries(self) -> None:
        data = synthetic_data()
        groups = np.arange(40)
        cases = [(np.array([], dtype=int), np.array([1]), data.order_key),
                 (np.array([0]), np.array([], dtype=int), data.order_key),
                 (np.array([0]), np.array([1]), data.order_key.mask(data.order_key.index == 1))]
        for train, valid, proxy in cases:
            with self.assertRaises(ValueError):
                review.clean_train(train, valid, groups, proxy)
            with self.assertRaises(ValueError):
                review.split_record(replace(data, order_key=proxy), groups, "test", 0,
                                    train, valid)
        with self.assertRaisesRegex(ValueError, "empty after"):
            review.clean_train(np.array([0]), np.array([1]), groups, pd.Series([1., 1.]))

    def test_clean_train_purges_both_linked_entities_and_boundary_ties(self) -> None:
        groups = np.array([0, 1, 4, 3, 4, 5])
        proxy = pd.Series([0., 1., 2., 3., 3., 4.])
        valid = np.array([4, 5])
        train = review.clean_train(np.array([0, 1, 2, 3]), valid, groups, proxy)
        self.assertEqual(train.tolist(), [0, 1])
        self.assertFalse(set(groups[train]) & set(groups[valid]))
        self.assertLess(proxy.iloc[train].max(), proxy.iloc[valid].min())

    def test_split_record_rejects_time_ties(self) -> None:
        data = replace(synthetic_data(), order_key=pd.Series([0., 1., 1.] + list(range(3, 40))))
        with self.assertRaisesRegex(AssertionError, "boundary violation"):
            review.split_record(data, np.arange(40), "test", 0,
                                np.array([0, 1]), np.array([2]))

    def test_split_record_rejects_entity_overlap_even_without_time_ties(self) -> None:
        data = synthetic_data()
        groups = np.arange(40)
        groups[2] = groups[0]
        with self.assertRaisesRegex(AssertionError, "boundary violation"):
            review.split_record(data, groups, "test", 0, np.array([0, 1]), np.array([2]))

    def test_candidate_column_routing_matches_declared_protocol(self) -> None:
        columns = ["raw__100X1", "missing__100X1", "cat__TOOL_a", "proc__100__mean_z",
                   "time__100__start_day", "route__100_to_200__gap_hours"]
        frame = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
        for candidate in review.PROTOCOL["candidates"]:
            routed = review.keep_columns(frame, candidate)
            self.assertEqual(routed, columns[:3] if candidate.startswith("raw_") else columns)
            model = review.model_for(candidate, 73)
            self.assertEqual(model.get_params()["max_depth"], int(candidate[-1]))
            self.assertEqual(model.get_params()["n_estimators"], review.PROTOCOL["n_estimators"])
            self.assertEqual(model.get_params()["random_state"], 73)

    def test_calibration_and_tail_labels_do_not_enter_frozen_fit(self) -> None:
        data = synthetic_data()
        fitted_rows = []

        class SpyBuilder:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, numeric, categorical, time_frame, y, order_key):
                fitted_rows.append((list(numeric.index), list(y.index)))
                return self

            def transform(self, numeric, categorical, time_frame):
                return pd.DataFrame({"raw__x": numeric["100X1"],
                                     "cat__tool": np.zeros(len(numeric)),
                                     "proc__100__mean_z": np.ones(len(numeric))},
                                    index=numeric.index)

        class SpyModel:
            def fit(self, X, y):
                self.value = float(y.mean())
                self.columns = list(X.columns)
                return self

            def predict(self, X):
                assert list(X.columns) == self.columns
                return np.full(len(X), self.value)

        train = np.arange(12)
        calibration, tail = np.arange(12, 20), np.arange(20, 40)
        changed_y = data.y.copy()
        changed_y.iloc[12:] = 99999
        with patch.object(review, "FoldFeatureBuilder", SpyBuilder), \
             patch.object(review, "model_for", side_effect=lambda *args: SpyModel()), \
             patch.dict(review.PROTOCOL, {"seeds": [42]}):
            original, bundles = review.fit_candidates(
                data, PipelineConfig(), train,
                {"cal": review.frames_at(data, calibration), "tail": review.frames_at(data, tail)},
                ["raw_d2", "full_d2"], keep_models=True,
            )
            revised, _ = review.fit_candidates(
                replace(data, y=changed_y), PipelineConfig(), train,
                {"cal": review.frames_at(data, calibration), "tail": review.frames_at(data, tail)},
                ["raw_d2", "full_d2"], keep_models=True,
            )
        for candidate in original:
            for key in original[candidate]:
                np.testing.assert_array_equal(original[candidate][key], revised[candidate][key])
        self.assertTrue(all(rows == list(train) and labels == list(train)
                            for rows, labels in fitted_rows))
        self.assertEqual(bundles["raw_d2"][0]["columns"], ["raw__x", "cat__tool"])
        self.assertEqual(len(bundles["full_d2"][0]["columns"]), 3)

    def test_time_branch_excludes_process_and_retains_route(self) -> None:
        columns = ["raw__x", "missing__x", "cat__tool", "proc__mean", "time__median", "route__gap"]
        X = pd.DataFrame([[1] * len(columns)], columns=columns)
        self.assertEqual(review.keep_columns(X, "time_d3"), columns[:3] + columns[4:])
        self.assertEqual(review.keep_columns(X, "process_d2"), columns[:4])
        with self.assertRaises(ValueError):
            review.keep_columns(X, "unknown_d2")

    def test_expanded_selection_can_select_time_without_evaluation_labels(self) -> None:
        data = synthetic_data(800)
        train = np.arange(300)
        candidates = [f"{v}_d{d}" for v in ["raw", "process", "time", "full"] for d in [2, 3]]
        protocol = {**review.PROTOCOL, "candidates": candidates}
        calls = []

        def mock_fit(current, cfg, fit, targets, names, **kwargs):
            valid = targets["valid"][0].index.to_numpy()
            calls.append((fit, valid, names, kwargs["seeds"]))
            # This oracle is a learner double used only to check selector routing.
            return {n: {"valid": current.y.iloc[valid].to_numpy() + (0 if n == "time_d3" else 1)} for n in names}, {}

        changed = data.y.copy(); changed.iloc[300:] = 1e9
        with patch.object(review, "fit_candidates", side_effect=mock_fit):
            for current in [data, replace(data, y=changed)]:
                rows, splits = [], []
                selected = review.select_inner(current, PipelineConfig(), train, np.arange(800), "test", rows, splits, protocol=protocol)
                self.assertEqual(selected, "time_d3")
                self.assertEqual(len(rows), 16)
        for fit, valid, names, seeds in calls:
            self.assertLess(max(fit), min(valid))
            self.assertLess(max(valid), 300)
            self.assertEqual(names, candidates)
            self.assertEqual(seeds, [42, 143, 244])

    def test_default_entry_without_a_answers_exports_training_only_b(self) -> None:
        self._assert_optional_answer_entry(answer_available=False)

    def test_default_entry_with_a_answers_explicitly_records_b_refit(self) -> None:
        self._assert_optional_answer_entry(answer_available=True)

    def _assert_optional_answer_entry(self, answer_available: bool) -> None:
        # Exercise the actual entrypoint and file/permission branches with an
        # inexpensive learner double; no test claims to reproduce model quality.
        data = synthetic_data(800)
        if answer_available:
            data = replace(data, test_a_time=pd.DataFrame({"100X6": [1000., 1001.]}))
        fit_calls = []

        def inexpensive_fit(current, config, train, targets, candidates, **kwargs):
            fit_calls.append({"rows": len(train), "data_rows": len(current.y),
                              "targets": set(targets)})
            predictions = {candidate: {key: np.full(len(frames[0]), current.y.iloc[train].mean())
                                       for key, frames in targets.items()}
                           for candidate in candidates}
            bundles = {candidate: [{"test_only_training_rows": len(train)}]
                       for candidate in candidates}
            return predictions, bundles

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            config = PipelineConfig(data_dir=folder)
            for name in [config.train_file, config.test_a_file, config.test_b_file]:
                (folder / name).touch()  # the loader is patched; hashes still exercise real files
            if answer_available:
                pd.DataFrame({"ID": data.test_a_ids, "Y": [2.0, 3.0]}).to_csv(
                    folder / config.test_a_answer_file, index=False, header=False)
            output = folder / "review"
            with patch.object(review, "REVIEW_DIR", output), \
                 patch.object(review, "prepare_data", return_value=data), \
                 patch.object(review, "select_inner", return_value="raw_d2"), \
                 patch.object(review, "fit_candidates", side_effect=inexpensive_fit):
                manifest = review.run_review_validation(config)
            self.assertTrue((output / "submission_A.csv").exists())
            b = pd.read_csv(output / "submission_B.csv", header=None)
            baseline_b = pd.read_csv(output / "submission_B_train_only.csv", header=None)
            self.assertEqual(b.iloc[:, 0].tolist(), data.test_b_ids.tolist())
            self.assertEqual(b.shape, (2, 2))
            self.assertEqual(manifest["A_answer_available"], answer_available)
            self.assertEqual(manifest["B_refit_with_released_A_performed"], answer_available)
            self.assertEqual(config.test_a_answer_file in manifest["input_sha256"], answer_available)
            split_audit = pd.read_csv(output / "split_audit.csv")
            self.assertIn("calibration_vs_tail", split_audit["stage"].tolist())
            stored = joblib.load(output / "model_B.joblib")
            if answer_available:
                self.assertIsInstance(manifest["retrospective_A_metrics"], dict)
                self.assertEqual(fit_calls[-1]["rows"], 802)
                self.assertEqual(fit_calls[-1]["targets"], {"B"})
                self.assertEqual(stored["training_scope"], "800_original_plus_released_A_uniform_weight")
            else:
                self.assertIsNone(manifest["retrospective_A_metrics"])
                pd.testing.assert_frame_equal(b, baseline_b)
                self.assertTrue(all(call["data_rows"] == 800 for call in fit_calls))
                self.assertEqual(fit_calls[-1]["targets"], {"A", "B"})
                self.assertEqual(stored["training_scope"], "800_original_labels_no_released_A")


if __name__ == "__main__":
    unittest.main()
