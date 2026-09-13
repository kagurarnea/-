"""Auditable retrospective validation; no pristine-test or coverage guarantee claims."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .config import PipelineConfig
from .pipeline import (FoldFeatureBuilder, RunTrace, metric_row, prepare_data,
                       purge_training_overlap, validation_groups, log)
from .phase_b_optimization import validate_released_a_answer

REVIEW_DIR = Path(__file__).resolve().parent / "outputs" / "review"
PROTOCOL = {
    "version": "review_v2", "interpretation": "retrospective_not_pristine_test",
    "outer_windows": [[320, 100], [420, 100], [520, 100]],
    "development_end": 620, "calibration_end": 700,
    "inner_splits": 2, "inner_block": 60, "seeds": [42, 143, 244],
    "candidates": ["raw_d2", "raw_d3", "full_d2", "full_d3"],
    "selection_metric": "pooled_inner_mse", "n_estimators": 1200,
    "raw_view": "selected_raw_plus_missing_plus_onehot",
    "time_order": "median_parsed_timestamp_proxy_not_release_order",
    "external_A": "retrospective_known_labels_not_used_in_numeric_fit",
    "B_refit": "fixed_uniform_weight_for_released_A_no_claim_of_optimal_weight",
    "interval": "frozen_development_model_separate_calibration_descriptive_only",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def model_for(candidate: str, seed: int) -> XGBRegressor:
    return XGBRegressor(objective="reg:squarederror", n_estimators=1200,
        max_depth=int(candidate[-1]), learning_rate=0.05, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.55, reg_alpha=0.01, reg_lambda=2,
        random_state=seed, n_jobs=4, tree_method="hist")


def keep_columns(X: pd.DataFrame, candidate: str) -> list[str]:
    if candidate.startswith("raw_"):
        return [c for c in X if c.startswith(("raw__", "missing__", "cat__"))]
    return list(X.columns)


def clean_train(train, valid, groups, order_key):
    """No linked entity or time-proxy tie crosses the split boundary."""
    train = purge_training_overlap(np.asarray(train), np.asarray(valid), groups)
    return train[order_key.iloc[train].to_numpy() < order_key.iloc[valid].min()]


def fit_candidates(data, config, train, targets, candidates, *, keep_models=False):
    predictions = {name: {key: [] for key in targets} for name in candidates}
    bundles = {name: [] for name in candidates}
    for seed in PROTOCOL["seeds"]:
        builder = FoldFeatureBuilder(data, config, seed=seed)
        builder.fit(data.train_numeric.iloc[train], data.train_categorical.iloc[train],
                    data.train_time.iloc[train], data.y.iloc[train], data.order_key.iloc[train])
        X = builder.transform(data.train_numeric.iloc[train],
                              data.train_categorical.iloc[train], data.train_time.iloc[train])
        matrices = {key: builder.transform(*frames) for key, frames in targets.items()}
        for name in candidates:
            columns = keep_columns(X, name)
            model = model_for(name, seed).fit(X[columns], data.y.iloc[train])
            for key in targets:
                predictions[name][key].append(model.predict(matrices[key][columns]).astype(float))
            if keep_models:
                bundles[name].append({"builder": builder, "model": model,
                                      "columns": columns, "seed": seed})
    averaged = {name: {key: np.mean(v, axis=0) for key, v in pred.items()}
                for name, pred in predictions.items()}
    return averaged, bundles


def frames_at(data, indices):
    return (data.train_numeric.iloc[indices], data.train_categorical.iloc[indices],
            data.train_time.iloc[indices])


def select_inner(data, config, train, groups, stage, selection_rows, split_rows):
    candidates = PROTOCOL["candidates"]
    squared_errors = {name: [] for name in candidates}
    splitter = TimeSeriesSplit(n_splits=2, test_size=60)
    for fold, (tp, vp) in enumerate(splitter.split(train), 1):
        valid = train[vp]
        fit = clean_train(train[tp], valid, groups, data.order_key)
        split_rows.append(split_record(data, groups, stage, fold, fit, valid))
        predictions, _ = fit_candidates(data, config, fit, {"valid": frames_at(data, valid)}, candidates)
        for name in candidates:
            errors = (predictions[name]["valid"] - data.y.iloc[valid].to_numpy()) ** 2
            squared_errors[name].extend(errors)
            selection_rows.append({"stage": stage, "inner_fold": fold, "candidate": name,
                "n_train": len(fit), "n_valid": len(valid), "mse": float(errors.mean())})
    scores = {name: float(np.mean(errors)) for name, errors in squared_errors.items()}
    selected = min(candidates, key=lambda name: scores[name])
    log(f"{stage}: inner selected {selected}; pooled MSE={scores[selected]:.6f}")
    return selected


def split_record(data, groups, stage, fold, train, valid):
    record = {"stage": stage, "fold": fold, "n_train": len(train), "n_valid": len(valid),
        "train_indices": json.dumps(list(map(int, train))),
        "valid_indices": json.dumps(list(map(int, valid))),
        "group_overlap": len(set(groups[train]) & set(groups[valid])),
        "max_train_proxy": float(data.order_key.iloc[train].max()),
        "min_valid_proxy": float(data.order_key.iloc[valid].min())}
    if record["group_overlap"] or record["max_train_proxy"] >= record["min_valid_proxy"]:
        raise AssertionError("Validation group/time boundary violation")
    return record


def run_review_validation(config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    output = REVIEW_DIR
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Freeze the candidate set before reading labels or fitting models.
    json_write(output / "protocol.json", PROTOCOL)
    data = prepare_data(config, RunTrace())
    if len(data.y) != 800 or not np.isfinite(data.order_key).all():
        raise ValueError("This fixed historical protocol requires 800 finite time-proxy records")
    groups = validation_groups(data)
    order = np.argsort(data.order_key.to_numpy(), kind="stable")
    selection_rows, split_rows, details = [], [], []
    for fold, (cut, size) in enumerate(PROTOCOL["outer_windows"], 1):
        valid = order[cut:cut + size]
        train = clean_train(order[:cut], valid, groups, data.order_key)
        split_rows.append(split_record(data, groups, "outer", fold, train, valid))
        selected = select_inner(data, config, train, groups, f"outer_{fold}", selection_rows, split_rows)
        names = list(dict.fromkeys([selected, "raw_d2", "full_d2"]))
        predictions, _ = fit_candidates(data, config, train, {"valid": frames_at(data, valid)}, names)
        outputs = {"nested_selected": predictions[selected]["valid"],
                   "fixed_raw_d2": predictions["raw_d2"]["valid"],
                   "fixed_full_d2": predictions["full_d2"]["valid"],
                   "mean_baseline": np.full(len(valid), data.y.iloc[train].mean())}
        for name, pred in outputs.items():
            for position, index in enumerate(valid):
                details.append({"fold": fold, "row_index": int(index),
                    "ID": data.train_ids.iloc[index], "group": int(groups[index]),
                    "time_proxy": float(data.order_key.iloc[index]), "model": name,
                    "selected_candidate": selected, "actual": float(data.y.iloc[index]),
                    "prediction": float(pred[position])})
        log(f"Nested retrospective outer fold {fold} complete")

    detail = pd.DataFrame(details)
    summary = pd.DataFrame([{"model": name, "n": len(frame),
        **metric_row(frame.actual.to_numpy(), frame.prediction.to_numpy())}
        for name, frame in detail.groupby("model")])
    detail.to_csv(output / "nested_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "nested_summary.csv", index=False, encoding="utf-8-sig")

    calibration = order[620:700]
    evaluation = order[700:]
    # Reserve evaluation entities before calibration entities and avoid time ties.
    calibration = clean_train(calibration, evaluation, groups, data.order_key)
    development = clean_train(order[:620], np.concatenate([calibration, evaluation]), groups, data.order_key)
    selected = select_inner(data, config, development, groups, "final_development", selection_rows, split_rows)
    split_rows.extend([split_record(data, groups, "calibration", 0, development, calibration),
                       split_record(data, groups, "tail_evaluation", 0, development, evaluation)])
    preds, bundles = fit_candidates(data, config, development,
        {"cal": frames_at(data, calibration), "tail": frames_at(data, evaluation)},
        [selected], keep_models=True)
    residuals = np.abs(data.y.iloc[calibration].to_numpy() - preds[selected]["cal"])
    rank = int(np.ceil((len(residuals) + 1) * .9))
    radius = float(np.sort(residuals)[min(rank - 1, len(residuals) - 1)])
    tail_pred = preds[selected]["tail"]
    tail_y = data.y.iloc[evaluation].to_numpy()
    tail = pd.DataFrame({"row_index": evaluation, "ID": data.train_ids.iloc[evaluation].to_numpy(),
        "actual": tail_y, "prediction": tail_pred,
        "empirical_lower": tail_pred - radius, "empirical_upper": tail_pred + radius})
    tail.to_csv(output / "frozen_tail_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_index": calibration, "absolute_error": residuals}).to_csv(
        output / "calibration_residuals.csv", index=False)
    joblib.dump({"members": bundles[selected], "radius": radius, "protocol": PROTOCOL,
        "coverage_guarantee": False}, output / "frozen_historical_model.joblib", compress=3)

    # Refit point predictors; frozen-model interval is deliberately NOT reused here.
    train_all = np.arange(len(data.y))
    external, final_bundle = fit_candidates(data, config, train_all,
        {"A": (data.test_a_numeric, data.test_a_categorical, data.test_a_time),
         "B": (data.test_b_numeric, data.test_b_categorical, data.test_b_time)}, [selected], keep_models=True)
    prediction_a = external[selected]["A"]
    answers = pd.read_csv(config.data_dir / config.test_a_answer_file, header=None)
    a_y = validate_released_a_answer(answers, data.test_a_ids)
    pd.DataFrame({"ID": data.test_a_ids, "prediction": prediction_a}).to_csv(
        output / "submission_A.csv", index=False, header=False)
    pd.DataFrame({"ID": data.test_b_ids, "prediction": external[selected]["B"]}).to_csv(
        output / "submission_B_train_only.csv", index=False, header=False)
    joblib.dump({"members": final_bundle[selected], "candidate": selected,
        "training_scope": "800_original_labels", "protocol": PROTOCOL},
        output / "model_A.joblib", compress=3)
    a_time = data.test_a_time.median(axis=1).reset_index(drop=True)
    if not np.isfinite(a_time).all():
        raise ValueError("A time proxy missing; cannot silently replace it with release order")
    combined = replace(data,
        train_numeric=pd.concat([data.train_numeric, data.test_a_numeric], ignore_index=True),
        train_categorical=pd.concat([data.train_categorical, data.test_a_categorical], ignore_index=True),
        train_time=pd.concat([data.train_time, data.test_a_time], ignore_index=True),
        y=pd.concat([data.y, a_y], ignore_index=True),
        train_ids=pd.concat([data.train_ids, data.test_a_ids], ignore_index=True),
        order_key=pd.concat([data.order_key, a_time], ignore_index=True))
    b_preds, b_bundle = fit_candidates(combined, config, np.arange(len(combined.y)),
        {"B": (data.test_b_numeric, data.test_b_categorical, data.test_b_time)}, [selected], keep_models=True)
    pd.DataFrame({"ID": data.test_b_ids, "prediction": b_preds[selected]["B"]}).to_csv(
        output / "submission_B.csv", index=False, header=False)
    joblib.dump({"members": b_bundle[selected], "candidate": selected,
        "training_scope": "800_original_plus_300_released_A_uniform_weight", "protocol": PROTOCOL},
        output / "model_B.joblib", compress=3)
    pd.DataFrame(selection_rows).to_csv(output / "inner_selection.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_audit.csv", index=False)
    source_files = [Path(__file__), Path(__file__).with_name("pipeline.py"), Path(__file__).with_name("config.py")]
    input_files = [config.data_dir / name for name in [config.train_file, config.test_a_file,
                    config.test_b_file, config.test_a_answer_file]]
    manifest = {"protocol": PROTOCOL, "selected_candidate": selected,
        "original_data_rows": len(data.y), "unique_validation_groups": int(len(np.unique(groups))),
        "n_development": len(development), "n_calibration": len(calibration), "n_tail": len(evaluation),
        "tail_metrics": metric_row(tail_y, tail_pred),
        "observed_tail_coverage": float(np.mean(np.abs(tail_y-tail_pred) <= radius)),
        "empirical_radius": radius, "exchangeability_established": False,
        "independent_test_available": False,
        "retrospective_A_metrics": metric_row(a_y.to_numpy(), prediction_a),
        "A_labels_used_in_numeric_training_or_selection": False,
        "A_labels_seen_in_previous_research": True,
        "B_labels_used": False, "deployment_intervals_exported": False,
        "elapsed_sec": time.perf_counter()-started,
        "input_sha256": {p.name: sha256(p) for p in input_files},
        "code_sha256": {p.name: sha256(p) for p in source_files},
        "protocol_sha256": sha256(output / "protocol.json"),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]}}
    json_write(output / "review_manifest.json", manifest)
    log(f"Review validation complete: historical tail MSE={manifest['tail_metrics']['mse']:.6f}")
    return manifest


if __name__ == "__main__":
    run_review_validation()
