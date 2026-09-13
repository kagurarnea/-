from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .pipeline import (
    FoldFeatureBuilder,
    PreparedData,
    RunTrace,
    conformal_quantile,
    log,
    make_models,
    metric_row,
    prepare_data,
)


def validate_released_a_answer(
    answer: pd.DataFrame, expected_ids: pd.Series
) -> pd.Series:
    if answer.shape[1] != 2:
        raise ValueError("A榜答案必须恰好包含ID和Y两列")
    if len(answer) != len(expected_ids):
        raise ValueError(
            f"A榜答案行数错误：期望{len(expected_ids)}，实际{len(answer)}"
        )

    answer_ids = answer.iloc[:, 0].astype(str).reset_index(drop=True)
    expected = expected_ids.astype(str).reset_index(drop=True)
    if not answer_ids.equals(expected):
        mismatch = np.flatnonzero(answer_ids.to_numpy() != expected.to_numpy())
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(f"A榜答案ID顺序与测试A不一致，首个差异行={first + 1}")

    values = pd.to_numeric(answer.iloc[:, 1], errors="coerce").reset_index(drop=True)
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("A榜答案Y包含缺失值或非有限数值")
    values.name = "Y"
    return values


def load_released_a_answer(config: PipelineConfig, data: PreparedData) -> pd.Series:
    path = config.data_dir / config.test_a_answer_file
    if not path.exists():
        raise FileNotFoundError(f"未找到已公布的A榜答案：{path}")
    answer = pd.read_csv(path, header=None)
    return validate_released_a_answer(answer, data.test_a_ids)


def _read_submission(path: Path, expected_ids: pd.Series) -> pd.Series:
    frame = pd.read_csv(path, header=None)
    if frame.shape != (len(expected_ids), 2):
        raise ValueError(f"提交文件结构错误：{path}，实际维度={frame.shape}")
    ids = frame.iloc[:, 0].astype(str).reset_index(drop=True)
    expected = expected_ids.astype(str).reset_index(drop=True)
    if not ids.equals(expected):
        raise ValueError(f"提交文件ID顺序错误：{path}")
    prediction = pd.to_numeric(frame.iloc[:, 1], errors="coerce").reset_index(drop=True)
    if prediction.isna().any() or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError(f"提交文件包含无效预测值：{path}")
    return prediction


def _training_frames(
    data: PreparedData, a_y: pd.Series, a_prefix: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    numeric = pd.concat(
        [data.train_numeric, data.test_a_numeric.iloc[:a_prefix]], ignore_index=True
    )
    categorical = pd.concat(
        [data.train_categorical, data.test_a_categorical.iloc[:a_prefix]],
        ignore_index=True,
    )
    time_frame = pd.concat(
        [data.train_time, data.test_a_time.iloc[:a_prefix]], ignore_index=True
    )
    target = pd.concat([data.y, a_y.iloc[:a_prefix]], ignore_index=True)

    # Competition phase order is used only for the selector's drift penalty.
    # Released A rows are later supervision than the original training block.
    phase_order = pd.Series(np.arange(len(target), dtype=float))
    return numeric, categorical, time_frame, target, phase_order


def _fit_xgb(
    config: PipelineConfig,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    original_train_rows: int,
    a_weight: float,
):
    model = make_models(config, seed)["XGBoost"]
    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[original_train_rows:] = a_weight
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _summary_from_predictions(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, frame in detail.groupby("strategy", sort=False):
        metrics = metric_row(
            frame["y_true"].to_numpy(dtype=float),
            frame["prediction"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "strategy": strategy,
                "a_weight": frame["a_weight"].iloc[0],
                "validation_rows": len(frame),
                **metrics,
                "bias": float((frame["prediction"] - frame["y_true"]).mean()),
                "worst_fold_mse": float(
                    frame.groupby("fold")["squared_error"].mean().max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def _plot_backtest(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    selected_strategy: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    plot = summary.sort_values("mse", ascending=False)
    labels = [
        "Train only" if name == "train_only_robust_blend" else name.rsplit("_x", 1)[-1] + "x A"
        for name in plot["strategy"]
    ]
    colors = [
        "#b42318" if name == "train_only_robust_blend" else
        "#087e8b" if name == selected_strategy else "#8aa6a9"
        for name in plot["strategy"]
    ]
    axes[0].barh(labels, plot["mse"], color=colors)
    axes[0].set_xlabel("Expanding-window MSE (lower is better)")
    axes[0].set_title("Released-A adaptation backtest")
    for position, value in enumerate(plot["mse"]):
        axes[0].text(value + 0.00025, position, f"{value:.5f}", va="center", fontsize=9)

    selected = detail.loc[detail["strategy"] == selected_strategy]
    baseline = detail.loc[detail["strategy"] == "train_only_robust_blend"]
    axes[1].scatter(
        baseline["y_true"], baseline["prediction"], s=22, alpha=0.45,
        color="#b42318", label="Train only",
    )
    axes[1].scatter(
        selected["y_true"], selected["prediction"], s=22, alpha=0.55,
        color="#087e8b", label="Train + released A",
    )
    low = min(detail["y_true"].min(), detail["prediction"].min())
    high = max(detail["y_true"].max(), detail["prediction"].max())
    axes[1].plot([low, high], [low, high], color="#333333", linewidth=1)
    axes[1].set_xlabel("Observed Y")
    axes[1].set_ylabel("Predicted Y")
    axes[1].set_title("Pseudo-future predictions")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phase_b_optimization(
    config: PipelineConfig,
    data: PreparedData | None = None,
    *,
    refresh_train_only_baseline: bool = False,
) -> dict[str, object]:
    """Use released A labels for a validated B-stage domain adaptation."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = data or prepare_data(config, RunTrace())
    a_y = load_released_a_answer(config, data)

    submission_a_path = config.output_dir / "submission_A.csv"
    submission_b_path = config.output_dir / "submission_B.csv"
    train_only_b_path = config.output_dir / "submission_B_train_only.csv"
    if not submission_a_path.exists() or not submission_b_path.exists():
        raise FileNotFoundError("请先运行主流程生成train-only的A/B提交文件")

    baseline_a = _read_submission(submission_a_path, data.test_a_ids)
    if refresh_train_only_baseline or not train_only_b_path.exists():
        baseline_b = _read_submission(submission_b_path, data.test_b_ids)
        pd.DataFrame({"ID": data.test_b_ids, "prediction": baseline_b}).to_csv(
            train_only_b_path, index=False, header=False
        )
        train_only_detail = config.output_dir / "predictions_B_detailed.csv"
        if train_only_detail.exists():
            shutil.copyfile(
                train_only_detail,
                config.output_dir / "predictions_B_train_only.csv",
            )
    else:
        baseline_b = _read_submission(train_only_b_path, data.test_b_ids)

    backtest_rows: list[dict[str, object]] = []
    for fold, (a_prefix, valid_size) in enumerate(
        config.phase_b_backtest_folds, start=1
    ):
        if a_prefix <= 0 or a_prefix + valid_size > len(a_y):
            raise ValueError(
                f"非法B阶段回测窗口：prefix={a_prefix}, valid={valid_size}"
            )
        log(
            f"B阶段扩展窗口 {fold}/{len(config.phase_b_backtest_folds)}："
            f"A训练前缀={a_prefix}，伪未来={valid_size}"
        )
        numeric, categorical, time_frame, target, phase_order = _training_frames(
            data, a_y, a_prefix
        )
        valid_slice = slice(a_prefix, a_prefix + valid_size)
        valid_numeric = data.test_a_numeric.iloc[valid_slice].reset_index(drop=True)
        valid_categorical = data.test_a_categorical.iloc[valid_slice].reset_index(
            drop=True
        )
        valid_time = data.test_a_time.iloc[valid_slice].reset_index(drop=True)
        valid_y = a_y.iloc[valid_slice].to_numpy(dtype=float)
        valid_ids = data.test_a_ids.iloc[valid_slice].reset_index(drop=True)

        strategies: list[tuple[str, float, np.ndarray]] = [
            (
                "train_only_robust_blend",
                0.0,
                baseline_a.iloc[valid_slice].to_numpy(dtype=float),
            )
        ]
        member_predictions = {
            float(a_weight): [] for a_weight in config.phase_b_a_weight_candidates
        }
        for member in range(config.phase_b_ensemble_members):
            seed = config.random_state + 900 + fold + member * 101
            builder = FoldFeatureBuilder(data, config, seed=seed)
            builder.fit(numeric, categorical, time_frame, target, phase_order)
            X_train = builder.transform(numeric, categorical, time_frame)
            X_valid = builder.transform(valid_numeric, valid_categorical, valid_time)
            for a_weight in config.phase_b_a_weight_candidates:
                model = _fit_xgb(
                    config,
                    seed,
                    X_train,
                    target,
                    len(data.y),
                    a_weight,
                )
                member_predictions[float(a_weight)].append(model.predict(X_valid))

        for a_weight in config.phase_b_a_weight_candidates:
            strategy = f"train_plus_released_a_x{a_weight:g}"
            prediction = np.mean(
                np.vstack(member_predictions[float(a_weight)]), axis=0
            )
            strategies.append((strategy, a_weight, prediction))

        for strategy, a_weight, prediction in strategies:
            residual = valid_y - np.asarray(prediction, dtype=float)
            for row_number in range(valid_size):
                backtest_rows.append(
                    {
                        "fold": fold,
                        "a_train_prefix": a_prefix,
                        "ID": valid_ids.iloc[row_number],
                        "y_true": valid_y[row_number],
                        "strategy": strategy,
                        "a_weight": a_weight,
                        "prediction": float(prediction[row_number]),
                        "residual": float(residual[row_number]),
                        "squared_error": float(residual[row_number] ** 2),
                    }
                )

    backtest = pd.DataFrame(backtest_rows)
    summary = _summary_from_predictions(backtest)
    candidate_summary = summary.loc[
        summary["strategy"].str.startswith("train_plus_released_a")
    ].sort_values("mse")
    selected = candidate_summary.iloc[0]
    selected_strategy = str(selected["strategy"])
    selected_weight = float(selected["a_weight"])
    baseline = summary.loc[summary["strategy"] == "train_only_robust_blend"].iloc[0]

    log(
        f"B阶段选择：{selected_strategy}，回测MSE={selected.mse:.6f}，"
        f"train-only={baseline.mse:.6f}"
    )
    numeric, categorical, time_frame, target, phase_order = _training_frames(
        data, a_y, len(a_y)
    )
    final_predictions: list[np.ndarray] = []
    final_feature_counts: list[int] = []
    for member in range(config.phase_b_ensemble_members):
        seed = config.random_state + 950 + member * 101
        final_builder = FoldFeatureBuilder(data, config, seed=seed)
        final_builder.fit(numeric, categorical, time_frame, target, phase_order)
        X_train = final_builder.transform(numeric, categorical, time_frame)
        X_test_b = final_builder.transform(
            data.test_b_numeric, data.test_b_categorical, data.test_b_time
        )
        final_model = _fit_xgb(
            config,
            seed,
            X_train,
            target,
            len(data.y),
            selected_weight,
        )
        final_predictions.append(
            np.asarray(final_model.predict(X_test_b), dtype=float)
        )
        final_feature_counts.append(X_train.shape[1])
    prediction_b = np.mean(np.vstack(final_predictions), axis=0)

    selected_backtest = backtest.loc[backtest["strategy"] == selected_strategy]
    interval_radius = conformal_quantile(
        selected_backtest["residual"].to_numpy(dtype=float), config.conformal_alpha
    )
    prediction_b_detail = pd.DataFrame(
        {
            "ID": data.test_b_ids,
            "prediction": prediction_b,
            "lower_90": prediction_b - interval_radius,
            "upper_90": prediction_b + interval_radius,
            "train_only_prediction": baseline_b.to_numpy(dtype=float),
            "adaptation_delta": prediction_b - baseline_b.to_numpy(dtype=float),
        }
    )

    backtest.to_csv(
        config.output_dir / "phase_b_backtest_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        config.output_dir / "phase_b_strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_b_detail.to_csv(
        config.output_dir / "predictions_B_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_b_detail[["ID", "prediction"]].to_csv(
        submission_b_path, index=False, header=False
    )

    relative_improvement = float((baseline.mse - selected.mse) / baseline.mse)
    manifest = {
        "method": "released_A_expanding_window_domain_adaptation",
        "a_answer_file": config.test_a_answer_file,
        "original_train_rows": len(data.y),
        "released_a_rows": len(a_y),
        "phase_b_training_rows": len(target),
        "test_b_rows": len(data.test_b_ids),
        "validation_rows": int(selected.validation_rows),
        "backtest_folds": [list(item) for item in config.phase_b_backtest_folds],
        "candidate_a_weights": list(config.phase_b_a_weight_candidates),
        "ensemble_members": config.phase_b_ensemble_members,
        "selected_strategy": selected_strategy,
        "selected_a_weight": selected_weight,
        "train_only_backtest_mse": float(baseline.mse),
        "selected_backtest_mse": float(selected.mse),
        "relative_mse_improvement": relative_improvement,
        "selected_backtest_rmse": float(selected.rmse),
        "selected_backtest_mae": float(selected.mae),
        "selected_backtest_r2": float(selected.r2),
        "selected_backtest_bias": float(selected.bias),
        "conformal_radius_90": interval_radius,
        "engineered_features_per_member": final_feature_counts,
        "b_prediction_mean": float(prediction_b.mean()),
        "b_prediction_std": float(prediction_b.std(ddof=1)),
        "mean_absolute_change_vs_train_only": float(
            np.mean(np.abs(prediction_b - baseline_b.to_numpy(dtype=float)))
        ),
        "b_labels_used": False,
        "assessment_role": "legacy_exploratory_file_order_development",
        "baseline_note": "train_only_robust_blend is a legacy key; values come from current submission_A.csv",
        "stable_weight_optimality_established": False,
        "interval_coverage_guarantee": False,
    }
    (config.output_dir / "phase_b_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_backtest(
        summary,
        backtest,
        selected_strategy,
        config.output_dir / "plots" / "29_phase_b_adaptation_backtest.png",
    )

    log(
        f"B阶段提交完成：{submission_b_path}；"
        f"伪未来MSE相对改善={relative_improvement:.1%}"
    )
    return {
        "backtest": backtest,
        "summary": summary,
        "manifest": manifest,
        "prediction_b_detail": prediction_b_detail,
    }


def main() -> None:
    from .review_validation import run_review_validation
    from .review_report import generate_review_report
    log("旧权重开发窗口已退役；执行修订版并等权使用公布A标签重训B。")
    run_review_validation()
    generate_review_report()


if __name__ == "__main__":
    main()
