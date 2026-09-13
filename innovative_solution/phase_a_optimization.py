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
from .phase_b_optimization import load_released_a_answer
from .pipeline import (
    FoldFeatureBuilder,
    PreparedData,
    RunTrace,
    conformal_quantile,
    log,
    make_models,
    metric_row,
    prepare_data,
    recency_weights,
)


def _strategy_name(half_life: float | None) -> str:
    return "uniform_xgb" if half_life is None else f"recency_half_{half_life:g}"


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


def _fit_model(
    config: PipelineConfig,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    phase_order: pd.Series,
    half_life: float | None,
):
    model = make_models(config, seed)["XGBoost"]
    sample_weight = (
        None if half_life is None else recency_weights(phase_order, half_life)
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, frame in detail.groupby("strategy", sort=False):
        metrics = metric_row(
            frame["y_true"].to_numpy(dtype=float),
            frame["prediction"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "strategy": strategy,
                "half_life_fraction": frame["half_life_fraction"].iloc[0],
                "validation_rows": len(frame),
                **metrics,
                "bias": float((frame["prediction"] - frame["y_true"]).mean()),
                "worst_fold_mse": float(
                    frame.groupby("fold")["squared_error"].mean().max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def _plot_phase_a(
    summary: pd.DataFrame,
    a_y: pd.Series,
    baseline: pd.Series,
    optimized: np.ndarray,
    selected_strategy: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    plot = summary.sort_values("mse", ascending=False)
    colors = [
        "#087e8b" if name == selected_strategy else "#8aa6a9"
        for name in plot["strategy"]
    ]
    axes[0].barh(plot["strategy"], plot["mse"], color=colors)
    axes[0].set_xlabel("Expanding-window MSE (lower is better)")
    axes[0].set_title("Training-only strategy selection for A")
    for position, value in enumerate(plot["mse"]):
        axes[0].text(value + 0.00012, position, f"{value:.5f}", va="center", fontsize=9)

    y_true = a_y.to_numpy(dtype=float)
    axes[1].scatter(
        y_true, baseline, s=22, alpha=0.42, color="#b42318", label="RobustBlend baseline"
    )
    axes[1].scatter(
        y_true, optimized, s=22, alpha=0.52, color="#087e8b", label="Selected train-only ensemble"
    )
    low = min(y_true.min(), baseline.min(), optimized.min())
    high = max(y_true.max(), baseline.max(), optimized.max())
    axes[1].plot([low, high], [low, high], color="#333333", linewidth=1)
    axes[1].set_xlabel("Published A answer Y (post-hoc only)")
    axes[1].set_ylabel("Predicted Y")
    axes[1].set_title("External A evaluation after strategy lock")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phase_a_optimization(
    config: PipelineConfig,
    data: PreparedData | None = None,
    *,
    refresh_train_only_baseline: bool = False,
) -> dict[str, object]:
    """Select and fit the A-stage model without using any A labels."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = data or prepare_data(config, RunTrace())
    submission_path = config.output_dir / "submission_A.csv"
    baseline_path = config.output_dir / "submission_A_train_only_blend.csv"
    if not submission_path.exists():
        raise FileNotFoundError("请先运行主流程生成A阶段RobustBlend基线")

    if refresh_train_only_baseline or not baseline_path.exists():
        baseline = _read_submission(submission_path, data.test_a_ids)
        pd.DataFrame({"ID": data.test_a_ids, "prediction": baseline}).to_csv(
            baseline_path, index=False, header=False
        )
        baseline_detail = config.output_dir / "predictions_A_detailed.csv"
        if baseline_detail.exists():
            shutil.copyfile(
                baseline_detail,
                config.output_dir / "predictions_A_blend_baseline.csv",
            )
    else:
        baseline = _read_submission(baseline_path, data.test_a_ids)

    backtest_rows: list[dict[str, object]] = []
    for fold, (train_size, valid_size) in enumerate(
        config.phase_a_backtest_folds, start=1
    ):
        if train_size <= 0 or train_size + valid_size > len(data.y):
            raise ValueError(
                f"非法A阶段回测窗口：train={train_size}, valid={valid_size}"
            )
        log(
            f"A阶段扩展窗口 {fold}/{len(config.phase_a_backtest_folds)}："
            f"train={train_size}，future={valid_size}"
        )
        train_slice = slice(0, train_size)
        valid_slice = slice(train_size, train_size + valid_size)
        phase_order = pd.Series(np.arange(train_size, dtype=float))
        valid_y = data.y.iloc[valid_slice].to_numpy(dtype=float)
        valid_ids = data.train_ids.iloc[valid_slice].reset_index(drop=True)
        predictions = {
            half_life: [] for half_life in config.phase_a_recency_candidates
        }

        for member in range(config.phase_a_ensemble_members):
            seed = config.random_state + 1200 + fold + member * 101
            builder = FoldFeatureBuilder(data, config, seed=seed)
            builder.fit(
                data.train_numeric.iloc[train_slice],
                data.train_categorical.iloc[train_slice],
                data.train_time.iloc[train_slice],
                data.y.iloc[train_slice],
                phase_order,
            )
            X_train = builder.transform(
                data.train_numeric.iloc[train_slice],
                data.train_categorical.iloc[train_slice],
                data.train_time.iloc[train_slice],
            )
            X_valid = builder.transform(
                data.train_numeric.iloc[valid_slice],
                data.train_categorical.iloc[valid_slice],
                data.train_time.iloc[valid_slice],
            )
            for half_life in config.phase_a_recency_candidates:
                model = _fit_model(
                    config,
                    seed,
                    X_train,
                    data.y.iloc[train_slice],
                    phase_order,
                    half_life,
                )
                predictions[half_life].append(model.predict(X_valid))

        for half_life, member_predictions in predictions.items():
            strategy = _strategy_name(half_life)
            prediction = np.mean(np.vstack(member_predictions), axis=0)
            residual = valid_y - prediction
            for row_number in range(valid_size):
                backtest_rows.append(
                    {
                        "fold": fold,
                        "train_rows": train_size,
                        "ID": valid_ids.iloc[row_number],
                        "y_true": valid_y[row_number],
                        "strategy": strategy,
                        "half_life_fraction": half_life,
                        "prediction": float(prediction[row_number]),
                        "residual": float(residual[row_number]),
                        "squared_error": float(residual[row_number] ** 2),
                    }
                )

    backtest = pd.DataFrame(backtest_rows)
    summary = _summarize(backtest)
    selected = summary.iloc[0]
    selected_strategy = str(selected["strategy"])
    selected_half_life = next(
        candidate
        for candidate in config.phase_a_recency_candidates
        if _strategy_name(candidate) == selected_strategy
    )
    log(
        f"A阶段选择：{selected_strategy}，训练内伪未来MSE={selected.mse:.6f}"
    )

    final_predictions: list[np.ndarray] = []
    final_feature_counts: list[int] = []
    phase_order = pd.Series(np.arange(len(data.y), dtype=float))
    for member in range(config.phase_a_ensemble_members):
        seed = config.random_state + 1600 + member * 101
        builder = FoldFeatureBuilder(data, config, seed=seed)
        builder.fit(
            data.train_numeric,
            data.train_categorical,
            data.train_time,
            data.y,
            phase_order,
        )
        X_train = builder.transform(
            data.train_numeric, data.train_categorical, data.train_time
        )
        X_test_a = builder.transform(
            data.test_a_numeric, data.test_a_categorical, data.test_a_time
        )
        model = _fit_model(
            config,
            seed,
            X_train,
            data.y,
            phase_order,
            selected_half_life,
        )
        final_predictions.append(np.asarray(model.predict(X_test_a), dtype=float))
        final_feature_counts.append(X_train.shape[1])
    prediction_a = np.mean(np.vstack(final_predictions), axis=0)

    selected_backtest = backtest.loc[backtest["strategy"] == selected_strategy]
    interval_radius = conformal_quantile(
        selected_backtest["residual"].to_numpy(dtype=float), config.conformal_alpha
    )

    # The released answer is loaded only after strategy selection and final fitting.
    a_y = load_released_a_answer(config, data)
    external_rows: list[dict[str, object]] = []
    for strategy, prediction in (
        ("robust_blend_baseline", baseline.to_numpy(dtype=float)),
        (selected_strategy, prediction_a),
    ):
        external_rows.append(
            {
                "strategy": strategy,
                **metric_row(a_y.to_numpy(dtype=float), prediction),
                "bias": float(np.mean(prediction - a_y.to_numpy(dtype=float))),
                "prediction_std": float(np.std(prediction, ddof=1)),
            }
        )
    external = pd.DataFrame(external_rows).sort_values("mse").reset_index(drop=True)
    baseline_external = external.loc[
        external["strategy"] == "robust_blend_baseline"
    ].iloc[0]
    selected_external = external.loc[external["strategy"] == selected_strategy].iloc[0]

    detail = pd.DataFrame(
        {
            "ID": data.test_a_ids,
            "prediction": prediction_a,
            "lower_90": prediction_a - interval_radius,
            "upper_90": prediction_a + interval_radius,
            "blend_baseline_prediction": baseline.to_numpy(dtype=float),
            "optimization_delta": prediction_a - baseline.to_numpy(dtype=float),
            "published_y": a_y.to_numpy(dtype=float),
            "residual": a_y.to_numpy(dtype=float) - prediction_a,
        }
    )

    backtest.to_csv(
        config.output_dir / "phase_a_backtest_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        config.output_dir / "phase_a_strategy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    external.to_csv(
        config.output_dir / "phase_a_external_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        config.output_dir / "predictions_A_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail[["ID", "prediction"]].to_csv(
        submission_path, index=False, header=False
    )

    external_improvement = float(
        (baseline_external.mse - selected_external.mse) / baseline_external.mse
    )
    manifest = {
        "method": "training_only_expanding_window_seed_ensemble",
        "training_rows": len(data.y),
        "test_a_rows": len(data.test_a_ids),
        "backtest_folds": [list(item) for item in config.phase_a_backtest_folds],
        "validation_rows": int(selected.validation_rows),
        "candidate_half_life_fractions": list(config.phase_a_recency_candidates),
        "ensemble_members": config.phase_a_ensemble_members,
        "selected_strategy": selected_strategy,
        "selected_half_life_fraction": selected_half_life,
        "selected_backtest_mse": float(selected.mse),
        "selected_backtest_rmse": float(selected.rmse),
        "selected_backtest_mae": float(selected.mae),
        "selected_backtest_r2": float(selected.r2),
        "selected_backtest_bias": float(selected.bias),
        "baseline_external_a_mse": float(baseline_external.mse),
        "selected_external_a_mse": float(selected_external.mse),
        "selected_external_a_rmse": float(selected_external.rmse),
        "selected_external_a_mae": float(selected_external.mae),
        "selected_external_a_r2": float(selected_external.r2),
        "selected_external_a_bias": float(selected_external.bias),
        "external_a_relative_mse_improvement": external_improvement,
        "conformal_radius_90": interval_radius,
        "engineered_features_per_member": final_feature_counts,
        "a_labels_used_for_selection_or_fit": False,
        "assessment_role": "legacy_exploratory_file_order_development",
        "A_labels_seen_in_prior_research": True,
        "independent_test": False,
        "interval_coverage_guarantee": False,
        "a_labels_used_for_posthoc_evaluation": True,
    }
    (config.output_dir / "phase_a_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_phase_a(
        summary,
        a_y,
        baseline,
        prediction_a,
        selected_strategy,
        config.output_dir / "plots" / "30_phase_a_training_only_backtest.png",
    )
    log(
        f"A阶段提交完成：{submission_path}；事后A MSE={selected_external.mse:.6f}，"
        f"相对原融合改善={external_improvement:.1%}"
    )
    return {
        "backtest": backtest,
        "summary": summary,
        "external": external,
        "manifest": manifest,
        "prediction_a_detail": detail,
    }


def main() -> None:
    from .review_validation import run_review_validation
    from .review_report import generate_review_report
    log("原文件顺序窗口已退役；执行分组隔离的修订版验证并输出review目录。")
    run_review_validation()
    generate_review_report()


if __name__ == "__main__":
    main()
