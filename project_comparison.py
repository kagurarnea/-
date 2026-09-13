# -*- coding: utf-8 -*-

"""汇总现有实验，生成统一口径的项目对比表、统计证据和论文图。

本脚本不重新训练模型，而是读取各实验已经保存的交叉验证结果。这样可以
快速审计当前结论，并明确哪些差异来自同口径 OOF 预测、哪些只是研发阶段
的结果演进。运行方式：

    python project_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "data" / "preprocess_missing"
OUT_DIR = RESULT_ROOT / "project_comparison"
PLOT_DIR = OUT_DIR / "plots"
RANDOM_STATE = 42


def configure_plot_style() -> None:
    """选择可用中文字体并统一科学绘图样式。"""

    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    font = next((name for name in candidates if name in available), "DejaVu Sans")

    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def read_csv(relative_path: str) -> pd.DataFrame:
    path = RESULT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"缺少实验结果文件：{path}")
    return pd.read_csv(path)


def save_figure(name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / name, bbox_inches="tight")
    plt.close()


def percent_improvement(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or reference == 0:
        return np.nan
    return 100.0 * (reference - candidate) / reference


def load_results() -> dict[str, pd.DataFrame]:
    return {
        "missing": read_csv("cv_results/missing_strategy_cv_summary.csv"),
        "feature": read_csv(
            "feature_engineering_results/02_feature_engineering_summary.csv"
        ),
        "feature_folds": read_csv(
            "feature_engineering_results/03_feature_engineering_folds.csv"
        ),
        "model": read_csv("cv_grid_ensemble/model_grid_results.csv"),
        "model_folds": read_csv("cv_grid_ensemble/cv_fold_results.csv"),
        "ensemble": read_csv("cv_grid_ensemble/ensemble_results.csv"),
        "ensemble_oof": read_csv("cv_grid_ensemble/oof_predictions.csv"),
        "feature_ablation": read_csv(
            "feature_diagnosis_experiment/experiment_summary.csv"
        ),
        "tool_feature": read_csv("tool_feature_experiment/experiment_summary.csv"),
        "tool_identity": read_csv("tool_identity_experiment/experiment_summary.csv"),
        "residual": read_csv(
            "residual_correction_experiment/experiment_summary.csv"
        ),
        "residual_lambda": read_csv(
            "residual_correction_experiment/lambda_summary.csv"
        ),
        "residual_folds": read_csv(
            "residual_correction_experiment/experiment_folds.csv"
        ),
        "residual_oof": read_csv(
            "residual_correction_experiment/oof_detail.csv"
        ),
        "hard_sample": read_csv("hard_sample_experiment/experiment_summary.csv"),
        "operation": read_csv(
            "residual_analysis/operation_feature_importance.csv"
        ),
        "tool_error": read_csv("residual_analysis/tool_error_analysis.csv"),
    }


def build_stage_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """提取研发流程中的代表性最优检查点。"""

    missing = data["missing"].sort_values("mse_mean").iloc[0]
    feature = data["feature"].sort_values("mse_mean").iloc[0]
    model = data["model"].sort_values("mse").iloc[0]
    ensemble = data["ensemble"].sort_values("mse").iloc[0]
    residual = data["residual"].sort_values("mse").iloc[0]

    rows = [
        {
            "stage": "缺失值基线",
            "method": str(missing["strategy"]),
            "mse": missing["mse_mean"],
            "mse_std": missing.get("mse_std", np.nan),
            "rmse": missing["rmse_mean"],
            "mae": missing["mae_mean"],
            "r2": missing["r2_mean"],
            "n_features": missing.get("feature_mean", np.nan),
            "source": "missing_strategy_cv_summary.csv",
        },
        {
            "stage": "特征筛选",
            "method": f"{feature['selector']} Top{int(feature['top_k'])}",
            "mse": feature["mse_mean"],
            "mse_std": feature.get("mse_std", np.nan),
            "rmse": feature["rmse_mean"],
            "mae": feature["mae_mean"],
            "r2": feature["r2_mean"],
            "n_features": feature.get("features_mean", np.nan),
            "source": "02_feature_engineering_summary.csv",
        },
        {
            "stage": "模型调优",
            "method": f"{model['model']} config {int(model['config_id'])}",
            "mse": model["mse"],
            "mse_std": model.get("mse_std", np.nan),
            "rmse": model["rmse"],
            "mae": model["mae"],
            "r2": model["r2"],
            "n_features": model.get("features_mean", np.nan),
            "source": "model_grid_results.csv",
        },
        {
            "stage": "模型融合",
            "method": str(ensemble["ensemble"]),
            "mse": ensemble["mse"],
            "mse_std": np.nan,
            "rmse": ensemble["rmse"],
            "mae": ensemble["mae"],
            "r2": ensemble["r2"],
            "n_features": np.nan,
            "source": "ensemble_results.csv",
        },
        {
            "stage": "残差校正",
            "method": f"alpha={residual['alpha']:.2f}",
            "mse": residual["mse"],
            "mse_std": data["residual_lambda"].sort_values("mse").iloc[0][
                "mse_std"
            ],
            "rmse": residual["rmse"],
            "mae": residual["mae"],
            "r2": residual["r2"],
            "n_features": np.nan,
            "source": "residual_correction_experiment/experiment_summary.csv",
        },
    ]

    summary = pd.DataFrame(rows)
    baseline = float(summary.iloc[0]["mse"])
    summary["improvement_vs_first_pct"] = summary["mse"].map(
        lambda value: percent_improvement(baseline, float(value))
    )
    summary["improvement_vs_previous_pct"] = np.nan
    for index in range(1, len(summary)):
        summary.loc[index, "improvement_vs_previous_pct"] = percent_improvement(
            float(summary.loc[index - 1, "mse"]), float(summary.loc[index, "mse"])
        )
    return summary


def add_ablation(
    rows: list[dict[str, object]],
    family: str,
    frame: pd.DataFrame,
    label_column: str,
    mse_column: str,
    reference_label: str,
) -> None:
    reference_rows = frame.loc[frame[label_column] == reference_label]
    if reference_rows.empty:
        return
    reference = reference_rows.iloc[0]
    best = frame.sort_values(mse_column).iloc[0]
    rows.append(
        {
            "comparison": family,
            "reference": reference[label_column],
            "best": best[label_column],
            "reference_mse": reference[mse_column],
            "best_mse": best[mse_column],
            "improvement_pct": percent_improvement(
                float(reference[mse_column]), float(best[mse_column])
            ),
            "supports_best": bool(float(best[mse_column]) < float(reference[mse_column])),
        }
    )


def build_ablation_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    add_ablation(
        rows,
        "缺失值处理",
        data["missing"],
        "strategy",
        "mse_mean",
        "median",
    )

    feature = data["feature"].copy()
    feature["feature_label"] = feature.apply(feature_label, axis=1)
    raw = feature.loc[
        (feature["selector"] == "baseline")
        & (feature["variance_threshold"] == 0)
        & (feature["transform"] == "none")
        & (feature["scaler"] == "none")
        & (feature["reducer"] == "none")
    ]
    if not raw.empty:
        reference = raw.iloc[0]
        best = feature.sort_values("mse_mean").iloc[0]
        rows.append(
            {
                "comparison": "特征工程",
                "reference": reference["feature_label"],
                "best": best["feature_label"],
                "reference_mse": reference["mse_mean"],
                "best_mse": best["mse_mean"],
                "improvement_pct": percent_improvement(
                    float(reference["mse_mean"]), float(best["mse_mean"])
                ),
                "supports_best": True,
            }
        )

    add_ablation(
        rows,
        "工具身份特征",
        data["tool_identity"],
        "experiment",
        "mse",
        "E0_baseline",
    )
    add_ablation(
        rows,
        "工序与异常特征",
        data["feature_ablation"],
        "experiment",
        "mse",
        "E0_baseline",
    )
    add_ablation(
        rows,
        "残差校正",
        data["residual"],
        "experiment",
        "mse",
        "global_baseline",
    )
    add_ablation(
        rows,
        "困难样本专家",
        data["hard_sample"],
        "experiment",
        "mse",
        "E0_global_only",
    )
    return pd.DataFrame(rows)


def feature_label(row: pd.Series) -> str:
    reducer = str(row["reducer"])
    selector = str(row["selector"])
    transform = str(row["transform"])
    scaler = str(row["scaler"])
    threshold = float(row["variance_threshold"])

    if reducer != "none":
        return reducer.upper()
    if selector != "baseline":
        return f"{selector} Top{int(row['top_k'])}"
    if transform != "none":
        return transform
    if scaler != "none":
        return f"{scaler} scaler"
    if threshold > 0:
        return f"variance>{threshold:g}"
    return "原始清洗特征"


def feature_family(row: pd.Series) -> str:
    label = feature_label(row)
    if label.startswith("extra_trees"):
        return "ExtraTrees筛选"
    if label.startswith("rf"):
        return "RF筛选"
    if label.startswith("xgb"):
        return "XGB筛选"
    if label.startswith("variance"):
        return "方差筛选"
    if label.startswith("standard") or label.startswith("minmax"):
        return label
    return label


def plot_stage_progression(summary: pd.DataFrame) -> None:
    frame = summary.iloc[::-1].reset_index(drop=True)
    labels = [f"{stage}\n{method}" for stage, method in zip(frame.stage, frame.method)]
    colors = ["#2f6f9f"] * len(frame)
    colors[0] = "#c44e52"

    plt.figure(figsize=(10.5, 5.8))
    bars = plt.barh(labels, frame["mse"], color=colors, alpha=0.9)
    plt.xlabel("5折交叉验证 MSE（越低越好）")
    plt.title("研发阶段性能演进")
    plt.xlim(0, float(frame["mse"].max()) * 1.18)
    for bar, mse, improvement in zip(
        bars, frame["mse"], frame["improvement_vs_first_pct"]
    ):
        plt.text(
            bar.get_width() + 0.00012,
            bar.get_y() + bar.get_height() / 2,
            f"{mse:.6f}  ({improvement:.1f}%)",
            va="center",
        )
    save_figure("01_stage_progression.png")


def plot_missing_strategies(frame: pd.DataFrame) -> None:
    plot = frame.sort_values("mse_mean", ascending=False).copy()
    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        plot["strategy"],
        plot["mse_mean"],
        xerr=plot["mse_std"],
        color="#4c78a8",
        alpha=0.88,
        capsize=3,
    )
    best_index = int(np.argmin(plot["mse_mean"].to_numpy()))
    bars[best_index].set_color("#c44e52")
    plt.xlabel("MSE均值 ± 折间标准差")
    plt.title("缺失值处理策略对比")
    save_figure("02_missing_strategy_comparison.png")


def plot_feature_ablation(frame: pd.DataFrame) -> None:
    data = frame.copy()
    data["label"] = data.apply(feature_label, axis=1)
    data["family"] = data.apply(feature_family, axis=1)
    representatives = (
        data.sort_values("mse_mean").groupby("family", as_index=False).first()
    )
    representatives = representatives.sort_values("mse_mean", ascending=False)

    plt.figure(figsize=(10.5, 7))
    bars = plt.barh(
        representatives["label"],
        representatives["mse_mean"],
        xerr=representatives["mse_std"],
        color="#59a14f",
        alpha=0.88,
        capsize=3,
    )
    best_index = int(np.argmin(representatives["mse_mean"].to_numpy()))
    bars[best_index].set_color("#c44e52")
    plt.xlabel("MSE均值 ± 折间标准差")
    plt.title("特征工程代表方案消融对比")
    save_figure("03_feature_engineering_ablation.png")


def plot_model_metrics(frame: pd.DataFrame) -> None:
    plot = frame.sort_values("mse").reset_index(drop=True)
    metrics = [
        ("mse", "MSE（越低越好）"),
        ("rmse", "RMSE（越低越好）"),
        ("mae", "MAE（越低越好）"),
        ("r2", "R²（越高越好）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (column, title) in zip(axes.flat, metrics):
        values = plot[column]
        bars = ax.barh(plot["ensemble"], values, color="#4c78a8", alpha=0.86)
        best_index = int(values.idxmin() if column != "r2" else values.idxmax())
        bars[best_index].set_color("#c44e52")
        ax.set_title(title)
        ax.invert_yaxis()
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.5f}",
                va="center",
            )
    fig.suptitle("单模型与融合模型多指标比较", fontsize=14)
    save_figure("04_model_metric_comparison.png")


def best_model_configs(model_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        model_summary.sort_values("mse")
        .groupby("model", as_index=False)
        .first()[["model", "config_id", "mse"]]
    )


def plot_fold_stability(
    model_summary: pd.DataFrame, model_folds: pd.DataFrame
) -> None:
    best = best_model_configs(model_summary)
    labels: list[str] = []
    values: list[np.ndarray] = []
    for row in best.itertuples(index=False):
        fold_values = model_folds.loc[
            (model_folds["model"] == row.model)
            & (model_folds["config_id"] == row.config_id),
            "mse",
        ].to_numpy()
        if fold_values.size:
            labels.append(f"{row.model}\nconfig {int(row.config_id)}")
            values.append(fold_values)

    plt.figure(figsize=(8.5, 6))
    box = plt.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#76b7b2")
        patch.set_alpha(0.72)
    for index, fold_values in enumerate(values, start=1):
        plt.scatter(
            np.full(fold_values.shape, index),
            fold_values,
            color="#2f2f2f",
            s=24,
            alpha=0.75,
            zorder=3,
        )
    plt.ylabel("单折 MSE")
    plt.title("最佳单模型的折间稳定性")
    save_figure("05_fold_stability.png")


def plot_residual_lambda(frame: pd.DataFrame) -> None:
    plot = frame.sort_values("alpha")
    best = plot.sort_values("mse").iloc[0]
    baseline = plot.loc[np.isclose(plot["alpha"], 0.0)].iloc[0]

    plt.figure(figsize=(9.5, 5.8))
    plt.errorbar(
        plot["alpha"],
        plot["mse"],
        yerr=plot["mse_std"],
        marker="o",
        linewidth=2,
        capsize=3,
        color="#4c78a8",
        label="残差校正",
    )
    plt.axhline(
        baseline["mse"], color="#777777", linestyle="--", label="全局模型基线"
    )
    plt.scatter([best["alpha"]], [best["mse"]], color="#c44e52", s=90, zorder=4)
    plt.annotate(
        f"最优 alpha={best['alpha']:.2f}\nMSE={best['mse']:.6f}",
        (best["alpha"], best["mse"]),
        xytext=(12, -38),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    plt.xlabel("残差修正强度 alpha")
    plt.ylabel("MSE均值 ± 折间标准差")
    plt.title("残差校正强度敏感性")
    plt.legend()
    save_figure("06_residual_lambda_sensitivity.png")


def plot_hard_sample_ablation(frame: pd.DataFrame) -> None:
    plot = frame.sort_values("alpha")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(plot["alpha"], plot["mse"], marker="o", color="#4c78a8")
    axes[0].set_xlabel("困难样本专家权重 alpha")
    axes[0].set_ylabel("整体 MSE")
    axes[0].set_title("整体样本")
    axes[1].plot(
        plot["alpha"], plot["tool_o_mse"], marker="s", color="#c44e52"
    )
    axes[1].set_xlabel("困难样本专家权重 alpha")
    axes[1].set_ylabel("Tool O MSE")
    axes[1].set_title("高误差 Tool O 子群")
    fig.suptitle("局部专家负向消融：权重增加后误差上升", fontsize=14)
    save_figure("07_hard_sample_expert_ablation.png")


def plot_prediction_diagnostics(oof: pd.DataFrame) -> None:
    y_true = oof["y_true"].to_numpy(dtype=float)
    pred = oof["ensemble_pred"].to_numpy(dtype=float)
    residual = y_true - pred
    low = float(min(y_true.min(), pred.min()))
    high = float(max(y_true.max(), pred.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(y_true, pred, s=18, alpha=0.48, color="#4c78a8")
    axes[0].plot([low, high], [low, high], linestyle="--", color="#555555")
    axes[0].set_xlabel("真实 Y")
    axes[0].set_ylabel("OOF预测 Y")
    axes[0].set_title("真实值与预测值")

    axes[1].hist(residual, bins=30, color="#59a14f", alpha=0.82)
    axes[1].axvline(0, linestyle="--", color="#555555")
    axes[1].set_xlabel("残差：真实值 - 预测值")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("残差分布")

    axes[2].scatter(pred, residual, s=18, alpha=0.48, color="#f28e2b")
    axes[2].axhline(0, linestyle="--", color="#555555")
    axes[2].set_xlabel("OOF预测 Y")
    axes[2].set_ylabel("残差")
    axes[2].set_title("异方差与系统偏差检查")
    fig.suptitle("最佳加权融合模型的OOF诊断", fontsize=14)
    save_figure("08_oof_prediction_diagnostics.png")


def plot_operation_importance(frame: pd.DataFrame) -> None:
    plot = frame.loc[frame["operation"].astype(str) != "UNKNOWN"].copy()
    plot["importance_share_pct"] = (
        100.0 * plot["total_importance"] / plot["total_importance"].sum()
    )
    plot = plot.nlargest(12, "importance_share_pct").sort_values(
        "importance_share_pct"
    )
    plt.figure(figsize=(9.5, 6.4))
    bars = plt.barh(
        plot["operation"].astype(str),
        plot["importance_share_pct"],
        color="#b07aa1",
        alpha=0.88,
    )
    for bar, value in zip(bars, plot["importance_share_pct"]):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1f}%",
            va="center",
        )
    plt.xlabel("总特征重要性占比")
    plt.ylabel("工序编号")
    plt.title("关键工序贡献排序（Top 12）")
    save_figure("09_operation_importance.png")


def plot_tool_error(frame: pd.DataFrame) -> None:
    plot = frame.sort_values("mse", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].bar(plot["Tool"], plot["mse"], color="#e15759", alpha=0.85)
    axes[0].set_xlabel("Tool")
    axes[0].set_ylabel("OOF MSE")
    axes[0].set_title("设备组误差")
    axes[1].bar(plot["Tool"], plot["n"], color="#4c78a8", alpha=0.85)
    axes[1].set_xlabel("Tool")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("设备组样本量")
    fig.suptitle("分设备鲁棒性检查", fontsize=14)
    save_figure("10_tool_error_and_sample_size.png")


def plot_accuracy_efficiency(
    model_summary: pd.DataFrame, model_folds: pd.DataFrame
) -> pd.DataFrame:
    timing = (
        model_folds.groupby(["model", "config_id"], as_index=False)["time_sec"]
        .mean()
        .rename(columns={"time_sec": "mean_fold_time_sec"})
    )
    joined = model_summary.merge(timing, on=["model", "config_id"], how="left")
    colors = {"XGB": "#4c78a8", "RF": "#59a14f", "ET": "#f28e2b"}

    plt.figure(figsize=(9.5, 6))
    for model, group in joined.groupby("model"):
        plt.scatter(
            group["mean_fold_time_sec"],
            group["mse"],
            s=50,
            alpha=0.68,
            label=model,
            color=colors.get(str(model), "#777777"),
        )
        best = group.sort_values("mse").iloc[0]
        plt.annotate(
            f"{model} best",
            (best["mean_fold_time_sec"], best["mse"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    plt.xlabel("平均单折训练时间（秒）")
    plt.ylabel("5折CV MSE")
    plt.title("精度与训练效率权衡")
    plt.legend()
    save_figure("11_accuracy_efficiency_tradeoff.png")
    return joined


def paired_bootstrap(
    y_true: np.ndarray,
    reference_pred: np.ndarray,
    candidate_pred: np.ndarray,
    n_bootstrap: int = 5000,
) -> dict[str, float]:
    """样本级配对Bootstrap；正差值表示候选模型更优。"""

    y_true = np.asarray(y_true, dtype=float)
    reference_pred = np.asarray(reference_pred, dtype=float)
    candidate_pred = np.asarray(candidate_pred, dtype=float)
    if not (
        len(y_true) == len(reference_pred) == len(candidate_pred)
        and len(y_true) > 0
    ):
        raise ValueError("配对Bootstrap输入长度不一致或为空")

    reference_loss = np.square(y_true - reference_pred)
    candidate_loss = np.square(y_true - candidate_pred)
    loss_delta = reference_loss - candidate_loss
    observed = float(loss_delta.mean())

    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(loss_delta), size=len(loss_delta))
        bootstrap[index] = float(loss_delta[sample].mean())

    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "delta_mse": observed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "bootstrap_positive_probability": float(np.mean(bootstrap > 0)),
        "n_samples": int(len(y_true)),
    }


def build_statistical_comparisons(
    ensemble_oof: pd.DataFrame, residual_oof: pd.DataFrame
) -> pd.DataFrame:
    comparisons: list[dict[str, object]] = []

    result = paired_bootstrap(
        ensemble_oof["y_true"],
        ensemble_oof["XGB_pred"],
        ensemble_oof["ensemble_pred"],
    )
    comparisons.append(
        {
            "comparison": "加权融合 vs XGB",
            "reference": "XGB",
            "candidate": "Weighted ensemble",
            **result,
        }
    )

    result = paired_bootstrap(
        residual_oof["y_true"],
        residual_oof["pred_alpha_0_0"],
        residual_oof["pred_alpha_0_3"],
    )
    comparisons.append(
        {
            "comparison": "残差校正 vs 全局模型",
            "reference": "alpha=0.0",
            "candidate": "alpha=0.3",
            **result,
        }
    )
    return pd.DataFrame(comparisons)


def plot_statistical_comparisons(frame: pd.DataFrame) -> None:
    plot = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    x = plot["delta_mse"].to_numpy()
    lower = x - plot["ci95_low"].to_numpy()
    upper = plot["ci95_high"].to_numpy() - x

    plt.figure(figsize=(9.5, 4.7))
    plt.errorbar(
        x,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=8,
        capsize=5,
        color="#4c78a8",
    )
    plt.axvline(0, color="#777777", linestyle="--")
    plt.yticks(y, plot["comparison"])
    plt.xlabel("参考模型MSE - 候选模型MSE（正值支持候选模型）")
    plt.title("同一样本OOF预测的配对Bootstrap比较")
    for yi, row in plot.iterrows():
        plt.text(
            row["ci95_high"],
            yi,
            f"  P(改善)={row['bootstrap_positive_probability']:.1%}",
            va="center",
        )
    save_figure("12_paired_bootstrap_comparison.png")


def write_report(
    stage: pd.DataFrame,
    ablation: pd.DataFrame,
    statistical: pd.DataFrame,
    n_train: int,
) -> None:
    best = stage.sort_values("mse").iloc[0]
    first = stage.iloc[0]
    ensemble_test = statistical.loc[
        statistical["comparison"] == "加权融合 vs XGB"
    ].iloc[0]
    residual_test = statistical.loc[
        statistical["comparison"] == "残差校正 vs 全局模型"
    ].iloc[0]

    ablation_lines = []
    for row in ablation.itertuples(index=False):
        direction = "改善" if row.improvement_pct > 0 else "未改善"
        ablation_lines.append(
            f"| {row.comparison} | {row.reference} | {row.best} | "
            f"{row.reference_mse:.6f} | {row.best_mse:.6f} | "
            f"{row.improvement_pct:.2f}%（{direction}） |"
        )

    report = f"""# 产品特性预测：统一对比与可视化报告

## 数据口径

当前结果文件包含 **{n_train} 条OOF训练样本**。这与初赛说明中的500条训练样本不是同一数据版本；论文和答辩必须将当前结果标注为后续阶段/扩充训练集结果，不能与初赛榜单直接混报。

本报告读取既有5折交叉验证和OOF预测，不重新训练模型。研发阶段演进图用于展示项目优化过程；严格的优劣判断优先使用同一批OOF样本上的配对比较。

## 当前结论

- 当前记录的最低MSE为 **{best.mse:.6f}**（{best.stage}，{best.method}），相对首个缺失值检查点 {first.mse:.6f} 下降 **{percent_improvement(float(first.mse), float(best.mse)):.2f}%**。
- 加权融合相对最佳XGB的MSE改善为 **{ensemble_test.delta_mse:.8f}**，样本级配对Bootstrap的95%区间为 **[{ensemble_test.ci95_low:.8f}, {ensemble_test.ci95_high:.8f}]**。
- alpha=0.3残差校正相对全局模型的MSE改善为 **{residual_test.delta_mse:.8f}**，样本级配对Bootstrap的95%区间为 **[{residual_test.ci95_low:.8f}, {residual_test.ci95_high:.8f}]**。
- 困难样本Tool O专用专家随着权重增加而恶化，因此当前证据支持保留全局模型，而不是为了“创新”强行加入局部专家。
- Bootstrap为样本级描述性证据；若ID代表生产时间，最终论文应补充按批次的Block Bootstrap和时间外推验证。

## 消融结论表

| 比较环节 | 参考方案 | 当前最优 | 参考MSE | 最优MSE | 相对变化 |
|---|---|---|---:|---:|---:|
{chr(10).join(ablation_lines)}

## 图表索引

1. [研发阶段性能演进](plots/01_stage_progression.png)
2. [缺失值处理策略对比](plots/02_missing_strategy_comparison.png)
3. [特征工程代表方案消融](plots/03_feature_engineering_ablation.png)
4. [单模型与融合模型多指标比较](plots/04_model_metric_comparison.png)
5. [最佳单模型折间稳定性](plots/05_fold_stability.png)
6. [残差校正强度敏感性](plots/06_residual_lambda_sensitivity.png)
7. [困难样本专家负向消融](plots/07_hard_sample_expert_ablation.png)
8. [OOF预测与残差诊断](plots/08_oof_prediction_diagnostics.png)
9. [关键工序贡献排序](plots/09_operation_importance.png)
10. [分设备误差与样本量](plots/10_tool_error_and_sample_size.png)
11. [精度与训练效率权衡](plots/11_accuracy_efficiency_tradeoff.png)
12. [同样本配对Bootstrap比较](plots/12_paired_bootstrap_comparison.png)

## “最优解”判定规则

最终方案不能只满足平均MSE最低，还应同时满足：主指标MSE最低或差异区间支持改善；折间标准差和最差折不明显恶化；MAE与R²方向一致；关键工序排名跨折稳定；训练与单样本推理耗时满足部署要求。若复杂模块只带来极小改善且置信区间跨0，应优先选择更简单的方案。
"""
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    stage = build_stage_summary(data)
    ablation = build_ablation_summary(data)
    statistical = build_statistical_comparisons(
        data["ensemble_oof"], data["residual_oof"]
    )

    stage.to_csv(OUT_DIR / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(OUT_DIR / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    statistical.to_csv(
        OUT_DIR / "statistical_comparison.csv", index=False, encoding="utf-8-sig"
    )

    plot_stage_progression(stage)
    plot_missing_strategies(data["missing"])
    plot_feature_ablation(data["feature"])
    plot_model_metrics(data["ensemble"])
    plot_fold_stability(data["model"], data["model_folds"])
    plot_residual_lambda(data["residual_lambda"])
    plot_hard_sample_ablation(data["hard_sample"])
    plot_prediction_diagnostics(data["ensemble_oof"])
    plot_operation_importance(data["operation"])
    plot_tool_error(data["tool_error"])
    efficiency = plot_accuracy_efficiency(data["model"], data["model_folds"])
    efficiency.to_csv(
        OUT_DIR / "accuracy_efficiency.csv", index=False, encoding="utf-8-sig"
    )
    plot_statistical_comparisons(statistical)
    write_report(
        stage,
        ablation,
        statistical,
        n_train=len(data["ensemble_oof"]),
    )

    print(f"统一对比报告：{OUT_DIR / 'REPORT.md'}")
    print(f"汇总表：{OUT_DIR / 'comparison_summary.csv'}")
    print(f"图表目录：{PLOT_DIR}")


if __name__ == "__main__":
    main()
