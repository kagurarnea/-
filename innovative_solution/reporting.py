from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .pipeline import PreparedData


COLORS = {
    "blue": "#4c78a8",
    "orange": "#f28e2b",
    "green": "#59a14f",
    "red": "#e15759",
    "purple": "#b07aa1",
    "teal": "#76b7b2",
    "gray": "#777777",
}


def configure_style() -> None:
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


def save(plot_dir: Path, name: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_dir / name, bbox_inches="tight")
    plt.close()


def plot_data_audit(data: PreparedData, plot_dir: Path) -> None:
    frame = data.operation_audit.loc[
        data.operation_audit["operation"].astype(str) != "META"
    ].copy()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(
        x,
        frame["kept_numeric"],
        label="保留数值特征",
        color=COLORS["blue"],
    )
    axes[0].bar(
        x,
        frame["time_features"],
        bottom=frame["kept_numeric"],
        label="重构时间特征",
        color=COLORS["orange"],
    )
    axes[0].set_ylabel("特征数量")
    axes[0].set_title("各工序的数据规模与时间字段")
    axes[0].legend()

    axes[1].plot(
        x,
        frame["mean_missing_rate"] * 100,
        marker="o",
        label="平均缺失率",
        color=COLORS["red"],
    )
    axes[1].plot(
        x,
        frame["mean_zero_rate"] * 100,
        marker="s",
        label="平均零值率",
        color=COLORS["green"],
    )
    axes[1].set_ylabel("占比（%）")
    axes[1].set_xlabel("工序编号")
    axes[1].set_xticks(x, frame["operation"].astype(str), rotation=35)
    axes[1].legend()
    save(plot_dir, "01_data_audit_by_operation.png")


def plot_target_split(
    data: PreparedData, oof: pd.DataFrame, holdout: pd.DataFrame, plot_dir: Path
) -> None:
    plt.figure(figsize=(10, 5.8))
    plt.hist(
        oof["y_true"],
        bins=25,
        density=True,
        alpha=0.58,
        label=f"开发集（n={len(oof)}）",
        color=COLORS["blue"],
    )
    plt.hist(
        holdout["y_true"],
        bins=18,
        density=True,
        alpha=0.55,
        label=f"时间外推集（n={len(holdout)}）",
        color=COLORS["orange"],
    )
    plt.axvline(data.y.mean(), color=COLORS["gray"], linestyle="--", label="全体均值")
    plt.xlabel("产品特性值 Y")
    plt.ylabel("密度")
    plt.title("开发集与锁定时间外推集的目标分布")
    plt.legend()
    save(plot_dir, "02_target_and_temporal_split.png")


def plot_cv_models(fold_metrics: pd.DataFrame, plot_dir: Path) -> None:
    order = (
        fold_metrics.loc[fold_metrics.validation_scheme == "rolling_cv"]
        .groupby("model")["mse"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    schemes = [
        ("random_cv", "随机五折CV"),
        ("rolling_cv", "滚动时间CV"),
    ]
    for ax, (scheme, title) in zip(axes, schemes):
        subset = fold_metrics.loc[fold_metrics.validation_scheme == scheme]
        values = [
            subset.loc[subset.model == model, "mse"].to_numpy() for model in order
        ]
        box = ax.boxplot(values, tick_labels=order, patch_artist=True, showmeans=True)
        mean_values = [float(np.mean(value)) for value in values]
        best_index = int(np.argmin(mean_values))
        for index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(
                COLORS["red"] if index == best_index else COLORS["teal"]
            )
            patch.set_alpha(0.78)
        for index, points in enumerate(values, start=1):
            ax.scatter(
                np.full(len(points), index),
                points,
                s=25,
                color="#303030",
                alpha=0.7,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xlabel("候选模型")
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("单折 MSE")
    fig.suptitle("随机精度与时间泛化的双验证", fontsize=14)
    save(plot_dir, "03_cv_model_stability.png")


def plot_oof_vs_holdout(model_summary: pd.DataFrame, plot_dir: Path) -> None:
    pivot = model_summary.pivot(index="model", columns="scope", values="mse")
    pivot = pivot.sort_values("temporal_holdout")
    x = np.arange(len(pivot))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width,
        pivot["random_oof"],
        width,
        label="随机OOF",
        color=COLORS["blue"],
    )
    plt.bar(
        x,
        pivot["rolling_oof"],
        width,
        label="滚动时间OOF",
        color=COLORS["green"],
    )
    plt.bar(
        x + width,
        pivot["temporal_holdout"],
        width,
        label="锁定时间外推集",
        color=COLORS["orange"],
    )
    plt.xticks(x, pivot.index, rotation=20)
    plt.ylabel("MSE（越低越好）")
    plt.title("随机交叉验证与时间外推性能对比")
    plt.legend()
    save(plot_dir, "04_oof_vs_temporal_holdout.png")


def plot_blend_weights(weights: pd.DataFrame, plot_dir: Path) -> None:
    frame = weights.sort_values("weight")
    plt.figure(figsize=(8.5, 4.8))
    bars = plt.barh(frame["model"], frame["weight"] * 100, color=COLORS["purple"])
    for bar, value in zip(bars, frame["weight"]):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1%}",
            va="center",
        )
    plt.xlabel("非负约束融合权重（%）")
    plt.title("开发集OOF学习得到的模型融合结构")
    save(plot_dir, "05_blend_weights.png")


def plot_prediction_correlation(oof: pd.DataFrame, plot_dir: Path) -> None:
    prediction_columns = [
        column
        for column in oof.columns
        if column.startswith("pred_") and column != "pred_RobustBlend"
    ]
    labels = [column.replace("pred_", "") for column in prediction_columns]
    correlation = oof[prediction_columns].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(correlation, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            color = "white" if correlation[row, column] > 0.78 else "black"
            ax.text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    fig.colorbar(image, ax=ax, label="OOF预测相关系数")
    ax.set_title("模型互补性检查")
    save(plot_dir, "06_model_prediction_correlation.png")


def plot_holdout_prediction(holdout: pd.DataFrame, plot_dir: Path) -> None:
    y_true = holdout["y_true"].to_numpy(dtype=float)
    prediction = holdout["pred_RobustBlend"].to_numpy(dtype=float)
    lower = holdout["lower_90"].to_numpy(dtype=float)
    upper = holdout["upper_90"].to_numpy(dtype=float)
    low = float(min(y_true.min(), lower.min()))
    high = float(max(y_true.max(), upper.max()))

    plt.figure(figsize=(7.2, 6.4))
    covered = holdout["covered_90"].astype(bool).to_numpy()
    plt.scatter(
        y_true[covered],
        prediction[covered],
        s=28,
        alpha=0.65,
        color=COLORS["blue"],
        label="区间覆盖",
    )
    plt.scatter(
        y_true[~covered],
        prediction[~covered],
        s=42,
        marker="x",
        color=COLORS["red"],
        label="区间未覆盖",
    )
    plt.plot([low, high], [low, high], linestyle="--", color=COLORS["gray"])
    plt.xlabel("真实 Y")
    plt.ylabel("时间外推预测 Y")
    plt.title("锁定时间外推集：真实值与预测值")
    plt.legend()
    save(plot_dir, "07_temporal_holdout_prediction.png")


def plot_residual_diagnostics(holdout: pd.DataFrame, plot_dir: Path) -> None:
    prediction = holdout["pred_RobustBlend"].to_numpy(dtype=float)
    residual = holdout["residual_RobustBlend"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(residual, bins=22, color=COLORS["green"], alpha=0.82)
    axes[0].axvline(0, linestyle="--", color=COLORS["gray"])
    axes[0].set_xlabel("残差：真实值 - 预测值")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("时间外推残差分布")
    axes[1].scatter(prediction, residual, s=26, alpha=0.65, color=COLORS["orange"])
    axes[1].axhline(0, linestyle="--", color=COLORS["gray"])
    axes[1].set_xlabel("预测 Y")
    axes[1].set_ylabel("残差")
    axes[1].set_title("系统偏差与异方差检查")
    save(plot_dir, "08_temporal_residual_diagnostics.png")


def plot_tool_error(holdout: pd.DataFrame, plot_dir: Path) -> None:
    if "Tool" not in holdout.columns:
        return
    frame = holdout.assign(
        squared_error=np.square(holdout["residual_RobustBlend"]),
        absolute_error=np.abs(holdout["residual_RobustBlend"]),
    )
    grouped = (
        frame.groupby("Tool", as_index=False)
        .agg(
            mse=("squared_error", "mean"),
            mae=("absolute_error", "mean"),
            n=("ID", "size"),
        )
        .sort_values("mse", ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(grouped["Tool"], grouped["mse"], color=COLORS["red"], alpha=0.84)
    axes[0].set_xlabel("Tool")
    axes[0].set_ylabel("时间外推 MSE")
    axes[0].set_title("分设备误差")
    axes[1].bar(grouped["Tool"], grouped["n"], color=COLORS["blue"], alpha=0.84)
    axes[1].set_xlabel("Tool")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("分设备样本量")
    save(plot_dir, "09_holdout_error_by_tool.png")


def plot_feature_stability(frame: pd.DataFrame, plot_dir: Path) -> None:
    plot = frame.head(20).sort_values("mean_stable_score")
    plt.figure(figsize=(10, 7.2))
    scatter = plt.scatter(
        plot["mean_stable_score"],
        np.arange(len(plot)),
        s=50 + 170 * plot["selection_rate"],
        c=plot["mean_temporal_drift"],
        cmap="OrRd",
        alpha=0.85,
    )
    plt.yticks(np.arange(len(plot)), plot["feature"])
    plt.xlabel("平均稳定性评分")
    plt.ylabel("原始特征")
    plt.title("Top 20稳定特征：大小=入选率，颜色=时间漂移")
    plt.colorbar(scatter, label="平均时间漂移")
    save(plot_dir, "10_feature_selection_stability.png")


def plot_operation_importance(frame: pd.DataFrame, plot_dir: Path) -> None:
    plot = frame.loc[~frame["operation"].isin(["META"])].head(14).copy()
    total = float(plot["mean_importance"].sum()) or 1.0
    plot["share"] = 100.0 * plot["mean_importance"] / total
    plot = plot.sort_values("share")
    plt.figure(figsize=(9.2, 6.2))
    plt.barh(plot["operation"], plot["share"], color=COLORS["purple"], alpha=0.85)
    plt.xlabel("Top工序重要性归一化占比（%）")
    plt.ylabel("工序编号")
    plt.title("跨折聚合的关键工序贡献")
    save(plot_dir, "11_operation_importance.png")


def plot_operation_drift(frame: pd.DataFrame, plot_dir: Path) -> None:
    plot = frame.head(14).sort_values("mean_shift")
    plt.figure(figsize=(9.2, 6.2))
    bars = plt.barh(
        plot["operation"], plot["mean_shift"], color=COLORS["orange"], alpha=0.85
    )
    for bar, maximum in zip(bars, plot["max_shift"]):
        plt.plot(
            maximum,
            bar.get_y() + bar.get_height() / 2,
            marker="|",
            markersize=14,
            color=COLORS["red"],
        )
    plt.xlabel("训练集与A/B测试集的稳健分布漂移")
    plt.ylabel("工序编号")
    plt.xscale("symlog", linthresh=0.05)
    plt.title("工序漂移（对数型坐标）：柱=平均，竖线=最大")
    save(plot_dir, "12_operation_distribution_drift.png")


def plot_conformal(frame: pd.DataFrame, plot_dir: Path) -> None:
    row = frame.iloc[0]
    labels = ["目标覆盖率", "时间外推实际覆盖率"]
    values = [row["nominal_coverage"] * 100, row["empirical_holdout_coverage"] * 100]
    plt.figure(figsize=(7.5, 4.8))
    bars = plt.bar(labels, values, color=[COLORS["blue"], COLORS["green"]], width=0.55)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    plt.ylim(0, 105)
    plt.ylabel("覆盖率（%）")
    plt.title(f"共形预测区间校准（平均宽度={row['mean_interval_width']:.3f}）")
    save(plot_dir, "13_conformal_interval_coverage.png")


def plot_efficiency(
    fold_metrics: pd.DataFrame, model_summary: pd.DataFrame, plot_dir: Path
) -> None:
    timing = (
        fold_metrics.loc[fold_metrics.validation_scheme == "rolling_cv"]
        .groupby("model", as_index=False)["fit_predict_sec"]
        .mean()
    )
    mse = model_summary.loc[model_summary.scope == "rolling_oof", ["model", "mse"]]
    frame = timing.merge(mse, on="model")
    plt.figure(figsize=(8.5, 5.8))
    plt.scatter(
        frame["fit_predict_sec"],
        frame["mse"],
        s=85,
        color=COLORS["blue"],
        alpha=0.8,
    )
    for row in frame.itertuples(index=False):
        plt.annotate(
            row.model,
            (row.fit_predict_sec, row.mse),
            xytext=(6, 5),
            textcoords="offset points",
        )
    plt.xlabel("平均单折拟合与预测时间（秒）")
    plt.ylabel("滚动时间OOF MSE")
    plt.title("时间泛化精度与训练效率权衡")
    save(plot_dir, "14_accuracy_efficiency.png")


def plot_run_trace(frame: pd.DataFrame, plot_dir: Path) -> None:
    plot = frame.loc[~frame["stage"].eq("完整新方案")].copy()
    plot = plot.sort_values("elapsed_sec")
    plt.figure(figsize=(10, 6.3))
    bars = plt.barh(plot["stage"], plot["elapsed_sec"], color=COLORS["teal"])
    for bar, value in zip(bars, plot["elapsed_sec"]):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1f}s",
            va="center",
        )
    plt.xlabel("耗时（秒）")
    plt.title("可复现工作痕迹：各流水线阶段耗时")
    save(plot_dir, "15_pipeline_run_trace.png")


def plot_error_concentration(holdout: pd.DataFrame, plot_dir: Path) -> None:
    frame = holdout.copy()
    frame["squared_error"] = np.square(frame["residual_RobustBlend"])
    frame = frame.sort_values("squared_error", ascending=False).reset_index(drop=True)
    total = float(frame["squared_error"].sum()) or 1.0
    frame["cumulative_share"] = frame["squared_error"].cumsum() / total
    top = frame.head(12).sort_values("squared_error")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(
        np.arange(1, len(frame) + 1),
        frame["cumulative_share"] * 100,
        color=COLORS["blue"],
        linewidth=2,
    )
    axes[0].axhline(50, color=COLORS["gray"], linestyle="--")
    axes[0].set_xlabel("按平方误差从高到低累计的样本数")
    axes[0].set_ylabel("累计MSE贡献（%）")
    axes[0].set_title("误差是否集中于少数样本")

    bars = axes[1].barh(
        top["ID"].astype(str), top["squared_error"], color=COLORS["red"], alpha=0.82
    )
    for bar, value in zip(bars, top["squared_error"] / total):
        axes[1].text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1%}",
            va="center",
        )
    axes[1].set_xlabel("平方误差")
    axes[1].set_title("Top 12高误差样本及MSE贡献")
    save(plot_dir, "16_error_concentration.png")


def markdown_model_table(summary: pd.DataFrame) -> str:
    pivot = summary.pivot(index="model", columns="scope", values=["mse", "rmse", "mae", "r2"])
    rows = []
    for model in pivot.index:
        rows.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.4f} |".format(
                model,
                pivot.loc[model, ("mse", "random_oof")],
                pivot.loc[model, ("mse", "rolling_oof")],
                pivot.loc[model, ("mse", "temporal_holdout")],
                pivot.loc[model, ("rmse", "temporal_holdout")],
                pivot.loc[model, ("mae", "temporal_holdout")],
                pivot.loc[model, ("r2", "temporal_holdout")],
            )
        )
    return "\n".join(rows)


def write_report(results: dict[str, object], config: PipelineConfig) -> None:
    manifest = results["manifest"]
    summary: pd.DataFrame = results["model_summary"]
    weights: pd.DataFrame = results["blend_weights"]
    importance: pd.DataFrame = results["operation_importance"]
    drift: pd.DataFrame = results["operation_drift"]
    conformal: pd.DataFrame = results["conformal_metrics"]

    random_result = summary.loc[summary.scope == "random_oof"].sort_values("mse")
    rolling_result = summary.loc[summary.scope == "rolling_oof"].sort_values("mse")
    holdout = summary.loc[summary.scope == "temporal_holdout"].sort_values("mse")
    best_random = random_result.iloc[0]
    best_rolling = rolling_result.iloc[0]
    best_holdout = holdout.iloc[0]
    blend_holdout = holdout.loc[holdout.model == "RobustBlend"].iloc[0]
    weight_text = "，".join(
        f"{row.model}={row.weight:.1%}" for row in weights.itertuples(index=False)
    )
    top_operations = "、".join(importance.head(5)["operation"].astype(str))
    drift_operations = "、".join(drift.head(5)["operation"].astype(str))
    conformal_row = conformal.iloc[0]
    holdout_detail: pd.DataFrame = results["holdout_detail"]
    error_frame = holdout_detail.assign(
        squared_error=np.square(holdout_detail["residual_RobustBlend"])
    ).sort_values("squared_error", ascending=False)
    worst = error_frame.iloc[0]
    worst_share = float(worst.squared_error / error_frame.squared_error.sum())
    mse_without_worst = float(error_frame.iloc[1:]["squared_error"].mean())
    plot_count = len(list((config.output_dir / "plots").glob("*.png")))

    advanced_path = config.output_dir / "advanced_summary.json"
    advanced_section = ""
    advanced_visual_index = ""
    if advanced_path.exists():
        advanced = json.loads(advanced_path.read_text(encoding="utf-8"))
        significant = "、".join(advanced["significant_residual_tests"]) or "无"
        reducer_names = "、".join(advanced["available_reducers"])
        distribution_fit = pd.read_csv(
            config.output_dir / "theoretical_distribution_fit.csv"
        )
        best_distribution_fit = distribution_fit.loc[
            distribution_fit["best_by_aic"].astype(bool)
        ]
        consistent_count = int(
            best_distribution_fit["distribution_consistent_at_5pct"].astype(bool).sum()
        )
        tested_distribution_count = int(best_distribution_fit["feature"].nunique())
        transform_result = pd.read_csv(
            config.output_dir / "transform_reducer_summary.csv"
        )
        yeo_result = transform_result.loc[
            transform_result["experiment"] == "yeo_johnson"
        ].iloc[0]
        advanced_section = f"""
## 完善分析结果

- 逐样本缺失审计发现，{advanced['rows_with_any_missing']}条样本至少存在一个缺失值，单样本最高缺失率为{advanced['row_missing_rate_max']:.1%}；缺失机制按工序和设备差异输出为统计候选，不能替代生产日志确认。
- 对{tested_distribution_count}个稳定重要特征分别拟合Normal、Laplace、Logistic和Student-t分布；其AIC最优分布中有{consistent_count}个通过5%水平KS一致性检验，说明不能直接采用统一正态假设。
- 可插拔降维接口支持：{reducer_names}。在统一Ridge滚动实验中，当前最佳变换/降维配置为 **{advanced['best_transform_reducer']}**，MSE={advanced['best_transform_reducer_mse']:.6f}；该结论只适用于线性降维分支，不覆盖主干XGBoost。`autoencoder16`已实现插件接口但不在默认九组实验中，避免把未执行结果写成结论。
- Yeo-Johnson在匿名混合量纲特征上出现明显数值失稳（MSE={yeo_result.mse_mean:.3g}），已作为负向实验保留，不能默认认为分布变换必然改善模型。
- 三折消融中最优配置为 **{advanced['best_ablation']}**，MSE={advanced['best_ablation_mse']:.6f}；但扩大到主流程四个滚动时间折后未形成稳定优势，因此最终特征视图以主流程更广的时间验证结果为准。嵌套网格搜索的外层时间MSE为{advanced['nested_grid_outer_mse']:.6f}。
- 标准树模型对比中，**{advanced['best_tree_model']}**最低，MSE={advanced['best_tree_model_mse']:.6f}。
- 残差显著检验项：{significant}。这表明残差正态性和生产顺序相关性需要在后续模型中继续处理。
- 已为最高误差样本 **{advanced['highest_error_sample']}** 生成逐特征稳健Z分数和关键工序诊断，不以删除样本代替原因分析。
"""
        advanced_visual_index = """
17. [逐行与逐列缺失分布](plots/17_row_and_column_missingness.png)
18. [样本×工序缺失热力图](plots/18_sample_operation_missing_heatmap.png)
19. [方差、贡献与漂移](plots/19_variance_importance_drift.png)
20. [实际与理论分布拟合](plots/20_theoretical_distribution_fit.png)
21. [特征变换与降维对比](plots/21_transform_reducer_comparison.png)
22. [降维维数与误差](plots/22_reducer_dimension_and_error.png)
23. [工序特征消融](plots/23_feature_ablation.png)
24. [嵌套网格搜索](plots/24_nested_grid_search.png)
25. [随机森林、ExtraTrees与XGBoost](plots/25_tree_model_comparison.png)
26. [高误差样本局部诊断](plots/26_high_error_sample_diagnosis.png)
27. [残差Q-Q与时间自相关](plots/27_residual_qq_and_autocorrelation.png)
28. [完善分析运行痕迹](plots/28_advanced_analysis_trace.png)
"""

    phase_a_path = config.output_dir / "phase_a_manifest.json"
    phase_a_section = ""
    phase_a_visual_index = ""
    if phase_a_path.exists():
        phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
        phase_a_section = f"""
## A阶段训练内扩展窗口优化

- A模型严格只使用{phase_a['training_rows']}条原训练标签。按竞赛样本顺序构造三个扩展窗口，共评价{phase_a['validation_rows']}个伪未来样本；每个候选均使用{phase_a['ensemble_members']}个独立种子的特征选择与XGBoost平均预测。
- 比较均匀权重与近期样本半衰期{phase_a['candidate_half_life_fractions'][1:]}，完全依据训练内回测锁定 **{phase_a['selected_strategy']}**，回测MSE={phase_a['selected_backtest_mse']:.6f}、RMSE={phase_a['selected_backtest_rmse']:.6f}、MAE={phase_a['selected_backtest_mae']:.6f}。
- 策略锁定并完成A预测后才读取公开答案。事后A集MSE由原RobustBlend的{phase_a['baseline_external_a_mse']:.6f}降至 **{phase_a['selected_external_a_mse']:.6f}**，相对下降 **{phase_a['external_a_relative_mse_improvement']:.1%}**；R²={phase_a['selected_external_a_r2']:.4f}。
- 公开A答案只用于事后评价，未参与A策略选择或拟合：{phase_a['a_labels_used_for_selection_or_fit']}。原融合版本保存在`submission_A_train_only_blend.csv`。
"""
        phase_a_visual_index = """
30. [A阶段训练内策略选择与事后评价](plots/30_phase_a_training_only_backtest.png)
"""

    phase_b_path = config.output_dir / "phase_b_manifest.json"
    phase_b_section = ""
    phase_b_visual_index = ""
    submission_stage_note = (
        "上述A/B预测均为不使用测试标签的train-only基线。"
    )
    if phase_b_path.exists():
        phase_b = json.loads(phase_b_path.read_text(encoding="utf-8"))
        submission_stage_note = (
            "上述模型比较属于不使用测试标签的第一阶段基线。A榜答案公布后，"
            "B阶段只通过A内扩展窗口回测选择增量训练规则，B标签未参与训练或选择。"
        )
        phase_b_section = f"""
## 公布A标签后的B阶段适配

- A榜结束并公布答案后，将{phase_b['released_a_rows']}条A样本作为新增监督数据；原训练集仍保留全部{phase_b['original_train_rows']}条样本。
- 使用三个扩展窗口，以A前缀训练、紧随其后的A区段模拟未知下一批次，共评价{phase_b['validation_rows']}个伪未来样本；所有特征选择与填充均在各窗口训练部分重新拟合，并对{phase_b['ensemble_members']}个独立种子的特征选择与XGBoost预测取平均。
- 回测比较A样本权重{phase_b['candidate_a_weights']}，最终选择 **{phase_b['selected_strategy']}**。train-only基线MSE为{phase_b['train_only_backtest_mse']:.6f}，适配后为 **{phase_b['selected_backtest_mse']:.6f}**，相对下降 **{phase_b['relative_mse_improvement']:.1%}**。
- 适配回测RMSE={phase_b['selected_backtest_rmse']:.6f}、MAE={phase_b['selected_backtest_mae']:.6f}、R²={phase_b['selected_backtest_r2']:.4f}、平均偏差={phase_b['selected_backtest_bias']:.6f}。
- 最终B模型使用训练集+A共{phase_b['phase_b_training_rows']}条标签，公开A样本权重为{phase_b['selected_a_weight']:g}；正式`submission_B.csv`由该模型生成，train-only版本保存在`submission_B_train_only.csv`。
- 该改进是无B标签条件下的代理验证结果，不表述为B榜真实得分；B标签参与状态：{phase_b['b_labels_used']}。
"""
        phase_b_visual_index = """
29. [公开A标签增量训练回测](plots/29_phase_b_adaptation_backtest.png)
"""

    report = f"""# 工序感知与漂移鲁棒的产品特性预测

## 方案定位

本方案面向TFT-LCD复杂多工序生产中的连续产品特性预测。针对样本数量有限、原始变量维度高、设备工况差异明显、传感器缺失异常和跨批次分布漂移等问题，构建工序感知、漂移鲁棒且可解释的回归预测系统。

系统以预测精度和时间泛化能力为核心，同时输出预测区间、关键工序、稳定特征、设备组误差和数据漂移信息，为在线质量预警、工艺参数排查和检测资源配置提供依据。

技术主线为：**结构审计 → 工序/工具条件填充 → 稳定且抗漂移的原始特征选择 → 工序稳健统计与相对时间重构 → 树模型、正则线性模型和轻量MLP集成 → OOF非负融合 → 锁定时间外推验证 → 共形预测区间 → A训练内扩展窗口优化 → 公布A标签后的扩展窗口回测与B阶段适配**。

## 数据与验证口径

- 训练样本：{manifest['train_rows']}；测试A：{manifest['test_a_rows']}；测试B：{manifest['test_b_rows']}。
- 清洗后数值特征：{manifest['clean_numeric_features']}；识别时间字段：{manifest['timestamp_features']}；类别字段：{manifest['categorical_features']}。
- 按生产时间代理顺序锁定最后{manifest['temporal_holdout_rows']}条作为外推集；开发集同时执行随机五折CV和仅用过去预测未来的滚动时间CV（有效校准样本{manifest['rolling_calibration_rows']}条）。
- 所有监督特征选择、工具条件填充、缩放和类别编码均在训练折内拟合。

## 模型结果

| 模型 | 随机OOF MSE | 滚动时间OOF MSE | 时间外推MSE | 时间外推RMSE | 时间外推MAE | 时间外推R² |
|---|---:|---:|---:|---:|---:|---:|
{markdown_model_table(summary)}

随机OOF最低模型为 **{best_random.model}**（MSE={best_random.mse:.6f}）；滚动时间OOF最低模型为 **{best_rolling.model}**（MSE={best_rolling.mse:.6f}）；时间外推集最低模型为 **{best_holdout.model}**（MSE={best_holdout.mse:.6f}）。融合权重只根据滚动时间OOF学习，RobustBlend在外推集上的MSE为 **{blend_holdout.mse:.6f}**，权重为：{weight_text}。

不能在看到时间外推集结果后重新调权重，否则该外推集将失去独立验证意义。{submission_stage_note}

{phase_a_section}

{phase_b_section}

## 可信度与工程结论

- 90%共形区间在时间外推集上的实际覆盖率为 **{conformal_row.empirical_holdout_coverage:.1%}**，平均区间宽度为 **{conformal_row.mean_interval_width:.4f}**。
- 跨折贡献最高的工序为：{top_operations}。
- 训练集与A/B测试集漂移较高的工序为：{drift_operations}。
- 最大误差样本为 **{worst.ID}**，单点贡献时间外推集总平方误差的 **{worst_share:.1%}**；保留该样本时MSE为{blend_holdout.mse:.6f}，仅作敏感性分析、排除该点后的描述性MSE为{mse_without_worst:.6f}。除非能证明标签或采集错误，否则不能为了提高指标直接删除。
- 完整运行耗时为 **{manifest['total_elapsed_sec'] / 60:.2f}分钟**；各阶段、各模型耗时保存在`run_trace.csv`和`fold_metrics.csv`。

本轮第一次运行曾用该时间外推集识别随机CV过于乐观的问题，随后验证设计升级为滚动时间CV。因此当前外推结果应解释为**时间泛化基准**，不是完全未触碰的最终测试。正式论文若要作统计确认，应使用新增生产周期或嵌套滚动验证作为最终盲测。

## 创新点

1. **工序条件缺失建模**：缺失值先按工序对应设备组统计填充，再由训练折全局中位数兜底，同时保留高缺失特征的缺失指示。
2. **抗漂移稳定选择**：特征得分联合ExtraTrees贡献、目标相关性和开发期前后段稳健中位数漂移，不把一次划分的重要性直接当成最终结论。
3. **量纲无关工序表征**：先按训练折中位数/IQR将匿名传感器转换为稳健Z分数，再计算每道工序的分布、异常率、缺失率和零值率。
4. **工序时间图特征**：将8/14/16位时间字段解析为秒，生成工序开始、结束、跨度以及相邻工序等待/重叠特征，不直接删除时间，也不把无序字段误当RNN序列。
5. **跨范式鲁棒融合**：同时训练Ridge、ExtraTrees、XGBoost和三成员轻量MLP，通过开发集OOF学习非负且和为1的权重。
6. **面向部署的可信预测**：除Y点预测外输出90%共形区间、工序贡献、测试漂移和时间外推误差。

## 近年方法如何进入本方案

- 2025年Nature的TabPFN表明表格基础模型在不超过约10000样本、500特征的小数据上具有很强潜力。本方案先稳定筛选至{config.top_k_raw_features}个原始特征，为后续加入TabPFN留下兼容接口；当前环境未安装模型及权重，因此没有伪造TabPFN实验结果。
- ICLR 2025 TabM强调参数高效的MLP集成和成员多样性。本方案当前使用可直接运行的三成员小型MLP集成验证这一方向，但明确不将其冒充为TabM。
- SCARF通过随机特征破坏学习表格表示，说明缺失/扰动机制可用于鲁棒表征。本方案将缺失掩码和工序级异常统计显式纳入特征；后续安装PyTorch后可增加SCARF预训练消融。
- 大规模表格基准显示中等规模结构化数据上树模型仍是强基线，因此XGBoost和ExtraTrees保持为主干，而深度模型作为互补分支而非预设赢家。

{advanced_section}

## 输出与工作痕迹

- `feature_audit.csv`：每列保留、删除或时间重构的原因。
- `fold_metrics.csv`、`model_summary.csv`：统一验证结果和训练耗时。
- `feature_selection_stability.csv`：每个原始特征的五折入选率、贡献、相关性和漂移。
- `engineered_feature_importance.csv`、`operation_importance.csv`：特征与工序两级解释。
- `feature_drift.csv`、`operation_drift.csv`：训练集到A/B测试集的漂移证据。
- `random_oof_predictions.csv`、`rolling_oof_predictions.csv`、`temporal_holdout_predictions.csv`：三种验证层级的逐样本预测与残差。
- `conformal_metrics.csv`：预测区间校准结果。
- `submission_A.csv`、`submission_B.csv`：无表头两列竞赛提交文件；A为训练内扩展窗口选择版本，B为公开A标签适配版本。
- `submission_A_train_only_blend.csv`：原RobustBlend的A基线，便于完整追溯。
- `phase_a_backtest_predictions.csv`、`phase_a_strategy_summary.csv`、`phase_a_external_evaluation.csv`：A训练内回测和答案公布后的事后评价。
- `submission_B_train_only.csv`：不使用A答案的B基线，便于完整追溯。
- `phase_b_backtest_predictions.csv`、`phase_b_strategy_summary.csv`、`phase_b_manifest.json`：B阶段扩展窗口证据、策略比较和配置清单。
- `WORKFLOW_DESCRIPTION.md`：从数据权限到A/B提交的完整工作过程文字稿。
- `row_missing_report.csv`、`row_operation_missing_matrix.csv`：逐样本和逐工序缺失证据。
- `feature_variance_report.csv`、`theoretical_distribution_fit.csv`：方差、AIC和KS检验结果。
- `transform_reducer_summary.csv`、`ablation_summary.csv`：变换、降维和工序特征消融。
- `nested_grid_search_outer.csv`、`tree_model_comparison_summary.csv`：折内搜索和树模型对比。
- `high_error_sample_feature_detail.csv`、`residual_statistical_tests.csv`：样本级解释和残差检验。
- `feature_semantics.csv`、`reproducibility_snapshot.json`：物理语义边界、输入哈希和软件版本。
- `plots/`：{plot_count}张数据、模型、残差、解释、漂移和运行过程图。

## 可视化索引

1. [工序数据审计](plots/01_data_audit_by_operation.png)
2. [目标分布与时间切分](plots/02_target_and_temporal_split.png)
3. [候选模型五折稳定性](plots/03_cv_model_stability.png)
4. [随机CV与时间外推差异](plots/04_oof_vs_temporal_holdout.png)
5. [融合权重](plots/05_blend_weights.png)
6. [模型预测相关性](plots/06_model_prediction_correlation.png)
7. [时间外推真实值与预测值](plots/07_temporal_holdout_prediction.png)
8. [时间外推残差诊断](plots/08_temporal_residual_diagnostics.png)
9. [分设备误差](plots/09_holdout_error_by_tool.png)
10. [原始特征选择稳定性](plots/10_feature_selection_stability.png)
11. [关键工序贡献](plots/11_operation_importance.png)
12. [工序分布漂移](plots/12_operation_distribution_drift.png)
13. [共形区间覆盖率](plots/13_conformal_interval_coverage.png)
14. [精度与训练效率](plots/14_accuracy_efficiency.png)
15. [流水线运行痕迹](plots/15_pipeline_run_trace.png)
16. [高误差样本集中度](plots/16_error_concentration.png)
{advanced_visual_index}
{phase_b_visual_index}
{phase_a_visual_index}

## 研究依据

- [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)
- [TabM官方实现与论文, ICLR 2025](https://github.com/yandex-research/tabm)
- [SCARF, ICLR 2022 Spotlight](https://arxiv.org/abs/2106.15147)
- [树模型与深度模型表格基准](https://arxiv.org/abs/2207.08815)
- [天池冠军工序特征方案](https://tianchi.aliyun.com/forum/post/4020)
- [天池SDAE与相对时间方案](https://tianchi.aliyun.com/forum/post/4087)
"""
    (config.output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def generate_report(results: dict[str, object], config: PipelineConfig) -> None:
    if (config.output_dir / "review" / "review_manifest.json").exists():
        from .review_report import generate_review_report
        generate_review_report()
        return
    configure_style()
    plot_dir = config.output_dir / "plots"
    data: PreparedData = results["data"]
    plot_data_audit(data, plot_dir)
    plot_target_split(
        data, results["random_oof_detail"], results["holdout_detail"], plot_dir
    )
    plot_cv_models(results["fold_metrics"], plot_dir)
    plot_oof_vs_holdout(results["model_summary"], plot_dir)
    plot_blend_weights(results["blend_weights"], plot_dir)
    plot_prediction_correlation(results["rolling_oof_detail"], plot_dir)
    plot_holdout_prediction(results["holdout_detail"], plot_dir)
    plot_residual_diagnostics(results["holdout_detail"], plot_dir)
    plot_tool_error(results["holdout_detail"], plot_dir)
    plot_feature_stability(results["selection_stability"], plot_dir)
    plot_operation_importance(results["operation_importance"], plot_dir)
    plot_operation_drift(results["operation_drift"], plot_dir)
    plot_conformal(results["conformal_metrics"], plot_dir)
    plot_efficiency(results["fold_metrics"], results["model_summary"], plot_dir)
    plot_run_trace(results["trace"], plot_dir)
    plot_error_concentration(results["holdout_detail"], plot_dir)
    write_report(results, config)
