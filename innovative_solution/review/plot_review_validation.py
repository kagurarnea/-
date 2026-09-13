"""Figures for corrected validation; never rewrites the three audit figures.

Run: python innovative_solution/review/plot_review_validation.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .plot_audit import AUDIT, BLUE, GRAY, ORANGE, PLOTS, TEAL, save
else:
    from plot_audit import AUDIT, BLUE, GRAY, ORANGE, PLOTS, TEAL, save


def nested_figure() -> None:
    detail = pd.read_csv(AUDIT / "nested_predictions.csv")
    summary = pd.read_csv(AUDIT / "nested_summary.csv").set_index("model")
    detail["squared_error"] = (detail.prediction - detail.actual) ** 2
    per_fold = detail.groupby(["fold", "model"]).squared_error.mean().unstack()
    names = ["mean_baseline", "fixed_raw_d2", "fixed_full_d2", "nested_selected"]
    labels = ["均值基线", "固定基础特征（深度 2）", "固定完整特征（深度 2）", "内层选择方案"]
    colors = ["#9EA9B3", BLUE, TEAL, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.095, right=.97, top=.77, bottom=.38, wspace=.30)
    fig.suptitle("不同特征表示与选参策略的外层预测误差", fontsize=16, y=.96)
    x = np.arange(len(per_fold))
    width = .19
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        axes[0].bar(x + (i-1.5)*width, per_fold[name], width=width*.94, color=color, label=label)
    axes[0].set_xticks(x, [f"外层 {int(fold)}\n(n=100)" for fold in per_fold.index])
    axes[0].set_ylabel("均方误差 MSE")
    axes[0].set_title("（a）三个外层验证窗口", loc="left", pad=20)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.25), ncol=2, fontsize=8.7)
    bars = axes[1].bar(np.arange(4), [summary.loc[n, "mse"] for n in names], color=colors, width=.60)
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.001, f"{bar.get_height():.5f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_xticks(np.arange(4), ["均值\n基线", "固定\n基础特征", "固定\n完整特征", "内层\n选择"])
    axes[1].set_title("（b）合并外层预测（n=300）", loc="left", pad=20)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.17)
        ax.set_ylim(0, max(per_fold.max().max(), summary.mse.max()) * 1.16)
    fig.text(.095, .128, "完整特征在深度 2 条件下获得最低合并 MSE；嵌套选择的误差高于固定配置。", fontsize=10, color="#283541")
    fig.text(.095, .077, "内层分别拟合预处理并选择候选；按 ID 分组隔离重叠记录。时间顺序来自字段中位数代理。", fontsize=9, color=GRAY)
    fig.text(.095, .032, "全部方法使用相同的 300 条外层记录，来源：nested_predictions.csv / nested_summary.csv。", fontsize=8.5, color=GRAY)
    save(fig, "review_nested_mse")


def tail_figure() -> None:
    tail = pd.read_csv(AUDIT / "frozen_tail_predictions.csv")
    manifest = json.loads((AUDIT / "review_manifest.json").read_text(encoding="utf-8"))
    covered = tail.actual.between(tail.empirical_lower, tail.empirical_upper)
    count, n = int(covered.sum()), len(tail)
    observed = count/n
    assert np.isclose(observed, manifest["observed_tail_coverage"])
    x = np.arange(1, n+1)
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    fig.subplots_adjust(left=.085, right=.97, top=.77, bottom=.38)
    fig.suptitle("冻结模型在尾段样本上的预测与经验区间", fontsize=16, y=.96)
    ax.fill_between(x, tail.empirical_lower, tail.empirical_upper, color=BLUE, alpha=.17, label="单独校准段残差构造的经验区间")
    ax.plot(x, tail.prediction, color=BLUE, linewidth=1.5, label="冻结模型预测")
    ax.scatter(x, tail.actual, s=22, color="#394B58", alpha=.8, zorder=3, label="实际 Y")
    ax.scatter(x[~covered], tail.actual.loc[~covered], s=47, facecolors="none", edgecolors=ORANGE, linewidths=1.3, zorder=4, label="实际值落在区间之外")
    ax.set_xlim(.2,n+.8)
    ax.set_xlabel("尾段样本顺序（依据时间代理排列）", labelpad=9)
    ax.set_ylabel("质量特性值 Y")
    ax.set_title(f"经验覆盖 {count}/{n} = {observed:.1%}　｜　名义水平 90%　｜　MSE = {manifest['tail_metrics']['mse']:.5f}", loc="left", pad=18, fontsize=11.5)
    ax.grid(axis="y", alpha=.17)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -.30), fontsize=8.7)
    fig.text(.085, .118, "模型在开发段训练后冻结；校准段 80 条样本仅确定区间半径；尾段 100 条样本用于报告误差与覆盖。", fontsize=9.5, color="#283541")
    fig.text(.085, .067, "覆盖率为尾段实测结果；时间依赖和分布变化可能影响校准残差对后续误差的代表性。", fontsize=9, color=GRAY)
    fig.text(.085, .025, "来源：frozen_tail_predictions.csv / review_manifest.json。", fontsize=8.5, color=GRAY)
    save(fig, "review_tail_interval")


def residual_figure() -> None:
    tail = pd.read_csv(AUDIT / "frozen_tail_predictions.csv")
    residual = tail.actual - tail.prediction
    worst_index = int(np.argmax(np.abs(residual.to_numpy())))
    worst = tail.iloc[worst_index]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.7))
    fig.subplots_adjust(left=.095, right=.97, top=.77, bottom=.25, wspace=.30)
    fig.suptitle("冻结尾段预测的拟合偏差与残差诊断", fontsize=16, y=.96)
    axes[0].scatter(tail.actual, tail.prediction, color=BLUE, s=27, alpha=.70)
    low = min(tail.actual.min(), tail.prediction.min())-.08
    high = max(tail.actual.max(), tail.prediction.max())+.08
    axes[0].plot([low, high], [low, high], color=GRAY, linestyle="--", linewidth=1.1, label="预测值 = 实际值")
    axes[0].set_xlim(low, high)
    axes[0].set_ylim(low, high)
    axes[0].set_xlabel("实际 Y")
    axes[0].set_ylabel("预测 Y")
    axes[0].set_title("（a）实际值与预测值", loc="left", pad=18)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].scatter(tail.prediction, residual, color=TEAL, s=27, alpha=.70)
    axes[1].axhline(0, color=GRAY, linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("预测 Y")
    axes[1].set_ylabel("残差（实际值 − 预测值）")
    axes[1].set_title("（b）残差与预测值", loc="left", pad=18)
    for ax, px, py in ((axes[0], worst.actual, worst.prediction), (axes[1], worst.prediction, residual.iloc[worst_index])):
        ax.scatter([px], [py], facecolors="none", edgecolors=ORANGE, linewidths=1.7, s=86, zorder=4)
        annotation_position = (.29, .85) if ax is axes[0] else (.30, .32)
        ax.annotate(str(worst.ID), xy=(px, py), xycoords="data", xytext=annotation_position, textcoords="axes fraction",
                    ha="center", va="center", color=ORANGE, fontsize=10,
                    bbox={"boxstyle":"round,pad=.30", "facecolor":"white", "edgecolor":"none", "alpha":.94},
                    arrowprops={"arrowstyle":"->", "color":ORANGE, "linewidth":1})
        ax.grid(alpha=.14)
        ax.set_axisbelow(True)
    fig.text(.095, .140, f"最大绝对残差样本 {worst.ID}：实际 {worst.actual:.4f}，预测 {worst.prediction:.4f}，残差 {residual.iloc[worst_index]:+.4f}。", fontsize=10, color="#283541")
    fig.text(.095, .085, "图中保留全部 100 条尾段样本；高误差记录用于诊断，不能在没有采集或标签错误证据时直接剔除。", fontsize=9, color=GRAY)
    fig.text(.095, .040, "来源：frozen_tail_predictions.csv；残差图反映预测关联，不提供工艺因果解释。", fontsize=8.5, color=GRAY)
    save(fig, "review_tail_residuals")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    nested_figure()
    tail_figure()
    residual_figure()
    print("Created review_nested_mse, review_tail_interval and review_tail_residuals as PNG (240 dpi) and SVG")


if __name__ == "__main__":
    main()
