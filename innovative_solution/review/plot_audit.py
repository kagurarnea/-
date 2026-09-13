"""Publication figures from the read-only statistical audit tables.

Run: python innovative_solution/review/plot_audit.py
Only outputs/review/plots is written. Existing experiment figures are untouched.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "innovative_solution" / "outputs"
AUDIT = OUT / "review"
PLOTS = AUDIT / "plots"
FONT = Path("C:/Windows/Fonts/msyh.ttc")
if not FONT.exists():
    raise FileNotFoundError("A Chinese font is required: C:/Windows/Fonts/msyh.ttc")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
                     "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "axes.unicode_minus": False, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
BLUE = "#266B93"
TEAL = "#178277"
ORANGE = "#C2782E"
GRAY = "#66727D"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(PLOTS / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(PLOTS / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def time_figure() -> None:
    frame = pd.read_csv(AUDIT / "dataset_temporal_audit.csv").set_index("dataset")
    overlap = pd.read_csv(AUDIT / "dataset_overlap.csv").set_index("dataset")
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.11, right=.97, top=.80, bottom=.33, wspace=.36)
    fig.suptitle("竞赛数据阶段与生产时间代理并不一致", x=.5, y=.965, fontsize=16)
    axes[0].set_title("（a）每个数据集的时间代理范围", loc="left", pad=24)
    labels = {"train": "训练集", "A": "测试 A", "B": "测试 B"}
    colors = {"train": BLUE, "A": TEAL, "B": ORANGE}
    for position, name in enumerate(("train", "A", "B")):
        row = frame.loc[name]
        start, end = pd.Timestamp(row.proxy_min), pd.Timestamp(row.proxy_max)
        axes[0].plot([start, end], [position, position], color=colors[name], linewidth=7, solid_capstyle="round")
        axes[0].scatter([start,end],[position,position],s=40,color=colors[name],zorder=3)
        axes[0].text(start, position-.20, start.strftime("%m-%d"), ha="center", va="center", fontsize=9)
        axes[0].text(end, position-.20, end.strftime("%m-%d"), ha="center", va="center", fontsize=9)
    axes[0].set_yticks(range(3), [f"{labels[n]}\n(n={int(frame.loc[n,'rows'])})" for n in ("train","A","B")])
    axes[0].set_ylim(2.55,-.65)
    axes[0].set_xlim(pd.Timestamp("2017-05-24"), pd.Timestamp("2017-07-29"))
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].set_xlabel("2017 年日期（由已识别时间字段推定）", labelpad=12)
    axes[0].grid(axis="x",alpha=.18)
    axes[1].set_title("（b）测试样本相对训练时间范围的位置", loc="left", pad=24)
    for position, name in enumerate(("A","B")):
        row=overlap.loc[name]
        before=100*row.rows_before_train_proxy_min/row.rows
        inside=100*row.rows_in_train_proxy_range/row.rows
        axes[1].barh(position,before,color=ORANGE,height=.32)
        axes[1].barh(position,inside,left=before,color=BLUE,height=.32)
        axes[1].text(before/2,position,f"{int(row.rows_before_train_proxy_min)} 条 / {before:.1f}%",ha="center",va="center",color="white",fontsize=10)
        axes[1].annotate(f"{int(row.rows_in_train_proxy_range)} 条 / {inside:.1f}%",xy=(before+inside/2,position),
                         xytext=(94,position-.29),ha="right",va="center",fontsize=9,
                         arrowprops={"arrowstyle":"-","color":BLUE,"lw":.8})
    axes[1].set_yticks([0,1],["测试 A","测试 B"])
    axes[1].set_ylim(1.90,-.65)
    axes[1].set_xlim(0,102)
    axes[1].set_xticks([0,25,50,75,100],["0%","25%","50%","75%","100%"])
    axes[1].set_xlabel("样本占比",labelpad=12)
    axes[1].grid(axis="x",alpha=.14)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=ORANGE,label="早于训练最早时间"),Patch(color=BLUE,label="处于训练时间范围内")],
                   frameon=False,loc="lower left",fontsize=8.5)
    fig.text(.11,.155,"A、B 均无样本晚于训练集的最晚时间代理；阶段发布顺序不能直接解释为生产先后。",fontsize=10,color="#283541")
    fig.text(.11,.095,"时间代理为每条样本 72 个已识别时间字段的中位数，字段的实际工序含义尚未核实。",fontsize=9,color=GRAY)
    fig.text(.11,.055,"依据：dataset_temporal_audit.csv 与 dataset_overlap.csv；所有范围和比例均来自原始数据审计。",fontsize=8.5,color=GRAY)
    save(fig,"audit_01_time_proxy_ranges")


def coverage_figure() -> None:
    table=pd.read_csv(AUDIT/"interval_coverage.csv").set_index("evaluation")
    row=table.loc["current_A"]
    covered=int(row.covered_rows)
    n=int(row.rows)
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.6),gridspec_kw={"width_ratios":[1.1,1]})
    fig.subplots_adjust(left=.11,right=.96,top=.79,bottom=.25,wspace=.44)
    fig.suptitle("当前 A 集经验区间覆盖低于名义水平",fontsize=16,y=.96)
    bars=axes[0].bar([0,1],[100*row.observed_coverage,100*row.nominal_coverage],color=[TEAL,"#B7C2CB"],width=.56)
    for bar,value,label in zip(bars,[100*row.observed_coverage,100*row.nominal_coverage],[f"{covered}/{n} 条", "目标水平"]):
        axes[0].text(bar.get_x()+bar.get_width()/2,value+2,f"{value:.2f}%\n{label}",ha="center",va="bottom",fontsize=11,linespacing=1.5)
    axes[0].set_xticks([0,1],["A 集实测覆盖","名义覆盖"])
    axes[0].set_ylabel("覆盖率（%）")
    axes[0].set_ylim(0,107)
    axes[0].set_yticks([0,20,40,60,80,100])
    axes[0].grid(axis="y",alpha=.17)
    axes[0].set_axisbelow(True)
    axes[0].set_title("（a）回顾性评价的实际覆盖率",loc="left",pad=18)
    axes[1].axis("off")
    axes[1].set_title("（b）样本与区间统计",loc="left",pad=18)
    info=[("评价样本数",f"{n} 条"),("被区间覆盖",f"{covered} 条"),("未被区间覆盖",f"{n-covered} 条"),
          ("低于名义水平",f"{100*(row.nominal_coverage-row.observed_coverage):.2f} 个百分点"),
          ("平均区间宽度",f"{row.mean_width:.6f}（Y 的数值尺度）")]
    for i,(label,value) in enumerate(info):
        y=.91-i*.17
        axes[1].text(0,y,label,fontsize=10,color=GRAY,va="center")
        axes[1].text(1,y,value,fontsize=11,color="#263746",va="center",ha="right")
        axes[1].axhline(y-.08,xmin=0,xmax=1,color="#E1E6EA",linewidth=.8)
    fig.text(.11,.145,"该区间由开发回测残差分位数构造；A 数据曾在研究过程中被查看，结果属于回顾性评价。",fontsize=9.5,color="#283541")
    fig.text(.11,.085,"未建立独立校准与样本交换性条件，因此本图不表示对未知数据的 90% 理论覆盖保证。",fontsize=9,color=GRAY)
    fig.text(.11,.042,"依据：interval_coverage.csv；预测区间仅作经验诊断，不作为自动放行产品的依据。",fontsize=8.5,color=GRAY)
    save(fig,"audit_02_A_empirical_coverage")


def weight_figure() -> None:
    folds=pd.read_csv(AUDIT/"phase_b_fold_comparison.csv")
    rows=[]
    for weight in (1,2,4):
        group=folds.loc[folds.strategy.eq(f"train_plus_released_a_x{weight}")]
        rows.append({"weight":weight,"mse":np.average(group.mse,weights=group.rows),"mae":np.average(group.mae,weights=group.rows),"rows":int(group.rows.sum())})
    values=pd.DataFrame(rows)
    relative=100*(values.iloc[0].mse-values.iloc[1].mse)/values.iloc[0].mse
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.6))
    fig.subplots_adjust(left=.10,right=.97,top=.78,bottom=.28,wspace=.30)
    fig.suptitle("已公开 A 样本权重的开发回测比较",fontsize=16,y=.96)
    for ax,metric,title in zip(axes,("mse","mae"),("（a）均方误差 MSE（越低越好）","（b）平均绝对误差 MAE（越低越好）")):
        colors=[BLUE,TEAL,ORANGE]
        bars=ax.bar(range(3),values[metric],color=colors,width=.58)
        for bar,value in zip(bars,values[metric]):
            ax.text(bar.get_x()+bar.get_width()/2,value+values[metric].max()*.035,f"{value:.6f}",ha="center",va="bottom",fontsize=10)
        ax.set_xticks(range(3),["A 权重 1","A 权重 2","A 权重 4"])
        ax.set_ylim(0,values[metric].max()*1.29)
        ax.set_title(title,loc="left",pad=18)
        ax.grid(axis="y",alpha=.17)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis="y",style="plain")
    fig.text(.10,.178,f"权重 2 相对权重 1 的 MSE 仅低 {relative:.3f}%；MAE 则以权重 1 更低。",fontsize=11,color="#283541")
    fig.text(.10,.119,"三种策略使用相同 200 条开发验证记录（分折 50 / 50 / 100），按样本数合并计算。",fontsize=9.5,color=GRAY)
    fig.text(.10,.071,"这些分数同时用于候选选择；行序分块重采样未显示权重 2 的稳定优势，不作显著性或 B 榜成绩声明。",fontsize=9,color=GRAY)
    fig.text(.10,.030,"依据：phase_b_fold_comparison.csv 与 paired_bootstrap_sensitivity.csv。",fontsize=8.5,color=GRAY)
    save(fig,"audit_03_A_weight_development_comparison")


def main() -> None:
    PLOTS.mkdir(parents=True,exist_ok=True)
    time_figure()
    coverage_figure()
    weight_figure()
    print("Generated 3 audit figures as PNG and SVG in",PLOTS)


if __name__=="__main__":
    main()
