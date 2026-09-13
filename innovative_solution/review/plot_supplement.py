"""Publication figures directly from the supplementary experiment tables."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/"outputs"/"supplement"
PLOTS=OUT/"plots"
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
BLUE="#236A92";TEAL="#008A80";ORANGE="#C57A25"


def save(fig,name):
    PLOTS.mkdir(parents=True,exist_ok=True)
    fig.savefig(PLOTS/(name+".png"),dpi=240,bbox_inches="tight",facecolor="white")
    fig.savefig(PLOTS/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def baselines():
    scores=pd.read_csv(OUT/"baseline_summary.csv").sort_values("mse")
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(scores.model,scores.mse,color=BLUE)
    ax.invert_yaxis();ax.set_xlabel("合并外层 MSE（300 条记录）");ax.set_title("统一特征与窗口下的回归方法比较",pad=15)
    for i,value in enumerate(scores.mse):ax.text(value+.0004,i,f"{value:.5f}",va="center",fontsize=9)
    ax.set_xlim(0,scores.mse.max()*1.2);ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
    fig.text(.13,.005,"各方法由内层窗口选参，外层三种子平均；排名受候选网格与迭代预算影响。",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,"supplement_baselines")


def ablation():
    scores=pd.read_csv(OUT/"feature_experiment_summary.csv");scores=scores[scores.scope.eq("ablation")].set_index("model")
    names=["measurements","basic","process_only","time_only","full","no_mask","median_only","group_min5","group_min5_shrink10"]
    labels=["测量及设备","基础特征","基础加工序","基础加时间","组合特征","去缺失指示","全局中位数","设备至少5个","至少5个并收缩"]
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),gridspec_kw={"width_ratios":[1,1.1]})
    axes[0].barh(labels,[scores.loc[n,"mse"] for n in names],color=[ORANGE if n=="time_only" else BLUE for n in names]);axes[0].invert_yaxis()
    axes[0].set_xlabel("合并 MSE");axes[0].set_title("（a）固定深度2 三成员平均",pad=15)
    inf=pd.read_csv(OUT/"paired_group_inference.csv").iloc[1:]
    if len(inf):
        d=inf.mse_difference_ref_minus_candidate.to_numpy();lo=inf.ci_low.to_numpy();hi=inf.ci_high.to_numpy()
        labs=[f"{r.reference} → {r.candidate}" for r in inf.itertuples()]
        axes[1].errorbar(d,np.arange(len(d)),xerr=[d-lo,hi-d],fmt="o",color=TEAL,capsize=3)
        axes[1].set_yticks(np.arange(len(d)),labs,fontsize=8);axes[1].invert_yaxis();axes[1].axvline(0,color="gray",linestyle="--")
        axes[1].set_xlabel("参考 MSE − 候选 MSE");axes[1].set_title("（b）配对关联组 95% 区间",pad=15)
    fig.text(.08,.005,"区间以现有预测为条件，未消除研究选择偏差；正差值表示候选误差更低。",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,"supplement_ablation")


def selection():
    scores=pd.read_csv(OUT/"selection_score_distribution.csv").pivot(index="stage",columns="candidate",values="pooled_mse")
    scores=scores.reindex(["outer_1","outer_2","outer_3","final_development"])
    fields=pd.read_csv(OUT/"selected_fields.csv")
    sets={(f,s):set(g.feature) for (f,s),g in fields.groupby(["fold","seed"])}
    keys=list(sets);J=np.array([[len(sets[a]&sets[b])/len(sets[a]|sets[b]) for b in keys] for a in keys])
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    axes[0].imshow(scores.values,cmap="YlGnBu",aspect="auto")
    axes[0].set_xticks(range(4),scores.columns,fontsize=9);axes[0].set_yticks(range(4),["外层1内层","外层2内层","外层3内层","最终开发内层"])
    for i in range(4):
        for j in range(4):axes[0].text(j,i,f"{scores.iloc[i,j]:.4f}",ha="center",va="center",color="white" if scores.iloc[i,j]>.03 else "black",fontsize=9)
    axes[0].set_title("内层合并 MSE",pad=12)
    img=axes[1].imshow(J,cmap="viridis",vmin=0,vmax=1)
    lab=[f"F{a} S{b}" for a,b in keys]
    axes[1].set_xticks(range(9),lab,rotation=90,fontsize=7);axes[1].set_yticks(range(9),lab,fontsize=7);axes[1].set_title("入选字段 Jaccard 重叠率",pad=12)
    fig.colorbar(img,ax=axes[1],shrink=.8);fig.tight_layout();save(fig,"supplement_selection")


def coverage():
    f=pd.read_csv(OUT/"conditional_coverage.csv")
    overall=f[f.stratifier.eq("overall")]
    group=f[(f.strategy.eq("fixed"))&((f.stratifier.eq("predicted_tercile"))|((f.stratifier.eq("TOOL"))&f.n.ge(20)))]
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for i,r in enumerate(group.itertuples()):
        axes[0].errorbar(r.coverage,i,xerr=[[r.coverage-r.wilson_low],[r.wilson_high-r.coverage]],fmt="o",capsize=3,color=BLUE)
    axes[0].set_yticks(range(len(group)),[f"{r.level} (n={r.n})" for r in group.itertuples()],fontsize=9);axes[0].invert_yaxis();axes[0].axvline(.9,color=ORANGE,ls="--")
    axes[0].set_xlim(.5,1);axes[0].set_xlabel("覆盖率与描述性 Wilson 区间");axes[0].set_title("（a）固定区间的条件覆盖",pad=15)
    axes[1].bar(["固定校准","滚动80条"],overall.mean_width,color=[BLUE,TEAL],width=.5)
    for i,r in enumerate(overall.itertuples()):axes[1].text(i,r.mean_width+.01,f"覆盖 {r.coverage:.0%}\n宽度 {r.mean_width:.3f}",ha="center",fontsize=10)
    axes[1].set_ylim(0,overall.mean_width.max()*1.25);axes[1].set_ylabel("平均区间宽度");axes[1].set_title("（b）覆盖与区间宽度",pad=15)
    fig.text(.08,.005,"滚动场景假设较早时间组标签立即可用；Wilson 区间未校正设备组内的时间相关。",fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1));save(fig,"supplement_coverage")


def main():
    baselines();ablation();selection();coverage()


if __name__=="__main__":main()
