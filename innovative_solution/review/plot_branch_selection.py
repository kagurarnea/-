"""Figures bound to the chosen branch experiment and its diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parents[1]/'outputs'/'branch_selection'
PLOTS=OUT/'plots'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
    'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#236A92';TEAL='#008A80';ORANGE='#C57A25'


def save(fig,name):
    PLOTS.mkdir(parents=True,exist_ok=True)
    fig.savefig(PLOTS/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(PLOTS/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    scores=pd.read_csv(OUT/'nested_summary.csv').set_index('model')
    names=['fixed_raw_d2','fixed_process_d2','fixed_time_d2','fixed_full_d2','nested_selected']
    labels=['基础 深度2','基础加工序 深度2','基础加时间 深度2','组合 深度2','八候选嵌套选择']
    fig,ax=plt.subplots(figsize=(10,4.6));values=[scores.loc[n,'mse'] for n in names]
    ax.barh(labels,values,color=[BLUE,BLUE,TEAL,BLUE,ORANGE]);ax.invert_yaxis()
    for i,v in enumerate(values):ax.text(v+.0002,i,f'{v:.6f}',va='center')
    ax.set_xlim(0,max(values)*1.22);ax.set_xlabel('合并外层 MSE（同一300条记录）');ax.grid(axis='x',alpha=.2)
    ax.set_title('固定表示与选择流程的误差比较',pad=15)
    fig.text(.1,.005,'固定表示与自适应选择是不同的评价对象；全部比较仍为回顾性分析。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'review_nested_mse')
    inner=pd.read_csv(OUT/'selection_score_distribution.csv').pivot(index='stage',columns='candidate',values='pooled_mse')
    names=[f'{v}_d{d}' for v in ['raw','process','time','full'] for d in [2,3]]
    inner=inner.reindex(index=['outer_1','outer_2','outer_3','final_development'],columns=names)
    fig,ax=plt.subplots(figsize=(11,4.4));ax.imshow(inner.values,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(8),names,rotation=20,ha='right',fontsize=9)
    ax.set_yticks(range(4),['外层1内部','外层2内部','外层3内部','最终开发段'])
    for i in range(4):
        for j in range(8):ax.text(j,i,f'{inner.iloc[i,j]:.5f}',ha='center',va='center',fontsize=8,color='white' if inner.iloc[i,j]>.03 else 'black')
        best=int(np.argmin(inner.iloc[i].to_numpy()));ax.add_patch(Rectangle((best-.5,i-.5),1,1,fill=False,edgecolor=ORANGE,lw=3))
    ax.set_title('八候选的内层合并 MSE（边框为该行选中项）',pad=16)
    fig.text(.08,.005,'raw 基础；process 基础加工序；time 基础加时间；full 组合。每项均为三种子平均。',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));save(fig,'supplement_selection')
    manifest=json.loads((OUT/'review_manifest.json').read_text(encoding='utf-8'))
    tail=pd.read_csv(OUT/'frozen_tail_predictions.csv');x=np.arange(1,len(tail)+1)
    fig,ax=plt.subplots(figsize=(11,4.7));ax.fill_between(x,tail.empirical_lower,tail.empirical_upper,color=BLUE,alpha=.2,label='经验区间')
    ax.plot(x,tail.prediction,color=BLUE,label='预测值');ax.scatter(x,tail.actual,s=14,color='#37444F',label='实际值')
    outside=~tail.actual.between(tail.empirical_lower,tail.empirical_upper)
    ax.scatter(x[outside],tail.actual[outside],s=35,facecolor='none',edgecolor=ORANGE,label='区间外观测')
    ax.set_xlabel('尾段顺序（时间代理排序）');ax.set_ylabel('Y');ax.legend(ncol=4,loc='lower left',fontsize=9)
    ax.set_title(f"开发段选定 {manifest['selected_candidate']}  |  MSE {manifest['tail_metrics']['mse']:.6f}  |  覆盖 {manifest['observed_tail_coverage']:.0%}",pad=15)
    fig.text(.08,.005,'620条开发、80条校准、100条尾段；覆盖为观测比例，不是对未知生产数据的保证。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'review_tail_interval')
    residual=tail.actual-tail.prediction;worst=int(abs(residual).argmax())
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    axes[0].scatter(tail.actual,tail.prediction,s=20,color=BLUE,alpha=.7);lim=[min(tail.actual.min(),tail.prediction.min())-.1,max(tail.actual.max(),tail.prediction.max())+.1]
    axes[0].plot(lim,lim,'--',color='gray');axes[0].set(xlabel='实际 Y',ylabel='预测 Y',title='实际值与预测值')
    axes[1].scatter(tail.prediction,residual,s=20,color=TEAL,alpha=.7);axes[1].axhline(0,color='gray',linestyle='--')
    axes[1].set(xlabel='预测 Y',ylabel='实际值 − 预测值',title='残差与预测值')
    for ax,px,py in [(axes[0],tail.actual.iloc[worst],tail.prediction.iloc[worst]),(axes[1],tail.prediction.iloc[worst],residual.iloc[worst])]:
        ax.annotate(tail.ID.iloc[worst],xy=(px,py),xytext=(.32,.35),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':ORANGE},color=ORANGE)
    fig.tight_layout();save(fig,'review_tail_residuals')
    coverage=pd.read_csv(OUT/'conditional_coverage.csv');coverage=coverage[coverage.stratifier.isin(['overall','predicted_tercile'])]
    levels=['all','lower_predicted','middle_predicted','upper_predicted'];labs=['总体','预测低段','预测中段','预测高段']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for offset,(strategy,label,color) in enumerate([('fixed','固定',BLUE),('rolling80_immediate_labels','滚动80条',TEAL)]):
        f=coverage[coverage.strategy.eq(strategy)].set_index('level').reindex(levels);pos=np.arange(4)+(offset-.5)*.28
        axes[0].errorbar(pos,f.coverage,yerr=[f.coverage-f.wilson_low,f.wilson_high-f.coverage],fmt='o',capsize=3,color=color,label=label)
        axes[1].bar(pos,f.mean_width,width=.26,color=color,label=label)
    axes[0].axhline(.9,color='gray',linestyle='--');axes[0].set_ylim(.5,1.03)
    for ax in axes:ax.set_xticks(range(4),labs);ax.legend();ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('覆盖率与Wilson区间');axes[1].set_ylabel('平均区间宽度')
    fig.text(.08,.005,'预测分层边界仅由校准预测值确定；滚动方法假设较早时间组标签立即可用。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'supplement_coverage')
    fair=pd.read_csv(OUT.parent/'fair_comparison'/'summary.csv').sort_values('mse')
    fig,ax=plt.subplots(figsize=(10,4.8));ax.barh(fair.model,fair.mse,color=BLUE);ax.invert_yaxis()
    for i,v in enumerate(fair.mse):ax.text(v+.0003,i,f'{v:.6f}',va='center',fontsize=9)
    ax.set_xlim(0,fair.mse.max()*1.2);ax.set_xlabel('合并外层 MSE');ax.set_title('统一候选数与树轮数的算法敏感性比较',pad=15)
    fig.text(.1,.005,'每类算法比较基础／基础加时间两个候选；树模型500轮；相同轮数不表示相同计算量。',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'fair_algorithms')


if __name__=='__main__':main()
