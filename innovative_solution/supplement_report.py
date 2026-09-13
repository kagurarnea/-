"""Research-facing experiment report, separate from reviewer correspondence."""
from pathlib import Path
import pandas as pd
from .supplement_common import OUT,BASE


def md(frame, columns):
    names=list(columns);lines=['| '+' | '.join(columns.values())+' |','|'+'---|'*len(names)]
    for _,r in frame.iterrows():
        values=[f'{r[n]:.6f}' if isinstance(r[n],float) else str(r[n]) for n in names]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def generate():
    baselines=pd.read_csv(OUT/'baseline_summary.csv').sort_values('mse')
    ab=pd.read_csv(OUT/'feature_experiment_summary.csv')
    stats=pd.read_csv(OUT/'paired_group_inference.csv')
    coverage=pd.read_csv(OUT/'conditional_coverage.csv')
    sections=['# TFT-LCD质量特性预测实验报告','',
        '研究对象为800条训练记录及A/B测试记录的连续Y回归。模型先在训练折拟合数据处理与工序特征，再进行嵌套窗口、固定对照、分支消融与误差诊断。所有主比较均保存逐记录结果。',
        '', '## 主要证据','',
        '基础与完整特征的300条外层记录属于296个关联组。MSE差约0.000853，整组bootstrap区间[-0.000628,0.002348]，组级符号翻转p约0.2683；尚未获得稳定改善证据。时间特征单独加入时误差差异更明确，不能把完整特征的全部差异归功于工序聚合。',
        '', '## 同样本算法比较','',md(baselines,{'model':'方法','mse':'MSE','rmse':'RMSE','mae':'MAE','r2':'R²'}),
        '', '算法由内层有限网格选参、外层三种子平均。各算法预算明示但不完全相等；Stacking为统一时间OOF协议下的适配对照。',
        '', '## 特征与填充消融','',md(ab[ab.scope.eq('ablation')],{'model':'配置','mse':'MSE','mae':'MAE'}),
        '', '## 单因素敏感性','',md(ab[ab.scope.eq('sensitivity')],{'model':'配置','mse':'MSE','mae':'MAE'}),
        '', '敏感性固定种子42，不与三成员结果混算改进率。',
        '', '## 区间与尾段','',md(coverage[coverage.stratifier.eq('overall')],{'strategy':'策略','n':'记录数','coverage':'覆盖率','mean_width':'平均宽度'}),
        '', '滚动校准假设较早时间组标签立即可用；其覆盖提升伴随宽度增加，不代表现场时延条件已成立。NH1835贡献约36.1%的尾段总平方误差，绝对损失与低值加权未消除该误差。',
        '', '## 使用与复现','',
        '训练集、A、B两两无相同ID或完全相同输入。A回顾性MSE为0.031630；B无标签。A标签训练B的具体竞赛条款尚未核实，合并标签模型按研究场景解释，另保留训练标签B预测。',
        '', '运行python -m innovative_solution.run生成主要模型，运行python -m innovative_solution.run_supplement完成有限补充比较。第三方数据权限、完整复现步骤及环境见REPRODUCE.md。',
        '', '论文正文与实验结果是同一项研究；意见处理和15项答辩回答单独保存于review/审稿意见处理与答辩回答.md。',
        '', '## 图表','']
    for name,title in [('supplement_baselines','多算法对比'),('supplement_ablation','消融和配对区间'),('supplement_selection','参数和字段稳定性'),('supplement_coverage','条件覆盖与宽度')]:
        sections += [f'![{title}](plots/{name}.png)','']
    text='\n'.join(sections)
    (OUT/'EXPERIMENT_REPORT.md').write_text(text,encoding='utf-8')
    # Correct relative image paths in the main output index.
    (OUT.parent/'REPORT.md').write_text(text.replace('](plots/','](supplement/plots/'),encoding='utf-8')


if __name__=='__main__':generate()
