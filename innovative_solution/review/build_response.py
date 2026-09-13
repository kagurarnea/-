"""Generate reviewer response and defense answers outside the thesis narrative."""
from pathlib import Path
import json
import pandas as pd
from innovative_solution.supplement_common import OUT


def main():
    primary=pd.read_csv(OUT/'paired_group_inference.csv').iloc[0]
    baseline=pd.read_csv(OUT/'baseline_summary.csv').set_index('model')
    features=pd.read_csv(OUT/'feature_experiment_summary.csv')
    ab=features[features.scope.eq('ablation')].set_index('model')
    jac=pd.read_csv(OUT/'selection_jaccard.csv');matched=jac[(jac.seed_a==jac.seed_b)&(jac.fold_a!=jac.fold_b)]
    dims=pd.read_csv(OUT/'fold_feature_dimensions.csv');dims=dims[dims.variant.eq('full')]
    latency=json.loads((OUT/'inference_latency.json').read_text(encoding='utf-8'))
    answers=[
      ('预测时点与泄漏边界',
       '本文明确定位为完整记录已采集后的离线终点特性估计。实际Y在何时测量、每个字段何时可用，数据没有提供可核实的业务字典。去掉时间特征已做实验，但仍无法证明其余匿名字段没有质检后信息；中途预警和自动放行不作为已完成成果。',
       'thesis/manuscript.md 第1.2、3.3、5.5节；timestamp_availability_audit.csv'),
      ('时间代理及方向不一致',
       '行中位数只是稳健排序代理。最小值/最大值代理与中位数的Spearman相关约0.959/0.948，尾段100条交集仅81/61。A/B大多数记录早于训练范围，故额外比较了相同300条评价记录上的反向回放；基础/完整MSE约0.028100/0.027704。反向训练样本数量和范围不同，不能用前后差值单独证明时间漂移。',
       'time_proxy_sensitivity.csv；feature_experiment_predictions.csv 的reverse_time范围'),
      ('训练与A/B交叉及规则',
       '训练-A、训练-B、A-B两两的相同ID数量和完全相同输入对均为0。检查包含全部输入，并在哈希匹配后再次核对实际值。竞赛官方页面只返回网站导航，未核到明确允许A训练B的条款；公开A答案本身不等同训练许可。B合并标签模型作为增量学习场景，并保留仅训练标签的B预测，未执行线上提交。',
       'cross_dataset_identity_audit.csv；review/competition_rule_evidence.md'),
      ('3.48%改善的统计证据',
       f'300条记录对应296个关联组。配对整组bootstrap 5000次，基础减完整的MSE差为{primary.mse_difference_ref_minus_candidate:.6f}，95%区间[{primary.ci_low:.6f},{primary.ci_high:.6f}]；组级符号翻转p={primary.group_sign_flip_p:.4f}。组等权及组块长5/10/20结果也跨0。已撤去稳定改善结论。区间以现有预测为条件，不能消除研究者选择历史；没有Y单位和业务阈值，也不能判断0.00275的RMSE差是否有实际工艺价值。',
       'paired_group_inference.csv；group_equal_metrics.csv'),
      ('内层选择频率与不稳定性',
       '三个外层选择full_d2、raw_d3、raw_d2，各一次；最终开发选择full_d3。已列全部内层分数、各候选范围与热图。少量验证样本及不同窗口会造成排序变化，选参流程外层得分较差是负向发现，但不能仅凭此区分过拟合、窗口差异及随机性各自影响。',
       'selected_configs.csv；selection_score_distribution.csv；selection_frequency.csv'),
      ('尾段误差升高的原因',
       'NH1835贡献尾段36.1%的平方误差，是明显影响点。另存开发、校准、尾段目标分布和320个入选变量的KS距离、缺失及中位数差异。深度2完整模型尾段MSE约0.042428，与full_d3的0.042649接近，因此不能把较大误差只归因于选到深度3。漂移、局部困难样本与模型失配尚不能做因果拆分。',
       'tail_error_influence.csv；target_segment_distribution.csv；tail_covariate_shift.csv；tail_model_metrics.csv'),
      ('参数敏感性与筛选稳定性',
       f'已做13组单种子参考/敏感性配置，包括重要性权重0.5/0.7/0.9、漂移幂0/0.25/0.5、字段160/320/640、缺失阈值0/0.5%/2%、迭代轮数学习率及正则。完整特征维度为{dims.engineered.min()}至{dims.engineered.max()}，固定种子跨折Jaccard为{matched.jaccard.min():.3f}至{matched.jaccard.max():.3f}。敏感性结果显示参数不是已证明最优，不按其最低分反向改写主要评价。',
       'feature_experiment_summary.csv；fold_feature_dimensions.csv；selected_fields.csv；selection_jaccard.csv'),
      ('设备组支持数与回退',
       '设备对应关系按字段命名及列位置推断，不能冒充已确认物理路线。有效计数按设备-字段的非缺失观测计算。新增至少5个有效观测及向训练中位数收缩10个先验观测的策略，并与至少1个及全局中位数对照。只有1条且该字段缺失时有效计数为0，直接回退；不会用缺失记录自身算均值。未知设备回退训练中位数是可复现工程规则，不是物理值恢复。',
       'supplement_common.py SupportedGroupBuilder；device_group_support.csv；test_supplement.py'),
      ('强基线与同样本比较',
       '已运行Ridge、RandomForest、GBR、SVR、LightGBM、CatBoost、XGBoost及时间OOF Stacking，全部使用相同300条外层记录。各方法在两个内层窗口选择预设参数，外层三个种子平均；参数网格与迭代预算明示，并非声称全球最优。Stacking保留GBR/XGB/RF/SVR加线性元模型的组合，但采用统一时间窗口，属于适配对照而非他人论文精确复现。',
       'baseline_grid.json；baseline_inner_scores.csv；baseline_choices.csv；baseline_summary.csv'),
      ('区间校准及条件覆盖',
       '固定区间为87/100，宽约0.490183；描述性Wilson区间约[79.0%,92.2%]，包含90%。在较早时间组标签立即可用的模拟下，滚动80残差得到90/100，宽约0.540555。已报告按14个类别字段及校准预测三分位分层的覆盖、样本数、宽度和Wilson区间；小于20条标注不稳定。没有真实质检延迟和交换性证据，不作分布无关保证。',
       'conditional_coverage.csv；conditional_interval_predictions.csv；supplement_tail.py'),
      ('极端样本改进与尾部指标',
       '低值阈值使用开发Y第10百分位2.539856，尾段共有5条，不是产品合格限。已比较full_d2、绝对损失、低值权重3和无时间分支。绝对损失与加权均未解决NH1835误差；无时间分支低值召回1/5但整体MSE约0.046684。报告低值MSE/MAE、低值召回、最大误差及中位数pinball损失，保留全部样本。',
       'tail_model_metrics.csv；NH1835_feature_diagnosis.csv'),
      ('实际应用价值与时延',
       f'Y单位、质量上下限、测量误差和误判成本未知，因此不能定量声称节省抽检或提高良率。本机模型/数据驻留内存的端到端三成员预测中位耗时：单记录{latency["medians"]["1"]:.3f}秒，100条批量{latency["medians"]["100"]:.3f}秒，包含特征转换，不含采集、排队、网络和质检。当前用途限于离线估计、变量筛查与人工复核研究。',
       'inference_latency.json；论文第5.9节'),
      ('各分支增量贡献',
       f'基础MSE {float(ab.loc["basic","mse"]):.6f}，基础加工序{float(ab.loc["process_only","mse"]):.6f}，基础加时间{float(ab.loc["time_only","mse"]):.6f}，完整{float(ab.loc["full","mse"]):.6f}。工序单独差异的区间跨0，时间分支差异区间不跨0（探索性Holm p约0.0048）。去缺失指示的未经校正p约0.048，经多比较后不显著；设备收缩策略区间也跨0。因此不能把所有分支都描述为有效改进。',
       'feature_experiment_summary.csv；paired_group_inference.csv'),
      ('第三方复现与获取方式',
       '本地提供源码与结果复现ZIP、REPRODUCE.md、协议、版本、哈希、逐记录预测和作图代码，尚未发布公共仓库。由于竞赛再分发许可未知，不包含原Excel或可能内嵌数据的joblib模型；本地原模型仍保留。完整重训需要读者合法获得相同数据，无数据权限时可以检查预测表的统计结论。不能用哈希替代访问权限或宣称跨环境位级一致。',
       'REPRODUCE.md；reproducibility_manifest.json；requirements-research.txt'),
      ('文字 图表 公式 引用核验',
       '核查实际正文为“冗余变量”“回顾性”，没有复现所述“回顺性”错字；“余变量”可能是对冗余变量的截取。图1-1有真实嵌图，并在Word导出中显示；可能存在阅读或提取方式差异。继续系统检查中英文摘要、公式变量和17条参考文献首次编号。新增LightGBM、CatBoost引用根据NeurIPS原始出版页面核对。最终论文保持学生研究者口吻，本答复与审稿记录单独保存。',
       'thesis/paper_render/structural_qa.json；thesis/citation_map.json；sources/source_access.json'),
    ]
    parts=['# 审稿意见处理与答辩回答','',
       '本文件记录意见核验与证据，论文正文仍以研究者口吻描述一项完整实验。已完成的计算不等于补齐业务数据：预测时点、Y物理意义、真实批次和A标签使用条款仍需外部确认。','',
       '## 处理结论','',
       '完成强基线、分支消融、单因素敏感性、关联组推断、字段稳定性、跨文件重复检查、时间方向检查、条件/滚动覆盖、低值样本策略及本机时延。未宣称3.48%为稳定提升，未虚构新盲测、B真值、物理参数或公开代码仓库。','']
    for i,(title,answer,evidence) in enumerate(answers,1):
        parts += [f'## {i}. {title}','',answer,'',f'证据：{evidence}。除另注明外，表格均位于outputs/supplement。','']
    parts += ['## 算法结果速查','','| 方法 | MSE | RMSE | MAE |','|---|---:|---:|---:|']
    for name,r in baseline.sort_values('mse').iterrows():parts.append(f'| {name} | {r.mse:.6f} | {r.rmse:.6f} | {r.mae:.6f} |')
    (Path(__file__).parent/'审稿意见处理与答辩回答.md').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':main()
