"""Evidence-bound prose and tables for the research manuscript, stdlib only."""
import csv,json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/"outputs"/"supplement"


def read(name):
    with (OUT/name).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def table(caption,heads,rows):
    return "\n".join([caption,"","| "+" | ".join(heads)+" |","|"+"---|"*len(heads)]+["| "+" | ".join(map(str,row))+" |" for row in rows])


def load():
    primary=read("paired_group_inference.csv")[0]
    values={"group_ci":f"[{float(primary['ci_low']):.6f}, {float(primary['ci_high']):.6f}]",
            "group_p":f"{float(primary['group_sign_flip_p']):.4f}"}
    base=read("baseline_summary.csv")
    values["strong_table"]=table("表5-2 相同外层样本上的算法比较",["方法","MSE","RMSE","MAE"],
        [[r['model'],*[f"{float(r[k]):.6f}" for k in ['mse','rmse','mae']]] for r in sorted(base,key=lambda r:float(r['mse']))])
    best=min(base,key=lambda r:float(r['mse']))
    values["strong_interpretation"]=(f"在预先限定的候选网格下，{best['model']}取得算法对照中的最低合并MSE，为{float(best['mse']):.6f}。"
        "算法均采用相同外层记录和折内特征处理，但候选数及迭代预算并非完全相等，因此排名仅适用于所列搜索范围。"
        "TemporalStacking在每个外层内用GBR、XGBoost、随机森林和SVR的时间OOF预测训练线性元模型，"
        "属于统一协议下的集成对照，未将其声称为其他论文的精确复现。")
    features=read("feature_experiment_summary.csv")
    ab={r['model']:r for r in features if r['scope']=='ablation'}
    labels={'measurements':'测量值及设备','basic':'基础特征','process_only':'基础加工序统计','time_only':'基础加时间特征','full':'组合特征',
            'no_mask':'组合特征去缺失指示','median_only':'全局中位数填充','group_min5':'设备有效观测至少5个','group_min5_shrink10':'至少5个并收缩10'}
    values['ablation_table']=table("表5-3 固定深度2模型的分支消融",['特征或填充配置','MSE','MAE'],
        [[labels[n],f"{float(ab[n]['mse']):.6f}",f"{float(ab[n]['mae']):.6f}"] for n in labels])
    values['ablation_interpretation']=(f"基础特征加入工序统计后MSE为{float(ab['process_only']['mse']):.6f}，"
        f"仅加入时间特征时为{float(ab['time_only']['mse']):.6f}。"
        "组合表示的差异不能全部归因于工序聚合，时间特征提供了更明显的样本区分信息。"
        f"设备组均值收缩后的MSE为{float(ab['group_min5_shrink10']['mse']):.6f}，"
        "说明填充规则值得比较。四种特征表示均纳入八候选嵌套搜索，固定深度消融与开发段选参分别报告。")
    selection=read('selected_configs.csv')
    stage_names={'outer_1':'外层1内部','outer_2':'外层2内部','outer_3':'外层3内部','final_development':'最终开发段'}
    values['selection_table']=table('表5-4 各开发窗口选择的配置',['开发范围','选中配置','内层合并MSE'],
        [[stage_names[r['stage']],r['candidate'],f"{float(r['pooled_mse']):.6f}"] for r in sorted(selection,key=lambda r:(r['stage']=='final_development',r['stage']))])
    sen=[r for r in features if r['scope']=='sensitivity']
    labels_s={'reference_seed42':'参考参数','importance_0.5':'重要性权重0.5','importance_0.9':'重要性权重0.9','drift_0':'漂移幂0','drift_0.5':'漂移幂0.5',
        'top160':'保留160字段','top640':'保留640字段','mask0':'缺失阈值0','mask0.02':'缺失阈值2%','rounds600':'600轮 学习率0.05',
        'lr0.025':'1200轮 学习率0.025','reg0_1':'L1为0 L2为1','reg0.1_10':'L1为0.1 L2为10'}
    values['sensitivity_table']=table('表5-5 单因素敏感性分析',['配置','MSE','MAE'],
        [[labels_s[r['model']],f"{float(r['mse']):.6f}",f"{float(r['mae']):.6f}"] for r in sen])
    jac=read('selection_jaccard.csv')
    pairs=[float(r['jaccard']) for r in jac if r['seed_a']==r['seed_b'] and r['fold_a']!=r['fold_b']]
    dims=read('fold_feature_dimensions.csv');dims=[r for r in dims if r['variant']=='full']
    values['stability_text']=(f"组合特征输入维度在各折及种子间为{min(int(r['engineered']) for r in dims)}至{max(int(r['engineered']) for r in dims)}。"
        f"固定种子跨窗口的字段集合Jaccard系数范围为{min(pairs):.3f}至{max(pairs):.3f}，平均为{sum(pairs)/len(pairs):.3f}。"
        "各次选择均保留至多320字段，字段集合会随训练范围变化，因此单次重要性排序不足以代表稳定工艺因子。")
    rev={r['model']:r for r in features if r['scope']=='reverse_time'}
    values['direction_text']=(f"在同一组300条评价记录上，以较晚记录训练并预测较早记录时，基础特征和组合特征MSE分别为"
        f"{float(rev['basic']['mse']):.6f}和{float(rev['full']['mse']):.6f}。"
        "反向回放改变了训练样本范围和数量，仅用于检查时间方向敏感性，不能将与前向回放的差值单独归因于漂移。"
        "A/B分布与前向窗口并不完全一致，因此前向分数不直接估计竞赛测试误差。")
    cond=[r for r in read('conditional_coverage.csv') if r['stratifier']=='overall']
    values['coverage_table']=table('表5-6 固定与滚动校准的尾段区间',['校准方式','覆盖记录','平均宽度','Wilson区间'],
        [[('固定校准' if r['strategy']=='fixed' else '滚动80条'),f"{r['covered']}/{r['n']}",f"{float(r['mean_width']):.6f}",
          f"[{100*float(r['wilson_low']):.1f}%, {100*float(r['wilson_high']):.1f}%]"] for r in cond])
    tail=read('tail_model_metrics.csv')
    tail_labels={'frozen_full_d3':'组合特征 深度3','full_d2':'组合特征 深度2','no_time_d2':'去时间特征 深度2','absolute_loss_d2':'绝对损失 深度2','low_tail_weight_d2':'低值权重3 深度2'}
    values['tail_table']=table('表5-7 尾段损失函数与低值样本分析',['方案','整体MSE','低值MSE','低值召回率'],
        [[tail_labels[r['model']],f"{float(r['mse']):.6f}",f"{float(r['low_tail_mse']):.6f}",f"{100*float(r['low_value_recall']):.0f}%"] for r in tail])
    fixed=next(r for r in tail if r['model']=='frozen_full_d3')
    values['tail_low_definition']=f"低值阈值取开发段Y的第10百分位，即{float(fixed['low_cutoff_from_development']):.6f}；尾段低于该阈值的记录共{fixed['n_low']}条。"
    latency=json.loads((OUT/'inference_latency.json').read_text(encoding='utf-8'))
    values['latency_text']=(f"在本机模型和输入驻留内存的条件下，三成员集成连同特征转换的单记录预测中位耗时为{latency['medians']['1']:.3f}秒，"
        f"100条批量预测为{latency['medians']['100']:.3f}秒。测试不包含生产采集、网络、排队和质检反馈时间，不能据此承诺产线实时性。")
    return values
