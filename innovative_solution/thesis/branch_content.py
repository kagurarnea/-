"""Manuscript content computed from the eight-candidate experiment."""
import csv
import json
from pathlib import Path
from .supplement_content import table

OUT = Path(__file__).resolve().parents[1] / "outputs" / "branch_selection"
VIEWS = {"raw": "基础特征", "process": "基础加工序统计", "time": "基础加时间特征", "full": "组合特征"}


def read(file):
    with (OUT / file).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def label(candidate):
    view, depth = candidate.rsplit("_d", 1)
    return f"{VIEWS[view]} 深度{depth}"


def effect(value):
    number=float(value)
    return f"{number:.2e}" if 0 < abs(number) < .0000005 else f"{number:.6f}"


def load():
    manifest = json.loads((OUT / "review_manifest.json").read_text(encoding="utf-8"))
    score = {r["model"]:r for r in read("nested_summary.csv")}
    selected = manifest["selected_candidate"]
    values = {"selected": selected, "selected_description": label(selected),
        "nested_mse": f"{float(score['nested_selected']['mse']):.6f}",
        "time_mse": f"{float(score['fixed_time_d2']['mse']):.6f}",
        "coverage": f"{manifest['observed_tail_coverage']:.0%}",
        "radius": f"{manifest['empirical_radius']:.6f}", "width": f"{2*manifest['empirical_radius']:.6f}",
        "covered_n": round(manifest['observed_tail_coverage']*manifest['n_tail']),
        "n_dev": manifest['n_development'], "n_cal": manifest['n_calibration'], "n_tail": manifest['n_tail']}
    for prefix, metrics in [("tail",manifest["tail_metrics"]),("A",manifest["retrospective_A_metrics"])]:
        values.update({f"{prefix}_{key}": f"{value:.6f}" for key,value in metrics.items()})
    old_file = OUT.parent / "review" / "nested_summary.csv"
    with old_file.open(encoding="utf-8-sig",newline="") as f:
        restricted = next(r for r in csv.DictReader(f) if r["model"]=="nested_selected")
    rows = []
    for name in ["mean_baseline","fixed_raw_d2","fixed_process_d2","fixed_time_d2","fixed_full_d2"]:
        r = score[name]
        rows.append(["训练均值基线" if name=="mean_baseline" else label(name.removeprefix("fixed_")),
                     *[f"{float(r[k]):.6f}" for k in ["mse","rmse","mae","r2"]]])
    for name,r in [("四候选嵌套选择",restricted),("八候选嵌套选择",score["nested_selected"])]:
        rows.append([name,*[f"{float(r[k]):.6f}" for k in ["mse","rmse","mae","r2"]]])
    values["nested_table"] = table("表5-1 同一300条外层记录的特征与选择策略比较",["方法","MSE","RMSE","MAE","R²"],rows)
    values["nested_interpretation"] = (f"四种深度2固定表示中，基础加时间特征的MSE最低，为{values['time_mse']}。"
        "加入工序统计并未在此基础上继续降低误差，组合特征不能据名称视为更优方案。"
        f"四候选与八候选选择流程的MSE分别为{float(restricted['mse']):.6f}和{values['nested_mse']}；"
        "二者评价的是内层选择规则在外层的表现，不等同于某个固定配置的分数。")
    selections = read("selected_configs.csv")
    stages = {"outer_1":"外层1内部","outer_2":"外层2内部","outer_3":"外层3内部","final_development":"最终开发段"}
    selections.sort(key=lambda r:(r["stage"]=="final_development",r["stage"]))
    values["selection_table"] = table("表5-4 八候选搜索的窗口选择结果",["开发范围","选中配置","内层合并MSE"],
        [[stages[r['stage']],r['candidate'],f"{float(r['pooled_mse']):.6f}"] for r in selections])
    inner = read("inner_selection.csv")
    final = {r["candidate"]:r for r in read("selection_score_distribution.csv") if r["stage"]=="final_development"}
    devrows = []
    for candidate in sorted(final,key=lambda c:float(final[c]['pooled_mse'])):
        folds = sorted([r for r in inner if r['stage']=='final_development' and r['candidate']==candidate],key=lambda r:int(r['inner_fold']))
        devrows.append([candidate,*[f"{float(r['mse']):.6f}" for r in folds],f"{float(final[candidate]['pooled_mse']):.6f}"])
    values["development_candidates_table"] = table("表5-4a 最终开发段全部候选的内层分数",["候选配置","内层1 MSE","内层2 MSE","合并MSE"],devrows)
    choices = "、".join(r['candidate'] for r in selections if r['stage'].startswith('outer_'))
    values["selection_interpretation"] = (f"三个外层的内层分别选择{choices}。最终开发段选择{selected}，即{label(selected)}，"
        f"其内层合并MSE为{float(final[selected]['pooled_mse']):.6f}。"
        "表5-4a同时展示全部八个候选，基础加时间的两个深度均实际参与比较。"
        "最终配置由开发段内层排名决定，固定外层最低分仅作为表示比较的结果，不用于再次覆盖选参决定。"
        f"固定{selected}的外层MSE为{float(score['fixed_'+selected]['mse']):.6f}，并非固定配置中最低。"
        "这种开发段与外层排名不一致体现了窗口依赖，不能由一次选中推断分支稳定有效。")
    infer = read("paired_group_inference.csv")
    compare = next(r for r in infer if r['reference']=='fixed_full_d2' and r['candidate']=='fixed_time_d2')
    values["time_vs_combined"] = (f"组合特征减去基础加时间的MSE差为{float(compare['mse_difference_ref_minus_candidate']):.6f}，"
        f"配对关联组95%区间为[{float(compare['ci_low']):.6f}, {float(compare['ci_high']):.6f}]。"
        "该比较衡量已有时间分支时增加工序统计的效果，区间以现有预测为条件。")
    worst = read("tail_error_influence.csv")[0]
    values["worst_sample_text"] = (f"尾段绝对残差最大的记录为{worst['ID']}，实际值为{float(worst['actual']):.4f}，"
        f"预测值为{float(worst['prediction']):.4f}，残差为{float(worst['residual']):.4f}。"
        f"该记录贡献{100*float(worst['fraction_total_squared_error']):.1f}%的尾段总平方误差。")
    coverage = [r for r in read("conditional_coverage.csv") if r['stratifier']=='overall']
    values["coverage_table"] = table("表5-6 选定模型的固定与滚动校准",["校准方式","覆盖记录","平均宽度","Wilson区间"],
        [["固定校准" if r['strategy']=='fixed' else "滚动80条",f"{r['covered']}/{r['n']}",f"{float(r['mean_width']):.6f}",
          f"[{100*float(r['wilson_low']):.1f}%, {100*float(r['wilson_high']):.1f}%]"] for r in coverage])
    rolling = next(r for r in coverage if r['strategy']!='fixed')
    values["rolling_text"] = f"滚动校准覆盖{rolling['covered']}/{rolling['n']}条记录，平均宽度为{float(rolling['mean_width']):.6f}。"
    tailrow = read("selected_tail_metrics.csv")[0]
    values["selected_low_text"] = (f"八候选选定模型在{tailrow['n_low']}条低值记录上的MSE为{float(tailrow['low_tail_mse']):.6f}，"
        f"低值召回率为{float(tailrow['low_value_recall']):.0%}。")
    latency = json.loads((OUT/'inference_latency.json').read_text(encoding='utf-8'))
    values['latency_text'] = (f"选定的{label(selected)}三成员模型在本机的单记录预测中位耗时为{latency['medians']['1']:.3f}秒，"
        f"100条批量预测为{latency['medians']['100']:.3f}秒，包含输入驻留内存时的特征转换与模型计算。"
        "测试不包含生产采集、网络、排队和质检反馈时间，不能据此承诺产线实时性。")
    with (OUT.parent/'fair_comparison'/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
        fair=sorted(csv.DictReader(f),key=lambda r:float(r['mse']))
    values['fair_table']=table('表5-2a 统一候选数与树轮数的敏感性比较',['算法','MSE','MAE','累计学习器秒数'],
        [[r['model'],f"{float(r['mse']):.6f}",f"{float(r['mae']):.6f}",f"{float(r['learner_seconds']):.1f}"] for r in fair])
    values['fair_interpretation']=(f"在两个特征候选及固定参数条件下，{fair[0]['model']}的外层MSE最低，为{float(fair[0]['mse']):.6f}。"
        "耗时累计该算法全部内层拟合与外层三成员拟合、预测，公共预处理耗时另存。"
        "测量来自本机运行，可能受并发负载影响，不能作为严格效率排名。"
        "该比较检验有限预算设置的敏感性，尚未覆盖CatBoost原生类别处理和表格基础模型，因此不作算法普遍优劣或前沿最佳性能的判断。")
    family=read('inference_family.csv')
    time_stat=next(r for r in family if r['reference']=='basic' and r['candidate']=='time_only')
    relative=100*float(time_stat['mse_difference_ref_minus_candidate'])/.02453254866743879
    values['time_stat_text']=(f"基础减去基础加时间的MSE差为{float(time_stat['mse_difference_ref_minus_candidate']):.6f}，"
        f"相对基础MSE降低约{relative:.2f}%；95%区间为[{float(time_stat['ci_low']):.6f}, {float(time_stat['ci_high']):.6f}]，"
        f"原始p为{float(time_stat['group_sign_flip_p']):.6f}，在{len(family)}项比较中Holm校正p为{float(time_stat['holm_p']):.6f}。")
    values['inference_table']=table('表5-3a 配对关联组误差差异与多重比较',['比较 参考减候选','MSE差及95%区间','原始p','Holm p'],
        [[f"{r['reference_label']}－{r['candidate_label']}",f"{effect(r['mse_difference_ref_minus_candidate'])} [{effect(r['ci_low'])}, {effect(r['ci_high'])}]",
          f"{float(r['group_sign_flip_p']):.6f}",f"{float(r['holm_p']):.6f}"] for r in family])
    return values
