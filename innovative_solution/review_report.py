"""Replace overclaims with versioned evidence and a reproducible revision ledger."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / "outputs"
REVIEW = OUTPUT / "review"

FINDINGS = [
    ("R01", "严重", "A/B阶段被等同生产未来", "A有250条、B有373条早于训练最早时间代理；A行序相关仅0.084", "用时间代理排序的历史回放；声明代理非真实完工时间", "已修正验证与论文"),
    ("R02", "严重", "同产品跨训练验证边界", "原随机CV有42条验证记录同ID已见，时间外推边界4条", "同ID或相同输入连通分组，逐边界移除训练侧冲突，保存索引审计", "已修复并重跑"),
    ("R03", "严重", "全800行提前筛列", "全空、常量、重复及时间类型先在全训练文件确定", "原始numeric保留到每折，训练内重新识别列类型并学习筛选", "已修复并测试"),
    ("R04", "严重", "内层网格泄漏", "先用整个外层y做特征选择再内层CV", "每个内层训练部分从原始字段重建预处理与监督选择", "已修复并重跑"),
    ("R05", "严重", "已看A答案仍宣称独立验证", "历史诊断已读取并比较A答案及多个候选", "区分程序未用A拟合与研究历史已看标签；A只报告回顾性结果", "已更正文稿与元数据"),
    ("R06", "严重", "共形区间覆盖承诺", "旧A实际256/300覆盖85.33%；同残差参与选择/校准且交换性未知", "冻结模型、分离校准和尾段评价；称经验区间；全量重训仅导出点预测", "已修正设计"),
    ("R07", "主要", "权重2显著优于1的暗示", "MSE仅差0.00004672，block10探索区间[-0.000749,0.000867]跨0", "撤回稳定最优主张；新B默认等权新增标签", "已修改默认策略"),
    ("R08", "主要", "B基线方法误标", "train_only_robust_blend实为当前uniform_xgb预测", "旧结果标记误名并说明真实来源，新实验从冻结方法生成基线", "已修正文稿与流程"),
    ("R09", "主要", "KS统计解释错误", "同一数据估参并用普通KS分布算p值，且传感器离散化", "保留AIC/KS距离作为描述，不作5%一致性结论", "已修复分析解释"),
    ("R10", "主要", "重要性与物理流程混淆", "设备映射来自列位置，路线邻接来自编号排序，重要性是增益而非因果", "明确启发式映射与匿名物理边界，撤回自动工艺放行建议", "已修改论文"),
    ("R11", "主要", "元数据核实被写成内容核实", "部分仅核DOI，未读全文；数字引用不按首次出现排序", "按支持度限缩引用、按首次出现编号、保存来源审计", "已修正引用"),
    ("R12", "主要", "任意组合图表与单次最低分即称创新", "分折样本不同、复杂度和数据量同时变化", "同一外折比较固定原始视图与完整视图，降低创新强度", "已补公平对照"),
    ("R13", "主要", "复现资料不足", "缺模型状态、缺阶段源码/数据哈希，无A答案时入口崩溃", "保存拟合器、模型、特征名、协议、版本与输入哈希；更新入口", "已修复"),
    ("R14", "主要", "缺失统计分母混淆", "原始5952列800条均有缺失，4.484%单元缺失；609条和13.1%来自旧2617列视图", "按原始和清理视图分别报告分母、单元缺失率与行缺失率", "已审计并修正文稿"),
]


def generate_review_report():
    results = json.loads((REVIEW / "review_manifest.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(REVIEW / "nested_summary.csv")
    a_metrics = results.get("retrospective_A_metrics")
    a_note = (f"当前A MSE={a_metrics['mse']:.6f}只能作为回顾性描述。B使用800+300条已公布标签等权重训，无B真值成绩。"
              if a_metrics else "本次未提供A答案，不计算A分数；B采用原训练标签拟合的模型。")
    ledger = pd.DataFrame(FINDINGS, columns=["编号", "等级", "原问题", "证据", "修正", "状态"])
    ledger.to_csv(REVIEW / "revision_ledger.csv", index=False, encoding="utf-8-sig")
    lines = ["# 质量预测方案审稿修订记录", "",
        "审稿结论：原稿存在验证设计和结论强度不匹配的问题，不能按原有表述送审。现版本完成代码修复与历史回放重跑，研究定位为匿名TFT-LCD数据的机器学习比较与验证方法研究；没有全新独立测试集，也没有上线验证。", "",
        "## 实质修订", ""]
    for no, severity, problem, evidence, fix, status in FINDINGS:
        lines += [f"### {no} {severity} {problem}", "", f"证据：{evidence}。", "", f"修正：{fix}。{status}。", ""]
    lines += ["## 本次重新验证", "",
        "历史800条先按既有时间代理排序。三个外层窗口分别以前320、420、520条为候选训练范围，评价随后100条。每个内层使用两个60条验证段，所有预处理与监督选择重新拟合；同实体及时间边界相交的训练记录被排除。", "",
        "| 方案 | 样本数 | 合并MSE | RMSE | MAE | R² |", "|---|---:|---:|---:|---:|---:|"]
    for row in scores.itertuples():
        lines += [f"| {row.model} | {row.n} | {row.mse:.6f} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.4f} |"]
    lines += ["", f"独立于本轮选参用途的校准段{results['n_calibration']}条、尾段{results['n_tail']}条；冻结模型只用前段{results['n_development']}条拟合，尾段MSE={results['tail_metrics']['mse']:.6f}，经验区间实测覆盖率={results['observed_tail_coverage']:.1%}。尾段在项目历史中仍曾被接触，因此不称盲测。", "",
        f"最终方法由前段内层分数选择为`{results['selected_candidate']}`。A标签没有用于本次数值拟合，但A此前已被反复查看。{a_note}全量重训文件不附未经验证的区间。", "",
        "嵌套选择的外层MSE高于两种固定对照，不能声称自动选择带来提升。冻结尾段87%的实测覆盖未达到名义90%。更严格的修订并不保证分数降低。", "",
        "## 输出与使用", "",
        "- 修订实验与新提交位于 `outputs/review/`；原 `outputs/` CSV和图表保留为历史探索，不能与新指标混算。",
        "- `split_audit.csv`保存每折索引、实体交集和时间边界，`inner_selection.csv`保存全部内层分数。",
        "- `protocol.json`保存运行前协议；`review_manifest.json`保存结果、软件版本、输入及源码SHA256。",
        "- `archive/review_validation_executed.py`对应本次执行哈希；`source_revision_after_run.json`记录其后边界断言和缺A兼容修改。原运行清单未改写。",
        "- `cache_source_verification.json`核验训练/A/B缓存与原Excel的列、行及全部单元格精确一致；源文件哈希与运行清单一致。",
        "- `model_A.joblib`和`model_B.joblib`包含三个成员的完整预处理器、字段顺序和预测器。",
        "- `frozen_historical_model.joblib`仅用于分离校准的历史评价，其区间不转移给全量重训模型。",
        "", "复核：23项针对性测试通过，其中无A答案分支使用轻量学习器替身验证数据权限与输出；完整模型已在实际数据上训练。重载A模型对头8条记录重新预测，与CSV最大差4.44×10⁻¹⁶。A/B提交分别300/412行、两列、无表头，ID原序不变，预测全部有限。", "",
        "", "运行 `python -m innovative_solution.run` 重做修订版实验；`python -m innovative_solution.report_only` 仅刷新文字报告。", "",
        "## 仍需外部证据", "",
        "新增生产周期盲测、字段单位与设备工艺字典、真批次与产品键、有效完工时间、质量合格限和产线部署时延均未取得。当前论文保留这些局限，不通过措辞将其替代为已完成验证。", ""]
    report = "\n".join(lines)
    (REVIEW / "REVIEW_REPORT.md").write_text(report, encoding="utf-8")
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    workflow = "\n".join(["# 修订后完整工作流程", "",
        "本流程是回顾性研究。第一版的参数、结果和论文仍保留用于审计，新结果统一来自outputs/review。", "",
        "1. 读取原始Excel并校验输入哈希；审计ID、缺失、时间代理和标签来源。", 
        "2. 将同ID或完全相同输入记录合并为验证实体，只用于切分隔离，不删除原始记录。", 
        "3. 按时间代理排序并生成嵌套窗口；真实工艺时间未验证，竞赛A/B顺序不参与排序。",
        "4. 每个训练折内重新识别全空/常量/重复/时间字段，再拟合设备条件填充、缺失指示和One-Hot。",
        "5. 从训练折内拟合ExtraTrees贡献、相关性、漂移排序，保留关键原始变量；计算稳健工序与时间代理特征。",
        "6. 内层比较原始视图/完整视图和深度2/3共四组配置；三个预设种子等权平均，依据合并MSE选择。",
        "7. 外层评价内层选择的完整算法，与固定raw_d2、full_d2及均值基线按同一批样本比较。",
        "8. 最终前段选参后冻结模型，在独立校准用途的数据上确定经验半径，在随后尾段检查误差与覆盖。",
        "9. 用全部800条重训生成A点预测；答案只作本次回顾性评价，承认历史已看A标签。",
        "10. 用800+300条已公布标签等权重训生成B点预测；缺少B真值，不报告B准确率或区间保证。",
        "11. 保存拟合器、模型、特征顺序、窗口索引、配置、版本、代码/数据哈希及两列无表头CSV。",
        "12. 论文引用按支持内容逐条限缩，历史指标标注开发性质，新指标直接从清单生成。", "",
        "具体修订与结果见review/REVIEW_REPORT.md。"])
    (OUTPUT / "WORKFLOW_DESCRIPTION.md").write_text(workflow, encoding="utf-8")
    return report


if __name__ == "__main__":
    generate_review_report()
