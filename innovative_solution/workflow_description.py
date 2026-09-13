from __future__ import annotations

import json

import pandas as pd

from .config import PipelineConfig


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def generate_workflow_description(config: PipelineConfig) -> None:
    output = config.output_dir
    if (output / "review" / "review_manifest.json").exists():
        from .review_report import generate_review_report
        generate_review_report()
        return
    base = _json(output / "run_manifest.json")
    advanced = _json(output / "advanced_summary.json")
    phase_a = _json(output / "phase_a_manifest.json")
    phase_b = _json(output / "phase_b_manifest.json")
    model_summary = pd.read_csv(output / "model_summary.csv")
    feature_audit = pd.read_csv(output / "feature_audit.csv")
    decision_log = pd.read_csv(output / "decision_log.csv")
    transform = pd.read_csv(output / "transform_reducer_summary.csv")
    ablation = pd.read_csv(output / "ablation_summary.csv")
    residual_tests = pd.read_csv(output / "residual_statistical_tests.csv")

    audit_counts = feature_audit["status"].value_counts().to_dict()
    random_best = model_summary.loc[model_summary.scope == "random_oof"].nsmallest(
        1, "mse"
    ).iloc[0]
    rolling_best = model_summary.loc[model_summary.scope == "rolling_oof"].nsmallest(
        1, "mse"
    ).iloc[0]
    holdout_best = model_summary.loc[
        model_summary.scope == "temporal_holdout"
    ].nsmallest(1, "mse").iloc[0]
    best_transform = transform.nsmallest(1, "mse_mean").iloc[0]
    best_ablation = ablation.nsmallest(1, "mse_mean").iloc[0]
    significant_tests = residual_tests.loc[residual_tests.pvalue < 0.05, "test"].tolist()

    modified = decision_log.loc[
        decision_log.status.isin(["modified", "reverted"])
    ]
    discarded = decision_log.loc[
        decision_log.status.isin(
            ["discarded", "discarded_from_final", "analysis_only", "reserved"]
        )
    ]
    modified_text = "\n".join(
        f"- **{row.stage} / {row.candidate_or_change}**：{row.evidence}；"
        f"处理为“{row.decision}”（状态：{row.status}）。"
        for row in modified.itertuples(index=False)
    )
    discarded_text = "\n".join(
        f"- **{row.stage} / {row.candidate_or_change}**：{row.evidence}；"
        f"处理为“{row.decision}”（状态：{row.status}）。"
        for row in discarded.itertuples(index=False)
    )

    text = f"""# 产品特性预测完整工作过程文字描述

## 1. 研究目标与方案定位

本项目针对TFT-LCD多工序生产过程建立连续特性值Y的回归预测系统。原始变量覆盖机台温度、气体与液体流量、功率、制程时间和设备标识等信息，具有典型的小样本、高维、强共线、混合量纲、缺失和批次漂移特征。方案目标不只是获得单次预测值，还要形成可复现的数据治理、验证、解释、残差诊断、样本追溯和竞赛提交链路。

当前数据包含{base['train_rows']}条原训练样本、{base['test_a_rows']}条测试A和{base['test_b_rows']}条测试B。原训练集是A阶段唯一可用于拟合和选择的数据；A榜答案公布后，可作为B阶段新增监督信息，但B标签始终不可见。项目通过两个独立阶段模块把这一权限边界固化，避免把事后答案泄漏包装成模型能力。

## 2. 数据读取、结构审计与复现

程序从原始Excel读取数据，并建立本地缓存以减少重复运行时间。第一列ID只用于顺序核验、样本追溯和提交，不直接作为普通连续变量建模；训练集目标列转为数值并检查有限性。清洗后保留{base['clean_numeric_features']}个数值变量、{base['categorical_features']}个设备/类别变量，并识别{base['timestamp_features']}个时间字段。

字段审计逐列记录位置、所属工序、缺失率、零值率、唯一值数量、时间戳比例和最终状态。审计结果中，全空删除{audit_counts.get('removed_all_nan', 0)}列、常量删除{audit_counts.get('removed_constant', 0)}列、重复删除{audit_counts.get('removed_duplicate', 0)}列、转换时间字段{audit_counts.get('converted_timestamp', 0)}列、保留数值字段{audit_counts.get('kept_numeric', 0)}列。所有配置、软件版本、输入哈希、随机种子和运行耗时分别保存在`run_manifest.json`、`reproducibility_snapshot.json`和运行轨迹文件中。

## 3. 缺失值处理及物理含义

缺失分析同时覆盖列、行、工序和设备层级。共有{advanced['rows_with_any_missing']}条训练样本至少含一个缺失值，单样本最高缺失率为{advanced['row_missing_rate_max']:.1%}。`row_missing_report.csv`逐行记录缺失数量与缺失最严重工序，`row_operation_missing_matrix.csv`保存样本乘工序的缺失矩阵，`missing_mechanism_report.csv`用于检查缺失是否与设备或工序有关。

填充严格在训练折内拟合。首先按字段所属工序找到对应设备列，使用同设备组均值填补；未见设备或设备组统计不可用时，再以训练折全局中位数兜底。对达到阈值的字段保留缺失指示变量，使模型能够区分“真实测量值”和“由缺失机制产生的替代值”。这一处理只给出统计含义：缺失可能来自仪表离线、工序未经过、设备版本差异或采集失败，真实物理原因仍需结合MES和设备日志确认。

## 4. 类别、时间、异常与量纲处理

TOOL、Tool、TOOL_ID等字段被视为名义类别，在训练折内执行未知类别安全的One-Hot编码，不把设备编号误解释为有大小顺序的连续量。8位、14位和16位时间字段被解析为秒，转换为各工序开始、结束、中位时间、持续跨度、有效率以及相邻工序等待或重叠特征。

匿名传感器可能属于温度、流量、功率或时间等不同量纲，因此不直接跨列求平均。工序统计前先使用训练折中位数和IQR构造稳健Z分数，并截断极端值，再形成每道工序的均值、标准差、分位数、范围、异常率、零值率和缺失率。原始异常不被粗暴删行；高误差样本进入单独诊断，只有取得采集或标签错误证据后才允许剔除。

## 5. 特征选择与工序特征工程

每个训练折先用ExtraTrees估计非线性贡献，再计算与Y的绝对相关性，并对训练前后段的稳健中位数漂移施加惩罚。最终稳定分数由贡献排名、相关排名和漂移稳定度共同组成，选择前{config.top_k_raw_features}个原始数值特征。这样避免直接根据全量标签筛选造成验证泄漏，也避免一次特征重要性把偶然相关当成稳定机制。

特征视图包括原始值、缺失指示、工序稳健统计、工序时间图和设备类别。当前主流程完整视图约{base['engineered_features']}维。工序重要性由跨折XGBoost贡献聚合，训练到测试漂移同时输出字段级和工序级结果。所有重要性只代表模型预测贡献，不自动等同于因果效应或可直接调节的工艺旋钮。

## 6. 分布检验、特征变换与可插拔降维

重要特征分别拟合Normal、Laplace、Logistic和Student-t分布，使用AIC选择相对最优分布，再用KS检验检查一致性。结果不支持把全部工业变量统一视为正态分布。变换和降维接口包括{', '.join(advanced['available_reducers'])}；统一Ridge滚动实验中最优为{best_transform.experiment}，MSE={best_transform.mse_mean:.6f}、输出{best_transform.output_features:.0f}维。

PCA、SVD和PLS保留为可插拔对照，没有进入XGBoost主干，因为线性降维可能破坏稀疏的非线性阈值关系。Yeo-Johnson在混合量纲特征上出现严重数值失稳，因此明确作为负向实验保留。Autoencoder接口已实现，但未执行的结果不会写成已验证结论。

## 7. 验证、网格搜索与模型对比

基础阶段设置随机五折、滚动时间CV和最后20%时间外推三层验证。所有填充、编码、缩放和监督选择都在折内重新拟合。随机OOF最优为{random_best.model}，MSE={random_best.mse:.6f}；滚动OOF最优为{rolling_best.model}，MSE={rolling_best.mse:.6f}；时间外推最优为{holdout_best.model}，MSE={holdout_best.mse:.6f}。三者差异说明随机切分会明显低估跨批次误差。

嵌套网格搜索只在外层训练折中寻优，外层时间MSE为{advanced['nested_grid_outer_mse']:.6f}。特征消融中最优视图为{best_ablation.experiment}，MSE={best_ablation.mse_mean:.6f}；但主流程根据更多滚动折的稳定性保留完整视图。模型对比覆盖Dummy、Ridge、RandomForest、ExtraTrees、XGBoost、近期加权XGBoost和三成员MLP，最终基础融合通过滚动OOF学习非负且和为1的权重。

## 8. A阶段训练内优化与真实答案事后评价

A阶段不使用A标签。按竞赛样本顺序，以训练前500、600、700条分别预测后续100条，形成{phase_a['validation_rows']}个互不重叠的伪未来样本。比较均匀权重以及半衰期{phase_a['candidate_half_life_fractions'][1:]}的近期加权XGBoost；每个策略都对{phase_a['ensemble_members']}个独立种子的特征选择与模型预测取平均。

训练内回测选择{phase_a['selected_strategy']}，MSE={phase_a['selected_backtest_mse']:.6f}、RMSE={phase_a['selected_backtest_rmse']:.6f}、MAE={phase_a['selected_backtest_mae']:.6f}、R²={phase_a['selected_backtest_r2']:.4f}。策略锁定并生成A预测后才读取公开答案。事后A集MSE由原RobustBlend的{phase_a['baseline_external_a_mse']:.6f}下降到{phase_a['selected_external_a_mse']:.6f}，相对下降{phase_a['external_a_relative_mse_improvement']:.1%}。这是真实A答案上的事后评价，但A答案没有参与该模型的选择或拟合。

正式A文件为`submission_A.csv`；原融合基线为`submission_A_train_only_blend.csv`。两者均保持300行、两列、无表头和原始ID顺序。

## 9. 公布A标签后的B阶段适配

A答案公布后，将300条A样本作为合法的新增监督数据。B阶段用A前100、150、200条训练，分别预测紧随其后的50、50、100条，构成{phase_b['validation_rows']}个伪未来样本；比较A样本权重{phase_b['candidate_a_weights']}，每个候选同样采用{phase_b['ensemble_members']}成员平均。

回测选择{phase_b['selected_strategy']}。相对于当前train-only A模型的伪未来基线MSE {phase_b['train_only_backtest_mse']:.6f}，适配后MSE为{phase_b['selected_backtest_mse']:.6f}，相对下降{phase_b['relative_mse_improvement']:.1%}；RMSE={phase_b['selected_backtest_rmse']:.6f}、MAE={phase_b['selected_backtest_mae']:.6f}、R²={phase_b['selected_backtest_r2']:.4f}。最终以800条原训练数据和300条A标签拟合B模型，B标签使用状态为{phase_b['b_labels_used']}。

正式B文件为`submission_B.csv`；未使用A答案的基线为`submission_B_train_only.csv`。B真实答案不可见，因此回测改善不能冒充B榜真实得分。

## 10. 残差、单样本与预测可信度

残差分析覆盖直方图、Q-Q图、残差与预测值关系、按生产顺序自相关、分设备误差和高误差样本集中度。显著检验包括：{', '.join(significant_tests)}。最高误差样本为{advanced['highest_error_sample']}，项目输出其关键特征稳健Z分数和工序定位，但不因单点影响大就删除。

基础阶段采用滚动OOF残差构造90%共形区间，时间外推覆盖率为{base['holdout_coverage']:.1%}。A阶段和B阶段分别使用自身扩展窗口残差重新估计区间半径，使区间与对应训练权限和验证分布一致。预测区间表达经验不确定性，不替代真实检测或工艺安全上下限。

## 11. 已修改和回退的步骤

{modified_text}

此外，A阶段由单一RobustBlend升级为训练内扩展窗口三成员XGBoost；B阶段由只使用800条训练样本升级为公开A标签增量训练。两阶段均先定义代理未来窗口再选择策略，并分别保留旧基线文件。

## 12. 舍弃、降级和保留接口的步骤

{discarded_text}

A阶段没有根据公开答案选择表现更好的衰减参数，因为这会导致公开标签泄漏；B阶段的A局部模型与残差修正单次探索弱于直接增量训练，因此不作为正式主干。所有未进入最终模型的方法仍保留结果或接口，便于论文说明负向实验和后续扩展。

## 13. 最终输出与复现方式

核心提交为`submission_A.csv`和`submission_B.csv`。逐样本预测、区间、基线差异、公开A事后残差分别保存在详细预测文件中；A/B阶段策略明细、汇总和清单分别使用`phase_a_*`、`phase_b_*`命名。数据审计、特征、工序、漂移、模型、残差、区间和运行过程共有30张图，位于`outputs/plots/`。

完整运行命令：

```powershell
& 'C:\\Users\\lenovo\\anaconda3\\python.exe' -m innovative_solution.run
```

仅运行A阶段或B阶段：

```powershell
& 'C:\\Users\\lenovo\\anaconda3\\python.exe' -m innovative_solution.phase_a_optimization
& 'C:\\Users\\lenovo\\anaconda3\\python.exe' -m innovative_solution.phase_b_optimization
```

项目最终结论应区分三类数字：训练内代理回测、公开A答案事后评价、未知B榜真实评分。前两类已有完整证据，第三类只能由竞赛平台返回，不能由本地实验推断为确定结果。
"""
    (output / "WORKFLOW_DESCRIPTION.md").write_text(text, encoding="utf-8")


def main() -> None:
    generate_workflow_description(PipelineConfig())


if __name__ == "__main__":
    main()
