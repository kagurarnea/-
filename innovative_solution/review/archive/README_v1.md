# 工序感知与漂移鲁棒的产品特性预测

## 方案定位

本方案面向TFT-LCD复杂多工序生产中的连续产品特性预测。针对样本数量有限、原始变量维度高、设备工况差异明显、传感器缺失异常和跨批次分布漂移等问题，构建工序感知、漂移鲁棒且可解释的回归预测系统。

系统以预测精度和时间泛化能力为核心，同时输出预测区间、关键工序、稳定特征、设备组误差和数据漂移信息，使模型能够支持在线质量预警、工艺参数排查和检测资源配置。

完整的阶段流程、修改原因和舍弃记录见[`PROJECT_PROCESS.md`](PROJECT_PROCESS.md)。

## 技术路线

原始`data/*.xlsx`依次经过数据审计、工序解析、时间重构、稳定特征选择、工序统计建模、多模型训练、滚动时间验证、非负融合、共形区间校准和A阶段结果生成。A榜答案公布后，再通过A内扩展窗口回测选择增量训练权重，以训练集+A标签生成正式B结果。每个阶段均保存结构化结果和可视化工作痕迹。

## 运行

在项目根目录执行：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m innovative_solution.run
```

第一次运行会读取Excel并在`innovative_solution/.cache/`生成本地缓存，后续运行会明显加快。所有结果写入`innovative_solution/outputs/`。

只修改图表或报告模板、不重新训练模型时执行：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m innovative_solution.report_only
```

只运行缺失审计、分布检验、降维、消融、网格搜索和残差诊断：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m innovative_solution.advanced_analysis
```

## 验证设计

1. 按解析后的生产时间代理顺序锁定最后20%训练样本，模拟未来批次。
2. 前80%作为开发集，同时执行随机五折CV和仅用过去预测未来的滚动时间CV。
3. 缺失填充、类别编码、稳健缩放和监督特征选择均在训练折内拟合。
4. 只在滚动时间OOF预测上学习非负融合权重和共形区间，避免随机CV掩盖批次漂移。
5. 权重确定后只评估一次锁定时间外推集，不根据外推结果反向调参。
6. 最后在全部训练数据上重训并预测测试A、测试B。

## 方案组成

- 工序及对应设备自动解析；
- 工序设备条件均值填充和缺失指示；
- ExtraTrees贡献、目标相关性和时间漂移联合稳定选择；
- 基于中位数/IQR的工序稳健统计；
- 8/14/16位时间字段解析及相邻工序等待/重叠特征；
- Dummy、Ridge、ExtraTrees、XGBoost、近期样本加权XGBoost和三成员MLP集成；
- OOF非负约束融合；
- 90%共形预测区间；
- 特征、工序、设备和训练/测试漂移四级诊断；
- 30张可用于论文与答辩的可视化工作痕迹。

`TabPFN`、`TabM`和`SCARF`被设计为可选研究扩展。未安装对应依赖时不会伪造实验结果，执行状态记录在`outputs/optional_modern_methods.csv`。

## 主要输出

- `REPORT.md`：完整方法、真实结果、创新点和图表索引；
- `WORKFLOW_DESCRIPTION.md`：可直接用于项目说明的完整工作过程文字稿；
- `run_manifest.json`、`run_trace.csv`：配置、维度、耗时和运行轨迹；
- `fold_metrics.csv`、`model_summary.csv`：统一CV与时间外推指标；
- `feature_audit.csv`、`feature_selection_stability.csv`：数据治理和特征证据；
- `operation_importance.csv`、`operation_drift.csv`：工序贡献和漂移；
- `random_oof_predictions.csv`、`rolling_oof_predictions.csv`、`temporal_holdout_predictions.csv`：三层逐样本预测与残差；
- `submission_A.csv`、`submission_B.csv`：无表头、两列、保持原始ID顺序的提交文件；
- `submission_A_train_only_blend.csv`：A阶段原RobustBlend基线；
- `phase_a_backtest_predictions.csv`、`phase_a_strategy_summary.csv`：只用训练标签的A扩展窗口策略证据；
- `phase_a_external_evaluation.csv`：策略锁定后与公开A答案的事后比较；
- `submission_B_train_only.csv`：未使用A答案的B基线版本；
- `phase_b_backtest_predictions.csv`、`phase_b_strategy_summary.csv`：A扩展窗口伪未来回测明细与策略汇总；
- `phase_b_manifest.json`：B阶段适配配置、选择结果和无B标签声明；
- `row_missing_report.csv`、`row_operation_missing_matrix.csv`：逐行、逐工序缺失明细；
- `feature_variance_report.csv`、`theoretical_distribution_fit.csv`：方差及理论分布检验；
- `transform_reducer_summary.csv`：特征变换和多种可插拔降维实验；
- `ablation_summary.csv`、`nested_grid_search_outer.csv`：消融与嵌套网格搜索；
- `high_error_sample_feature_detail.csv`、`residual_statistical_tests.csv`：样本诊断和残差检验；
- `feature_semantics.csv`、`reproducibility_snapshot.json`：物理语义边界及复现信息；
- `plots/`：30张静态论文图，第29张展示B阶段公开A标签适配，第30张展示A训练内策略选择与事后评价。

只重新执行A答案公布后的B阶段适配：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m innovative_solution.phase_b_optimization
```

只重新执行A阶段训练内优化：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m innovative_solution.phase_a_optimization
```

只重新生成完整工作过程文字稿：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m innovative_solution.workflow_description
```
