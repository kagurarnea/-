# 产品特性预测：统一对比与可视化报告

## 数据口径

当前结果文件包含 **800 条OOF训练样本**。这与初赛说明中的500条训练样本不是同一数据版本；论文和答辩必须将当前结果标注为后续阶段/扩充训练集结果，不能与初赛榜单直接混报。

本报告读取既有5折交叉验证和OOF预测，不重新训练模型。研发阶段演进图用于展示项目优化过程；严格的优劣判断优先使用同一批OOF样本上的配对比较。

## 当前结论

- 当前记录的最低MSE为 **0.011786**（残差校正，alpha=0.30），相对首个缺失值检查点 0.013539 下降 **12.95%**。
- 加权融合相对最佳XGB的MSE改善为 **0.00004290**，样本级配对Bootstrap的95%区间为 **[-0.00006457, 0.00014462]**。
- alpha=0.3残差校正相对全局模型的MSE改善为 **0.00017432**，样本级配对Bootstrap的95%区间为 **[-0.00005008, 0.00040355]**。
- 困难样本Tool O专用专家随着权重增加而恶化，因此当前证据支持保留全局模型，而不是为了“创新”强行加入局部专家。
- Bootstrap为样本级描述性证据；若ID代表生产时间，最终论文应补充按批次的Block Bootstrap和时间外推验证。

## 消融结论表

| 比较环节 | 参考方案 | 当前最优 | 参考MSE | 最优MSE | 相对变化 |
|---|---|---|---:|---:|---:|
| 缺失值处理 | median | operation_tool_mean | 0.013897 | 0.013539 | 2.57%（改善） |
| 特征工程 | 原始清洗特征 | extra_trees Top300 | 0.013478 | 0.012155 | 9.81%（改善） |
| 工具身份特征 | E0_baseline | E3_tool_target_mean_std | 0.011979 | 0.011961 | 0.16%（改善） |
| 工序与异常特征 | E0_baseline | E6_tool_operation | 0.012794 | 0.012138 | 5.12%（改善） |
| 残差校正 | global_baseline | best_residual_corrected | 0.011961 | 0.011786 | 1.46%（改善） |
| 困难样本专家 | E0_global_only | E0_global_only | 0.011979 | 0.011979 | 0.00%（未改善） |

## 图表索引

1. [研发阶段性能演进](plots/01_stage_progression.png)
2. [缺失值处理策略对比](plots/02_missing_strategy_comparison.png)
3. [特征工程代表方案消融](plots/03_feature_engineering_ablation.png)
4. [单模型与融合模型多指标比较](plots/04_model_metric_comparison.png)
5. [最佳单模型折间稳定性](plots/05_fold_stability.png)
6. [残差校正强度敏感性](plots/06_residual_lambda_sensitivity.png)
7. [困难样本专家负向消融](plots/07_hard_sample_expert_ablation.png)
8. [OOF预测与残差诊断](plots/08_oof_prediction_diagnostics.png)
9. [关键工序贡献排序](plots/09_operation_importance.png)
10. [分设备误差与样本量](plots/10_tool_error_and_sample_size.png)
11. [精度与训练效率权衡](plots/11_accuracy_efficiency_tradeoff.png)
12. [同样本配对Bootstrap比较](plots/12_paired_bootstrap_comparison.png)

## “最优解”判定规则

最终方案不能只满足平均MSE最低，还应同时满足：主指标MSE最低或差异区间支持改善；折间标准差和最差折不明显恶化；MAE与R²方向一致；关键工序排名跨折稳定；训练与单样本推理耗时满足部署要求。若复杂模块只带来极小改善且置信区间跨0，应优先选择更简单的方案。
