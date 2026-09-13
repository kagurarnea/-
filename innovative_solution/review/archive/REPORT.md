# 工序感知与漂移鲁棒的产品特性预测

## 方案定位

本方案面向TFT-LCD复杂多工序生产中的连续产品特性预测。针对样本数量有限、原始变量维度高、设备工况差异明显、传感器缺失异常和跨批次分布漂移等问题，构建工序感知、漂移鲁棒且可解释的回归预测系统。

系统以预测精度和时间泛化能力为核心，同时输出预测区间、关键工序、稳定特征、设备组误差和数据漂移信息，为在线质量预警、工艺参数排查和检测资源配置提供依据。

技术主线为：**结构审计 → 工序/工具条件填充 → 稳定且抗漂移的原始特征选择 → 工序稳健统计与相对时间重构 → 树模型、正则线性模型和轻量MLP集成 → OOF非负融合 → 锁定时间外推验证 → 共形预测区间 → A训练内扩展窗口优化 → 公布A标签后的扩展窗口回测与B阶段适配**。

## 数据与验证口径

- 训练样本：800；测试A：300；测试B：412。
- 清洗后数值特征：2617；识别时间字段：72；类别字段：14。
- 按生产时间代理顺序锁定最后160条作为外推集；开发集同时执行随机五折CV和仅用过去预测未来的滚动时间CV（有效校准样本320条）。
- 所有监督特征选择、工具条件填充、缩放和类别编码均在训练折内拟合。

## 模型结果

| 模型 | 随机OOF MSE | 滚动时间OOF MSE | 时间外推MSE | 时间外推RMSE | 时间外推MAE | 时间外推R² |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0.042437 | 0.041108 | 0.051930 | 0.227881 | 0.173636 | -0.1285 |
| ExtraTrees | 0.016237 | 0.029420 | 0.037400 | 0.193390 | 0.138387 | 0.1872 |
| MLP-Ensemble | 0.019062 | 0.040658 | 0.038767 | 0.196894 | 0.137893 | 0.1575 |
| Ridge | 0.028782 | 0.042624 | 0.046961 | 0.216706 | 0.143896 | -0.0206 |
| RobustBlend | 0.010085 | 0.025261 | 0.032885 | 0.181343 | 0.126268 | 0.2853 |
| XGB-Recency | 0.010429 | 0.025707 | 0.033417 | 0.182804 | 0.126831 | 0.2738 |
| XGBoost | 0.010063 | 0.025421 | 0.032888 | 0.181350 | 0.127077 | 0.2853 |

随机OOF最低模型为 **XGBoost**（MSE=0.010063）；滚动时间OOF最低模型为 **RobustBlend**（MSE=0.025261）；时间外推集最低模型为 **RobustBlend**（MSE=0.032885）。融合权重只根据滚动时间OOF学习，RobustBlend在外推集上的MSE为 **0.032885**，权重为：Ridge=0.0%，ExtraTrees=5.5%，XGBoost=60.6%，XGB-Recency=33.9%，MLP-Ensemble=0.0%。

不能在看到时间外推集结果后重新调权重，否则该外推集将失去独立验证意义。上述模型比较属于不使用测试标签的第一阶段基线。A榜答案公布后，B阶段只通过A内扩展窗口回测选择增量训练规则，B标签未参与训练或选择。


## A阶段训练内扩展窗口优化

- A模型严格只使用800条原训练标签。按竞赛样本顺序构造三个扩展窗口，共评价300个伪未来样本；每个候选均使用3个独立种子的特征选择与XGBoost平均预测。
- 比较均匀权重与近期样本半衰期[0.2, 0.35, 0.5]，完全依据训练内回测锁定 **uniform_xgb**，回测MSE=0.025067、RMSE=0.158327、MAE=0.117136。
- 策略锁定并完成A预测后才读取公开答案。事后A集MSE由原RobustBlend的0.030016降至 **0.027525**，相对下降 **8.3%**；R²=0.5017。
- 公开A答案只用于事后评价，未参与A策略选择或拟合：False。原融合版本保存在`submission_A_train_only_blend.csv`。



## 公布A标签后的B阶段适配

- A榜结束并公布答案后，将300条A样本作为新增监督数据；原训练集仍保留全部800条样本。
- 使用三个扩展窗口，以A前缀训练、紧随其后的A区段模拟未知下一批次，共评价200个伪未来样本；所有特征选择与填充均在各窗口训练部分重新拟合，并对3个独立种子的特征选择与XGBoost预测取平均。
- 回测比较A样本权重[1.0, 2.0, 4.0]，最终选择 **train_plus_released_a_x2**。train-only基线MSE为0.029490，适配后为 **0.020916**，相对下降 **29.1%**。
- 适配回测RMSE=0.144622、MAE=0.113227、R²=0.6641、平均偏差=-0.011496。
- 最终B模型使用训练集+A共1100条标签，公开A样本权重为2；正式`submission_B.csv`由该模型生成，train-only版本保存在`submission_B_train_only.csv`。
- 该改进是无B标签条件下的代理验证结果，不表述为B榜真实得分；B标签参与状态：False。


## 可信度与工程结论

- 90%共形区间在时间外推集上的实际覆盖率为 **90.0%**，平均区间宽度为 **0.5223**。
- 跨折贡献最高的工序为：311、210、420、520、344。
- 训练集与A/B测试集漂移较高的工序为：344、220、400、META、520。
- 最大误差样本为 **NH1835**，单点贡献时间外推集总平方误差的 **28.5%**；保留该样本时MSE为0.032885，仅作敏感性分析、排除该点后的描述性MSE为0.023656。除非能证明标签或采集错误，否则不能为了提高指标直接删除。
- 完整运行耗时为 **2.17分钟**；各阶段、各模型耗时保存在`run_trace.csv`和`fold_metrics.csv`。

本轮第一次运行曾用该时间外推集识别随机CV过于乐观的问题，随后验证设计升级为滚动时间CV。因此当前外推结果应解释为**时间泛化基准**，不是完全未触碰的最终测试。正式论文若要作统计确认，应使用新增生产周期或嵌套滚动验证作为最终盲测。

## 创新点

1. **工序条件缺失建模**：缺失值先按工序对应设备组统计填充，再由训练折全局中位数兜底，同时保留高缺失特征的缺失指示。
2. **抗漂移稳定选择**：特征得分联合ExtraTrees贡献、目标相关性和开发期前后段稳健中位数漂移，不把一次划分的重要性直接当成最终结论。
3. **量纲无关工序表征**：先按训练折中位数/IQR将匿名传感器转换为稳健Z分数，再计算每道工序的分布、异常率、缺失率和零值率。
4. **工序时间图特征**：将8/14/16位时间字段解析为秒，生成工序开始、结束、跨度以及相邻工序等待/重叠特征，不直接删除时间，也不把无序字段误当RNN序列。
5. **跨范式鲁棒融合**：同时训练Ridge、ExtraTrees、XGBoost和三成员轻量MLP，通过开发集OOF学习非负且和为1的权重。
6. **面向部署的可信预测**：除Y点预测外输出90%共形区间、工序贡献、测试漂移和时间外推误差。

## 近年方法如何进入本方案

- 2025年Nature的TabPFN表明表格基础模型在不超过约10000样本、500特征的小数据上具有很强潜力。本方案先稳定筛选至320个原始特征，为后续加入TabPFN留下兼容接口；当前环境未安装模型及权重，因此没有伪造TabPFN实验结果。
- ICLR 2025 TabM强调参数高效的MLP集成和成员多样性。本方案当前使用可直接运行的三成员小型MLP集成验证这一方向，但明确不将其冒充为TabM。
- SCARF通过随机特征破坏学习表格表示，说明缺失/扰动机制可用于鲁棒表征。本方案将缺失掩码和工序级异常统计显式纳入特征；后续安装PyTorch后可增加SCARF预训练消融。
- 大规模表格基准显示中等规模结构化数据上树模型仍是强基线，因此XGBoost和ExtraTrees保持为主干，而深度模型作为互补分支而非预设赢家。


## 完善分析结果

- 逐样本缺失审计发现，609条样本至少存在一个缺失值，单样本最高缺失率为13.1%；缺失机制按工序和设备差异输出为统计候选，不能替代生产日志确认。
- 对12个稳定重要特征分别拟合Normal、Laplace、Logistic和Student-t分布；其AIC最优分布中有0个通过5%水平KS一致性检验，说明不能直接采用统一正态假设。
- 可插拔降维接口支持：none、pca50、pca95、svd50、pls20、autoencoder16。在统一Ridge滚动实验中，当前最佳变换/降维配置为 **pca50**，MSE=0.038703；该结论只适用于线性降维分支，不覆盖主干XGBoost。`autoencoder16`已实现插件接口但不在默认九组实验中，避免把未执行结果写成结论。
- Yeo-Johnson在匿名混合量纲特征上出现明显数值失稳（MSE=5.08e+05），已作为负向实验保留，不能默认认为分布变换必然改善模型。
- 三折消融中最优配置为 **A3_process_time**，MSE=0.027388；但扩大到主流程四个滚动时间折后未形成稳定优势，因此最终特征视图以主流程更广的时间验证结果为准。嵌套网格搜索的外层时间MSE为0.027636。
- 标准树模型对比中，**XGBoost**最低，MSE=0.028343。
- 残差显著检验项：Shapiro-Wilk normality、D'Agostino normality、Ljung-Box autocorrelation lag=5、Ljung-Box autocorrelation lag=10。这表明残差正态性和生产顺序相关性需要在后续模型中继续处理。
- 已为最高误差样本 **NH1835** 生成逐特征稳健Z分数和关键工序诊断，不以删除样本代替原因分析。


## 输出与工作痕迹

- `feature_audit.csv`：每列保留、删除或时间重构的原因。
- `fold_metrics.csv`、`model_summary.csv`：统一验证结果和训练耗时。
- `feature_selection_stability.csv`：每个原始特征的五折入选率、贡献、相关性和漂移。
- `engineered_feature_importance.csv`、`operation_importance.csv`：特征与工序两级解释。
- `feature_drift.csv`、`operation_drift.csv`：训练集到A/B测试集的漂移证据。
- `random_oof_predictions.csv`、`rolling_oof_predictions.csv`、`temporal_holdout_predictions.csv`：三种验证层级的逐样本预测与残差。
- `conformal_metrics.csv`：预测区间校准结果。
- `submission_A.csv`、`submission_B.csv`：无表头两列竞赛提交文件；A为训练内扩展窗口选择版本，B为公开A标签适配版本。
- `submission_A_train_only_blend.csv`：原RobustBlend的A基线，便于完整追溯。
- `phase_a_backtest_predictions.csv`、`phase_a_strategy_summary.csv`、`phase_a_external_evaluation.csv`：A训练内回测和答案公布后的事后评价。
- `submission_B_train_only.csv`：不使用A答案的B基线，便于完整追溯。
- `phase_b_backtest_predictions.csv`、`phase_b_strategy_summary.csv`、`phase_b_manifest.json`：B阶段扩展窗口证据、策略比较和配置清单。
- `WORKFLOW_DESCRIPTION.md`：从数据权限到A/B提交的完整工作过程文字稿。
- `row_missing_report.csv`、`row_operation_missing_matrix.csv`：逐样本和逐工序缺失证据。
- `feature_variance_report.csv`、`theoretical_distribution_fit.csv`：方差、AIC和KS检验结果。
- `transform_reducer_summary.csv`、`ablation_summary.csv`：变换、降维和工序特征消融。
- `nested_grid_search_outer.csv`、`tree_model_comparison_summary.csv`：折内搜索和树模型对比。
- `high_error_sample_feature_detail.csv`、`residual_statistical_tests.csv`：样本级解释和残差检验。
- `feature_semantics.csv`、`reproducibility_snapshot.json`：物理语义边界、输入哈希和软件版本。
- `plots/`：30张数据、模型、残差、解释、漂移和运行过程图。

## 可视化索引

1. [工序数据审计](plots/01_data_audit_by_operation.png)
2. [目标分布与时间切分](plots/02_target_and_temporal_split.png)
3. [候选模型五折稳定性](plots/03_cv_model_stability.png)
4. [随机CV与时间外推差异](plots/04_oof_vs_temporal_holdout.png)
5. [融合权重](plots/05_blend_weights.png)
6. [模型预测相关性](plots/06_model_prediction_correlation.png)
7. [时间外推真实值与预测值](plots/07_temporal_holdout_prediction.png)
8. [时间外推残差诊断](plots/08_temporal_residual_diagnostics.png)
9. [分设备误差](plots/09_holdout_error_by_tool.png)
10. [原始特征选择稳定性](plots/10_feature_selection_stability.png)
11. [关键工序贡献](plots/11_operation_importance.png)
12. [工序分布漂移](plots/12_operation_distribution_drift.png)
13. [共形区间覆盖率](plots/13_conformal_interval_coverage.png)
14. [精度与训练效率](plots/14_accuracy_efficiency.png)
15. [流水线运行痕迹](plots/15_pipeline_run_trace.png)
16. [高误差样本集中度](plots/16_error_concentration.png)

17. [逐行与逐列缺失分布](plots/17_row_and_column_missingness.png)
18. [样本×工序缺失热力图](plots/18_sample_operation_missing_heatmap.png)
19. [方差、贡献与漂移](plots/19_variance_importance_drift.png)
20. [实际与理论分布拟合](plots/20_theoretical_distribution_fit.png)
21. [特征变换与降维对比](plots/21_transform_reducer_comparison.png)
22. [降维维数与误差](plots/22_reducer_dimension_and_error.png)
23. [工序特征消融](plots/23_feature_ablation.png)
24. [嵌套网格搜索](plots/24_nested_grid_search.png)
25. [随机森林、ExtraTrees与XGBoost](plots/25_tree_model_comparison.png)
26. [高误差样本局部诊断](plots/26_high_error_sample_diagnosis.png)
27. [残差Q-Q与时间自相关](plots/27_residual_qq_and_autocorrelation.png)
28. [完善分析运行痕迹](plots/28_advanced_analysis_trace.png)


29. [公开A标签增量训练回测](plots/29_phase_b_adaptation_backtest.png)


30. [A阶段训练内策略选择与事后评价](plots/30_phase_a_training_only_backtest.png)


## 研究依据

- [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)
- [TabM官方实现与论文, ICLR 2025](https://github.com/yandex-research/tabm)
- [SCARF, ICLR 2022 Spotlight](https://arxiv.org/abs/2106.15147)
- [树模型与深度模型表格基准](https://arxiv.org/abs/2207.08815)
- [天池冠军工序特征方案](https://tianchi.aliyun.com/forum/post/4020)
- [天池SDAE与相对时间方案](https://tianchi.aliyun.com/forum/post/4087)
