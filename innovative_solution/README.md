# TFT-LCD质量特性预测方法比较

本项目研究匿名TFT-LCD完整记录上的离线质量特性估计，比较特征分支、回归算法与有限样本选择策略。研究重点是增量信息是否有效、评价是否可追溯，以及预测关联的适用边界。模型尚未获得现场可用性或前瞻预测验证。

训练集800条、5952个输入字段，A集300条、B集412条。全部输入的物理含义与相对Y的可用时点均未核实。A/B多数记录的匿名时间代理早于训练范围，因此A/B预测属于回顾性估计，不能视为从过去预测未来。

## 方法与证据

每个训练折独立完成结构筛选、设备条件填充、缺失指示、编码和监督筛选。基础、基础加工序、基础加时间、组合表示分别与XGBoost树深2/3组合，共八个候选，各使用三个固定种子。内层选参，外层隔离相关记录与时间边界。开发、校准和尾段分别使用620、80、100条记录。

同一300条外层记录上，固定基础加时间深度2的MSE为0.022162，组合深度2为0.023679，八候选选择流程为0.023242。工序统计未获得稳定的增量改善证据。开发段内层选中process_d3，其固定外层MSE为0.025181，说明不同窗口的配置排名并不一致；不称为全局最优。

七类算法各比较基础／基础加时间两个候选、树模型统一500轮的敏感性实验中，CatBoost MSE为0.022253，XGBoost为0.023893。相同轮数不代表相同计算量；CatBoost使用统一编码，未比较原生类别处理。表格基础模型未运行，不作前沿最佳性能的判断。

11个不重复的特征与选择策略比较统一进行Holm校正。时间分支的MSE差为0.002370，原始p约0.000600，校正p约0.005999。组级推断以保存的预测为条件，不能消除研究历史中的适应性偏差。

选定process_d3的冻结尾段MSE为0.044408、区间覆盖87%，A集回顾性MSE为0.032528。候选范围扩展并未使所有评价段都改善。B无标签；合并公开A标签训练B仅作为研究场景，另保留仅训练标签的B预测。

## 运行与复现

完整依赖、数据权限与执行顺序见 [REPRODUCE.md](REPRODUCE.md)。在项目根目录执行：

```powershell
python -m innovative_solution.run
python -m innovative_solution.report_only
python innovative_solution/thesis/build_thesis.py
```

run执行八候选选择、选定模型诊断、统一预算算法比较、字段可用性清点、统计和报告。它依赖本项目已保存的四候选参照与消融；从零重训时先按复现说明执行对应入口。report_only仅由保存的预测刷新图表与报告。

## 输出

| 位置 | 内容 |
|---|---|
| outputs/REPORT.md | 全流程研究报告 |
| outputs/branch_selection/ | 八候选协议、内外层分数、统计、选定模型和逐记录结果 |
| outputs/branch_selection/submission_A.csv | A预测，300行、两列、无表头 |
| outputs/branch_selection/submission_B.csv | 合并A标签的B研究场景预测，412行 |
| outputs/branch_selection/submission_B_train_only.csv | 仅训练集拟合的B预测，412行 |
| outputs/branch_selection/submission_format_check.json | 行数、列数、有限数值及ID顺序验证 |
| outputs/fair_comparison/ | 统一候选数、树轮数的算法比较与耗时 |
| outputs/availability/ | 全部5952字段的可用性清点及时间方向审计 |
| outputs/supplement/ | 算法、消融、敏感性、字段稳定性及固定模型诊断 |
| outputs/review/ | 四候选参照及固定配置结果 |
| thesis/基于机器学习的TFT-LCD多工序质量特性预测研究.docx | 论文 |

论文由正文模板及结果表生成，以本文的研究叙述展开。修改和舍弃原因单独记录在 [PROJECT_PROCESS.md](PROJECT_PROCESS.md)，针对意见的说明位于review/特征候选与结论定位处理说明.md。源码、模型、竞赛数据、实验结果和论文已发布到公开仓库 [github.com/kagurarnea/-](https://github.com/kagurarnea/-)。原创源码与项目文档采用MIT许可证；该许可证不重新授权竞赛数据或第三方资料。仓库发布不等于向竞赛平台提交结果，本项目没有官方线上成绩。
