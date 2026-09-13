# TFT-LCD Quality Characteristic Prediction

本项目面向TFT-LCD多工序制造数据，使用机器学习预测连续质量特性Y。仓库公开保存可执行代码、竞赛数据、训练模型、逐样本实验结果、可视化和论文，便于核对从数据审计到预测文件生成的完整过程。

项目地址：[github.com/kagurarnea/-](https://github.com/kagurarnea/-)

## 研究方案

训练集包含800条记录和5952个输入字段，测试A、B分别包含300条和412条记录。当前流程在每个训练折内独立完成结构筛选、设备条件填充、缺失指示、类别编码和监督特征选择，并比较以下四种输入表示：

- 基础特征；
- 基础特征加工序统计；
- 基础特征加时间代理；
- 同时包含工序统计与时间代理的组合特征。

四种表示分别与XGBoost树深2、3组合，形成八个候选配置。内层窗口负责选参，外层窗口评价选择流程；开发、区间校准和尾段评价使用分离的数据范围。项目还包含七类回归算法对照、特征消融、统一候选预算比较、关联组重采样、区间覆盖和逐样本残差诊断。

## 主要结果

| 方法或配置 | 外层MSE |
|---|---:|
| 基础特征 深度2 | 0.024533 |
| 基础加工序统计 深度2 | 0.024291 |
| 基础加时间特征 深度2 | 0.022162 |
| 组合特征 深度2 | 0.023679 |
| 八候选嵌套选择 | 0.023242 |

开发段内层选择 `process_d3`。该配置的冻结尾段MSE为0.044408，使用全部训练记录拟合后的A集回顾性MSE为0.032528。不同窗口中的候选排名存在变化，因此仓库分别报告固定配置、选参流程和最终开发配置，不将单一分数称为全局最优结果。

完整数字、口径和图表见[实验报告](innovative_solution/outputs/REPORT.md)。

## 快速开始

建议使用Python 3.13。在仓库根目录执行：

```bash
python -m venv .venv
```

激活环境后安装依赖：

```bash
python -m pip install -r innovative_solution/requirements-research.txt
```

根据已保存的逐样本结果刷新报告和图表：

```bash
python -m innovative_solution.report_only
```

重新运行当前八候选流程：

```bash
python -m innovative_solution.run
```

从数据审计开始复现四候选参照、补充实验和当前方案的完整顺序见[复现说明](innovative_solution/REPRODUCE.md)。

## 目录

```text
.
|-- data/                         竞赛数据及探索阶段结果
|-- innovative_solution/         当前研究代码
|   |-- outputs/                 指标、预测、模型和可视化
|   |-- review/                  数据与引用核验资料
|   |-- tests/                   自动化检查
|   |-- thesis/                  论文来源、生成脚本和Word论文
|   |-- README.md                方案与结果说明
|   `-- REPRODUCE.md             完整复现步骤
|-- .gitignore                   版本控制排除规则
|-- LICENSE                      源码与项目文档许可证
`-- README.md                    仓库入口
```

## 论文与结果文件

- [论文正文](innovative_solution/thesis/论文正文.md)
- [Word论文](innovative_solution/thesis/基于机器学习的TFT-LCD多工序质量特性预测研究.docx)
- [研究方案说明](innovative_solution/README.md)
- [实验复现说明](innovative_solution/REPRODUCE.md)
- [研究决策记录](innovative_solution/PROJECT_PROCESS.md)
- [A集预测](innovative_solution/outputs/branch_selection/submission_A.csv)
- [B集预测](innovative_solution/outputs/branch_selection/submission_B.csv)
- [仅训练集标签拟合的B集预测](innovative_solution/outputs/branch_selection/submission_B_train_only.csv)

A/B预测文件均为两列无表头CSV，第一列保留测试文件ID及行序，第二列为预测Y。格式核验记录位于 `innovative_solution/outputs/branch_selection/submission_format_check.json`。

## 数据说明

`data/`包含本研究使用的训练集、测试A、测试B和A答案。项目所有者授权将这些竞赛文件随本仓库公开，以支持结果复核。数据来源于阿里云天池智能制造质量预测赛题；公开仓库不改变比赛平台、数据提供方或其他权利人的原有权利。使用者仍需遵守适用的比赛规则和数据条款。

输入字段、目标Y和时间字段均为匿名信息。现有结果适用于完整记录条件下的离线回顾性质量特性估计，不代表已经完成生产现场的输入可用时点验证或在线预警验证。

## License

仓库中的原创源码与项目文档采用[MIT License](LICENSE)。该许可证不重新授权 `data/` 中的竞赛数据，也不改变论文引用资料及第三方内容的原有权利，具体见[数据说明](DATA_NOTICE.md)。
