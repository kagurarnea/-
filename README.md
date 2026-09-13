# TFT-LCD质量特性预测方法比较

本项目使用机器学习研究TFT-LCD多工序生产记录与连续质量特性Y之间的预测关系，提供训练代码、竞赛数据、模型、实验结果、可视化与论文。

当前研究采用折内预处理、四种特征表示与八候选嵌套选择，并进行算法对照、消融、预算敏感性、配对关联组推断和逐样本诊断。研究定位为匿名完整记录上的离线回顾性比较。

## 项目入口

- [方案与主要结果](innovative_solution/README.md)
- [完整实验报告](innovative_solution/outputs/REPORT.md)
- [安装及复现说明](innovative_solution/REPRODUCE.md)
- [研究决策与修改过程](innovative_solution/PROJECT_PROCESS.md)
- [论文正文](innovative_solution/thesis/论文正文.md)
- [Word论文](innovative_solution/thesis/基于机器学习的TFT-LCD多工序质量特性预测研究.docx)

## 快速开始

建议使用Python 3.13，在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r innovative_solution/requirements-research.txt
python -m innovative_solution.report_only
```

上述命令使用保存的实验记录刷新报告与图表。重新训练当前方案：

```bash
python -m innovative_solution.run
```

从头运行所有比较及论文构建的完整顺序见复现说明。训练与推理模型为本项目生成的joblib文件；环境版本记录在依赖清单与运行结果中。

## 数据与输出

data目录包含训练集、测试A、测试B和A答案，由项目所有者授权一并上传至本私有仓库。训练集800条、A集300条、B集412条，输入5952个字段。

当前预测文件位于innovative_solution/outputs/branch_selection/：

- submission_A.csv：A集预测，300行。
- submission_B.csv：加入公开A标签训练的B研究场景预测，412行。
- submission_B_train_only.csv：仅使用训练集标签的B预测，412行。

三个文件均为两列无表头CSV，第一列保留测试ID及行序，第二列为预测Y。该仓库上传不涉及向竞赛平台提交结果，也不改变数据的原有权利归属。

## 主要实验结果

| 比较或配置 | 外层MSE |
|---|---:|
| 基础特征 深度2 | 0.024533 |
| 基础加时间 深度2 | 0.022162 |
| 组合特征 深度2 | 0.023679 |
| 八候选嵌套选择 | 0.023242 |

开发段按内层分数选中process_d3；其尾段MSE为0.044408，A集回顾性MSE为0.032528。所有比较的样本范围、选择规则和实际结果见实验报告。

## 文件组织

- innovative_solution/：当前研究的代码、依赖、测试、说明与论文。
- innovative_solution/outputs/：逐记录预测、统计、图表及可复用模型。
- data/：竞赛数据和过程分析产物。
- 仓库根目录其他Python脚本：项目探索与对比实验代码。

依赖安装目录、缓存、临时材料和论文排版检查图片不纳入版本控制。运行时会按需生成缓存。输入字段与Y的物理语义、采集时点尚未核实，结果不代表已验证的在线提前预警能力。
