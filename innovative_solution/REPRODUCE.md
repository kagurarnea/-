# 实验复现说明

本项目通过公开GitHub仓库 [github.com/kagurarnea/-](https://github.com/kagurarnea/-) 保存代码、竞赛数据、模型、实验记录及论文。项目所有者授权将竞赛数据随仓库公开。原创源码与项目文档采用MIT许可证；该许可证不重新授权竞赛数据或第三方资料。单独生成的源码与结果ZIP仍采用不含数据与模型对象的精简范围，两种分发形式应分别理解。

## 数据

仓库根目录data中包含训练集.xlsx、测试集A.xlsx、测试集B.xlsx及测试集A_答案.csv。文件SHA256应与innovative_solution/outputs/review/review_manifest.json中的input_sha256一致。Git属性保留文件字节，避免克隆时自动换行转换影响哈希。缺少A答案时主预测入口支持仅训练标签的A/B预测，但本文A分数和合并标签B实验需要该答案文件。

赛题入口：https://tianchi.aliyun.com/competition/entrance/231633/introduction 。当前网页未取得具体下载与使用条款，不能保证该入口仍能匿名下载全部实验文件。数据权限应向比赛平台或合法资料提供者确认。

## 环境与命令

实验运行在Windows和Python 3.13.9下。建议建立独立环境，安装requirements-research.txt中记录的版本。没有修改计算协议时，随机种子和字段顺序保持固定；跨平台、BLAS和线程实现仍可能产生浮点差异。

在项目根目录执行：

```powershell
python -m pip install -r innovative_solution/requirements-research.txt
python -m innovative_solution.review_validation
python -m innovative_solution.run_supplement
python -m innovative_solution.run
```

三个实验入口依次生成四候选参照、算法与消融比较、八候选选择及当前结果。最后一个入口还运行统一候选预算算法比较、全字段可用性清点、选定模型尾段诊断和11比较统计表。已有相同数据的参照与消融文件时，可仅执行最后一个入口；不要把不同输入哈希的文件混用。数据审计另有review/statistical_audit.py和review/verify_cache_source.py。

```powershell
python -m unittest discover -s innovative_solution/tests -v
python innovative_solution/thesis/plot_workflow.py
python innovative_solution/thesis/build_thesis.py
```

Word稿从thesis/manuscript.md及已保存实验指标生成。目录更新及内部PDF检查可用安装了Microsoft Word的Windows环境执行thesis/render_review.ps1；PDF布局核验脚本需要pypdf与lxml。所有源码哈希保存在supplement结果清单中，输入缓存与Excel逐单元一致性已有独立核验。

## 实验口径

- 主验证和补充比较使用相同300条外层记录；校准80条与尾段100条有独立的数据用途。
- 算法比较在各内层选择参数，外层三成员平均；敏感性固定单种子，不与三成员结果混作改进率。
- bootstrap以关联组为重采样单位，区间和p值是回顾性探索证据。
- 滚动区间假设较早时间组标签立即可用，该假设不等同已知生产质检延迟。
- 包内预测表支持指标复算；完整重训还需要相同数据权限。哈希用于识别数据，不提供下载授权。

## 结果位置

outputs/branch_selection保存当前八候选选择、模型、A/B点预测、区间诊断和11项推断表；outputs/fair_comparison为统一候选与树轮数对照，outputs/availability为全字段可用性清点。outputs/review保留四候选参照，outputs/supplement保存组合表示下的算法、消融、敏感性、固定模型诊断与来源。修改过程的意见与答复在review目录，论文正文不包含审稿口吻或版本比较。
