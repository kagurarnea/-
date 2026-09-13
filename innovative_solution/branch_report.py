"""Current research report, with decisions separate from manuscript prose."""
import json
from .branch_selection import OUT
from .thesis.branch_content import load


def generate():
    v=load()
    manifest=json.loads((OUT/'review_manifest.json').read_text(encoding='utf-8'))
    text='\n\n'.join(['# TFT-LCD质量特性预测方法比较报告',
        '本研究比较匿名完整记录上的特征分支、算法与配置选择策略。工序统计未获得稳定增量改善证据；时间分支在所列固定比较中更有预测信息，但全部5952项输入的目标测量前可用性均未核实。研究定位为离线回顾性质量特性估计。',
        '## 特征与选择流程',v['nested_table'],v['nested_interpretation'],
        '## 正式候选与开发段选择',v['selection_table'],v['development_candidates_table'],v['selection_interpretation'],
        '所有八个候选均经历两个内层60条窗口和三个固定种子；切分与四候选对照逐索引一致。外层得分不用于覆盖开发段内层的选定配置。最终process_d3的内层MSE只比full_d3低约0.000137，三个外层却均选择时间分支，说明排名依赖窗口。',
        '## 同样本算法与预算敏感性',v['strong_table'] if 'strong_table' in v else '组合特征下的多算法对照见 outputs/supplement/baseline_summary.csv。',
        v['fair_table'],v['fair_interpretation'],
        '## 统计推断',v['inference_table'],v['time_stat_text'],
        '校正族共11项不同的特征与选择策略比较，排除了同一假设的重复实现。95%区间为个别区间；Holm校正不覆盖所有研究历史或算法搜索。已有预测上的关联组重采样仍不能消除适应性分析偏差。',
        '## 时间方向与输入可用性',
        '逐字段清点位于 ../availability/all_input_availability.csv，5952项均未核实早于Y测量可用。用户确认未收到字段字典或采集时点说明。72个时间字段的行中位数用于排序，时间分支另包含工序时间统计和路线代理差。删除显式时间输入也不能认证其他字段。',
        'A有250/300、B有373/412条记录的代理早于训练范围下界，均无晚于训练范围上界的记录。全量训练拟合A/B属于跨时间范围离线估计；按代理顺向回放的外层结果不能代替实际生产前瞻评价。',
        '## 最终配置的尾段与A/B输出',
        f"最终开发配置为{v['selected']}。冻结尾段MSE {v['tail_mse']}，MAE {v['tail_mae']}，R² {v['tail_r2']}；A回顾性MSE {v['A_mse']}。尾段及A误差并未因扩大候选而降低，不能把外层选择流程改善写成所有场景均改善。",
        v['coverage_table'],v['rolling_text'],v['worst_sample_text'],v['selected_low_text'],v['latency_text'],
        'submission_A.csv为800条训练标签拟合后的A点预测；submission_B_train_only.csv只用训练标签；submission_B.csv为合并公开A标签的研究场景。全部为两列无表头，ID和测试行序一致。B无真值，A标签使用条款未核实，未上传竞赛平台。',
        '## 完整过程与复现',
        '数据核验与关联组 → 字段可用性清点 → 折内处理 → 固定表示比较 → 八候选嵌套选择 → 开发段选定配置 → 冻结校准与尾段 → A/B重拟合 → 逐记录统计和作图。算法预算比较、低值诊断和敏感性是同一研究的独立比较环节。',
        '数据与处理记录、强基线和消融保存在 ../supplement/；八候选协议、逐折分数、逐记录预测、模型和提交保存在本目录；预算实验保存在 ../fair_comparison/。全部实际数字由CSV/JSON生成。修改与舍弃原因另见 PROJECT_PROCESS.md，论文正文保持研究者叙述。',
        '## 可视化',*[f'![{label}](plots/{name}.png)' for name,label in [('review_nested_mse','外层误差'),('supplement_selection','八候选选择'),('fair_algorithms','算法预算比较'),('review_tail_interval','尾段区间'),('review_tail_residuals','逐样本残差'),('supplement_coverage','条件覆盖')]]])
    (OUT/'EXPERIMENT_REPORT.md').write_text(text,encoding='utf-8')
    (OUT.parent/'REPORT.md').write_text(text.replace('](plots/','](branch_selection/plots/').replace('../availability/','availability/').replace('../supplement/','supplement/').replace('../fair_comparison/','fair_comparison/'),encoding='utf-8')
    response='\n\n'.join(['# 特征候选与结论定位处理说明',
        '已修正代码中的候选遗漏：四种表示乘两个树深全部参与内层选择，不根据已看过的外层最低值直接更换模型。',
        v['selection_table'],v['development_candidates_table'],v['nested_interpretation'],
        '最终process_d3并不代表工序统计稳定有效。其选择依据仅是开发段内层最低MSE，外层固定process_d3为0.025181，差于固定time_d2的0.022162；论文同时披露这一点及窗口间排名变化。',
        '统一候选与树轮数下CatBoost得到最低MSE 0.022253，XGBoost为0.023893。因此删除XGBoost普遍最优的暗示，分别限定算法输入、预算和类别编码条件。',
        v['time_stat_text'],
        '现行Holm族为11个不重复比较。此前0.004799来自8行输出，其中基础对组合有一次重复实现；该数仅是当时输出族的校正值，现行正文采用统一11假设族，原始p值与效应不变。不得用新的校正结果声称消除了反复分析偏差。',
        '全部字段清点已输出，但物理可用性核实未完成。用户确认没有字段和事件时点说明。模型仍是匿名完整记录的回顾性关联研究，既不认证无泄漏，也不声称实现提前预警。',
        '摘要、研究目标、特征定义、表5-1至表5-4a、区间、A/B结果和结论同步更新；组合表示不再暗示最好，正文以本文的研究叙述展开。'])
    (OUT.parents[1]/'review'/'特征候选与结论定位处理说明.md').write_text(response,encoding='utf-8')
    return manifest


if __name__=='__main__':generate()
