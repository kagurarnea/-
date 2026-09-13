"""Standalone process diagram for the quality-prediction paper."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

DEST = Path(__file__).resolve().parents[1] / "outputs" / "review" / "plots"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(9, 4.9))
fig.subplots_adjust(left=.015, right=.985, top=.94, bottom=.09)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, title, body):
    ax.add_patch(Rectangle((x-.145, y-.16), .29, .32, facecolor="#F4F7F9", edgecolor="#A7B4BE", lw=1))
    ax.text(x, y+.080, title, ha="center", va="center", fontsize=13, fontweight="bold", color="#152F42")
    ax.text(x, y-.045, body, ha="center", va="center", fontsize=11.5, linespacing=1.55, color="#263643")


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color="#506B7E", lw=1.2))


box(.165, .79, "数据与记录分组", "训练集及测试 A/B\n字段 缺失 时间分析\n关联记录隔离")
box(.5, .79, "训练折内特征工程", "填充 编码 监督筛选\n四种特征分支表示\n工序与时间增量比较")
box(.835, .79, "XGBoost 集成", "八组候选配置\n三种子等权预测\n拟合器随模型保存")
arrow((.31, .79), (.35, .79)); arrow((.645, .79), (.685, .79))
ax.plot([.835, .835, .165], [.63, .48, .48], color="#506B7E", lw=1.2)
for x in [.165, .5, .835]:
    arrow((x, .48), (x, .37))
box(.165, .20, "嵌套窗口评价", "内层选择 外层比较\n固定特征对照\n同样本误差分析")
box(.5, .20, "冻结模型评价", "开发 校准 尾段分离\n预测区间与覆盖\n残差及单样本分析")
box(.835, .20, "全量拟合与输出", "训练集拟合 A 预测\n公开 A 标签加入 B 拟合\n两列无表头 CSV")
fig.text(.04, .027, "各评价环节按自身训练范围拟合处理器；校准区间仅对应冻结模型。", fontsize=10.5, color="#50606B")
DEST.mkdir(parents=True, exist_ok=True)
fig.savefig(DEST / "thesis_workflow.png", dpi=240, facecolor="white")
fig.savefig(DEST / "thesis_workflow.svg", facecolor="white")
plt.close(fig)
