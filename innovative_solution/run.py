from __future__ import annotations

import importlib.util

from .advanced_analysis import run_advanced_analysis
from .config import PipelineConfig
from .phase_a_optimization import run_phase_a_optimization
from .phase_b_optimization import run_phase_b_optimization
from .pipeline import log, run_pipeline
from .reporting import generate_report
from .workflow_description import generate_workflow_description


def main() -> None:
    from .branch_selection import main as branches
    from .branch_diagnostics import main as diagnostics
    from .fair_algorithm_comparison import main as fair
    from .availability_audit import main as availability
    from .inference_audit import main as inference
    from .review.plot_branch_selection import main as plot_current_results
    from .branch_report import generate
    branches()
    diagnostics()
    fair()
    availability()
    inference()
    plot_current_results()
    generate()
    log("实验结果与提交文件：innovative_solution/outputs/branch_selection/")


def run_legacy_exploratory_pipeline() -> None:
    """Historical comparison only. Never use this path for independence claims."""
    config = PipelineConfig()
    results = run_pipeline(config)
    if (config.data_dir / config.test_a_answer_file).exists():
        run_phase_a_optimization(
            config,
            results["data"],
            refresh_train_only_baseline=True,
        )
        run_phase_b_optimization(
            config,
            results["data"],
            refresh_train_only_baseline=True,
        )
    run_advanced_analysis(config)
    generate_report(results, config)
    if (config.output_dir / "phase_a_manifest.json").exists() and (
        config.output_dir / "phase_b_manifest.json"
    ).exists():
        generate_workflow_description(config)

    optional_status = [
        {
            "method": "TabPFN v2",
            "role": "小样本表格基础模型候选",
            "installed": bool(importlib.util.find_spec("tabpfn")),
            "executed": False,
            "note": "主流程先筛选至320维；仅在安装模型与权重后单独评测",
        },
        {
            "method": "TabM",
            "role": "参数高效MLP集成候选",
            "installed": bool(importlib.util.find_spec("tabm")),
            "executed": False,
            "note": "当前使用可运行的MLP-Ensemble验证集成方向，不冒充TabM",
        },
        {
            "method": "SCARF",
            "role": "随机特征破坏自监督预训练候选",
            "installed": bool(importlib.util.find_spec("torch")),
            "executed": False,
            "note": "需要PyTorch；主流程已显式建模缺失掩码与工序异常统计",
        },
    ]
    import pandas as pd

    pd.DataFrame(optional_status).to_csv(
        config.output_dir / "optional_modern_methods.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log(f"报告：{config.output_dir / 'REPORT.md'}")
    log(f"图片：{config.output_dir / 'plots'}")
    log(f"提交文件：{config.output_dir / 'submission_A.csv'}")
    log(f"提交文件：{config.output_dir / 'submission_B.csv'}")


if __name__ == "__main__":
    main()
