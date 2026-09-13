from __future__ import annotations

import json
import pandas as pd

from .config import PipelineConfig
from .pipeline import RunTrace, prepare_data
from .reporting import generate_report
from .workflow_description import generate_workflow_description


def main() -> None:
    config = PipelineConfig()
    output = config.output_dir
    if (output / "branch_selection" / "inference_family.csv").exists():
        from .review.plot_branch_selection import main as plot_branches
        from .review.plot_supplement import main as plot_supplement
        from .branch_report import generate
        plot_supplement()
        plot_branches()
        generate()
        print(f"实验报告已更新：{output / 'REPORT.md'}")
        return
    if (output / "supplement" / "baseline_summary.csv").exists():
        from .review.plot_supplement import main as plot_supplement
        from .supplement_report import generate
        plot_supplement()
        generate()
        print(f"实验报告已更新：{output / 'REPORT.md'}")
        return
    if (output / "review" / "review_manifest.json").exists():
        from .review_report import generate_review_report
        from .review.plot_review_validation import main as plot_current_results
        plot_current_results()
        generate_review_report()
        print(f"审稿修订报告已更新：{output / 'review' / 'REVIEW_REPORT.md'}")
        return
    data = prepare_data(config, RunTrace())
    results = {
        "data": data,
        "fold_metrics": pd.read_csv(output / "fold_metrics.csv"),
        "model_summary": pd.read_csv(output / "model_summary.csv"),
        "blend_weights": pd.read_csv(output / "blend_weights.csv"),
        "random_oof_detail": pd.read_csv(output / "random_oof_predictions.csv"),
        "rolling_oof_detail": pd.read_csv(output / "rolling_oof_predictions.csv"),
        "oof_detail": pd.read_csv(output / "rolling_oof_predictions.csv"),
        "holdout_detail": pd.read_csv(output / "temporal_holdout_predictions.csv"),
        "selection_stability": pd.read_csv(output / "feature_selection_stability.csv"),
        "feature_importance": pd.read_csv(output / "engineered_feature_importance.csv"),
        "operation_importance": pd.read_csv(output / "operation_importance.csv"),
        "operation_drift": pd.read_csv(output / "operation_drift.csv"),
        "conformal_metrics": pd.read_csv(output / "conformal_metrics.csv"),
        "trace": pd.read_csv(output / "run_trace.csv"),
        "manifest": json.loads((output / "run_manifest.json").read_text(encoding="utf-8")),
    }
    generate_report(results, config)
    if (output / "phase_a_manifest.json").exists() and (
        output / "phase_b_manifest.json"
    ).exists():
        generate_workflow_description(config)
    print(f"报告已更新：{output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
