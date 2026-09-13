"""Declare finite comparisons before model fitting; exploratory, not a new blind test."""
from .supplement_common import OUT, write_json

PROTOCOL = {
    "purpose": "supplementary_analysis_of_one_research_project",
    "status": "retrospective_exploratory_no_independent_new_test",
    "primary_rows": "exact three outer splits saved in outputs/review/split_audit.csv",
    "seeds_final": [42,143,244], "seed_inner": 42,
    "baseline_search": "two inner 60-record forward windows; raw-to-features rebuilt each inner fit",
    "baseline_families": ["Ridge", "RandomForest", "GBR", "SVR", "LightGBM", "CatBoost", "XGBoost", "TemporalStacking"],
    "stacking": "GBR+XGB+RF+SVR, linear meta fit on inner temporal OOF, evaluated only on outer; adapted design not exact published replication",
    "ablation": ["measurements", "basic", "process_only", "time_only", "full", "no_mask", "median_only", "group_min5", "group_min5_shrink10"],
    "ablation_fixed_model": "XGBoost d2,1200 rounds,lr.05,3 seeds,otherwise existing config",
    "sensitivity_seed": 42,
    "sensitivity": {"importance_weight": [.5,.7,.9], "drift_power": [0,.25,.5], "top_k": [160,320,640],
        "missing_threshold": [0,.005,.02], "boosting_round_lr": [[600,.05],[1200,.05],[1200,.025]],
        "l1_l2": [[0,1],[.01,2],[.1,10]]},
    "direction_check": "same 300 validation rows; forward past-only vs reverse future-only; 3 seeds d2 basic/full; direction sensitivity not deployment claim",
    "tail_models": ["frozen_full_d3", "full_d2", "no_time_d2", "absolute_loss_d2", "low_tail_weight_d2"],
    "low_tail_cutoff": "10th percentile of development y; weight 3 for training values below it",
    "intervals": "fixed radius; sequential rolling80 residuals only after each strictly later proxy time; same-proxy groups predicted before their outcomes enter calibration",
    "statistics": "paired group-cluster bootstrap,group sign flip,cluster ordered block sensitivity;B=5000;conditional on saved predictions;no correction of researcher adaptivity",
    "conditional_coverage": "device groups, predicted-value terciles defined by calibration predictions; report n and exploratory Wilson interval",
    "A_B": "no A labels used for supplementary model selection; audit raw cross-dataset ID and exact X; do not claim unpublished B accuracy",
    "final_model_promotion": False,
}

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        import json
        if json.loads(path.read_text(encoding="utf-8")) != PROTOCOL:
            raise RuntimeError("Existing supplementary protocol differs; use a new version instead of overwriting")
    else:
        write_json(path, PROTOCOL)

if __name__ == "__main__":
    freeze()
