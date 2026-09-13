from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = PROJECT_DIR / "data"
    output_dir: Path = PACKAGE_DIR / "outputs"
    cache_dir: Path = PACKAGE_DIR / ".cache"

    train_file: str = "训练集.xlsx"
    test_a_file: str = "测试集A.xlsx"
    test_b_file: str = "测试集B.xlsx"
    test_a_answer_file: str = "测试集A_答案.csv"

    random_state: int = 42
    n_splits: int = 5
    rolling_n_splits: int = 4
    temporal_holdout_fraction: float = 0.20
    top_k_raw_features: int = 320
    feature_view: str = "full"
    missing_indicator_threshold: float = 0.005
    conformal_alpha: float = 0.10
    n_bootstrap: int = 3000

    selector_estimators: int = 250
    extra_trees_estimators: int = 400
    xgb_estimators: int = 1200
    mlp_members: int = 3

    phase_a_recency_candidates: tuple[float | None, ...] = field(
        default_factory=lambda: (None, 0.20, 0.35, 0.50)
    )
    phase_a_backtest_folds: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: ((500, 100), (600, 100), (700, 100))
    )
    phase_a_ensemble_members: int = 3

    phase_b_a_weight_candidates: tuple[float, ...] = field(
        default_factory=lambda: (1.0, 2.0, 4.0)
    )
    phase_b_backtest_folds: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: ((100, 50), (150, 50), (200, 100))
    )
    phase_b_ensemble_members: int = 3

    ridge_alphas: tuple[float, ...] = field(
        default_factory=lambda: (
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
        )
    )
