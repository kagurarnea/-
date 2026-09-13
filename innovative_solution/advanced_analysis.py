from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import acf
from xgboost import XGBRegressor

from .config import PipelineConfig
from .pipeline import (
    FoldFeatureBuilder,
    PreparedData,
    RunTrace,
    infer_operation,
    log,
    metric_row,
    operation_sort_key,
    prepare_data,
    build_fold_features,
    validation_groups,
    purge_training_overlap,
)
from .reducers import available_reducers, make_reducer
from .reporting import COLORS, configure_style, save


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reproducibility_snapshot(config: PipelineConfig) -> None:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "statsmodels",
        "xgboost",
        "openpyxl",
    ]
    snapshot = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "random_state": config.random_state,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(config).items()
        },
        "packages": {
            package: importlib.metadata.version(package) for package in packages
        },
        "input_files": {},
    }
    for filename in (config.train_file, config.test_a_file, config.test_b_file):
        path = config.data_dir / filename
        snapshot["input_files"][filename] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    (config.output_dir / "reproducibility_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_missing_audit(
    data: PreparedData, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    operations = sorted(
        set(data.feature_operation.values()) - {"META"}, key=operation_sort_key
    )
    operation_matrix = pd.DataFrame(index=data.train_numeric.index)
    all_missing_matrix = pd.DataFrame(index=data.train_numeric.index)
    mechanism_rows: list[dict[str, object]] = []

    for operation in operations:
        columns = [
            column
            for column in data.train_numeric.columns
            if data.feature_operation.get(column) == operation
        ]
        if not columns:
            continue
        missing_rate = data.train_numeric[columns].isna().mean(axis=1)
        operation_matrix[operation] = missing_rate
        all_missing_matrix[operation] = missing_rate.eq(1.0)

        tool = data.operation_tool.get(operation)
        if tool and tool in data.train_categorical.columns:
            grouped = missing_rate.groupby(data.train_categorical[tool]).mean()
            tool_disparity = float(grouped.max() - grouped.min()) if len(grouped) else 0.0
        else:
            tool_disparity = 0.0

        overall = float(missing_rate.mean())
        if missing_rate.nunique(dropna=True) > 1:
            target_correlation = float(
                pd.Series(missing_rate).corr(data.y, method="spearman")
            )
        else:
            target_correlation = 0.0
        target_correlation = target_correlation if np.isfinite(target_correlation) else 0.0
        if tool_disparity >= 0.10:
            mechanism = "设备/工艺条件相关缺失候选"
        elif overall >= 0.20:
            mechanism = "结构性高缺失候选"
        elif overall > 0:
            mechanism = "低比例随机缺失候选"
        else:
            mechanism = "无缺失"
        mechanism_rows.append(
            {
                "operation": operation,
                "feature_count": len(columns),
                "mean_missing_rate": overall,
                "max_row_missing_rate": float(missing_rate.max()),
                "all_missing_rows": int(all_missing_matrix[operation].sum()),
                "associated_tool": tool or "",
                "tool_missing_disparity": tool_disparity,
                "spearman_missing_vs_y": target_correlation,
                "mechanism_candidate": mechanism,
                "interpretation": (
                    "统计候选结论，需要结合设备停机、未经过工序或采集故障记录确认"
                ),
            }
        )

    row_summary = pd.DataFrame(
        {
            "row_index": data.train_numeric.index,
            "ID": data.train_ids,
            "missing_count": data.train_numeric.isna().sum(axis=1),
            "missing_rate": data.train_numeric.isna().mean(axis=1),
            "all_missing_operations": all_missing_matrix.sum(axis=1),
            "partial_missing_operations": operation_matrix.gt(0).sum(axis=1)
            - all_missing_matrix.sum(axis=1),
            "Y": data.y,
        }
    )
    if data.primary_tool_column:
        row_summary["Tool"] = data.train_categorical[data.primary_tool_column]
    row_summary["highest_missing_operation"] = operation_matrix.idxmax(axis=1)
    row_summary["highest_operation_missing_rate"] = operation_matrix.max(axis=1)

    operation_matrix_out = operation_matrix.copy()
    operation_matrix_out.insert(0, "ID", data.train_ids)
    mechanism = pd.DataFrame(mechanism_rows)
    row_summary.to_csv(
        config.output_dir / "row_missing_report.csv", index=False, encoding="utf-8-sig"
    )
    operation_matrix_out.to_csv(
        config.output_dir / "row_operation_missing_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mechanism.to_csv(
        config.output_dir / "missing_mechanism_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return row_summary, operation_matrix, mechanism


def plot_missing_audit(
    data: PreparedData,
    row_summary: pd.DataFrame,
    operation_matrix: pd.DataFrame,
    plot_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(
        row_summary["missing_rate"] * 100,
        bins=25,
        color=COLORS["blue"],
        alpha=0.82,
    )
    axes[0].set_xlabel("单个样本缺失率（%）")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("逐行缺失分布")
    column_missing = data.train_numeric.isna().mean(axis=0) * 100
    axes[1].hist(column_missing, bins=25, color=COLORS["orange"], alpha=0.82)
    axes[1].set_xlabel("单个特征缺失率（%）")
    axes[1].set_ylabel("特征数")
    axes[1].set_title("逐列缺失分布")
    save(plot_dir, "17_row_and_column_missingness.png")

    ordered = operation_matrix.loc[
        row_summary.sort_values("missing_rate", ascending=False).index
    ]
    plt.figure(figsize=(12, 7))
    image = plt.imshow(
        ordered.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1
    )
    plt.xticks(np.arange(len(ordered.columns)), ordered.columns, rotation=35)
    plt.xlabel("工序编号")
    plt.ylabel("按总缺失率排序的样本")
    plt.title("样本×工序缺失率热力图")
    plt.colorbar(image, label="该样本在该工序的缺失比例")
    save(plot_dir, "18_sample_operation_missing_heatmap.png")


def build_variance_report(
    data: PreparedData, selection: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    numeric = data.train_numeric
    q75 = numeric.quantile(0.75)
    q25 = numeric.quantile(0.25)
    med = numeric.median()
    variance = numeric.var(ddof=1)
    report = pd.DataFrame(
        {
            "feature": numeric.columns,
            "operation": [data.feature_operation.get(column, "META") for column in numeric],
            "variance": variance.to_numpy(),
            "log10_variance": np.log10(np.maximum(variance.to_numpy(), 1e-30)),
            "iqr": (q75 - q25).to_numpy(),
            "median": med.to_numpy(),
            "coefficient_of_variation": (
                np.sqrt(variance) / med.abs().replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).to_numpy(),
            "missing_rate": numeric.isna().mean().to_numpy(),
            "skewness": numeric.skew().to_numpy(),
            "kurtosis": numeric.kurt().to_numpy(),
        }
    )
    report["variance_percentile"] = report["variance"].rank(pct=True)
    report = report.merge(
        selection[
            [
                "feature",
                "selection_rate",
                "mean_importance",
                "mean_abs_correlation",
                "mean_temporal_drift",
            ]
        ],
        on="feature",
        how="left",
    )
    report["variance_interpretation"] = np.select(
        [
            report["variance_percentile"] <= 0.05,
            report["variance_percentile"] >= 0.95,
        ],
        ["低方差候选，需结合量纲和预测贡献判断", "高方差候选，检查量纲和异常值"],
        default="中等方差",
    )
    report.sort_values("variance", ascending=False).to_csv(
        config.output_dir / "feature_variance_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return report


def fit_theoretical_distributions(
    data: PreparedData,
    selection: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, list[str]]:
    features = selection.head(12)["feature"].tolist()
    candidates = {
        "normal": stats.norm,
        "laplace": stats.laplace,
        "logistic": stats.logistic,
        "student_t": stats.t,
    }
    rows: list[dict[str, object]] = []
    for feature in features:
        values = data.train_numeric[feature].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 20 or np.isclose(np.std(values), 0):
            continue
        for name, distribution in candidates.items():
            try:
                parameters = distribution.fit(values)
                log_likelihood = float(np.sum(distribution.logpdf(values, *parameters)))
                aic = 2 * len(parameters) - 2 * log_likelihood
                ks_statistic, ks_pvalue = stats.kstest(values, distribution.cdf, args=parameters)
                rows.append(
                    {
                        "feature": feature,
                        "operation": data.feature_operation.get(feature, "META"),
                        "distribution": name,
                        "aic": aic,
                        "ks_statistic": ks_statistic,
                        "ks_pvalue_naive_unadjusted": ks_pvalue,
                        "inference_status": "descriptive_only_parameters_fitted_on_same_sample",
                        "parameters": repr(tuple(float(value) for value in parameters)),
                    }
                )
            except (ValueError, FloatingPointError):
                continue
    result = pd.DataFrame(rows)
    if not result.empty:
        result["best_by_aic"] = result.groupby("feature")["aic"].transform("min").eq(
            result["aic"]
        )
        # The ordinary KS null distribution assumes parameters fixed before
        # seeing these observations. A same-sample fit invalidates that p-value;
        # neither a rejection nor a consistency conclusion is justified here.
        result["interpretation"] = "AIC compares candidates; KS distance is descriptive, no calibrated goodness-of-fit claim"
    result.to_csv(
        config.output_dir / "theoretical_distribution_fit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result, features


def plot_variance_and_distributions(
    data: PreparedData,
    variance_report: pd.DataFrame,
    distribution_report: pd.DataFrame,
    features: list[str],
    plot_dir: Path,
) -> None:
    frame = variance_report.dropna(subset=["mean_importance"]).copy()
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        frame["variance"].clip(lower=1e-20),
        frame["mean_importance"],
        c=frame["mean_temporal_drift"].fillna(0),
        s=18 + 80 * frame["selection_rate"].fillna(0),
        cmap="OrRd",
        alpha=0.62,
    )
    plt.xscale("log")
    plt.xlabel("原始方差（对数坐标，受量纲影响）")
    plt.ylabel("跨折ExtraTrees平均贡献")
    plt.title("方差、预测贡献与时间漂移的关系")
    plt.colorbar(scatter, label="时间漂移")
    save(plot_dir, "19_variance_importance_drift.png")

    best = distribution_report.loc[distribution_report.get("best_by_aic", False)].copy()
    best_map = best.set_index("feature")["distribution"].to_dict() if not best.empty else {}
    distributions = {
        "normal": stats.norm,
        "laplace": stats.laplace,
        "logistic": stats.logistic,
        "student_t": stats.t,
    }
    selected = [feature for feature in features if feature in best_map][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax, feature in zip(axes.flat, selected):
        values = data.train_numeric[feature].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        distribution_name = best_map[feature]
        row = best.loc[best.feature == feature].iloc[0]
        parameters = tuple(float(value) for value in row["parameters"].strip("()").split(",") if value.strip())
        lower, upper = np.quantile(values, [0.005, 0.995])
        grid = np.linspace(lower, upper, 350)
        ax.hist(values, bins=28, density=True, alpha=0.55, color=COLORS["blue"])
        ax.plot(
            grid,
            distributions[distribution_name].pdf(grid, *parameters),
            color=COLORS["red"],
            linewidth=2,
        )
        ax.set_title(
            f"{feature}: {distribution_name}\nKS D={row['ks_statistic']:.3g}（描述性）"
        )
        ax.set_xlabel("观测值")
        ax.set_ylabel("密度")
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")
    fig.suptitle("实际分布与AIC最优理论分布", fontsize=14)
    save(plot_dir, "20_theoretical_distribution_fit.png")


def make_development_folds(
    data: PreparedData, config: PipelineConfig, n_splits: int = 3
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    order = np.argsort(data.order_key.to_numpy(dtype=float), kind="stable")
    holdout_size = max(1, int(round(len(order) * config.temporal_holdout_fraction)))
    groups = validation_groups(data)
    development = purge_training_overlap(order[:-holdout_size], order[-holdout_size:], groups)
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=max(20, len(development) // (n_splits + 5)),
    )
    folds = [
        (purge_training_overlap(development[train_position], development[valid_position], groups),
         development[valid_position])
        for train_position, valid_position in splitter.split(development)
    ]
    return development, folds


def build_fold_feature_cache(
    data: PreparedData,
    config: PipelineConfig,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    cache: list[dict[str, object]] = []
    groups = validation_groups(data)
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        train_index = purge_training_overlap(train_index, valid_index, groups)
        log(f"完善实验：构建滚动Fold {fold}/{len(folds)}特征")
        builder = FoldFeatureBuilder(data, config, config.random_state + 900 + fold)
        builder.fit(
            data.train_numeric.iloc[train_index],
            data.train_categorical.iloc[train_index],
            data.train_time.iloc[train_index],
            data.y.iloc[train_index],
            data.order_key.iloc[train_index],
        )
        cache.append(
            {
                "fold": fold,
                "data": data,
                "groups": groups,
                "train_index": train_index,
                "valid_index": valid_index,
                "X_train": builder.transform(
                    data.train_numeric.iloc[train_index],
                    data.train_categorical.iloc[train_index],
                    data.train_time.iloc[train_index],
                    feature_view="full",
                ),
                "X_valid": builder.transform(
                    data.train_numeric.iloc[valid_index],
                    data.train_categorical.iloc[valid_index],
                    data.train_time.iloc[valid_index],
                    feature_view="full",
                ),
                "y_train": data.y.iloc[train_index],
                "y_valid": data.y.iloc[valid_index],
            }
        )
    return cache


def make_transformer(name: str, n_samples: int):
    if name == "none":
        return None
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler(quantile_range=(25, 75))
    if name == "yeo_johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if name == "quantile_normal":
        return QuantileTransformer(
            n_quantiles=min(100, n_samples),
            output_distribution="normal",
            random_state=42,
        )
    raise ValueError(f"Unknown transformer: {name}")


def run_transform_reducer_experiment(
    fold_cache: list[dict[str, object]], config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments = [
        ("raw_none", "none", "none"),
        ("standard", "standard", "none"),
        ("robust", "robust", "none"),
        ("yeo_johnson", "yeo_johnson", "none"),
        ("quantile_normal", "quantile_normal", "none"),
        ("pca50", "standard", "pca50"),
        ("pca95", "standard", "pca95"),
        ("svd50", "standard", "svd50"),
        ("pls20", "standard", "pls20"),
    ]
    rows: list[dict[str, object]] = []
    for experiment, transform_name, reducer_name in experiments:
        log(f"完善实验：特征变换/降维 {experiment}")
        for item in fold_cache:
            started = time.perf_counter()
            X_train_frame: pd.DataFrame = item["X_train"]
            X_valid_frame: pd.DataFrame = item["X_valid"]
            nonconstant = X_train_frame.std(axis=0).gt(1e-12)
            X_train = X_train_frame.loc[:, nonconstant].to_numpy(dtype=float)
            X_valid = X_valid_frame.loc[:, nonconstant].to_numpy(dtype=float)
            y_train = item["y_train"].to_numpy(dtype=float)
            y_valid = item["y_valid"].to_numpy(dtype=float)

            transformer = make_transformer(transform_name, len(X_train))
            if transformer is not None:
                X_train = transformer.fit_transform(X_train)
                X_valid = transformer.transform(X_valid)
            reducer = make_reducer(reducer_name, config.random_state + int(item["fold"]))
            X_train_reduced = reducer.fit_transform(X_train, y_train)
            X_valid_reduced = reducer.transform(X_valid)
            model = Ridge(alpha=100.0)
            model.fit(X_train_reduced, y_train)
            prediction = model.predict(X_valid_reduced)
            rows.append(
                {
                    "experiment": experiment,
                    "transform": transform_name,
                    "reducer": reducer_name,
                    "fold": item["fold"],
                    **metric_row(y_valid, prediction),
                    "input_features": X_train.shape[1],
                    "output_features": X_train_reduced.shape[1],
                    "explained_variance": getattr(reducer, "explained_variance_", np.nan),
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["experiment", "transform", "reducer"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            output_features=("output_features", "mean"),
            explained_variance=("explained_variance", "mean"),
            elapsed_sec=("elapsed_sec", "sum"),
        )
        .sort_values("mse_mean")
    )
    folds.to_csv(
        config.output_dir / "transform_reducer_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        config.output_dir / "transform_reducer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return folds, summary


def xgb_for_experiment(seed: int, **overrides) -> XGBRegressor:
    parameters = {
        "objective": "reg:squarederror",
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.55,
        "reg_alpha": 0.01,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    parameters.update(overrides)
    return XGBRegressor(**parameters)


def ablation_columns(frame: pd.DataFrame, experiment: str) -> list[str]:
    prefixes = {
        "A0_raw": ("raw__",),
        "A1_raw_missing": ("raw__", "missing__"),
        "A2_process": ("raw__", "missing__", "proc__"),
        "A3_process_time": ("raw__", "missing__", "proc__", "time__", "route__"),
        "A4_full": ("raw__", "missing__", "proc__", "time__", "route__", "cat__"),
        "A5_process_only": ("proc__", "time__", "route__", "cat__"),
    }
    return [column for column in frame.columns if column.startswith(prefixes[experiment])]


def run_ablation_experiment(
    fold_cache: list[dict[str, object]], config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments = [
        "A0_raw",
        "A1_raw_missing",
        "A2_process",
        "A3_process_time",
        "A4_full",
        "A5_process_only",
    ]
    rows = []
    for experiment in experiments:
        log(f"完善实验：消融 {experiment}")
        for item in fold_cache:
            columns = ablation_columns(item["X_train"], experiment)
            model = xgb_for_experiment(config.random_state + int(item["fold"]))
            started = time.perf_counter()
            model.fit(item["X_train"][columns], item["y_train"])
            prediction = model.predict(item["X_valid"][columns])
            rows.append(
                {
                    "experiment": experiment,
                    "fold": item["fold"],
                    **metric_row(item["y_valid"].to_numpy(), prediction),
                    "n_features": len(columns),
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("experiment", as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            n_features=("n_features", "mean"),
            elapsed_sec=("elapsed_sec", "sum"),
        )
        .sort_values("mse_mean")
    )
    folds.to_csv(
        config.output_dir / "ablation_folds.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        config.output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig"
    )
    return folds, summary


def build_inner_feature_cache(
    item: dict[str, object], config: PipelineConfig, n_splits: int = 3
) -> list[dict[str, object]]:
    """Relearn all preprocessing/target-dependent selection inside inner folds."""
    if "data" not in item:
        raise ValueError("Nested validation requires raw PreparedData, not only preselected matrices")
    data: PreparedData = item["data"]
    indices = np.asarray(item["train_index"], dtype=int)
    indices = indices[np.argsort(data.order_key.iloc[indices].to_numpy(), kind="stable")]
    groups = item.get("groups")
    if groups is None:
        groups = validation_groups(data)
    cache = []
    for inner_fold, (train_position, valid_position) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(indices), start=1
    ):
        valid_index = indices[valid_position]
        train_index = purge_training_overlap(indices[train_position], valid_index, groups)
        builder, X_train, X_valid = build_fold_features(
            data, config, train_index, valid_index,
            seed=config.random_state + 1200 + 10 * int(item["fold"]) + inner_fold,
            feature_view="full",
        )
        cache.append({"train_index": train_index, "valid_index": valid_index,
                      "X_train": X_train, "X_valid": X_valid,
                      "y_train": data.y.iloc[train_index],
                      "y_valid": data.y.iloc[valid_index],
                      "structural_report": builder.structural_report_})
    return cache


def run_nested_grid_search(
    fold_cache: list[dict[str, object]], config: PipelineConfig,
    parameter_grid: dict[str, list[object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_grid = parameter_grid or {
        "n_estimators": [800, 1200],
        "max_depth": [2, 3],
        "min_child_weight": [1, 3],
    }
    rows: list[dict[str, object]] = []
    candidates: list[pd.DataFrame] = []
    for item in fold_cache:
        fold = int(item["fold"])
        log(f"完善实验：Fold {fold}内层网格搜索")
        started = time.perf_counter()
        inner_cache = build_inner_feature_cache(item, config)
        candidate_rows = []
        for parameters in ParameterGrid(parameter_grid):
            losses = []
            for inner_fold, inner in enumerate(inner_cache, start=1):
                estimator = xgb_for_experiment(
                    config.random_state + 1200 + fold + inner_fold,
                    n_jobs=1, **parameters,
                )
                estimator.fit(inner["X_train"], inner["y_train"])
                losses.append(mean_squared_error(inner["y_valid"],
                                                 estimator.predict(inner["X_valid"])))
            candidate_rows.append({"params": json.dumps(parameters, sort_keys=True),
                                   "mean_test_score": -float(np.mean(losses)),
                                   "std_test_score": float(np.std(losses)),
                                   "inner_mse": float(np.mean(losses)),
                                   "outer_fold": fold,
                                   "preprocessing_scope": "inner_training_rows_only",
                                   "inner_split": "purged_expanding_order_proxy"})
        candidate = pd.DataFrame(candidate_rows)
        candidate["rank_test_score"] = candidate["inner_mse"].rank(method="min").astype(int)
        best = candidate.sort_values(["inner_mse", "params"], kind="stable").iloc[0]
        best_params = json.loads(best["params"])
        estimator = xgb_for_experiment(config.random_state + 1200 + fold,
                                       n_jobs=1, **best_params)
        estimator.fit(item["X_train"], item["y_train"])
        prediction = estimator.predict(item["X_valid"])
        rows.append(
            {
                "outer_fold": fold,
                **metric_row(item["y_valid"].to_numpy(), prediction),
                "inner_best_mse": float(best["inner_mse"]),
                "best_params": best["params"],
                "preprocessing_scope": "relearned_in_each_inner_and_outer_training_fold",
                "elapsed_sec": time.perf_counter() - started,
            }
        )
        candidates.append(candidate)
    outer = pd.DataFrame(rows)
    candidate_frame = pd.concat(candidates, ignore_index=True)
    outer.to_csv(
        config.output_dir / "nested_grid_search_outer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_frame.to_csv(
        config.output_dir / "nested_grid_search_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return outer, candidate_frame


def run_tree_model_comparison(
    fold_cache: list[dict[str, object]], config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for item in fold_cache:
        fold = int(item["fold"])
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=config.random_state + fold,
                n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=config.random_state + fold,
                n_jobs=-1,
            ),
            "XGBoost": xgb_for_experiment(config.random_state + fold),
        }
        for model_name, model in models.items():
            started = time.perf_counter()
            model.fit(item["X_train"], item["y_train"])
            prediction = model.predict(item["X_valid"])
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    **metric_row(item["y_valid"].to_numpy(), prediction),
                    "elapsed_sec": time.perf_counter() - started,
                }
            )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("model", as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            elapsed_sec=("elapsed_sec", "sum"),
        )
        .sort_values("mse_mean")
    )
    folds.to_csv(
        config.output_dir / "tree_model_comparison_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        config.output_dir / "tree_model_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return folds, summary


def plot_experiment_results(
    transform_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    grid_outer: pd.DataFrame,
    grid_candidates: pd.DataFrame,
    tree_summary: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot = transform_summary.sort_values("mse_mean", ascending=False)
    plt.figure(figsize=(10.5, 6.8))
    lower_error = np.minimum(
        plot["mse_std"].fillna(0).to_numpy(),
        plot["mse_mean"].to_numpy() * 0.95,
    )
    plt.errorbar(
        plot["mse_mean"],
        np.arange(len(plot)),
        xerr=np.vstack([lower_error, plot["mse_std"].fillna(0).to_numpy()]),
        fmt="o",
        capsize=4,
        color=COLORS["blue"],
    )
    plt.yticks(np.arange(len(plot)), plot["experiment"])
    plt.xlabel("滚动CV MSE均值 ± 折间标准差")
    plt.ylabel("特征变换/降维配置")
    plt.xscale("log")
    plt.title("可插拔特征变换与降维对比（MSE对数坐标）")
    save(plot_dir, "21_transform_reducer_comparison.png")

    reducer_plot = transform_summary.loc[transform_summary.reducer != "none"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        reducer_plot["experiment"],
        reducer_plot["output_features"],
        color=COLORS["purple"],
    )
    axes[0].set_ylabel("降维后特征数")
    axes[0].set_title("维度压缩程度")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(
        reducer_plot["experiment"],
        reducer_plot["mse_mean"],
        color=COLORS["green"],
    )
    axes[1].set_ylabel("滚动CV MSE")
    axes[1].set_title("压缩后的预测误差")
    axes[1].tick_params(axis="x", rotation=30)
    save(plot_dir, "22_reducer_dimension_and_error.png")

    ablation = ablation_summary.sort_values("mse_mean", ascending=False)
    plt.figure(figsize=(10, 5.8))
    plt.errorbar(
        ablation["mse_mean"],
        np.arange(len(ablation)),
        xerr=ablation["mse_std"],
        fmt="o",
        capsize=4,
        color=COLORS["red"],
    )
    plt.yticks(np.arange(len(ablation)), ablation["experiment"])
    plt.xlabel("滚动CV MSE均值 ± 折间标准差")
    plt.title("工序感知特征的逐步消融")
    save(plot_dir, "23_feature_ablation.png")

    candidate = (
        grid_candidates.groupby("params", as_index=False)
        .agg(inner_mse=("inner_mse", "mean"), inner_std=("inner_mse", "std"))
        .sort_values("inner_mse")
        .head(8)
        .iloc[::-1]
    )
    labels = [
        value.replace('"', "").replace("n_estimators", "n").replace("min_child_weight", "child")
        for value in candidate["params"]
    ]
    plt.figure(figsize=(11.5, 6.5))
    plt.errorbar(
        candidate["inner_mse"],
        np.arange(len(candidate)),
        xerr=candidate["inner_std"].fillna(0),
        fmt="o",
        capsize=4,
        color=COLORS["blue"],
    )
    plt.yticks(np.arange(len(candidate)), labels)
    plt.xlabel("内层CV MSE")
    plt.title(
        f"嵌套网格搜索候选；外层滚动MSE={grid_outer['mse'].mean():.6f}"
    )
    save(plot_dir, "24_nested_grid_search.png")

    tree_plot = tree_summary.sort_values("mse_mean", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    plt.errorbar(
        tree_plot["mse_mean"],
        np.arange(len(tree_plot)),
        xerr=tree_plot["mse_std"],
        fmt="o",
        capsize=5,
        color=COLORS["green"],
    )
    plt.yticks(np.arange(len(tree_plot)), tree_plot["model"])
    plt.xlabel("滚动CV MSE均值 ± 折间标准差")
    plt.title("随机森林、极端随机树与XGBoost对比")
    save(plot_dir, "25_tree_model_comparison.png")


def build_sample_and_residual_diagnostics(
    data: PreparedData,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = pd.read_csv(config.output_dir / "temporal_holdout_predictions.csv")
    order = np.argsort(data.order_key.to_numpy(dtype=float), kind="stable")
    holdout_size = max(1, int(round(len(order) * config.temporal_holdout_fraction)))
    holdout_index = order[-holdout_size:]
    development_index = purge_training_overlap(order[:-holdout_size], holdout_index,
                                                validation_groups(data))
    builder = FoldFeatureBuilder(data, config, config.random_state + 1500)
    builder.fit(
        data.train_numeric.iloc[development_index],
        data.train_categorical.iloc[development_index],
        data.train_time.iloc[development_index],
        data.y.iloc[development_index],
        data.order_key.iloc[development_index],
    )
    holdout_numeric = data.train_numeric.iloc[holdout_index]
    imputed = builder.impute(
        holdout_numeric, data.train_categorical.iloc[holdout_index]
    )
    assert builder.robust_median is not None and builder.robust_iqr is not None
    robust_z = ((imputed - builder.robust_median) / builder.robust_iqr).clip(-20, 20)
    holdout_by_index = holdout.set_index("row_index")
    selected_score = builder.selection_report.set_index("feature")["stable_score"]
    worst_indices = (
        holdout.assign(abs_error=holdout["residual_RobustBlend"].abs())
        .nlargest(20, "abs_error")["row_index"]
        .astype(int)
        .tolist()
    )
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row_index in worst_indices:
        z = robust_z.loc[row_index, builder.selected_features].abs().nlargest(12)
        original = data.train_numeric.loc[row_index]
        top_operations = (
            pd.Series(
                {
                    operation: float(
                        z[[feature for feature in z.index if infer_operation(feature) == operation]].sum()
                    )
                    for operation in {infer_operation(feature) for feature in z.index}
                }
            )
            .sort_values(ascending=False)
            .head(3)
            .index.astype(str)
            .tolist()
        )
        prediction_row = holdout_by_index.loc[row_index]
        summary_rows.append(
            {
                "row_index": row_index,
                "ID": prediction_row["ID"],
                "Tool": prediction_row.get("Tool", ""),
                "y_true": prediction_row["y_true"],
                "prediction": prediction_row["pred_RobustBlend"],
                "residual": prediction_row["residual_RobustBlend"],
                "absolute_error": abs(prediction_row["residual_RobustBlend"]),
                "missing_rate": float(original.isna().mean()),
                "top_abnormal_operations": ",".join(top_operations),
                "top_abnormal_features": ",".join(z.index[:5]),
            }
        )
        for rank, feature in enumerate(z.index, start=1):
            detail_rows.append(
                {
                    "row_index": row_index,
                    "ID": prediction_row["ID"],
                    "rank": rank,
                    "feature": feature,
                    "operation": infer_operation(feature),
                    "original_value": original[feature],
                    "training_median": builder.robust_median[feature],
                    "robust_z": robust_z.loc[row_index, feature],
                    "was_missing": bool(pd.isna(original[feature])),
                    "stable_selection_score": selected_score.get(feature, np.nan),
                }
            )
    sample_summary = pd.DataFrame(summary_rows)
    sample_detail = pd.DataFrame(detail_rows)

    residual = holdout["residual_RobustBlend"].to_numpy(dtype=float)
    prediction = holdout["pred_RobustBlend"].to_numpy(dtype=float)
    shapiro_stat, shapiro_p = stats.shapiro(residual)
    normal_stat, normal_p = stats.normaltest(residual)
    bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(
        residual, np.column_stack([np.ones(len(prediction)), prediction])
    )
    ljung = acorr_ljungbox(residual, lags=[5, 10], return_df=True)
    tests = [
        {
            "test": "Shapiro-Wilk normality",
            "statistic": shapiro_stat,
            "pvalue": shapiro_p,
            "interpretation": "独立样本假设下的未校正诊断p值；顺序依赖会影响其解释",
        },
        {
            "test": "D'Agostino normality",
            "statistic": normal_stat,
            "pvalue": normal_p,
            "interpretation": "独立样本假设下的偏度/峰度诊断；非正态不直接否定回归预测",
        },
        {
            "test": "Breusch-Pagan heteroscedasticity",
            "statistic": bp_stat,
            "pvalue": bp_p,
            "interpretation": "仅检验与预测值线性相关的方差变化；未拒绝不证明同方差",
        },
    ]
    for lag, row in ljung.iterrows():
        tests.append(
            {
                "test": f"Ljung-Box autocorrelation lag={lag}",
                "statistic": row["lb_stat"],
                "pvalue": row["lb_pvalue"],
                "interpretation": "顺序代理下残差自相关的探索性检验；时间与批次语义未经生产日志确认",
            }
        )
    residual_tests = pd.DataFrame(tests)
    sample_summary.to_csv(
        config.output_dir / "high_error_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sample_detail.to_csv(
        config.output_dir / "high_error_sample_feature_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residual_tests.to_csv(
        config.output_dir / "residual_statistical_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return sample_summary, sample_detail, residual_tests


def plot_sample_and_residual_diagnostics(
    holdout: pd.DataFrame,
    sample_summary: pd.DataFrame,
    sample_detail: pd.DataFrame,
    plot_dir: Path,
) -> None:
    top_samples = sample_summary.head(12)
    top_rows = top_samples["row_index"].tolist()
    pivot = (
        sample_detail.loc[sample_detail.row_index.isin(top_rows) & (sample_detail["rank"] <= 8)]
        .pivot(index="row_index", columns="feature", values="robust_z")
        .reindex(top_rows)
        .fillna(0.0)
    )
    plt.figure(figsize=(13, 6.8))
    image = plt.imshow(
        pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-8, vmax=8
    )
    plt.yticks(np.arange(len(pivot.index)),
               [f"{row.ID} [row {row.row_index}]" for row in top_samples.itertuples()])
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=65, ha="right")
    plt.xlabel("进入样本局部诊断的异常特征")
    plt.ylabel("高误差样本")
    plt.title("高误差样本的稳健Z分数局部诊断")
    plt.colorbar(image, label="相对训练分布的稳健Z分数")
    save(plot_dir, "26_high_error_sample_diagnosis.png")

    residual = holdout["residual_RobustBlend"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(residual, dist="norm", plot=axes[0])
    axes[0].set_title("残差正态Q-Q图")
    autocorrelation = acf(residual, nlags=min(20, len(residual) // 4), fft=False)
    axes[1].stem(np.arange(len(autocorrelation)), autocorrelation, basefmt=" ")
    confidence = 1.96 / np.sqrt(len(residual))
    axes[1].axhline(confidence, color=COLORS["red"], linestyle="--")
    axes[1].axhline(-confidence, color=COLORS["red"], linestyle="--")
    axes[1].set_xlabel("顺序代理滞后阶数")
    axes[1].set_ylabel("残差自相关")
    axes[1].set_title("残差时间自相关")
    save(plot_dir, "27_residual_qq_and_autocorrelation.png")


def feature_semantics(config: PipelineConfig) -> pd.DataFrame:
    importance = pd.read_csv(config.output_dir / "engineered_feature_importance.csv")

    def describe(feature: str) -> tuple[str, str, str, str]:
        if feature.startswith("raw__"):
            return (
                "匿名原始过程变量",
                "原始量纲未知",
                "对应工序中的机台或工艺测量值",
                "需由字段数据字典确认是温度、流量、功率或其他参数",
            )
        if feature.startswith("missing__"):
            return (
                "缺失指示",
                "0/1",
                "设备、工序未采集或测量失败状态",
                "缺失机制需结合生产日志确认",
            )
        if feature.startswith("proc__"):
            return (
                "工序稳健统计",
                "无量纲或比例",
                "同编号工序字段相对训练样本中位数与IQR的统计偏离",
                "训练样本未被证实均属正常工况；这是统计汇总，不等同于单一物理量",
            )
        if feature.startswith("time__"):
            return (
                "工序时间",
                "天或分钟",
                "同编号时间字段的最小值、最大值、中位数与跨度",
                "格式仅支持时间候选判断，不能证明分别是工序开始/结束",
            )
        if feature.startswith("route__"):
            return (
                "跨工序时间",
                "小时或0/1",
                "相邻编号时间组的差值和负差值指示",
                "编号顺序不一定是工艺路线，等待或重叠的解释需要业务确认",
            )
        if feature.startswith("cat__"):
            return (
                "设备/工艺类别",
                "类别指示",
                "机台、腔体或工艺路线身份",
                "匿名类别不表示性能高低",
            )
        return ("其他", "未知", "待确认", "需要字段元数据")

    descriptions = importance["feature"].map(describe)
    importance[["feature", "operation"]].assign(
        feature_type=descriptions.map(lambda value: value[0]),
        unit=descriptions.map(lambda value: value[1]),
        physical_interpretation=descriptions.map(lambda value: value[2]),
        interpretation_limit=descriptions.map(lambda value: value[3]),
    ).drop_duplicates().to_csv(
        config.output_dir / "feature_semantics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return pd.read_csv(config.output_dir / "feature_semantics.csv")


def write_advanced_summary(
    config: PipelineConfig,
    row_summary: pd.DataFrame,
    mechanism: pd.DataFrame,
    transform_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    grid_outer: pd.DataFrame,
    tree_summary: pd.DataFrame,
    sample_summary: pd.DataFrame,
    residual_tests: pd.DataFrame,
    elapsed_sec: float,
) -> None:
    best_transform = transform_summary.iloc[0]
    best_ablation = ablation_summary.iloc[0]
    summary = {
        "row_missing_rate_max": float(row_summary["missing_rate"].max()),
        "rows_with_any_missing": int(row_summary["missing_count"].gt(0).sum()),
        "operations_with_device_dependent_missing": int(
            mechanism["mechanism_candidate"].str.contains("设备").sum()
        ),
        "best_transform_reducer": str(best_transform["experiment"]),
        "best_transform_reducer_mse": float(best_transform["mse_mean"]),
        "best_ablation": str(best_ablation["experiment"]),
        "best_ablation_mse": float(best_ablation["mse_mean"]),
        "nested_grid_outer_mse": float(grid_outer["mse"].mean()),
        "best_tree_model": str(tree_summary.iloc[0]["model"]),
        "best_tree_model_mse": float(tree_summary.iloc[0]["mse_mean"]),
        "highest_error_sample": str(sample_summary.iloc[0]["ID"]),
        "significant_residual_tests": residual_tests.loc[
            residual_tests.pvalue < 0.05, "test"
        ].tolist(),
        "available_reducers": available_reducers(),
        "elapsed_sec": elapsed_sec,
        "new_visualizations": 12,
    }
    (config.output_dir / "advanced_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_advanced_trace(frame: pd.DataFrame, plot_dir: Path) -> None:
    plot = frame.loc[frame.stage != "完善分析总计"].sort_values("elapsed_sec")
    plt.figure(figsize=(10, 5.8))
    bars = plt.barh(plot["stage"], plot["elapsed_sec"], color=COLORS["teal"])
    for bar, value in zip(bars, plot["elapsed_sec"]):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1f}s",
            va="center",
        )
    plt.xlabel("耗时（秒）")
    plt.title("完善分析工作痕迹：各阶段实际耗时")
    save(plot_dir, "28_advanced_analysis_trace.png")


def run_advanced_analysis(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.output_dir / "plots"
    configure_style()
    started = time.perf_counter()
    advanced_trace: list[dict[str, object]] = []

    def record(stage: str, stage_started: float, note: str = "") -> None:
        advanced_trace.append(
            {
                "stage": stage,
                "elapsed_sec": time.perf_counter() - stage_started,
                "note": note,
            }
        )

    trace = RunTrace()
    stage_started = time.perf_counter()
    data = prepare_data(config, trace)
    write_reproducibility_snapshot(config)
    record("数据与环境快照", stage_started, "输入SHA256、软件版本和运行配置")

    selection = pd.read_csv(config.output_dir / "feature_selection_stability.csv")
    stage_started = time.perf_counter()
    row_summary, operation_matrix, mechanism = build_missing_audit(data, config)
    plot_missing_audit(data, row_summary, operation_matrix, plot_dir)
    record("逐行逐列缺失审计", stage_started, "行、列、工序和缺失机制候选")

    stage_started = time.perf_counter()
    variance_report = build_variance_report(data, selection, config)
    distribution_report, distribution_features = fit_theoretical_distributions(
        data, selection, config
    )
    plot_variance_and_distributions(
        data,
        variance_report,
        distribution_report,
        distribution_features,
        plot_dir,
    )
    record("方差与理论分布", stage_started, "方差、AIC、KS检验和理论分布图")

    stage_started = time.perf_counter()
    _, folds = make_development_folds(data, config, n_splits=3)
    fold_cache = build_fold_feature_cache(data, config, folds)
    record("统一滚动实验特征", stage_started, "三折共用同一特征构建口径")
    stage_started = time.perf_counter()
    transform_folds, transform_summary = run_transform_reducer_experiment(
        fold_cache, config
    )
    record("特征变换与降维", stage_started, "9组变换/降维方案")
    stage_started = time.perf_counter()
    ablation_folds, ablation_summary = run_ablation_experiment(fold_cache, config)
    record("工序特征消融", stage_started, "6组累计和独立特征视图")
    stage_started = time.perf_counter()
    grid_outer, grid_candidates = run_nested_grid_search(fold_cache, config)
    record("嵌套网格搜索", stage_started, "3个外层时间折、每折24个内层拟合")
    stage_started = time.perf_counter()
    tree_folds, tree_summary = run_tree_model_comparison(fold_cache, config)
    plot_experiment_results(
        transform_summary,
        ablation_summary,
        grid_outer,
        grid_candidates,
        tree_summary,
        plot_dir,
    )
    record("树模型对比与实验图", stage_started, "RF、ExtraTrees和XGBoost")

    stage_started = time.perf_counter()
    sample_summary, sample_detail, residual_tests = build_sample_and_residual_diagnostics(
        data, config
    )
    holdout = pd.read_csv(config.output_dir / "temporal_holdout_predictions.csv")
    plot_sample_and_residual_diagnostics(
        holdout, sample_summary, sample_detail, plot_dir
    )
    feature_semantics(config)
    record("样本、残差与物理语义", stage_started, "逐样本特征偏离和统计检验")
    write_advanced_summary(
        config,
        row_summary,
        mechanism,
        transform_summary,
        ablation_summary,
        grid_outer,
        tree_summary,
        sample_summary,
        residual_tests,
        time.perf_counter() - started,
    )
    advanced_trace.append(
        {
            "stage": "完善分析总计",
            "elapsed_sec": time.perf_counter() - started,
            "note": "从审计到消融、搜索、诊断和可视化",
        }
    )
    advanced_trace_frame = pd.DataFrame(advanced_trace)
    advanced_trace_frame.to_csv(
        config.output_dir / "advanced_run_trace.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_advanced_trace(advanced_trace_frame, plot_dir)
    log(f"完善分析完成：{config.output_dir}")


if __name__ == "__main__":
    run_advanced_analysis()
