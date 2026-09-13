# -*- coding: utf-8 -*-

"""
main.py

工业AI特征工程实验主程序。

============================================================
已经确定
============================================================

缺失值：

    operation_tool_mean

CV MSE：

    0.0135389961

因此：
    不重新比较缺失值。

============================================================
本阶段
============================================================

1. 数据分布
2. 理论分布
3. 方差
4. 特征变换
5. RF
6. ExtraTrees
7. XGB Importance
8. Top-K
9. PCA
10. SVD
11. StandardScaler
12. MinMaxScaler

所有实验：
    5 Fold CV

============================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

import missing_value
import feature_engineering


# =========================================================
# 路径
# =========================================================

DATA_DIR = Path(
    r"C:\Users\lenovo\Downloads\my method\data"
)

RESULT_DIR = (
    DATA_DIR
    /
    "preprocess_missing"
    /
    "feature_engineering_results"
)


# =========================================================
# 已确定的基线
# =========================================================

BEST_MISSING_STRATEGY = (
    "operation_tool_mean"
)

BEST_MISSING_MSE = (
    0.0135389961
)


# =========================================================
# 日志
# =========================================================

def log(message):

    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True
    )


def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


# =========================================================
# 模型
# =========================================================

def make_model(
    seed=42
):

    return XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=
            1200,

        learning_rate=
            0.03,

        max_depth=
            3,

        min_child_weight=
            1,

        subsample=
            0.8,

        colsample_bytree=
            0.5,

        reg_alpha=
            0.01,

        reg_lambda=
            1.0,

        gamma=
            0,

        random_state=
            seed,

        n_jobs=
            -1,

        tree_method=
            "hist"
    )


# =========================================================
# 数据准备
# =========================================================

def prepare_data():

    train = (
        missing_value.load_train(
            DATA_DIR
        )
    )

    target = (
        missing_value.first_col(
            train,
            [
                "Value",
                "value",
                "Y",
                "y"
            ]
        )
    )

    id_col = (
        missing_value.first_col(
            train,
            [
                "ID",
                "id",
                "Id"
            ]
        )
    )

    tool_col = (
        missing_value.first_col(
            train,
            [
                "TOOL",
                "Tool",
                "tool",
                "TOOL_ID",
                "Tool_ID"
            ]
        )
    )

    if target is None:

        raise RuntimeError(
            "找不到目标列 Value/Y"
        )

    # =====================================================
    # 数值特征
    # =====================================================

    features = (
        missing_value
        .detect_feature_cols(

            train,

            target,

            id_col,

            tool_col
        )
    )

    log(
        f"原始数值特征："
        f"{len(features)}"
    )

    # =====================================================
    # 0值暂不处理
    # =====================================================

    train = (
        missing_value
        .apply_zero_mode(

            train,

            features,

            mode="keep"
        )
    )

    # =====================================================
    # 结构清洗
    # =====================================================

    (
        features,
        all_nan,
        constant,
        duplicate
    ) = (
        missing_value
        .structural_clean(

            train,

            features
        )
    )

    log(
        f"全空特征："
        f"{len(all_nan)}"
    )

    log(
        f"常量特征："
        f"{len(constant)}"
    )

    log(
        f"重复特征："
        f"{len(duplicate)}"
    )

    log(
        f"结构清洗后："
        f"{len(features)}"
    )

    # =====================================================
    # Target
    # =====================================================

    y = pd.to_numeric(
        train[target],
        errors="coerce"
    )

    valid_target = (
        y.notna()
    )

    train = (
        train
        .loc[
            valid_target
        ]
        .reset_index(
            drop=True
        )
    )

    y = (
        y
        .loc[
            valid_target
        ]
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # X
    # =====================================================

    X = (
        train[
            features
        ]
        .copy()
        .astype(
            np.float64
        )
    )

    # =====================================================
    # Tool
    # =====================================================

    if tool_col is not None:

        meta = (
            train[
                [tool_col]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    else:

        meta = pd.DataFrame(
            index=X.index
        )

    return (
        X,
        y,
        meta,
        features,
        tool_col
    )


# =========================================================
# 固定缺失值处理
# =========================================================

def apply_fixed_missing(
    X_train,
    X_valid,
    meta_train,
    meta_valid,
    tool_col
):

    return (
        missing_value
        .apply_best_missing_strategy(

            X_train,

            X_valid,

            meta_train,

            meta_valid,

            tool_col
        )
    )


# =========================================================
# 单个实验
# =========================================================

def run_experiment(

    X,
    y,
    meta,

    features,
    tool_col,

    selector,

    variance_threshold,

    transform,

    scaler,

    reducer,

    top_k,

    cv_seed=42,

    model_seed=42
):

    # =====================================================
    # CV
    # =====================================================

    kf = KFold(

        n_splits=5,

        shuffle=True,

        random_state=cv_seed
    )

    rows = []

    # =====================================================
    # Fold
    # =====================================================

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(

        kf.split(X),

        start=1
    ):

        fold_start = (
            time.perf_counter()
        )

        # -------------------------------------------------
        # Raw
        # -------------------------------------------------

        X_train_raw = (
            X.iloc[
                train_idx
            ]
            .copy()
        )

        X_valid_raw = (
            X.iloc[
                valid_idx
            ]
            .copy()
        )

        meta_train = (
            meta.iloc[
                train_idx
            ]
            .copy()
        )

        meta_valid = (
            meta.iloc[
                valid_idx
            ]
            .copy()
        )

        y_train = (
            y.iloc[
                train_idx
            ]
        )

        y_valid = (
            y.iloc[
                valid_idx
            ]
        )

        # =================================================
        # 1. 固定缺失处理
        # =================================================

        (
            Xtr,
            Xva
        ) = apply_fixed_missing(

            X_train_raw,

            X_valid_raw,

            meta_train,

            meta_valid,

            tool_col
        )

        # =================================================
        # 2. 特征工程
        # =================================================

        (
            Xtr,
            Xva,
            info
        ) = (
            feature_engineering
            .apply_feature_engineering(

                Xtr,

                Xva,

                y_train,

                selector=
                    selector,

                variance_threshold=
                    variance_threshold,

                transform=
                    transform,

                top_k=
                    top_k,

                seed=
                    model_seed
            )
        )

        # =================================================
        # 3. 降维
        # =================================================

        if reducer == "none":

            Xtr_final = Xtr

            Xva_final = Xva

            explained_variance = (
                1.0
            )

        else:

            (
                Xtr_final,
                Xva_final,
                reduce_info
            ) = (
                feature_engineering
                .apply_pca(

                    Xtr,

                    Xva,

                    reducer
                )
            )

            explained_variance = (
                reduce_info[
                    "explained_variance"
                ]
            )

        # =================================================
        # 4. 非降维情况下单独Scaler
        # =================================================

        if (
            reducer == "none"
            and
            scaler != "none"
        ):

            (
                Xtr_final,
                Xva_final,
                _scaler
            ) = (
                feature_engineering
                .scale_features(

                    Xtr,

                    Xva,

                    scaler
                )
            )

        # =================================================
        # 5. 最终安全检查
        # =================================================

        Xtr_final, Xva_final = (
            feature_engineering
            .sanitize_features(

                Xtr_final,

                Xva_final
            )
        )

        # =================================================
        # 6. XGBoost
        # =================================================

        model = (
            make_model(
                model_seed
            )
        )

        log(
            f"Fold {fold}/5 | "
            f"{selector} | "
            f"var={variance_threshold} | "
            f"transform={transform} | "
            f"scaler={scaler} | "
            f"reducer={reducer} | "
            f"features={Xtr_final.shape[1]}"
        )

        model.fit(

            Xtr_final,

            y_train,

            eval_set=[
                (
                    Xva_final,

                    y_valid
                )
            ],

            verbose=False
        )

        # =================================================
        # 预测
        # =================================================

        pred = (
            model
            .predict(
                Xva_final
            )
        )

        # =================================================
        # 指标
        # =================================================

        mse = (
            mean_squared_error(
                y_valid,
                pred
            )
        )

        rmse = (
            np.sqrt(
                mse
            )
        )

        mae = (
            mean_absolute_error(
                y_valid,
                pred
            )
        )

        r2 = (
            r2_score(
                y_valid,
                pred
            )
        )

        rows.append({

            "selector":
                selector,

            "variance_threshold":
                variance_threshold,

            "transform":
                transform,

            "scaler":
                scaler,

            "reducer":
                reducer,

            "top_k":
                top_k,

            "fold":
                fold,

            "mse":
                mse,

            "rmse":
                rmse,

            "mae":
                mae,

            "r2":
                r2,

            "n_features":
                Xtr_final.shape[1],

            "explained_variance":
                explained_variance,

            "elapsed_sec":
                time.perf_counter()
                -
                fold_start
        })

        log(
            f"Fold {fold}/5完成 | "
            f"MSE={mse:.8f} | "
            f"RMSE={rmse:.8f} | "
            f"MAE={mae:.8f} | "
            f"R2={r2:.6f}"
        )

    # =====================================================
    # 汇总
    # =====================================================

    folds = pd.DataFrame(
        rows
    )

    summary = {

        "selector":
            selector,

        "variance_threshold":
            variance_threshold,

        "transform":
            transform,

        "scaler":
            scaler,

        "reducer":
            reducer,

        "top_k":
            top_k,

        "mse_mean":
            folds[
                "mse"
            ].mean(),

        "mse_std":
            folds[
                "mse"
            ].std(
                ddof=1
            ),

        "rmse_mean":
            folds[
                "rmse"
            ].mean(),

        "rmse_std":
            folds[
                "rmse"
            ].std(
                ddof=1
            ),

        "mae_mean":
            folds[
                "mae"
            ].mean(),

        "r2_mean":
            folds[
                "r2"
            ].mean(),

        "r2_std":
            folds[
                "r2"
            ].std(
                ddof=1
            ),

        "features_mean":
            folds[
                "n_features"
            ].mean(),

        "explained_variance_mean":
            folds[
                "explained_variance"
            ].mean(),

        "elapsed_sec":
            folds[
                "elapsed_sec"
            ].sum()
    }

    return (
        summary,
        folds
    )


# =========================================================
# MAIN
# =========================================================

def main():

    total_start = (
        time.perf_counter()
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # 1. 基线
    # =====================================================

    section(
        "第一阶段：固定缺失值方案"
    )

    print(
        "缺失值方法："
        f"{BEST_MISSING_STRATEGY}"
    )

    print(
        "历史CV MSE："
        f"{BEST_MISSING_MSE:.10f}"
    )

    print(
        "\n缺失值阶段已经结束，"
        "本程序不会重新搜索。"
    )

    # =====================================================
    # 2. 数据
    # =====================================================

    section(
        "第二阶段：数据准备"
    )

    (
        X,
        y,
        meta,
        features,
        tool_col
    ) = prepare_data()

    # =====================================================
    # 3. 数据分布
    # =====================================================

    section(
        "第三阶段：数据分布"
    )

    distribution = (
        feature_engineering
        .distribution_report(
            X
        )
    )

    distribution.to_csv(

        RESULT_DIR
        /
        "01_distribution_report.csv",

        index=False,

        encoding="utf-8-sig"
    )

    if not distribution.empty:

        top_skew = (
            distribution
            .assign(
                abs_skew=
                    distribution[
                        "skewness"
                    ].abs()
            )
            .sort_values(
                "abs_skew",
                ascending=False
            )
            .head(20)
        )

        print(
            "\n绝对偏度最高20个特征："
        )

        print(
            top_skew.to_string(
                index=False
            )
        )

    # =====================================================
    # 4. 理论分布
    # =====================================================

    section(
        "第四阶段：实际分布 vs 理论分布"
    )

    distribution_dir = (
        RESULT_DIR
        /
        "distribution"
    )

    theoretical = (
        feature_engineering
        .compare_theoretical_distributions(

            X,

            distribution_dir,

            top_n=30
        )
    )

    if theoretical.empty:

        print(
            "没有得到有效理论分布结果。"
        )

    else:

        print(
            "\n理论分布结果："
        )

        print(

            theoretical[
                [
                    "feature",
                    "distribution",
                    "ks_stat",
                    "ks_pvalue",
                    "aic",
                    "best_distribution"
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

        plots = (
            feature_engineering
            .plot_distribution_comparison(

                X,

                theoretical,

                distribution_dir,

                top_n=10
            )
        )

        print(
            f"\n已生成分布图片："
            f"{len(plots)}张"
        )

        print(
            f"图片目录："
            f"{distribution_dir}"
        )

    # =====================================================
    # 5. 实验矩阵
    # =====================================================

    section(
        "第五阶段：特征工程实验设计"
    )

    experiments = []

    # -----------------------------------------------------
    # A. 方差
    # -----------------------------------------------------

    for threshold in [

        0.0,

        1e-12,

        1e-10,

        1e-8,

        1e-6
    ]:

        experiments.append({

            "selector":
                "baseline",

            "variance_threshold":
                threshold,

            "transform":
                "none",

            "scaler":
                "none",

            "reducer":
                "none",

            "top_k":
                0
        })

    # -----------------------------------------------------
    # B. 变换
    # -----------------------------------------------------

    for transform in [

        "log1p",

        "yeo_johnson"
    ]:

        experiments.append({

            "selector":
                "baseline",

            "variance_threshold":
                0.0,

            "transform":
                transform,

            "scaler":
                "none",

            "reducer":
                "none",

            "top_k":
                0
        })

    # -----------------------------------------------------
    # C. 模型贡献度
    # -----------------------------------------------------

    for selector in [

        "rf",

        "extra_trees",

        "xgb"
    ]:

        for top_k in [

            50,

            100,

            200,

            300,

            500
        ]:

            experiments.append({

                "selector":
                    selector,

                "variance_threshold":
                    0.0,

                "transform":
                    "none",

                "scaler":
                    "none",

                "reducer":
                    "none",

                "top_k":
                    top_k
            })

    # -----------------------------------------------------
    # D. PCA / SVD
    # -----------------------------------------------------

    for reducer in [

        "pca90",

        "pca95",

        "pca50",

        "svd50"
    ]:

        experiments.append({

            "selector":
                "baseline",

            "variance_threshold":
                0.0,

            "transform":
                "none",

            "scaler":
                "standard",

            "reducer":
                reducer,

            "top_k":
                0
        })

    # -----------------------------------------------------
    # E. Scaler
    # -----------------------------------------------------

    for scaler in [

        "standard",

        "minmax"
    ]:

        experiments.append({

            "selector":
                "baseline",

            "variance_threshold":
                0.0,

            "transform":
                "none",

            "scaler":
                scaler,

            "reducer":
                "none",

            "top_k":
                0
        })

    print(
        f"实验数量："
        f"{len(experiments)}"
    )

    # =====================================================
    # 6. 执行
    # =====================================================

    section(
        "第六阶段：5折CV"
    )

    summaries = []

    fold_results = []

    total_exp = len(
        experiments
    )

    for i, config in enumerate(

        experiments,

        start=1
    ):

        print()
        print(
            "-" * 100
        )

        print(
            f"实验 {i}/{total_exp}"
        )

        print(
            config
        )

        try:

            summary, folds = (
                run_experiment(

                    X,

                    y,

                    meta,

                    features,

                    tool_col,

                    selector=
                        config[
                            "selector"
                        ],

                    variance_threshold=
                        config[
                            "variance_threshold"
                        ],

                    transform=
                        config[
                            "transform"
                        ],

                    scaler=
                        config[
                            "scaler"
                        ],

                    reducer=
                        config[
                            "reducer"
                        ],

                    top_k=
                        config[
                            "top_k"
                        ]
                )
            )

            summaries.append(
                summary
            )

            fold_results.append(
                folds
            )

            current_best = (
                pd.DataFrame(
                    summaries
                )
                .sort_values(
                    "mse_mean"
                )
                .iloc[0]
            )

            print(
                "\n当前最佳："
                f"{current_best['selector']} / "
                f"{current_best['transform']} / "
                f"{current_best['reducer']} | "
                f"MSE="
                f"{current_best['mse_mean']:.8f}"
            )

        except Exception as e:

            print(
                "\n实验失败："
                f"{config}"
            )

            print(
                f"错误：{type(e).__name__}: {e}"
            )

            # 不中断整个实验
            continue

    # =====================================================
    # 7. 结果
    # =====================================================

    if not summaries:

        raise RuntimeError(
            "所有特征工程实验均失败。"
        )

    summary_df = (

        pd.DataFrame(
            summaries
        )

        .sort_values(
            "mse_mean"
        )

        .reset_index(
            drop=True
        )
    )

    fold_df = (
        pd.concat(
            fold_results,
            ignore_index=True
        )
    )

    # =====================================================
    # 8. 保存
    # =====================================================

    summary_path = (
        RESULT_DIR
        /
        "02_feature_engineering_summary.csv"
    )

    fold_path = (
        RESULT_DIR
        /
        "03_feature_engineering_folds.csv"
    )

    summary_df.to_csv(

        summary_path,

        index=False,

        encoding="utf-8-sig"
    )

    fold_df.to_csv(

        fold_path,

        index=False,

        encoding="utf-8-sig"
    )

    # =====================================================
    # 9. 排名
    # =====================================================

    section(
        "第七阶段：最终排名"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # =====================================================
    # 10. 最优方案
    # =====================================================

    best = (
        summary_df.iloc[0]
    )

    section(
        "第八阶段：当前最优特征工程"
    )

    print(
        best.to_string()
    )

    # =====================================================
    # 11. 基线比较
    # =====================================================

    print()

    print(
        f"固定缺失值基线："
        f"{BEST_MISSING_MSE:.10f}"
    )

    print(
        f"当前特征工程："
        f"{best['mse_mean']:.10f}"
    )

    difference = (
        BEST_MISSING_MSE
        -
        best["mse_mean"]
    )

    percentage = (
        difference
        /
        BEST_MISSING_MSE
        *
        100
    )

    print(
        f"MSE变化："
        f"{difference:.10f}"
    )

    print(
        f"相对变化："
        f"{percentage:.2f}%"
    )

    # =====================================================
    # 12. 配置
    # =====================================================

    config_path = (
        RESULT_DIR
        /
        "04_experiment_config.txt"
    )

    config_text = f"""
工业AI特征工程实验配置

固定缺失值：
{BEST_MISSING_STRATEGY}

历史缺失值CV MSE：
{BEST_MISSING_MSE}

CV：
5 Fold
shuffle=True
random_state=42

XGBoost：

n_estimators=1200
learning_rate=0.03
max_depth=3
min_child_weight=1
subsample=0.8
colsample_bytree=0.5
reg_alpha=0.01
reg_lambda=1.0
gamma=0
tree_method=hist

理论分布：

Normal
LogNormal
Gamma
Weibull

特征变换：

none
Log1p
Yeo-Johnson

模型特征选择：

RandomForest
ExtraTrees
XGBoost

Top-K：

50
100
200
300
500

降维：

none
PCA90
PCA95
PCA50
TruncatedSVD50

量纲：

none
StandardScaler
MinMaxScaler

数据安全：

NaN -> Train Median
Inf -> NaN -> Train Median
极端值 -> NaN -> Train Median

严格原则：

1. 缺失值不再重新比较。
2. 缺失处理固定为operation_tool_mean。
3. 所有CV统计量只来自Train Fold。
4. 特征变换只在Train Fold fit。
5. 特征选择模型只在Train Fold训练。
6. Scaler只在Train Fold fit。
7. PCA只在Train Fold fit。
8. Valid Fold只执行transform。
9. 树模型不强制标准化。
10. PCA采用StandardScaler -> PCA。
"""

    config_path.write_text(
        config_text.strip(),
        encoding="utf-8"
    )

    # =====================================================
    # 13. 完成
    # =====================================================

    section(
        "实验完成"
    )

    print(
        f"结果目录："
        f"{RESULT_DIR}"
    )

    print()

    print(
        "01_distribution_report.csv"
    )

    print(
        "distribution/"
    )

    print(
        "02_feature_engineering_summary.csv"
    )

    print(
        "03_feature_engineering_folds.csv"
    )

    print(
        "04_experiment_config.txt"
    )

    print()

    print(
        f"总耗时："
        f"{(time.perf_counter()-total_start)/60:.2f} 分钟"
    )


if __name__ == "__main__":

    main()