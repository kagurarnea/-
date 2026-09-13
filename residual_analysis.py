# -*- coding: utf-8 -*-

"""
residual_analysis.py

工业AI：最优特征工程方案的 OOF 残差分析

当前固定方案
------------------------------------------------------------
缺失值：
    operation_tool_mean

特征选择：
    ExtraTrees Top300

最终模型：
    XGBoost

分析内容
------------------------------------------------------------
1. 5折 OOF 预测
2. 整体 MSE / RMSE / MAE / R2
3. 每折误差
4. 样本残差
5. Top 异常样本
6. Tool 分组误差
7. Operation 分组特征重要性
8. 特征重要性稳定性
9. 缺失率与误差关系
10. 0值率与误差关系
11. 可视化
12. 自动生成 summary.txt

输出目录
------------------------------------------------------------
data/
└─ preprocess_missing/
   └─ residual_analysis/
      ├─ oof_predictions.csv
      ├─ sample_error_ranking.csv
      ├─ fold_results.csv
      ├─ tool_error_analysis.csv
      ├─ operation_feature_importance.csv
      ├─ feature_importance_by_fold.csv
      ├─ feature_importance.csv
      ├─ summary.txt
      └─ plots/
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import missing_value


# =========================================================
# 配置
# =========================================================

DATA_DIR = Path(
    r"C:\Users\lenovo\Downloads\my method\data"
)

OUT_DIR = (
    DATA_DIR
    / "preprocess_missing"
    / "residual_analysis"
)

PLOT_DIR = (
    OUT_DIR
    / "plots"
)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 300


# =========================================================
# 日志
# =========================================================

def log(message: str):
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True
    )


def section(title: str):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


# =========================================================
# 查找列
# =========================================================

def find_col(df, names):

    for name in names:

        if name in df.columns:
            return name

    return None


# =========================================================
# 工序识别
# =========================================================

def infer_operation(feature):

    text = str(feature).strip()

    match = re.match(
        r"^(\d+)\s*[Xx]",
        text
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


# =========================================================
# XGBoost
# =========================================================

def make_xgb():

    return XGBRegressor(

        objective="reg:squarederror",

        n_estimators=1200,

        learning_rate=0.03,

        max_depth=3,

        min_child_weight=1,

        subsample=0.8,

        colsample_bytree=0.5,

        reg_alpha=0.01,

        reg_lambda=1.0,

        gamma=0,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )


# =========================================================
# 数据准备
# =========================================================

def prepare_data():

    section("1. 数据准备")

    df = missing_value.load_train(
        DATA_DIR
    )

    target_col = find_col(
        df,
        [
            "Value",
            "value",
            "Y",
            "y"
        ]
    )

    id_col = find_col(
        df,
        [
            "ID",
            "id",
            "Id"
        ]
    )

    tool_col = find_col(
        df,
        [
            "TOOL",
            "Tool",
            "tool",
            "TOOL_ID",
            "Tool_ID"
        ]
    )

    if target_col is None:

        raise RuntimeError(
            "找不到目标列 Value/Y。"
        )

    log(
        f"Target = {target_col}"
    )

    log(
        f"ID = {id_col}"
    )

    log(
        f"Tool = {tool_col}"
    )

    # -----------------------------------------------------
    # 数值特征
    # -----------------------------------------------------

    features = (
        missing_value
        .detect_feature_cols(
            df,
            target_col,
            id_col,
            tool_col
        )
    )

    original_count = len(
        features
    )

    # -----------------------------------------------------
    # 结构清洗
    # -----------------------------------------------------

    (
        features,
        all_nan,
        constant,
        duplicate
    ) = (
        missing_value
        .structural_clean(
            df,
            features
        )
    )

    log(
        f"原始数值特征 = "
        f"{original_count}"
    )

    log(
        f"全空特征 = "
        f"{len(all_nan)}"
    )

    log(
        f"常量特征 = "
        f"{len(constant)}"
    )

    log(
        f"重复特征 = "
        f"{len(duplicate)}"
    )

    log(
        f"最终候选特征 = "
        f"{len(features)}"
    )

    # -----------------------------------------------------
    # Y
    # -----------------------------------------------------

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    valid_target = y.notna()

    df = (
        df
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

    # -----------------------------------------------------
    # X
    # -----------------------------------------------------

    X = (
        df[
            features
        ]
        .copy()
        .astype(
            np.float64
        )
    )

    # -----------------------------------------------------
    # Meta
    # -----------------------------------------------------

    meta = pd.DataFrame(
        index=np.arange(
            len(df)
        )
    )

    if id_col is not None:

        meta["ID"] = (
            df[id_col]
            .astype(str)
            .to_numpy()
        )

    else:

        meta["ID"] = np.arange(
            len(df)
        )

    if tool_col is not None:

        meta["Tool"] = (
            df[tool_col]
            .astype(str)
            .fillna(
                "MISSING_TOOL"
            )
            .to_numpy()
        )

    else:

        meta["Tool"] = (
            "UNKNOWN_TOOL"
        )

    # -----------------------------------------------------
    # 样本原始缺失率
    # -----------------------------------------------------

    meta["missing_rate"] = (
        X.isna()
        .mean(axis=1)
        .to_numpy()
    )

    # -----------------------------------------------------
    # 样本0值率
    # -----------------------------------------------------

    meta["zero_rate"] = (
        (X == 0)
        .mean(axis=1)
        .to_numpy()
    )

    return (
        X,
        y,
        meta,
        features,
        tool_col
    )


# =========================================================
# Fold缺失值处理
# =========================================================

def preprocess_fold(
    X_train,
    X_valid,
    meta_train,
    meta_valid,
    tool_col
):

    # -----------------------------------------------------
    # 关键：
    # 所有Fold内部重新编号
    # -----------------------------------------------------

    X_train = (
        X_train
        .reset_index(drop=True)
        .copy()
    )

    X_valid = (
        X_valid
        .reset_index(drop=True)
        .copy()
    )

    meta_train = (
        meta_train
        .reset_index(drop=True)
        .copy()
    )

    meta_valid = (
        meta_valid
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------------------------------
    # 构造与Tool列完全对齐的meta
    # -----------------------------------------------------

    if tool_col is not None:

        meta_train_for_impute = (
            pd.DataFrame(
                {
                    tool_col:
                        meta_train[
                            "Tool"
                        ].to_numpy()
                }
            )
        )

        meta_valid_for_impute = (
            pd.DataFrame(
                {
                    tool_col:
                        meta_valid[
                            "Tool"
                        ].to_numpy()
                }
            )
        )

    else:

        meta_train_for_impute = (
            pd.DataFrame(
                index=np.arange(
                    len(X_train)
                )
            )
        )

        meta_valid_for_impute = (
            pd.DataFrame(
                index=np.arange(
                    len(X_valid)
                )
            )
        )

    # -----------------------------------------------------
    # 最佳缺失值方法
    # -----------------------------------------------------

    (
        X_train,
        X_valid
    ) = (
        missing_value
        .apply_best_missing_strategy(

            X_train,

            X_valid,

            meta_train_for_impute,

            meta_valid_for_impute,

            tool_col
        )
    )

    # -----------------------------------------------------
    # 最终数值安全
    # -----------------------------------------------------

    X_train = (
        X_train
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )

    X_valid = (
        X_valid
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )

    medians = (
        X_train
        .median()
    )

    # 如果某列Train仍没有有效值，使用0兜底
    medians = (
        medians
        .fillna(0.0)
    )

    X_train = (
        X_train
        .fillna(
            medians
        )
        .astype(
            np.float64
        )
    )

    X_valid = (
        X_valid
        .fillna(
            medians
        )
        .astype(
            np.float64
        )
    )

    # -----------------------------------------------------
    # 强制检查
    # -----------------------------------------------------

    if not np.isfinite(
        X_train.to_numpy()
    ).all():

        raise ValueError(
            "Train Fold存在非法数值。"
        )

    if not np.isfinite(
        X_valid.to_numpy()
    ).all():

        raise ValueError(
            "Valid Fold存在非法数值。"
        )

    return (
        X_train,
        X_valid
    )


# =========================================================
# ExtraTrees特征选择
# =========================================================

def select_features(
    X_train,
    y_train
):

    selector = (
        ExtraTreesRegressor(

            n_estimators=300,

            max_features="sqrt",

            min_samples_leaf=2,

            random_state=RANDOM_STATE,

            n_jobs=-1
        )
    )

    selector.fit(
        X_train,
        y_train
    )

    importance = (
        pd.Series(

            selector.feature_importances_,

            index=X_train.columns
        )
        .sort_values(
            ascending=False
        )
    )

    selected = (
        importance
        .head(
            min(
                TOP_K,
                len(importance)
            )
        )
        .index
        .tolist()
    )

    return (
        selected,
        importance
    )


# =========================================================
# OOF训练
# =========================================================

def run_oof(
    X,
    y,
    meta,
    tool_col
):

    section("2. 5折 OOF 预测")

    kf = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    oof_pred = np.full(
        len(y),
        np.nan,
        dtype=np.float64
    )

    fold_rows = []

    importance_rows = []

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

        log(
            f"开始 Fold "
            f"{fold}/{N_SPLITS}"
        )

        # -------------------------------------------------
        # Fold数据
        # -------------------------------------------------

        X_train = (
            X.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        X_valid = (
            X.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        y_train = (
            y.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        y_valid = (
            y.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        meta_train = (
            meta.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        meta_valid = (
            meta.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        # -------------------------------------------------
        # 缺失值
        # -------------------------------------------------

        (
            X_train,
            X_valid
        ) = preprocess_fold(

            X_train,

            X_valid,

            meta_train,

            meta_valid,

            tool_col
        )

        # -------------------------------------------------
        # ExtraTrees Top300
        # -------------------------------------------------

        (
            selected_features,
            importance
        ) = select_features(

            X_train,

            y_train
        )

        log(
            f"Fold {fold}: "
            f"选择 {len(selected_features)} "
            f"个特征"
        )

        # -------------------------------------------------
        # 保存Top300重要性
        # -------------------------------------------------

        for rank, (
            feature,
            value
        ) in enumerate(

            importance
            .head(TOP_K)
            .items(),

            start=1
        ):

            importance_rows.append({

                "fold":
                    fold,

                "rank":
                    rank,

                "feature":
                    feature,

                "importance":
                    float(value)
            })

        # -------------------------------------------------
        # XGBoost
        # -------------------------------------------------

        model = make_xgb()

        model.fit(

            X_train[
                selected_features
            ],

            y_train,

            eval_set=[

                (
                    X_valid[
                        selected_features
                    ],

                    y_valid
                )

            ],

            verbose=False
        )

        pred = (
            model
            .predict(
                X_valid[
                    selected_features
                ]
            )
        )

        # -------------------------------------------------
        # OOF
        # -------------------------------------------------

        oof_pred[
            valid_idx
        ] = pred

        # -------------------------------------------------
        # Fold指标
        # -------------------------------------------------

        mse = mean_squared_error(
            y_valid,
            pred
        )

        rmse = np.sqrt(
            mse
        )

        mae = mean_absolute_error(
            y_valid,
            pred
        )

        r2 = r2_score(
            y_valid,
            pred
        )

        fold_rows.append({

            "fold":
                fold,

            "mse":
                float(mse),

            "rmse":
                float(rmse),

            "mae":
                float(mae),

            "r2":
                float(r2),

            "n_features":
                len(
                    selected_features
                ),

            "time_sec":
                (
                    time.perf_counter()
                    -
                    fold_start
                )
        })

        log(
            f"Fold {fold}完成 | "
            f"MSE={mse:.8f} | "
            f"RMSE={rmse:.8f} | "
            f"MAE={mae:.8f} | "
            f"R2={r2:.6f}"
        )

    if np.isnan(
        oof_pred
    ).any():

        raise RuntimeError(
            "OOF预测未覆盖全部样本。"
        )

    return (
        oof_pred,

        pd.DataFrame(
            fold_rows
        ),

        pd.DataFrame(
            importance_rows
        )
    )


# =========================================================
# Tool分析
# =========================================================

def analyze_tool(
    result
):

    return (

        result

        .groupby(
            "Tool",
            dropna=False
        )

        .agg(

            n=(
                "y_true",
                "size"
            ),

            mse=(
                "squared_error",
                "mean"
            ),

            rmse=(
                "squared_error",
                lambda x:
                np.sqrt(
                    np.mean(x)
                )
            ),

            mae=(
                "absolute_residual",
                "mean"
            ),

            max_abs_error=(
                "absolute_residual",
                "max"
            ),

            mean_missing_rate=(
                "missing_rate",
                "mean"
            ),

            mean_zero_rate=(
                "zero_rate",
                "mean"
            ),

            mean_y=(
                "y_true",
                "mean"
            )
        )

        .reset_index()

        .sort_values(
            "mse",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


# =========================================================
# Operation分析
# =========================================================

def analyze_operation(
    importance_df
):

    if importance_df.empty:

        return pd.DataFrame()

    data = (
        importance_df.copy()
    )

    data["operation"] = (
        data["feature"]
        .map(
            infer_operation
        )
    )

    return (

        data

        .groupby(
            "operation"
        )

        .agg(

            feature_count=(
                "feature",
                "nunique"
            ),

            mean_importance=(
                "importance",
                "mean"
            ),

            total_importance=(
                "importance",
                "sum"
            )
        )

        .reset_index()

        .sort_values(
            "total_importance",
            ascending=False
        )
    )


# =========================================================
# 特征稳定性
# =========================================================

def analyze_feature_stability(
    importance_df
):

    if importance_df.empty:

        return pd.DataFrame()

    result = (

        importance_df

        .groupby(
            "feature"
        )

        .agg(

            mean_importance=(
                "importance",
                "mean"
            ),

            std_importance=(
                "importance",
                "std"
            ),

            max_importance=(
                "importance",
                "max"
            ),

            min_importance=(
                "importance",
                "min"
            ),

            selected_folds=(
                "fold",
                "nunique"
            )
        )

        .reset_index()
    )

    # 变异程度
    result["cv_importance"] = (

        result[
            "std_importance"
        ]

        /
        (
            result[
                "mean_importance"
            ]
            +
            1e-12
        )
    )

    return (

        result

        .sort_values(
            "mean_importance",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


# =========================================================
# 画图通用函数
# =========================================================

def save_current_plot(
    filename
):

    path = (
        PLOT_DIR
        /
        filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    return path


# =========================================================
# 可视化
# =========================================================

def make_plots(
    result,
    tool_df,
    fold_df,
    feature_df
):

    section(
        "4. 生成可视化"
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # 1. True vs Pred
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        result["y_true"],
        result["y_pred"],
        alpha=0.7
    )

    low = min(
        result["y_true"].min(),
        result["y_pred"].min()
    )

    high = max(
        result["y_true"].max(),
        result["y_pred"].max()
    )

    plt.plot(
        [
            low,
            high
        ],
        [
            low,
            high
        ],
        linewidth=2
    )

    plt.xlabel(
        "True"
    )

    plt.ylabel(
        "Predicted"
    )

    plt.title(
        "True vs Predicted"
    )

    save_current_plot(
        "01_true_vs_pred.png"
    )

    # =====================================================
    # 2. Residual
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        result["residual"],
        bins=40,
        alpha=0.75
    )

    plt.axvline(
        0,
        linewidth=2
    )

    plt.xlabel(
        "Residual"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Residual Distribution"
    )

    save_current_plot(
        "02_residual_distribution.png"
    )

    # =====================================================
    # 3. Residual vs Prediction
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        result["y_pred"],
        result["residual"],
        alpha=0.7
    )

    plt.axhline(
        0,
        linewidth=2
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual vs Predicted"
    )

    save_current_plot(
        "03_residual_vs_prediction.png"
    )

    # =====================================================
    # 4. Top Error Samples
    # =====================================================

    top_error = (
        result
        .nlargest(
            20,
            "absolute_residual"
        )
        .sort_values(
            "absolute_residual"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        np.arange(
            len(top_error)
        ),
        top_error[
            "absolute_residual"
        ]
    )

    plt.yticks(
        np.arange(
            len(top_error)
        ),
        top_error[
            "ID"
        ].astype(str)
    )

    plt.xlabel(
        "Absolute Residual"
    )

    plt.title(
        "Top 20 Error Samples"
    )

    save_current_plot(
        "04_top_error_samples.png"
    )

    # =====================================================
    # 5. Tool MSE
    # =====================================================

    if not tool_df.empty:

        tool_plot = (
            tool_df
            .head(20)
            .sort_values(
                "mse"
            )
        )

        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            np.arange(
                len(tool_plot)
            ),
            tool_plot[
                "mse"
            ]
        )

        plt.yticks(
            np.arange(
                len(tool_plot)
            ),
            tool_plot[
                "Tool"
            ].astype(str)
        )

        plt.xlabel(
            "MSE"
        )

        plt.title(
            "Tool MSE - Worst 20"
        )

        save_current_plot(
            "05_tool_mse.png"
        )

    # =====================================================
    # 6. Missing Rate
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        result[
            "missing_rate"
        ],
        result[
            "absolute_residual"
        ],
        alpha=0.7
    )

    plt.xlabel(
        "Missing Rate"
    )

    plt.ylabel(
        "Absolute Residual"
    )

    plt.title(
        "Missing Rate vs Error"
    )

    save_current_plot(
        "06_missing_rate_vs_error.png"
    )

    # =====================================================
    # 7. Zero Rate
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        result[
            "zero_rate"
        ],
        result[
            "absolute_residual"
        ],
        alpha=0.7
    )

    plt.xlabel(
        "Zero Rate"
    )

    plt.ylabel(
        "Absolute Residual"
    )

    plt.title(
        "Zero Rate vs Error"
    )

    save_current_plot(
        "07_zero_rate_vs_error.png"
    )

    # =====================================================
    # 8. Fold MSE
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.bar(
        fold_df[
            "fold"
        ].astype(str),
        fold_df[
            "mse"
        ]
    )

    plt.xlabel(
        "Fold"
    )

    plt.ylabel(
        "MSE"
    )

    plt.title(
        "OOF Fold MSE"
    )

    save_current_plot(
        "08_fold_mse.png"
    )

    # =====================================================
    # 9. Feature Importance
    # =====================================================

    if not feature_df.empty:

        top_features = (
            feature_df
            .head(30)
            .sort_values(
                "mean_importance"
            )
        )

        plt.figure(
            figsize=(10, 9)
        )

        plt.barh(
            np.arange(
                len(top_features)
            ),
            top_features[
                "mean_importance"
            ]
        )

        plt.yticks(
            np.arange(
                len(top_features)
            ),
            top_features[
                "feature"
            ]
        )

        plt.xlabel(
            "Mean Importance"
        )

        plt.title(
            "Top 30 Feature Importance"
        )

        save_current_plot(
            "09_feature_importance.png"
        )


# =========================================================
# 主程序
# =========================================================

def main():

    total_start = (
        time.perf_counter()
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # 数据
    # =====================================================

    (
        X,
        y,
        meta,
        features,
        tool_col
    ) = prepare_data()

    # =====================================================
    # OOF
    # =====================================================

    (
        oof_pred,
        fold_df,
        importance_by_fold
    ) = run_oof(

        X,
        y,
        meta,
        tool_col
    )

    # =====================================================
    # 残差
    # =====================================================

    section(
        "3. 残差分析"
    )

    result = meta.copy()

    result["y_true"] = (
        y.to_numpy()
    )

    result["y_pred"] = (
        oof_pred
    )

    result["residual"] = (
        result["y_true"]
        -
        result["y_pred"]
    )

    result["absolute_residual"] = (
        result["residual"]
        .abs()
    )

    result["squared_error"] = (
        result["residual"]
        ** 2
    )

    result["relative_error"] = (

        result[
            "absolute_residual"
        ]

        /

        (
            result[
                "y_true"
            ]
            .abs()
            +
            1e-12
        )
    )

    # =====================================================
    # OOF指标
    # =====================================================

    mse = mean_squared_error(
        result["y_true"],
        result["y_pred"]
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        result["y_true"],
        result["y_pred"]
    )

    r2 = r2_score(
        result["y_true"],
        result["y_pred"]
    )

    print(
        f"OOF MSE  = {mse:.10f}"
    )

    print(
        f"OOF RMSE = {rmse:.10f}"
    )

    print(
        f"OOF MAE  = {mae:.10f}"
    )

    print(
        f"OOF R2   = {r2:.6f}"
    )

    # =====================================================
    # Tool
    # =====================================================

    tool_df = analyze_tool(
        result
    )

    # =====================================================
    # Operation
    # =====================================================

    operation_df = (
        analyze_operation(
            importance_by_fold
        )
    )

    # =====================================================
    # Feature Stability
    # =====================================================

    feature_df = (
        analyze_feature_stability(
            importance_by_fold
        )
    )

    # =====================================================
    # 样本异常排名
    # =====================================================

    error_df = (
        result
        .sort_values(
            "absolute_residual",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    error_df.insert(
        0,
        "error_rank",
        np.arange(
            1,
            len(error_df) + 1
        )
    )

    # =====================================================
    # CSV
    # =====================================================

    result.to_csv(

        OUT_DIR
        /
        "oof_predictions.csv",

        index=False,

        encoding="utf-8-sig"
    )

    error_df.to_csv(

        OUT_DIR
        /
        "sample_error_ranking.csv",

        index=False,

        encoding="utf-8-sig"
    )

    fold_df.to_csv(

        OUT_DIR
        /
        "fold_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    tool_df.to_csv(

        OUT_DIR
        /
        "tool_error_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    operation_df.to_csv(

        OUT_DIR
        /
        "operation_feature_importance.csv",

        index=False,

        encoding="utf-8-sig"
    )

    importance_by_fold.to_csv(

        OUT_DIR
        /
        "feature_importance_by_fold.csv",

        index=False,

        encoding="utf-8-sig"
    )

    feature_df.to_csv(

        OUT_DIR
        /
        "feature_importance.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # =====================================================
    # 图片
    # =====================================================

    make_plots(

        result,

        tool_df,

        fold_df,

        feature_df
    )

    # =====================================================
    # Top异常样本
    # =====================================================

    section(
        "5. Top 20 异常样本"
    )

    print(

        error_df[
            [
                "error_rank",
                "ID",
                "Tool",
                "y_true",
                "y_pred",
                "residual",
                "absolute_residual",
                "squared_error",
                "missing_rate",
                "zero_rate"
            ]
        ]

        .head(20)

        .to_string(
            index=False
        )
    )

    # =====================================================
    # Tool
    # =====================================================

    section(
        "6. Tool误差排名"
    )

    print(
        tool_df
        .head(20)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # Fold
    # =====================================================

    section(
        "7. Fold结果"
    )

    print(
        fold_df
        .to_string(
            index=False
        )
    )

    # =====================================================
    # 最差Fold
    # =====================================================

    worst_fold = (
        fold_df
        .sort_values(
            "mse",
            ascending=False
        )
        .iloc[0]
    )

    # =====================================================
    # 最差Tool
    # =====================================================

    worst_tool = None

    if not tool_df.empty:

        worst_tool = (
            tool_df.iloc[0]
        )

    # =====================================================
    # Summary
    # =====================================================

    lines = [

        "工业AI OOF残差分析报告",

        "",

        "==================================================",

        "固定方案",

        "==================================================",

        "缺失值 = operation_tool_mean",

        "特征选择 = ExtraTrees Top300",

        "模型 = XGBoost",

        "",

        "==================================================",

        "OOF指标",

        "==================================================",

        f"MSE  = {mse:.10f}",

        f"RMSE = {rmse:.10f}",

        f"MAE  = {mae:.10f}",

        f"R2   = {r2:.6f}",

        "",

        "==================================================",

        "Fold结果",

        "=================================================="
    ]

    for _, row in fold_df.iterrows():

        lines.append(

            f"Fold {int(row['fold'])}: "

            f"MSE={row['mse']:.10f} | "

            f"RMSE={row['rmse']:.10f} | "

            f"MAE={row['mae']:.10f} | "

            f"R2={row['r2']:.6f}"
        )

    lines.extend([

        "",

        "最差Fold："
        f"{int(worst_fold['fold'])}",

        f"最差Fold MSE："
        f"{worst_fold['mse']:.10f}",

        "",

        "==================================================",

        "Top 10异常样本",

        "=================================================="
    ])

    for _, row in error_df.head(10).iterrows():

        lines.append(

            f"Rank {int(row['error_rank'])} | "

            f"ID={row['ID']} | "

            f"Tool={row['Tool']} | "

            f"Y={row['y_true']:.6f} | "

            f"Pred={row['y_pred']:.6f} | "

            f"Residual={row['residual']:.6f} | "

            f"AbsResidual="
            f"{row['absolute_residual']:.6f}"
        )

    if worst_tool is not None:

        lines.extend([

            "",

            "==================================================",

            "最差Tool",

            "==================================================",

            f"Tool = {worst_tool['Tool']}",

            f"样本数 = {int(worst_tool['n'])}",

            f"MSE = {worst_tool['mse']:.10f}",

            f"RMSE = {worst_tool['rmse']:.10f}",

            f"MAE = {worst_tool['mae']:.10f}",

            f"平均缺失率 = "
            f"{worst_tool['mean_missing_rate']:.6f}",

            f"平均0值率 = "
            f"{worst_tool['mean_zero_rate']:.6f}"
        ])

    # -----------------------------------------------------
    # Top稳定特征
    # -----------------------------------------------------

    if not feature_df.empty:

        lines.extend([

            "",

            "==================================================",

            "Top 10稳定重要特征",

            "=================================================="
        ])

        for _, row in (
            feature_df
            .head(10)
            .iterrows()
        ):

            lines.append(

                f"{row['feature']} | "

                f"MeanImportance="
                f"{row['mean_importance']:.8f} | "

                f"Std="
                f"{row['std_importance']:.8f} | "

                f"SelectedFolds="
                f"{int(row['selected_folds'])}"
            )

    lines.extend([

        "",

        "==================================================",

        "输出",

        "==================================================",

        f"{OUT_DIR}",

        "",

        f"总耗时："
        f"{(time.perf_counter()-total_start)/60:.2f} 分钟"
    ])

    (OUT_DIR / "summary.txt").write_text(

        "\n".join(
            lines
        ),

        encoding="utf-8"
    )

    # =====================================================
    # 完成
    # =====================================================

    section(
        "8. 分析完成"
    )

    print(
        f"结果目录：{OUT_DIR}"
    )

    print(
        f"OOF MSE：{mse:.10f}"
    )

    print(
        "已生成CSV、PNG和summary.txt"
    )


if __name__ == "__main__":

    main()