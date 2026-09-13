# -*- coding: utf-8 -*-

"""
deep_diagnosis.py

工业AI第二阶段深度诊断

当前固定最佳方案
------------------------------------------------------------
缺失值：
    operation_tool_mean

特征选择：
    ExtraTrees Top300

最终模型：
    XGBoost

本程序不再比较模型，而是回答三个问题：

    1. 为什么某个Fold误差明显更高？
    2. 为什么某个Tool误差明显更高？
    3. 极端异常样本到底异常在哪里？

------------------------------------------------------------
主要输出
------------------------------------------------------------

deep_diagnosis/
├── oof_detail.csv
├── top_error_samples.csv
├── fold_analysis.csv
├── tool_analysis.csv
├── tool_fold_analysis.csv
├── feature_importance_stability.csv
├── operation_importance.csv
├── sample_feature_anomaly.csv
├── worst_tool_features.csv
├── worst_sample_features.csv
├── summary.txt
│
└── plots/
    ├── 01_fold_mse.png
    ├── 02_tool_mse.png
    ├── 03_tool_sample_count.png
    ├── 04_missing_rate_vs_error.png
    ├── 05_zero_rate_vs_error.png
    ├── 06_y_vs_error.png
    ├── 07_feature_stability.png
    ├── 08_operation_importance.png
    └── 09_worst_sample_feature_zscore.png
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
    / "deep_diagnosis"
)

PLOT_DIR = (
    OUT_DIR
    / "plots"
)

RANDOM_STATE = 42

N_SPLITS = 5

TOP_K = 300

TOP_ERROR_SAMPLES = 30

TOP_FEATURES_PER_SAMPLE = 30


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
    print("=" * 90)
    print(title)
    print("=" * 90)


# =========================================================
# 查找列
# =========================================================

def find_col(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:

            return col

    return None


# =========================================================
# Operation
# =========================================================

def infer_operation(
    feature
):

    match = re.match(
        r"^\s*(\d+)\s*[Xx]",
        str(feature)
    )

    if match:

        return match.group(1)

    return "UNKNOWN"


# =========================================================
# XGBoost
# =========================================================

def make_xgb():

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
            RANDOM_STATE,

        n_jobs=
            -1,

        tree_method=
            "hist"
    )


# =========================================================
# 数据准备
# =========================================================

def prepare_data():

    section(
        "1. 数据准备"
    )

    df = (
        missing_value
        .load_train(
            DATA_DIR
        )
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
        f"全空 = "
        f"{len(all_nan)}"
    )

    log(
        f"常量 = "
        f"{len(constant)}"
    )

    log(
        f"重复 = "
        f"{len(duplicate)}"
    )

    log(
        f"候选特征 = "
        f"{len(features)}"
    )

    # -----------------------------------------------------
    # Y
    # -----------------------------------------------------

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    mask = y.notna()

    df = (
        df
        .loc[mask]
        .reset_index(
            drop=True
        )
    )

    y = (
        y
        .loc[mask]
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
            len(X)
        )
    )

    if id_col is not None:

        meta["ID"] = (
            df[
                id_col
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        meta["ID"] = (
            np.arange(
                len(X
                )
            )
        )

    if tool_col is not None:

        meta["Tool"] = (
            df[
                tool_col
            ]
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
    # 原始缺失率
    # -----------------------------------------------------

    meta["missing_rate"] = (
        X.isna()
        .mean(axis=1)
        .to_numpy()
    )

    # -----------------------------------------------------
    # 原始0值率
    # -----------------------------------------------------

    meta["zero_rate"] = (
        (X == 0)
        .mean(axis=1)
        .to_numpy()
    )

    # -----------------------------------------------------
    # 行索引
    # -----------------------------------------------------

    meta["row_index"] = (
        np.arange(
            len(X)
        )
    )

    return (
        X,
        y,
        meta,
        features,
        tool_col
    )


# =========================================================
# Fold预处理
# =========================================================

def preprocess_fold(
    X_train,
    X_valid,
    meta_train,
    meta_valid,
    tool_col
):

    # =====================================================
    # 强制重新编号
    # =====================================================

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

    # =====================================================
    # Tool
    # =====================================================

    if tool_col is not None:

        train_meta = pd.DataFrame({

            tool_col:
                meta_train[
                    "Tool"
                ].to_numpy()

        })

        valid_meta = pd.DataFrame({

            tool_col:
                meta_valid[
                    "Tool"
                ].to_numpy()

        })

    else:

        train_meta = pd.DataFrame(
            index=np.arange(
                len(X_train)
            )
        )

        valid_meta = pd.DataFrame(
            index=np.arange(
                len(X_valid)
            )
        )

    # =====================================================
    # 最佳缺失值方案
    # =====================================================

    X_train, X_valid = (
        missing_value
        .apply_best_missing_strategy(

            X_train,

            X_valid,

            train_meta,

            valid_meta,

            tool_col
        )
    )

    # =====================================================
    # 最终安全
    # =====================================================

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

    if not np.isfinite(
        X_train.to_numpy()
    ).all():

        raise ValueError(
            "Train存在非法数值。"
        )

    if not np.isfinite(
        X_valid.to_numpy()
    ).all():

        raise ValueError(
            "Valid存在非法数值。"
        )

    return (
        X_train,
        X_valid
    )


# =========================================================
# ExtraTrees
# =========================================================

def select_features(
    X_train,
    y_train
):

    selector = (
        ExtraTreesRegressor(

            n_estimators=
                300,

            max_features=
                "sqrt",

            min_samples_leaf=
                2,

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1
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
# OOF
# =========================================================

def run_oof(
    X,
    y,
    meta,
    tool_col
):

    section(
        "2. OOF建模"
    )

    kf = KFold(

        n_splits=
            N_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE
    )

    oof_pred = np.full(
        len(y),
        np.nan,
        dtype=np.float64
    )

    sample_fold = np.full(
        len(y),
        -1,
        dtype=int
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

        start = time.perf_counter()

        log(
            f"Fold {fold}/{N_SPLITS}"
        )

        X_train = (
            X.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        X_valid = (
            X.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
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
        )

        meta_valid = (
            meta.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

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
        # 特征选择
        # -------------------------------------------------

        (
            selected,
            importance
        ) = select_features(

            X_train,

            y_train
        )

        # -------------------------------------------------
        # 记录fold
        # -------------------------------------------------

        sample_fold[
            valid_idx
        ] = fold

        # -------------------------------------------------
        # 保存重要性
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
                    float(value),

                "operation":
                    infer_operation(
                        feature
                    )
            })

        # -------------------------------------------------
        # XGBoost
        # -------------------------------------------------

        model = make_xgb()

        model.fit(

            X_train[
                selected
            ],

            y_train,

            eval_set=[

                (
                    X_valid[
                        selected
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
                    selected
                ]
            )
        )

        oof_pred[
            valid_idx
        ] = pred

        mse = (
            mean_squared_error(
                y_valid,
                pred
            )
        )

        rmse = np.sqrt(
            mse
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

        fold_rows.append({

            "fold":
                fold,

            "n_samples":
                len(valid_idx),

            "mse":
                float(mse),

            "rmse":
                float(rmse),

            "mae":
                float(mae),

            "r2":
                float(r2),

            "n_features":
                len(selected),

            "time_sec":
                time.perf_counter()
                -
                start
        })

        log(
            f"Fold {fold}完成 | "
            f"MSE={mse:.8f} | "
            f"RMSE={rmse:.8f}"
        )

    if np.isnan(
        oof_pred
    ).any():

        raise RuntimeError(
            "OOF没有覆盖全部样本。"
        )

    return (
        oof_pred,
        sample_fold,
        pd.DataFrame(
            fold_rows
        ),
        pd.DataFrame(
            importance_rows
        )
    )


# =========================================================
# 构造样本结果
# =========================================================

def build_sample_result(
    meta,
    y,
    pred,
    fold
):

    result = meta.copy()

    result["fold"] = fold

    result["y_true"] = (
        y.to_numpy()
    )

    result["y_pred"] = pred

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

    return result


# =========================================================
# Fold分析
# =========================================================

def analyze_fold(
    result
):

    return (

        result

        .groupby(
            "fold"
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
            ),

            std_y=(
                "y_true",
                "std"
            )
        )

        .reset_index()

        .sort_values(
            "mse",
            ascending=False
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

            max_error=(
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
            ),

            std_y=(
                "y_true",
                "std"
            )
        )

        .reset_index()

        .sort_values(
            "mse",
            ascending=False
        )
    )


# =========================================================
# Tool × Fold
# =========================================================

def analyze_tool_fold(
    result
):

    return (

        result

        .groupby(
            [
                "Tool",
                "fold"
            ],
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

            mae=(
                "absolute_residual",
                "mean"
            ),

            mean_abs_residual=(
                "absolute_residual",
                "mean"
            ),

            mean_missing_rate=(
                "missing_rate",
                "mean"
            ),

            mean_zero_rate=(
                "zero_rate",
                "mean"
            )
        )

        .reset_index()
    )


# =========================================================
# 特征稳定性
# =========================================================

def feature_stability(
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

            min_importance=(
                "importance",
                "min"
            ),

            max_importance=(
                "importance",
                "max"
            ),

            selected_folds=(
                "fold",
                "nunique"
            )
        )

        .reset_index()
    )

    result["stability"] = (

        result[
            "selected_folds"
        ]

        /
        N_SPLITS
    )

    result["importance_cv"] = (

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
            [
                "selected_folds",
                "mean_importance"
            ],
            ascending=[
                False,
                False
            ]
        )

        .reset_index(
            drop=True
        )
    )


# =========================================================
# Operation重要性
# =========================================================

def operation_importance(
    importance_df
):

    if importance_df.empty:

        return pd.DataFrame()

    return (

        importance_df

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
            ),

            selected_features=(
                "feature",
                "nunique"
            )
        )

        .reset_index()

        .sort_values(
            "total_importance",
            ascending=False
        )
    )


# =========================================================
# 找样本异常特征
# =========================================================

def find_sample_anomalies(
    X,
    sample_result,
    features,
    top_samples=TOP_ERROR_SAMPLES
):

    top = (
        sample_result
        .nlargest(
            top_samples,
            "absolute_residual"
        )
    )

    rows = []

    # -----------------------------------------------------
    # 用全体原始数据计算基准
    # -----------------------------------------------------

    means = (
        X[features]
        .mean()
    )

    stds = (
        X[features]
        .std()
        .replace(
            0,
            np.nan
        )
    )

    # -----------------------------------------------------
    # 每个异常样本
    # -----------------------------------------------------

    for _, sample in top.iterrows():

        row_index = int(
            sample[
                "row_index"
            ]
        )

        values = (
            X.iloc[
                row_index
            ][
                features
            ]
        )

        z = (
            (
                values
                -
                means
            )
            /
            stds
        )

        z_abs = (
            z.abs()
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
            .sort_values(
                ascending=False
            )
            .head(
                TOP_FEATURES_PER_SAMPLE
            )
        )

        for rank, (
            feature,
            z_value
        ) in enumerate(

            z_abs.items(),

            start=1
        ):

            rows.append({

                "ID":
                    sample["ID"],

                "Tool":
                    sample["Tool"],

                "row_index":
                    row_index,

                "fold":
                    sample["fold"],

                "y_true":
                    sample["y_true"],

                "y_pred":
                    sample["y_pred"],

                "residual":
                    sample["residual"],

                "absolute_residual":
                    sample[
                        "absolute_residual"
                    ],

                "rank":
                    rank,

                "feature":
                    feature,

                "value":
                    values[
                        feature
                    ],

                "z_score":
                    z.loc[
                        feature
                    ],

                "abs_z_score":
                    z_value,

                "operation":
                    infer_operation(
                        feature
                    )
            })

    return pd.DataFrame(
        rows
    )


# =========================================================
# 找最差Tool的特征
# =========================================================

def analyze_worst_tool_features(
    X,
    sample_result,
    features,
    top_n_tools=1,
    top_features=30
):

    tool_df = analyze_tool(
        sample_result
    )

    if tool_df.empty:

        return pd.DataFrame()

    worst_tools = (
        tool_df
        .head(
            top_n_tools
        )[
            "Tool"
        ]
        .tolist()
    )

    # -----------------------------------------------------
    # 全体基准
    # -----------------------------------------------------

    global_mean = (
        X[features]
        .mean()
    )

    global_std = (
        X[features]
        .std()
        .replace(
            0,
            np.nan
        )
    )

    rows = []

    for tool in worst_tools:

        tool_indices = (
            sample_result
            .index[
                sample_result[
                    "Tool"
                ]
                ==
                tool
            ]
        )

        # 注意：
        # sample_result索引就是原始X行号
        values = (
            X.loc[
                tool_indices,
                features
            ]
        )

        tool_mean = (
            values
            .mean()
        )

        shift = (
            (
                tool_mean
                -
                global_mean
            )
            /
            global_std
        )

        top = (
            shift.abs()
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
            .sort_values(
                ascending=False
            )
            .head(
                top_features
            )
        )

        for rank, (
            feature,
            z_value
        ) in enumerate(

            top.items(),

            start=1
        ):

            rows.append({

                "Tool":
                    tool,

                "rank":
                    rank,

                "feature":
                    feature,

                "tool_mean":
                    tool_mean[
                        feature
                    ],

                "global_mean":
                    global_mean[
                        feature
                    ],

                "shift_z":
                    z_value,

                "operation":
                    infer_operation(
                        feature
                    )
            })

    return pd.DataFrame(
        rows
    )


# =========================================================
# 图片
# =========================================================

def save_plot(
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


def make_plots(
    result,
    fold_df,
    tool_df,
    feature_df,
    operation_df,
    worst_sample_features
):

    section(
        "4. 生成诊断图片"
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # Fold MSE
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plot_df = (
        fold_df
        .sort_values(
            "fold"
        )
    )

    plt.bar(
        plot_df[
            "fold"
        ].astype(str),
        plot_df[
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
        "Fold MSE"
    )

    save_plot(
        "01_fold_mse.png"
    )

    # =====================================================
    # Tool MSE
    # =====================================================

    if not tool_df.empty:

        temp = (
            tool_df
            .sort_values(
                "mse"
            )
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.barh(
            np.arange(
                len(temp)
            ),
            temp["mse"]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp["Tool"].astype(str)
        )

        plt.xlabel(
            "MSE"
        )

        plt.title(
            "Tool MSE"
        )

        save_plot(
            "02_tool_mse.png"
        )

    # =====================================================
    # Tool样本数
    # =====================================================

    if not tool_df.empty:

        temp = (
            tool_df
            .sort_values(
                "n"
            )
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.barh(
            np.arange(
                len(temp)
            ),
            temp["n"]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp["Tool"].astype(str)
        )

        plt.xlabel(
            "Sample Count"
        )

        plt.title(
            "Tool Sample Count"
        )

        save_plot(
            "03_tool_sample_count.png"
        )

    # =====================================================
    # 缺失率 vs误差
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

    save_plot(
        "04_missing_rate_vs_error.png"
    )

    # =====================================================
    # 0值率 vs误差
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

    save_plot(
        "05_zero_rate_vs_error.png"
    )

    # =====================================================
    # Y vs误差
    # =====================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        result[
            "y_true"
        ],
        result[
            "absolute_residual"
        ],
        alpha=0.7
    )

    plt.xlabel(
        "Y"
    )

    plt.ylabel(
        "Absolute Residual"
    )

    plt.title(
        "Y vs Absolute Residual"
    )

    save_plot(
        "06_y_vs_error.png"
    )

    # =====================================================
    # 特征稳定性
    # =====================================================

    if not feature_df.empty:

        temp = (
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
                len(temp)
            ),
            temp[
                "mean_importance"
            ]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp[
                "feature"
            ]
        )

        plt.xlabel(
            "Mean Importance"
        )

        plt.title(
            "Top Stable Features"
        )

        save_plot(
            "07_feature_stability.png"
        )

    # =====================================================
    # Operation
    # =====================================================

    if not operation_df.empty:

        temp = (
            operation_df
            .head(20)
            .sort_values(
                "total_importance"
            )
        )

        plt.figure(
            figsize=(9, 8)
        )

        plt.barh(
            np.arange(
                len(temp)
            ),
            temp[
                "total_importance"
            ]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp[
                "operation"
            ].astype(str)
        )

        plt.xlabel(
            "Total Feature Importance"
        )

        plt.title(
            "Operation Importance"
        )

        save_plot(
            "08_operation_importance.png"
        )

    # =====================================================
    # 最差样本特征Z-score
    # =====================================================

    if not worst_sample_features.empty:

        first_id = (
            worst_sample_features
            .iloc[0]["ID"]
        )

        temp = (
            worst_sample_features[
                worst_sample_features[
                    "ID"
                ]
                ==
                first_id
            ]
            .sort_values(
                "abs_z_score"
            )
        )

        plt.figure(
            figsize=(10, 8)
        )

        plt.barh(
            np.arange(
                len(temp)
            ),
            temp[
                "z_score"
            ]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp[
                "feature"
            ]
        )

        plt.xlabel(
            "Z-score"
        )

        plt.title(
            f"Extreme Feature Deviations: {first_id}"
        )

        save_plot(
            "09_worst_sample_feature_zscore.png"
        )


# =========================================================
# Main
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
        pred,
        fold_id,
        fold_df,
        importance_df
    ) = run_oof(

        X,
        y,
        meta,
        tool_col
    )

    # =====================================================
    # 样本结果
    # =====================================================

    section(
        "3. 构建样本级结果"
    )

    result = build_sample_result(

        meta,

        y,

        pred,

        fold_id
    )

    # =====================================================
    # 整体指标
    # =====================================================

    overall_mse = mean_squared_error(
        result["y_true"],
        result["y_pred"]
    )

    overall_rmse = np.sqrt(
        overall_mse
    )

    overall_mae = mean_absolute_error(
        result["y_true"],
        result["y_pred"]
    )

    overall_r2 = r2_score(
        result["y_true"],
        result["y_pred"]
    )

    print(
        f"OOF MSE  = "
        f"{overall_mse:.10f}"
    )

    print(
        f"OOF RMSE = "
        f"{overall_rmse:.10f}"
    )

    print(
        f"OOF MAE  = "
        f"{overall_mae:.10f}"
    )

    print(
        f"OOF R2   = "
        f"{overall_r2:.6f}"
    )

    # =====================================================
    # 分析
    # =====================================================

    fold_analysis = analyze_fold(
        result
    )

    tool_analysis_df = analyze_tool(
        result
    )

    tool_fold_df = analyze_tool_fold(
        result
    )

    feature_stability_df = (
        feature_stability(
            importance_df
        )
    )

    operation_df = (
        operation_importance(
            importance_df
        )
    )

    # =====================================================
    # 异常样本特征
    # =====================================================

    sample_feature_df = (
        find_sample_anomalies(

            X,

            result,

            features,

            TOP_ERROR_SAMPLES
        )
    )

    # =====================================================
    # 最差Tool特征
    # =====================================================

    worst_tool_feature_df = (
        analyze_worst_tool_features(

            X,

            result,

            features,

            top_n_tools=1,

            top_features=30
        )
    )

    # =====================================================
    # 保存
    # =====================================================

    result.to_csv(

        OUT_DIR
        /
        "oof_detail.csv",

        index=False,

        encoding="utf-8-sig"
    )

    (
        result
        .nlargest(
            TOP_ERROR_SAMPLES,
            "absolute_residual"
        )
        .to_csv(

            OUT_DIR
            /
            "top_error_samples.csv",

            index=False,

            encoding="utf-8-sig"
        )
    )

    fold_analysis.to_csv(

        OUT_DIR
        /
        "fold_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    tool_analysis_df.to_csv(

        OUT_DIR
        /
        "tool_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    tool_fold_df.to_csv(

        OUT_DIR
        /
        "tool_fold_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    importance_df.to_csv(

        OUT_DIR
        /
        "feature_importance_by_fold.csv",

        index=False,

        encoding="utf-8-sig"
    )

    feature_stability_df.to_csv(

        OUT_DIR
        /
        "feature_importance_stability.csv",

        index=False,

        encoding="utf-8-sig"
    )

    operation_df.to_csv(

        OUT_DIR
        /
        "operation_importance.csv",

        index=False,

        encoding="utf-8-sig"
    )

    sample_feature_df.to_csv(

        OUT_DIR
        /
        "sample_feature_anomaly.csv",

        index=False,

        encoding="utf-8-sig"
    )

    worst_tool_feature_df.to_csv(

        OUT_DIR
        /
        "worst_tool_features.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # =====================================================
    # 可视化
    # =====================================================

    make_plots(

        result,

        fold_analysis,

        tool_analysis_df,

        feature_stability_df,

        operation_df,

        sample_feature_df
    )

    # =====================================================
    # 打印异常样本
    # =====================================================

    section(
        "5. Top异常样本"
    )

    print(

        result
        .nlargest(
            TOP_ERROR_SAMPLES,
            "absolute_residual"
        )
        [
            [
                "error_rank",
                "ID",
                "Tool",
                "fold",
                "y_true",
                "y_pred",
                "residual",
                "absolute_residual",
                "missing_rate",
                "zero_rate"
            ]
        ]

        if "error_rank" in result.columns
        else
        result
        .nlargest(
            TOP_ERROR_SAMPLES,
            "absolute_residual"
        )
        [
            [
                "ID",
                "Tool",
                "fold",
                "y_true",
                "y_pred",
                "residual",
                "absolute_residual",
                "missing_rate",
                "zero_rate"
            ]
        ]

        .to_string(
            index=False
        )
    )

    # =====================================================
    # Tool
    # =====================================================

    section(
        "6. Tool误差"
    )

    print(
        tool_analysis_df
        .to_string(
            index=False
        )
    )

    # =====================================================
    # Tool × Fold
    # =====================================================

    section(
        "7. Tool × Fold"
    )

    print(
        tool_fold_df
        .sort_values(
            "mse",
            ascending=False
        )
        .head(20)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # 最差Tool特征
    # =====================================================

    section(
        "8. 最差Tool的异常特征"
    )

    if worst_tool_feature_df.empty:

        print(
            "没有得到有效结果。"
        )

    else:

        print(
            worst_tool_feature_df
            .to_string(
                index=False
            )
        )

    # =====================================================
    # 最差样本特征
    # =====================================================

    section(
        "9. 极端异常样本的Top特征"
    )

    if sample_feature_df.empty:

        print(
            "没有得到有效结果。"
        )

    else:

        first_id = (
            sample_feature_df
            .iloc[0]["ID"]
        )

        print(

            sample_feature_df[
                sample_feature_df[
                    "ID"
                ]
                ==
                first_id
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # =====================================================
    # 找最差Fold
    # =====================================================

    worst_fold = (
        fold_analysis
        .sort_values(
            "mse",
            ascending=False
        )
        .iloc[0]
    )

    # =====================================================
    # 找最差Tool
    # =====================================================

    worst_tool = None

    if not tool_analysis_df.empty:

        worst_tool = (
            tool_analysis_df
            .iloc[0]
        )

    # =====================================================
    # Summary
    # =====================================================

    lines = [

        "工业AI深度诊断报告",

        "",

        "==================================================",

        "当前固定模型",

        "==================================================",

        "缺失值：operation_tool_mean",

        "特征选择：ExtraTrees Top300",

        "最终模型：XGBoost",

        "",

        "==================================================",

        "OOF总体结果",

        "==================================================",

        f"MSE = {overall_mse:.10f}",

        f"RMSE = {overall_rmse:.10f}",

        f"MAE = {overall_mae:.10f}",

        f"R2 = {overall_r2:.6f}",

        "",

        "==================================================",

        "最差Fold",

        "==================================================",

        f"Fold = {int(worst_fold['fold'])}",

        f"MSE = {worst_fold['mse']:.10f}",

        f"样本数 = {int(worst_fold['n'])}",

        f"平均缺失率 = {worst_fold['mean_missing_rate']:.6f}",

        f"平均0值率 = {worst_fold['mean_zero_rate']:.6f}",

        f"Y均值 = {worst_fold['mean_y']:.6f}",

        f"Y标准差 = {worst_fold['std_y']:.6f}"
    ]

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

            f"平均缺失率 = {worst_tool['mean_missing_rate']:.6f}",

            f"平均0值率 = {worst_tool['mean_zero_rate']:.6f}",

            f"Y均值 = {worst_tool['mean_y']:.6f}",

            f"Y标准差 = {worst_tool['std_y']:.6f}"
        ])

    # =====================================================
    # Top样本
    # =====================================================

    lines.extend([

        "",

        "==================================================",

        "Top 10异常样本",

        "=================================================="
    ])

    for _, row in (

        result
        .nlargest(
            10,
            "absolute_residual"
        )
        .iterrows()

    ):

        lines.append(

            f"ID={row['ID']} | "
            f"Tool={row['Tool']} | "
            f"Fold={int(row['fold'])} | "
            f"Y={row['y_true']:.6f} | "
            f"Pred={row['y_pred']:.6f} | "
            f"Residual={row['residual']:.6f} | "
            f"AbsResidual={row['absolute_residual']:.6f}"
        )

    # =====================================================
    # 稳定特征
    # =====================================================

    if not feature_stability_df.empty:

        lines.extend([

            "",

            "==================================================",

            "Top 10稳定特征",

            "=================================================="
        ])

        for _, row in (
            feature_stability_df
            .head(10)
            .iterrows()
        ):

            lines.append(

                f"{row['feature']} | "
                f"Importance="
                f"{row['mean_importance']:.8f} | "
                f"SelectedFolds="
                f"{int(row['selected_folds'])}/{N_SPLITS} | "
                f"Stability="
                f"{row['stability']:.2f}"
            )

    # =====================================================
    # Operation
    # =====================================================

    if not operation_df.empty:

        lines.extend([

            "",

            "==================================================",

            "Top Operation",

            "=================================================="
        ])

        for _, row in (
            operation_df
            .head(10)
            .iterrows()
        ):

            lines.append(

                f"Operation={row['operation']} | "
                f"Features={int(row['feature_count'])} | "
                f"Importance="
                f"{row['total_importance']:.8f}"
            )

    lines.extend([

        "",

        "==================================================",

        "输出目录",

        "==================================================",

        str(OUT_DIR),

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
        "10. 深度诊断完成"
    )

    print(
        f"结果目录：{OUT_DIR}"
    )

    print(
        f"OOF MSE：{overall_mse:.10f}"
    )

    print(
        "已生成CSV、PNG和summary.txt"
    )


if __name__ == "__main__":

    main()