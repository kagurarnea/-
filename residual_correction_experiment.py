# -*- coding: utf-8 -*-

"""
residual_correction_experiment.py

============================================================
工业AI：二阶段残差校正实验
============================================================

第一阶段固定最佳方案
------------------------------------------------------------
缺失值：
    operation_tool_mean

基础特征：
    ExtraTrees Top300

Tool条件：
    Tool-Z Top100

Operation：
    Mean
    Std
    Range
    Min
    Max
    Median

Tool身份：
    One-Hot

Tool目标统计：
    Smoothed Mean
    Smoothed Std
    Relative Mean
    Mean Ratio

第一阶段模型：
    XGBoost

============================================================
第二阶段
============================================================

使用第一阶段OOF预测：

    residual = y - y_global

然后使用Residual XGBoost学习：

    X -> residual

最终：

    y_final =
        y_global + lambda * residual_pred

测试：

    lambda =
        0.00
        0.10
        0.20
        0.30
        0.50
        0.75
        1.00

============================================================
严格防止泄漏
============================================================

每个Outer Fold：

    Train
      ↓
    第一阶段内部5折
      ↓
    获得Train OOF prediction
      ↓
    residual = y - OOF_pred
      ↓
    用整个Outer Train重新训练Global Model
      ↓
    生成Outer Valid global prediction
      ↓
    用Train OOF residual训练Residual Model
      ↓
    预测Outer Valid residual
      ↓
    global + lambda * residual

这样Residual Model看不到Outer Valid目标。

============================================================
输出
============================================================

data/
└── preprocess_missing/
    └── residual_correction_experiment/
        ├── experiment_summary.csv
        ├── experiment_folds.csv
        ├── lambda_summary.csv
        ├── tool_analysis.csv
        ├── residual_analysis.csv
        ├── oof_detail.csv
        ├── top_error_samples.csv
        ├── summary.txt
        └── plots/
            ├── 01_lambda_mse.png
            ├── 02_lambda_rmse.png
            ├── 03_residual_distribution.png
            ├── 04_residual_vs_prediction.png
            ├── 05_tool_error.png
            └── 06_absolute_error_before_after.png
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


# ============================================================
# 配置
# ============================================================

DATA_DIR = Path(
    r"C:\Users\lenovo\Downloads\my method\data"
)

OUT_DIR = (
    DATA_DIR
    / "preprocess_missing"
    / "residual_correction_experiment"
)

PLOT_DIR = OUT_DIR / "plots"

RANDOM_STATE = 42

OUTER_SPLITS = 5

INNER_SPLITS = 5

BASE_TOP_K = 300

TOOL_TOP_K = 100

TARGET_SMOOTH_ALPHA = 20.0

LAMBDAS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00
]


# ============================================================
# 日志
# ============================================================

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


# ============================================================
# 基础函数
# ============================================================

def find_col(df, names):

    for name in names:

        if name in df.columns:

            return name

    return None


def infer_operation(feature):

    m = re.match(
        r"^\s*(\d+)\s*[Xx]",
        str(feature)
    )

    if m:

        return m.group(1)

    return "UNKNOWN"


def safe_numeric(df):

    return (
        df
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )


def safe_fill(train, valid):

    train = safe_numeric(train)

    valid = safe_numeric(valid)

    medians = (
        train
        .median()
        .fillna(0.0)
    )

    train = (
        train
        .fillna(medians)
        .replace(
            [
                np.inf,
                -np.inf
            ],
            0.0
        )
        .astype(
            np.float64
        )
    )

    valid = (
        valid
        .fillna(medians)
        .replace(
            [
                np.inf,
                -np.inf
            ],
            0.0
        )
        .astype(
            np.float64
        )
    )

    return (
        train,
        valid
    )


def remove_duplicate_columns(
    train,
    valid
):

    mask = (
        ~train
        .columns
        .duplicated()
    )

    train = (
        train
        .loc[
            :,
            mask
        ]
        .copy()
    )

    valid = (
        valid
        .reindex(
            columns=train.columns
        )
        .copy()
    )

    return (
        train,
        valid
    )


def check_matrix(
    X,
    y=None,
    name="X"
):

    if y is not None:

        if len(X) != len(y):

            raise ValueError(
                f"{name}与y长度不一致："
                f"{len(X)} vs {len(y)}"
            )

    duplicated = (
        X.columns[
            X.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated:

        raise ValueError(
            f"{name}存在重复列："
            f"{duplicated[:20]}"
        )

    values = X.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            f"{name}存在NaN/Inf"
        )


# ============================================================
# XGBoost
# ============================================================

def make_global_xgb():

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


def make_residual_xgb():

    return XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=
            800,

        learning_rate=
            0.03,

        max_depth=
            2,

        min_child_weight=
            2,

        subsample=
            0.8,

        colsample_bytree=
            0.7,

        reg_alpha=
            0.05,

        reg_lambda=
            2.0,

        gamma=
            0,

        random_state=
            RANDOM_STATE,

        n_jobs=
            -1,

        tree_method=
            "hist"
    )


# ============================================================
# 数据
# ============================================================

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
            "找不到Value/Y目标列"
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

    raw_count = len(
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
        f"原始数值特征 = {raw_count}"
    )

    log(
        f"全空 = {len(all_nan)}"
    )

    log(
        f"常量 = {len(constant)}"
    )

    log(
        f"重复 = {len(duplicate)}"
    )

    log(
        f"最终候选 = {len(features)}"
    )

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    mask = y.notna()

    df = (
        df
        .loc[
            mask
        ]
        .reset_index(
            drop=True
        )
    )

    y = (
        y
        .loc[
            mask
        ]
        .reset_index(
            drop=True
        )
    )

    X = safe_numeric(
        df[
            features
        ]
    )

    meta = pd.DataFrame(
        {
            "row_index":
                np.arange(
                    len(df)
                )
        }
    )

    if id_col is not None:

        meta[
            "ID"
        ] = (
            df[
                id_col
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        meta[
            "ID"
        ] = (
            meta[
                "row_index"
            ]
            .astype(str)
        )

    if tool_col is not None:

        meta[
            "Tool"
        ] = (
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

        meta[
            "Tool"
        ] = "UNKNOWN_TOOL"

    meta[
        "missing_rate"
    ] = (
        X
        .isna()
        .mean(axis=1)
        .to_numpy()
    )

    meta[
        "zero_rate"
    ] = (
        (X == 0)
        .mean(axis=1)
        .to_numpy()
    )

    return (
        X,
        y,
        meta,
        tool_col
    )


# ============================================================
# 缺失值
# ============================================================

def preprocess_fold(
    X_train,
    X_valid,
    meta_train,
    meta_valid,
    tool_col
):

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

    if tool_col is not None:

        train_meta = pd.DataFrame(
            {
                tool_col:
                    meta_train[
                        "Tool"
                    ]
                    .to_numpy()
            }
        )

        valid_meta = pd.DataFrame(
            {
                tool_col:
                    meta_valid[
                        "Tool"
                    ]
                    .to_numpy()
            }
        )

    else:

        train_meta = pd.DataFrame()

        valid_meta = pd.DataFrame()

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

    return safe_fill(
        X_train,
        X_valid
    )


# ============================================================
# Base Top300
# ============================================================

def select_base_features(
    X,
    y
):

    model = ExtraTreesRegressor(

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

    model.fit(
        X,
        y
    )

    importance = (
        pd.Series(
            model.feature_importances_,
            index=X.columns
        )
        .sort_values(
            ascending=False
        )
    )

    selected = (
        importance
        .head(
            min(
                BASE_TOP_K,
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


# ============================================================
# Tool统计
# ============================================================

def fit_tool_statistics(
    X,
    tools
):

    tools = np.asarray(
        tools,
        dtype=str
    )

    if len(tools) != len(X):

        raise ValueError(
            "Tool统计长度错误"
        )

    temp = X.copy()

    temp[
        "__TOOL__"
    ] = tools

    grouped = (
        temp
        .groupby(
            "__TOOL__"
        )
    )

    tool_mean = (
        grouped
        .mean(
            numeric_only=True
        )
    )

    tool_std = (
        grouped
        .std(
            numeric_only=True
        )
    )

    return {

        "tool_mean":
            tool_mean,

        "tool_std":
            tool_std,

        "global_mean":
            X.mean(),

        "global_std":
            X.std()
            .replace(
                0,
                np.nan
            )
            .fillna(1.0)
    }


# ============================================================
# Tool-Z
# ============================================================

def build_tool_z(
    X,
    tools,
    statistics
):

    X = (
        X
        .reset_index(
            drop=True
        )
    )

    tools = np.asarray(
        tools,
        dtype=str
    )

    n_samples = len(X)

    n_features = X.shape[1]

    if len(tools) != n_samples:

        raise ValueError(
            "Tool-Z长度错误："
            f"tools={len(tools)}, "
            f"X={n_samples}"
        )

    columns = list(
        X.columns
    )

    tool_mean = (
        statistics[
            "tool_mean"
        ]
    )

    tool_std = (
        statistics[
            "tool_std"
        ]
    )

    global_mean = (
        statistics[
            "global_mean"
        ]
        .reindex(
            columns
        )
        .fillna(0.0)
    )

    global_std = (
        statistics[
            "global_std"
        ]
        .reindex(
            columns
        )
        .replace(
            0,
            np.nan
        )
        .fillna(1.0)
    )

    global_mean_array = np.array(
        global_mean.to_numpy(
            dtype=np.float64
        ),
        dtype=np.float64,
        copy=True
    )

    global_std_array = np.array(
        global_std.to_numpy(
            dtype=np.float64
        ),
        dtype=np.float64,
        copy=True
    )

    mean_matrix = np.empty(
        (
            n_samples,
            n_features
        ),
        dtype=np.float64
    )

    std_matrix = np.empty(
        (
            n_samples,
            n_features
        ),
        dtype=np.float64
    )

    for i, tool in enumerate(
        tools
    ):

        if tool in tool_mean.index:

            mean_values = np.array(
                tool_mean
                .loc[
                    tool,
                    columns
                ]
                .to_numpy(
                    dtype=np.float64
                ),
                dtype=np.float64,
                copy=True
            )

            std_values = np.array(
                tool_std
                .loc[
                    tool,
                    columns
                ]
                .to_numpy(
                    dtype=np.float64
                ),
                dtype=np.float64,
                copy=True
            )

            bad_std = (

                ~np.isfinite(
                    std_values
                )

                |

                (
                    np.abs(
                        std_values
                    )
                    < 1e-12
                )
            )

            if bad_std.any():

                std_values[
                    bad_std
                ] = (
                    global_std_array[
                        bad_std
                    ]
                )

            bad_mean = ~np.isfinite(
                mean_values
            )

            if bad_mean.any():

                mean_values[
                    bad_mean
                ] = (
                    global_mean_array[
                        bad_mean
                    ]
                )

            mean_matrix[
                i,
                :
            ] = mean_values

            std_matrix[
                i,
                :
            ] = std_values

        else:

            mean_matrix[
                i,
                :
            ] = global_mean_array

            std_matrix[
                i,
                :
            ] = global_std_array

    X_array = np.array(
        X.to_numpy(
            dtype=np.float64
        ),
        dtype=np.float64,
        copy=True
    )

    z_array = (
        X_array
        -
        mean_matrix
    ) / std_matrix

    bad = ~np.isfinite(
        z_array
    )

    z_array[
        bad
    ] = 0.0

    z_array = np.clip(
        z_array,
        -50.0,
        50.0
    )

    z = pd.DataFrame(
        z_array,
        columns=[
            f"TZ_{c}"
            for c in columns
        ]
    )

    abs_z = np.abs(
        z_array
    )

    k5 = min(
        5,
        n_features
    )

    k10 = min(
        10,
        n_features
    )

    top5 = np.partition(
        abs_z,
        n_features - k5,
        axis=1
    )[
        :,
        n_features - k5:
    ]

    top10 = np.partition(
        abs_z,
        n_features - k10,
        axis=1
    )[
        :,
        n_features - k10:
    ]

    anomaly = pd.DataFrame(
        {

            "AN_tool_max_z":
                abs_z.max(
                    axis=1
                ),

            "AN_tool_top5_mean_z":
                top5.mean(
                    axis=1
                ),

            "AN_tool_top10_mean_z":
                top10.mean(
                    axis=1
                ),

            "AN_tool_count_z3":
                (
                    abs_z > 3
                ).sum(axis=1),

            "AN_tool_count_z5":
                (
                    abs_z > 5
                ).sum(axis=1)
        }
    )

    return (
        z,
        anomaly
    )


# ============================================================
# Tool-Z选择
# ============================================================

def select_tool_z_features(
    tool_z,
    y
):

    if len(tool_z) != len(y):

        raise ValueError(
            "Tool-Z特征与Y长度不一致"
        )

    if tool_z.shape[1] == 0:

        return (
            [],
            pd.Series(
                dtype=float
            )
        )

    model = (
        ExtraTreesRegressor(

            n_estimators=
                200,

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

    model.fit(
        tool_z,
        y
    )

    importance = (
        pd.Series(
            model.feature_importances_,
            index=tool_z.columns
        )
        .sort_values(
            ascending=False
        )
    )

    selected = (
        importance
        .head(
            min(
                TOOL_TOP_K,
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


# ============================================================
# Operation
# ============================================================

def build_operation_features(
    X,
    features
):

    groups = {}

    for feature in features:

        operation = (
            infer_operation(
                feature
            )
        )

        groups.setdefault(
            operation,
            []
        )

        groups[
            operation
        ].append(
            feature
        )

    blocks = []

    for operation, cols in (
        groups.items()
    ):

        cols = [
            c
            for c in cols
            if c in X.columns
        ]

        if not cols:
            continue

        values = X[
            cols
        ]

        block = pd.DataFrame(
            index=np.arange(
                len(X)
            )
        )

        block[
            f"OP_{operation}_mean"
        ] = (
            values
            .mean(axis=1)
            .to_numpy()
        )

        block[
            f"OP_{operation}_std"
        ] = (
            values
            .std(axis=1)
            .to_numpy()
        )

        block[
            f"OP_{operation}_range"
        ] = (
            values.max(axis=1)
            -
            values.min(axis=1)
        ).to_numpy()

        block[
            f"OP_{operation}_min"
        ] = (
            values
            .min(axis=1)
            .to_numpy()
        )

        block[
            f"OP_{operation}_max"
        ] = (
            values
            .max(axis=1)
            .to_numpy()
        )

        block[
            f"OP_{operation}_median"
        ] = (
            values
            .median(axis=1)
            .to_numpy()
        )

        blocks.append(
            block
        )

    if not blocks:

        return pd.DataFrame(
            index=np.arange(
                len(X)
            )
        )

    return pd.concat(
        blocks,
        axis=1
    )


# ============================================================
# Tool One-Hot
# ============================================================

def build_tool_onehot(
    train_tools,
    valid_tools
):

    train_tools = (
        pd.Series(
            train_tools
        )
        .astype(str)
        .reset_index(drop=True)
    )

    valid_tools = (
        pd.Series(
            valid_tools
        )
        .astype(str)
        .reset_index(drop=True)
    )

    categories = sorted(
        train_tools.unique()
    )

    train_data = {}

    valid_data = {}

    for category in categories:

        column = (
            f"TOOL_{category}"
        )

        train_data[
            column
        ] = (
            train_tools
            .eq(category)
            .astype(float)
            .to_numpy()
        )

        valid_data[
            column
        ] = (
            valid_tools
            .eq(category)
            .astype(float)
            .to_numpy()
        )

    return (
        pd.DataFrame(
            train_data
        ),
        pd.DataFrame(
            valid_data
        )
    )


# ============================================================
# Target Encoding
# ============================================================

def fit_tool_target_encoding(
    y,
    tools
):

    y = (
        pd.Series(
            y
        )
        .astype(float)
        .reset_index(drop=True)
    )

    tools = (
        pd.Series(
            tools
        )
        .astype(str)
        .reset_index(drop=True)
    )

    if len(y) != len(tools):

        raise ValueError(
            "Target Encoding长度错误"
        )

    global_mean = float(
        y.mean()
    )

    data = pd.DataFrame(
        {
            "Tool":
                tools,

            "Y":
                y
        }
    )

    grouped = (
        data
        .groupby(
            "Tool"
        )[
            "Y"
        ]
        .agg(
            [
                "mean",
                "std",
                "count"
            ]
        )
    )

    grouped[
        "smooth_mean"
    ] = (
        (
            grouped[
                "count"
            ]
            *
            grouped[
                "mean"
            ]
        )
        +
        TARGET_SMOOTH_ALPHA
        *
        global_mean
    ) / (
        grouped[
            "count"
        ]
        +
        TARGET_SMOOTH_ALPHA
    )

    grouped[
        "relative_mean"
    ] = (
        grouped[
            "smooth_mean"
        ]
        -
        global_mean
    )

    if abs(
        global_mean
    ) > 1e-12:

        grouped[
            "mean_ratio"
        ] = (
            grouped[
                "smooth_mean"
            ]
            /
            global_mean
        )

    else:

        grouped[
            "mean_ratio"
        ] = 1.0

    return {

        "table":
            grouped,

        "global_mean":
            global_mean
    }


def transform_tool_target_encoding(
    tools,
    encoding,
    include_std=True
):

    tools = (
        pd.Series(
            tools
        )
        .astype(str)
        .reset_index(drop=True)
    )

    table = (
        encoding[
            "table"
        ]
    )

    global_mean = (
        encoding[
            "global_mean"
        ]
    )

    rows = []

    for tool in tools:

        if tool in table.index:

            row = (
                table.loc[
                    tool
                ]
            )

            rows.append(
                {

                    "TE_tool_mean":
                        float(
                            row[
                                "smooth_mean"
                            ]
                        ),

                    "TE_tool_std":
                        (
                            float(
                                row[
                                    "std"
                                ]
                            )
                            if pd.notna(
                                row[
                                    "std"
                                ]
                            )
                            else 0.0
                        ),

                    "TE_tool_relative_mean":
                        float(
                            row[
                                "relative_mean"
                            ]
                        ),

                    "TE_tool_mean_ratio":
                        float(
                            row[
                                "mean_ratio"
                            ]
                        )
                }
            )

        else:

            rows.append(
                {

                    "TE_tool_mean":
                        global_mean,

                    "TE_tool_std":
                        0.0,

                    "TE_tool_relative_mean":
                        0.0,

                    "TE_tool_mean_ratio":
                        1.0
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not include_std:

        result = result[
            [
                "TE_tool_mean",
                "TE_tool_relative_mean",
                "TE_tool_mean_ratio"
            ]
        ]

    return result


# ============================================================
# 第一阶段最终特征
# ============================================================

def build_features(
    X_train,
    X_valid,
    y_train,
    meta_train,
    meta_valid,
    use_tool_onehot=True,
    use_target_encoding=True,
    use_target_std=True,
    use_tool_anomaly=False,
    use_global_anomaly=False
):

    # --------------------------------------------------------
    # Base Top300
    # --------------------------------------------------------

    (
        base_features,
        base_importance
    ) = select_base_features(

        X_train,

        y_train
    )

    base_train = (
        X_train[
            base_features
        ]
        .copy()
    )

    base_valid = (
        X_valid[
            base_features
        ]
        .copy()
    )

    train_blocks = [
        base_train
    ]

    valid_blocks = [
        base_valid
    ]

    # --------------------------------------------------------
    # Tool-Z
    # --------------------------------------------------------

    tool_stats = (
        fit_tool_statistics(

            base_train,

            meta_train[
                "Tool"
            ]
            .to_numpy()
        )
    )

    (
        tool_z_train,
        tool_anomaly_train
    ) = build_tool_z(

        base_train,

        meta_train[
            "Tool"
        ]
        .to_numpy(),

        tool_stats
    )

    (
        tool_z_valid,
        tool_anomaly_valid
    ) = build_tool_z(

        base_valid,

        meta_valid[
            "Tool"
        ]
        .to_numpy(),

        tool_stats
    )

    (
        tool_features,
        tool_importance
    ) = select_tool_z_features(

        tool_z_train,

        y_train
    )

    if tool_features:

        train_blocks.append(
            tool_z_train[
                tool_features
            ]
        )

        valid_blocks.append(
            tool_z_valid[
                tool_features
            ]
        )

    # --------------------------------------------------------
    # Operation
    # --------------------------------------------------------

    operation_train = (
        build_operation_features(

            base_train,

            base_features
        )
    )

    operation_valid = (
        build_operation_features(

            base_valid,

            base_features
        )
    )

    if not operation_train.empty:

        train_blocks.append(
            operation_train
        )

        valid_blocks.append(
            operation_valid
        )

    # --------------------------------------------------------
    # Tool One-Hot
    # --------------------------------------------------------

    if use_tool_onehot:

        (
            onehot_train,
            onehot_valid
        ) = build_tool_onehot(

            meta_train[
                "Tool"
            ],

            meta_valid[
                "Tool"
            ]
        )

        train_blocks.append(
            onehot_train
        )

        valid_blocks.append(
            onehot_valid
        )

    # --------------------------------------------------------
    # Target Encoding
    # --------------------------------------------------------

    if use_target_encoding:

        encoding = (
            fit_tool_target_encoding(

                y_train,

                meta_train[
                    "Tool"
                ]
            )
        )

        te_train = (
            transform_tool_target_encoding(

                meta_train[
                    "Tool"
                ],

                encoding,

                include_std=
                    use_target_std
            )
        )

        te_valid = (
            transform_tool_target_encoding(

                meta_valid[
                    "Tool"
                ],

                encoding,

                include_std=
                    use_target_std
            )
        )

        train_blocks.append(
            te_train
        )

        valid_blocks.append(
            te_valid
        )

    # --------------------------------------------------------
    # Tool异常
    # --------------------------------------------------------

    if use_tool_anomaly:

        train_blocks.append(
            tool_anomaly_train
        )

        valid_blocks.append(
            tool_anomaly_valid
        )

    # --------------------------------------------------------
    # Global异常
    # --------------------------------------------------------

    if use_global_anomaly:

        global_train = (
            build_global_anomaly(

                base_train,

                base_train
            )
        )

        global_valid = (
            build_global_anomaly(

                base_train,

                base_valid
            )
        )

        train_blocks.append(
            global_train
        )

        valid_blocks.append(
            global_valid
        )

    # --------------------------------------------------------
    # 拼接
    # --------------------------------------------------------

    train_final = pd.concat(
        train_blocks,
        axis=1
    )

    valid_final = pd.concat(
        valid_blocks,
        axis=1
    )

    train_final, valid_final = (
        remove_duplicate_columns(

            train_final,

            valid_final
        )
    )

    train_final, valid_final = safe_fill(
        train_final,
        valid_final
    )

    check_matrix(
        train_final,
        y_train,
        "Final Train"
    )

    check_matrix(
        valid_final,
        None,
        "Final Valid"
    )

    if len(
        train_final
    ) != len(
        y_train
    ):

        raise RuntimeError(
            "Final Train行数错误"
        )

    if len(
        valid_final
    ) != len(
        meta_valid
    ):

        raise RuntimeError(
            "Final Valid行数错误"
        )

    if list(
        train_final.columns
    ) != list(
        valid_final.columns
    ):

        raise RuntimeError(
            "Train/Valid特征不一致"
        )

    return {

        "train":
            train_final,

        "valid":
            valid_final,

        "base_features":
            base_features,

        "tool_features":
            tool_features,

        "base_importance":
            base_importance,

        "tool_importance":
            tool_importance
    }


# ============================================================
# 第一阶段完整预处理
# ============================================================

def preprocess_and_build(
    X_train_raw,
    X_valid_raw,
    y_train,
    meta_train,
    meta_valid,
    tool_col
):

    (
        X_train,
        X_valid
    ) = preprocess_fold(

        X_train_raw,

        X_valid_raw,

        meta_train,

        meta_valid,

        tool_col
    )

    return build_features(

        X_train,

        X_valid,

        y_train,

        meta_train,

        meta_valid,

        use_tool_onehot=True,

        use_target_encoding=True,

        use_target_std=True,

        use_tool_anomaly=False,

        use_global_anomaly=False
    )


# ============================================================
# 第一阶段内部OOF
# ============================================================

def generate_inner_oof(
    X_outer_train,
    y_outer_train,
    meta_outer_train,
    tool_col
):

    inner_kfold = KFold(

        n_splits=
            INNER_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE + 100
    )

    inner_oof = np.full(
        len(
            y_outer_train
        ),
        np.nan,
        dtype=np.float64
    )

    for inner_fold, (
        inner_train_idx,
        inner_valid_idx
    ) in enumerate(

        inner_kfold.split(
            X_outer_train
        ),

        start=1
    ):

        Xtr_raw = (
            X_outer_train
            .iloc[
                inner_train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        Xva_raw = (
            X_outer_train
            .iloc[
                inner_valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        ytr = (
            y_outer_train
            .iloc[
                inner_train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        mtr = (
            meta_outer_train
            .iloc[
                inner_train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        mva = (
            meta_outer_train
            .iloc[
                inner_valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        features = preprocess_and_build(

            Xtr_raw,

            Xva_raw,

            ytr,

            mtr,

            mva,

            tool_col
        )

        model = make_global_xgb()

        model.fit(

            features[
                "train"
            ],

            ytr,

            verbose=False
        )

        pred = model.predict(
            features[
                "valid"
            ]
        )

        if not np.isfinite(
            pred
        ).all():

            raise RuntimeError(
                f"Inner Fold {inner_fold} "
                "预测存在NaN/Inf"
            )

        inner_oof[
            inner_valid_idx
        ] = pred

        log(
            f"      Inner Fold "
            f"{inner_fold}/{INNER_SPLITS}"
        )

    if np.isnan(
        inner_oof
    ).any():

        raise RuntimeError(
            "内部OOF仍然存在NaN"
        )

    return inner_oof


# ============================================================
# Residual模型
# ============================================================

def train_residual_model(
    X_outer_train,
    y_outer_train,
    residual_target,
    meta_outer_train,
    tool_col
):

    residual_features = preprocess_and_build(

        X_outer_train,

        X_outer_train.copy(),

        residual_target,

        meta_outer_train,

        meta_outer_train.copy(),

        tool_col
    )

    model = make_residual_xgb()

    model.fit(

        residual_features[
            "train"
        ],

        residual_target,

        verbose=False
    )

    return (
        model,
        residual_features
    )


# ============================================================
# Outer Fold
# ============================================================

def run_outer_fold(
    X,
    y,
    meta,
    train_idx,
    valid_idx,
    tool_col
):

    X_outer_train = (
        X.iloc[
            train_idx
        ]
        .reset_index(
            drop=True
        )
    )

    X_outer_valid = (
        X.iloc[
            valid_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_outer_train = (
        y.iloc[
            train_idx
        ]
        .reset_index(
            drop=True
        )
    )

    y_outer_valid = (
        y.iloc[
            valid_idx
        ]
        .reset_index(
            drop=True
        )
    )

    meta_outer_train = (
        meta.iloc[
            train_idx
        ]
        .reset_index(
            drop=True
        )
    )

    meta_outer_valid = (
        meta.iloc[
            valid_idx
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Step 1：内部OOF
    # ========================================================

    inner_oof = generate_inner_oof(

        X_outer_train,

        y_outer_train,

        meta_outer_train,

        tool_col
    )

    residual_target = (
        y_outer_train.to_numpy()
        -
        inner_oof
    )

    residual_target = np.asarray(
        residual_target,
        dtype=np.float64
    )

    if not np.isfinite(
        residual_target
    ).all():

        raise RuntimeError(
            "Residual target存在NaN/Inf"
        )

    # ========================================================
    # Step 2：Outer Train训练Global
    # ========================================================

    global_features = preprocess_and_build(

        X_outer_train,

        X_outer_valid,

        y_outer_train,

        meta_outer_train,

        meta_outer_valid,

        tool_col
    )

    global_train = (
        global_features[
            "train"
        ]
    )

    global_valid = (
        global_features[
            "valid"
        ]
    )

    global_model = (
        make_global_xgb()
    )

    global_model.fit(

        global_train,

        y_outer_train,

        eval_set=[
            (
                global_valid,
                y_outer_valid
            )
        ],

        verbose=False
    )

    global_pred = (
        global_model
        .predict(
            global_valid
        )
    )

    # ========================================================
    # Step 3：
    # 构造Residual Model的训练特征
    #
    # 必须使用Outer Train的特征体系
    # ========================================================

    residual_train_features = (
        preprocess_and_build(

            X_outer_train,

            X_outer_train.copy(),

            y_outer_train,

            meta_outer_train,

            meta_outer_train.copy(),

            tool_col
        )
    )

    residual_train_X = (
        residual_train_features[
            "train"
        ]
    )

    residual_valid = (
        preprocess_and_build(

            X_outer_train,

            X_outer_valid,

            y_outer_train,

            meta_outer_train,

            meta_outer_valid,

            tool_col
        )
    )

    residual_valid_X = (
        residual_valid[
            "valid"
        ]
    )

    # ========================================================
    # Step 4：Residual Model
    # ========================================================

    residual_model = (
        make_residual_xgb()
    )

    residual_model.fit(

        residual_train_X,

        residual_target,

        verbose=False
    )

    residual_pred = (
        residual_model
        .predict(
            residual_valid_X
        )
    )

    if not np.isfinite(
        residual_pred
    ).all():

        raise RuntimeError(
            "Residual预测存在NaN/Inf"
        )

    return {

        "y_valid":
            y_outer_valid.to_numpy(),

        "global_pred":
            global_pred,

        "residual_pred":
            residual_pred,

        "meta_valid":
            meta_outer_valid,

        "global_features":
            global_features,

        "inner_oof":
            inner_oof
    }


# ============================================================
# 主实验
# ============================================================

def run_experiment(
    X,
    y,
    meta,
    tool_col
):

    section(
        "2. 二阶段残差校正"
    )

    kfold = KFold(

        n_splits=
            OUTER_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE
    )

    n = len(y)

    global_oof = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    residual_oof = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    folds_all = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(

        kfold.split(X),

        start=1
    ):

        fold_start = (
            time.perf_counter()
        )

        log(
            f"Outer Fold "
            f"{fold}/{OUTER_SPLITS}"
        )

        result = run_outer_fold(

            X,

            y,

            meta,

            train_idx,

            valid_idx,

            tool_col
        )

        y_valid = (
            result[
                "y_valid"
            ]
        )

        global_pred = (
            result[
                "global_pred"
            ]
        )

        residual_pred = (
            result[
                "residual_pred"
            ]
        )

        global_oof[
            valid_idx
        ] = global_pred

        residual_oof[
            valid_idx
        ] = residual_pred

        fold_rows = []

        for alpha in LAMBDAS:

            corrected_pred = (
                global_pred
                +
                alpha
                *
                residual_pred
            )

            mse = (
                mean_squared_error(
                    y_valid,
                    corrected_pred
                )
            )

            rmse = np.sqrt(
                mse
            )

            mae = (
                mean_absolute_error(
                    y_valid,
                    corrected_pred
                )
            )

            r2 = (
                r2_score(
                    y_valid,
                    corrected_pred
                )
            )

            fold_rows.append(

                {

                    "fold":
                        fold,

                    "alpha":
                        alpha,

                    "mse":
                        mse,

                    "rmse":
                        rmse,

                    "mae":
                        mae,

                    "r2":
                        r2,

                    "n_features":
                        result[
                            "global_features"
                        ][
                            "train"
                        ]
                        .shape[1],

                    "time_sec":
                        (
                            time.perf_counter()
                            -
                            fold_start
                        )
                }
            )

        fold_df = (
            pd.DataFrame(
                fold_rows
            )
        )

        folds_all.append(
            fold_df
        )

        best_fold = (
            fold_df
            .sort_values(
                "mse"
            )
            .iloc[0]
        )

        log(
            f"Fold {fold}完成 | "
            f"Global MSE="
            f"{fold_df.loc[fold_df['alpha']==0,'mse'].iloc[0]:.8f} | "
            f"Best alpha="
            f"{best_fold['alpha']:.2f} | "
            f"Best MSE="
            f"{best_fold['mse']:.8f}"
        )

    if np.isnan(
        global_oof
    ).any():

        raise RuntimeError(
            "Global OOF存在NaN"
        )

    if np.isnan(
        residual_oof
    ).any():

        raise RuntimeError(
            "Residual OOF存在NaN"
        )

    folds_df = pd.concat(
        folds_all,
        ignore_index=True
    )

    # ========================================================
    # OOF Lambda评价
    # ========================================================

    lambda_rows = []

    for alpha in LAMBDAS:

        corrected_oof = (
            global_oof
            +
            alpha
            *
            residual_oof
        )

        mse = (
            mean_squared_error(
                y,
                corrected_oof
            )
        )

        rmse = np.sqrt(
            mse
        )

        mae = (
            mean_absolute_error(
                y,
                corrected_oof
            )
        )

        r2 = (
            r2_score(
                y,
                corrected_oof
            )
        )

        lambda_rows.append(

            {

                "alpha":
                    alpha,

                "mse":
                    mse,

                "rmse":
                    rmse,

                "mae":
                    mae,

                "r2":
                    r2,

                "mse_std":
                    folds_df[
                        folds_df[
                            "alpha"
                        ]
                        ==
                        alpha
                    ][
                        "mse"
                    ]
                    .std()
            }
        )

    lambda_df = (
        pd.DataFrame(
            lambda_rows
        )
        .sort_values(
            "mse"
        )
        .reset_index(
            drop=True
        )
    )

    return {

        "global_oof":
            global_oof,

        "residual_oof":
            residual_oof,

        "folds":
            folds_df,

        "lambda":
            lambda_df
    }


# ============================================================
# Tool分析
# ============================================================

def analyze_tools(
    y,
    pred,
    meta
):

    df = pd.DataFrame(
        {

            "Tool":
                meta[
                    "Tool"
                ]
                .astype(str)
                .to_numpy(),

            "y_true":
                y.to_numpy(),

            "y_pred":
                pred
        }
    )

    df[
        "residual"
    ] = (
        df[
            "y_true"
        ]
        -
        df[
            "y_pred"
        ]
    )

    df[
        "absolute_error"
    ] = (
        df[
            "residual"
        ]
        .abs()
    )

    df[
        "squared_error"
    ] = (
        df[
            "residual"
        ]
        ** 2
    )

    return (
        df
        .groupby(
            "Tool"
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
                "absolute_error",
                "mean"
            ),

            max_error=(
                "absolute_error",
                "max"
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


# ============================================================
# 残差分析
# ============================================================

def build_residual_analysis(
    y,
    global_pred,
    residual_pred,
    best_alpha,
    meta
):

    corrected_pred = (
        global_pred
        +
        best_alpha
        *
        residual_pred
    )

    result = meta.copy()

    result[
        "y_true"
    ] = y.to_numpy()

    result[
        "global_pred"
    ] = global_pred

    result[
        "residual_pred"
    ] = residual_pred

    result[
        "corrected_pred"
    ] = corrected_pred

    result[
        "global_residual"
    ] = (
        result[
            "y_true"
        ]
        -
        result[
            "global_pred"
        ]
    )

    result[
        "corrected_residual"
    ] = (
        result[
            "y_true"
        ]
        -
        result[
            "corrected_pred"
        ]
    )

    result[
        "global_absolute_error"
    ] = (
        result[
            "global_residual"
        ]
        .abs()
    )

    result[
        "corrected_absolute_error"
    ] = (
        result[
            "corrected_residual"
        ]
        .abs()
    )

    result[
        "absolute_error_change"
    ] = (
        result[
            "global_absolute_error"
        ]
        -
        result[
            "corrected_absolute_error"
        ]
    )

    result[
        "improved"
    ] = (
        result[
            "corrected_absolute_error"
        ]
        <
        result[
            "global_absolute_error"
        ]
    )

    return result


# ============================================================
# 绘图
# ============================================================

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


def make_plots(
    lambda_df,
    tool_df,
    residual_df
):

    section(
        "5. 生成可视化"
    )

    # --------------------------------------------------------
    # Lambda MSE
    # --------------------------------------------------------

    temp = (
        lambda_df
        .sort_values(
            "alpha"
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        temp[
            "alpha"
        ],
        temp[
            "mse"
        ],
        marker="o"
    )

    plt.xlabel(
        "Residual Correction Alpha"
    )

    plt.ylabel(
        "OOF MSE"
    )

    plt.title(
        "MSE vs Residual Correction Strength"
    )

    save_plot(
        "01_lambda_mse.png"
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        temp[
            "alpha"
        ],
        temp[
            "rmse"
        ],
        marker="o"
    )

    plt.xlabel(
        "Residual Correction Alpha"
    )

    plt.ylabel(
        "OOF RMSE"
    )

    plt.title(
        "RMSE vs Residual Correction Strength"
    )

    save_plot(
        "02_lambda_rmse.png"
    )

    # --------------------------------------------------------
    # Residual distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        residual_df[
            "global_residual"
        ],
        bins=40,
        alpha=0.7
    )

    plt.xlabel(
        "Global Residual"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Global Residual Distribution"
    )

    save_plot(
        "03_residual_distribution.png"
    )

    # --------------------------------------------------------
    # Residual vs prediction
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        residual_df[
            "global_pred"
        ],
        residual_df[
            "global_residual"
        ],
        alpha=0.7
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Global Prediction"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual vs Global Prediction"
    )

    save_plot(
        "04_residual_vs_prediction.png"
    )

    # --------------------------------------------------------
    # Tool
    # --------------------------------------------------------

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
            temp[
                "mse"
            ]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp[
                "Tool"
            ]
            .astype(str)
        )

        plt.xlabel(
            "MSE"
        )

        plt.title(
            "MSE by Tool"
        )

        save_plot(
            "05_tool_error.png"
        )

    # --------------------------------------------------------
    # Before / After
    # --------------------------------------------------------

    if not residual_df.empty:

        before = (
            residual_df[
                "global_absolute_error"
            ]
            .mean()
        )

        after = (
            residual_df[
                "corrected_absolute_error"
            ]
            .mean()
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.bar(
            [
                "Global",
                "Corrected"
            ],
            [
                before,
                after
            ]
        )

        plt.ylabel(
            "Mean Absolute Error"
        )

        plt.title(
            "Absolute Error Before vs After Correction"
        )

        save_plot(
            "06_absolute_error_before_after.png"
        )


# ============================================================
# Main
# ============================================================

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

    # ========================================================
    # 数据
    # ========================================================

    (
        X,
        y,
        meta,
        tool_col
    ) = prepare_data()

    # ========================================================
    # 实验
    # ========================================================

    result = run_experiment(

        X,

        y,

        meta,

        tool_col
    )

    global_oof = (
        result[
            "global_oof"
        ]
    )

    residual_oof = (
        result[
            "residual_oof"
        ]
    )

    folds_df = (
        result[
            "folds"
        ]
    )

    lambda_df = (
        result[
            "lambda"
        ]
    )

    # ========================================================
    # 当前最优
    # ========================================================

    best = (
        lambda_df
        .iloc[0]
    )

    baseline = (
        lambda_df
        .loc[
            lambda_df[
                "alpha"
            ]
            ==
            0.0
        ]
        .iloc[0]
    )

    best_alpha = float(
        best[
            "alpha"
        ]
    )

    improvement = (

        (
            baseline[
                "mse"
            ]
            -
            best[
                "mse"
            ]
        )

        /

        baseline[
            "mse"
        ]

        *

        100.0
    )

    # ========================================================
    # 最终Residual分析
    # ========================================================

    residual_df = (
        build_residual_analysis(

            y,

            global_oof,

            residual_oof,

            best_alpha,

            meta
        )
    )

    corrected_oof = (
        residual_df[
            "corrected_pred"
        ]
        .to_numpy()
    )

    # ========================================================
    # Tool
    # ========================================================

    tool_df = analyze_tools(

        y,

        corrected_oof,

        meta
    )

    # ========================================================
    # Top异常
    # ========================================================

    top_error = (
        residual_df
        .sort_values(
            "corrected_absolute_error",
            ascending=False
        )
        .head(50)
    )

    # ========================================================
    # 残差模型效果
    # ========================================================

    residual_corr = (
        np.corrcoef(

            residual_df[
                "global_residual"
            ],

            residual_df[
                "residual_pred"
            ]

        )[0, 1]
    )

    if not np.isfinite(
        residual_corr
    ):

        residual_corr = 0.0

    # ========================================================
    # 保存
    # ========================================================

    folds_df.to_csv(

        OUT_DIR
        /
        "experiment_folds.csv",

        index=False,

        encoding="utf-8-sig"
    )

    lambda_df.to_csv(

        OUT_DIR
        /
        "lambda_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )

    residual_df.to_csv(

        OUT_DIR
        /
        "residual_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    tool_df.to_csv(

        OUT_DIR
        /
        "tool_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    top_error.to_csv(

        OUT_DIR
        /
        "top_error_samples.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # OOF完整结果
    # ========================================================

    oof_detail = (
        meta.copy()
    )

    oof_detail[
        "y_true"
    ] = y.to_numpy()

    oof_detail[
        "global_pred"
    ] = global_oof

    oof_detail[
        "residual_pred"
    ] = residual_oof

    for alpha in LAMBDAS:

        oof_detail[
            f"pred_alpha_{str(alpha).replace('.', '_')}"
        ] = (

            global_oof
            +
            alpha
            *
            residual_oof
        )

    oof_detail.to_csv(

        OUT_DIR
        /
        "oof_detail.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # Experiment Summary
    # ========================================================

    summary_df = pd.DataFrame(
        [

            {

                "experiment":
                    "global_baseline",

                "alpha":
                    0.0,

                "mse":
                    baseline[
                        "mse"
                    ],

                "rmse":
                    baseline[
                        "rmse"
                    ],

                "mae":
                    baseline[
                        "mae"
                    ],

                "r2":
                    baseline[
                        "r2"
                    ]
            },

            {

                "experiment":
                    "best_residual_corrected",

                "alpha":
                    best_alpha,

                "mse":
                    best[
                        "mse"
                    ],

                "rmse":
                    best[
                        "rmse"
                    ],

                "mae":
                    best[
                        "mae"
                    ],

                "r2":
                    best[
                        "r2"
                    ]
            }
        ]
    )

    summary_df.to_csv(

        OUT_DIR
        /
        "experiment_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # 可视化
    # ========================================================

    make_plots(

        lambda_df,

        tool_df,

        residual_df
    )

    # ========================================================
    # 输出
    # ========================================================

    section(
        "3. Lambda实验排名"
    )

    print(
        lambda_df.to_string(
            index=False
        )
    )

    section(
        "4. 当前最优"
    )

    print(
        f"最佳Alpha："
        f"{best_alpha:.2f}"
    )

    print(
        f"Baseline MSE："
        f"{baseline['mse']:.10f}"
    )

    print(
        f"Corrected MSE："
        f"{best['mse']:.10f}"
    )

    print(
        f"Baseline RMSE："
        f"{baseline['rmse']:.10f}"
    )

    print(
        f"Corrected RMSE："
        f"{best['rmse']:.10f}"
    )

    print(
        f"Baseline MAE："
        f"{baseline['mae']:.10f}"
    )

    print(
        f"Corrected MAE："
        f"{best['mae']:.10f}"
    )

    print(
        f"Baseline R2："
        f"{baseline['r2']:.6f}"
    )

    print(
        f"Corrected R2："
        f"{best['r2']:.6f}"
    )

    print(
        f"MSE变化："
        f"{baseline['mse'] - best['mse']:.10f}"
    )

    print(
        f"相对变化："
        f"{improvement:.2f}%"
    )

    print(
        f"Residual预测相关系数："
        f"{residual_corr:.6f}"
    )

    section(
        "5. Tool误差"
    )

    print(
        tool_df.to_string(
            index=False
        )
    )

    section(
        "6. Top 20异常样本"
    )

    print(

        top_error[
            [
                "ID",
                "Tool",
                "row_index",
                "y_true",
                "global_pred",
                "residual_pred",
                "corrected_pred",
                "global_absolute_error",
                "corrected_absolute_error",
                "absolute_error_change",
                "improved"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    lines = [

        "工业AI二阶段残差校正实验",

        "",

        "第一阶段固定方案：",

        "缺失值 = operation_tool_mean",

        f"ExtraTrees = Top{BASE_TOP_K}",

        f"Tool-Z = Top{TOOL_TOP_K}",

        "Operation = All",

        "Tool One-Hot = True",

        "Tool Target Mean = True",

        "Tool Target Std = True",

        "第一阶段模型 = XGBoost",

        "",

        "第二阶段：",

        "Residual Model = XGBoost",

        "Residual Target = y - Inner OOF Prediction",

        "",

        "Lambda结果："
    ]

    for _, row in (
        lambda_df
        .iterrows()
    ):

        lines.append(

            f"alpha={row['alpha']:.2f} | "
            f"MSE={row['mse']:.10f} | "
            f"RMSE={row['rmse']:.10f} | "
            f"MAE={row['mae']:.10f} | "
            f"R2={row['r2']:.6f}"
        )

    lines.extend([

        "",

        f"最佳Alpha={best_alpha:.2f}",

        f"Baseline MSE="
        f"{baseline['mse']:.10f}",

        f"Corrected MSE="
        f"{best['mse']:.10f}",

        f"Relative improvement="
        f"{improvement:.2f}%",

        "",

        f"Residual prediction correlation="
        f"{residual_corr:.6f}",

        "",

        f"输出目录："
        f"{OUT_DIR}",

        "",

        f"总耗时："
        f"{(time.perf_counter()-total_start)/60:.2f}分钟"
    ])

    (
        OUT_DIR
        /
        "summary.txt"
    ).write_text(

        "\n".join(
            lines
        ),

        encoding="utf-8"
    )

    # ========================================================
    # 完成
    # ========================================================

    section(
        "7. 完成"
    )

    print(
        f"结果目录："
        f"{OUT_DIR}"
    )

    print(
        f"最佳Alpha："
        f"{best_alpha:.2f}"
    )

    print(
        f"最佳MSE："
        f"{best['mse']:.10f}"
    )

    print(
        f"相对Baseline："
        f"{improvement:.2f}%"
    )


if __name__ == "__main__":

    main()