# -*- coding: utf-8 -*-

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
    / "tool_identity_experiment"
)

PLOT_DIR = OUT_DIR / "plots"

RANDOM_STATE = 42

N_SPLITS = 5

BASE_TOP_K = 300

TOOL_TOP_K = 100

TARGET_SMOOTH_ALPHA = 20.0


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

    match = re.match(
        r"^\s*(\d+)\s*[Xx]",
        str(feature)
    )

    if match:

        return match.group(1)

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


def safe_fill(
    train,
    valid
):

    train = safe_numeric(
        train
    )

    valid = safe_numeric(
        valid
    )

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

    return train, valid


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

    duplicate = (
        X.columns[
            X.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate:

        raise ValueError(
            f"{name}存在重复列："
            f"{duplicate[:20]}"
        )

    values = X.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            f"{name}中存在NaN/Inf"
        )


# ============================================================
# XGBoost
# ============================================================

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

    valid_mask = y.notna()

    df = (
        df
        .loc[
            valid_mask
        ]
        .reset_index(
            drop=True
        )
    )

    y = (
        y
        .loc[
            valid_mask
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
        ] = (
            "UNKNOWN_TOOL"
        )

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
# Fold预处理
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
        .reset_index(
            drop=True
        )
        .copy()
    )

    X_valid = (
        X_valid
        .reset_index(
            drop=True
        )
        .copy()
    )

    meta_train = (
        meta_train
        .reset_index(
            drop=True
        )
        .copy()
    )

    meta_valid = (
        meta_valid
        .reset_index(
            drop=True
        )
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
# Base特征
# ============================================================

def select_base_features(
    X,
    y
):

    check_matrix(
        X,
        y,
        "Base Train"
    )

    model = (
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
            "Tool统计长度错误："
            f"tools={len(tools)}, "
            f"X={len(X)}"
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

    return {

        "tool_mean":
            grouped
            .mean(
                numeric_only=True
            ),

        "tool_std":
            grouped
            .std(
                numeric_only=True
            ),

        "global_mean":
            X.mean(),

        "global_std":
            X.std()
            .replace(
                0,
                np.nan
            )
            .fillna(
                1.0
            )
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
            "Tool-Z输入长度不一致："
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
        .fillna(
            0.0
        )
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
        .fillna(
            1.0
        )
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
                tool_mean.loc[
                    tool,
                    columns
                ].to_numpy(
                    dtype=np.float64
                ),
                dtype=np.float64,
                copy=True
            )

            # 这里必须copy=True
            # 避免read-only数组

            std_values = np.array(
                tool_std.loc[
                    tool,
                    columns
                ].to_numpy(
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

    if mean_matrix.shape != (
        n_samples,
        n_features
    ):

        raise RuntimeError(
            f"Tool mean shape错误："
            f"{mean_matrix.shape}"
        )

    if std_matrix.shape != (
        n_samples,
        n_features
    ):

        raise RuntimeError(
            f"Tool std shape错误："
            f"{std_matrix.shape}"
        )

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

    z_columns = [
        f"TZ_{col}"
        for col in columns
    ]

    z = pd.DataFrame(
        z_array,
        columns=z_columns
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
                )
                .sum(axis=1),

            "AN_tool_count_z5":
                (
                    abs_z > 5
                )
                .sum(axis=1)
        }
    )

    return (
        z,
        anomaly
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
            index=X.index
        )

        block[
            f"OP_{operation}_mean"
        ] = (
            values
            .mean(axis=1)
        )

        block[
            f"OP_{operation}_std"
        ] = (
            values
            .std(axis=1)
        )

        block[
            f"OP_{operation}_range"
        ] = (
            values.max(axis=1)
            -
            values.min(axis=1)
        )

        block[
            f"OP_{operation}_min"
        ] = (
            values.min(axis=1)
        )

        block[
            f"OP_{operation}_max"
        ] = (
            values.max(axis=1)
        )

        block[
            f"OP_{operation}_median"
        ] = (
            values.median(axis=1)
        )

        blocks.append(
            block
        )

    if not blocks:

        return pd.DataFrame(
            index=X.index
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
# Tool Target Encoding
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
            "Target Encoding长度错误："
            f"y={len(y)}, "
            f"tools={len(tools)}"
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

    if abs(global_mean) > 1e-12:

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
# 构建最终特征
# ============================================================

def build_final_features(
    X_train,
    X_valid,
    y_train,
    meta_train,
    meta_valid,
    config
):

    # --------------------------------------------------------
    # Base
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
        ].to_numpy(),

        tool_stats
    )

    (
        tool_z_valid,
        tool_anomaly_valid
    ) = build_tool_z(

        base_valid,

        meta_valid[
            "Tool"
        ].to_numpy(),

        tool_stats
    )

    (
        tool_columns,
        tool_importance
    ) = select_tool_z_features(

        tool_z_train,

        y_train,

        TOOL_TOP_K
    )

    if tool_columns:

        train_blocks.append(
            tool_z_train[
                tool_columns
            ]
        )

        valid_blocks.append(
            tool_z_valid[
                tool_columns
            ]
        )

    # --------------------------------------------------------
    # Operation All
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

    if config[
        "tool_onehot"
    ]:

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

    if config[
        "target_encoding"
    ]:

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
                    config[
                        "target_std"
                    ]
            )
        )

        te_valid = (
            transform_tool_target_encoding(

                meta_valid[
                    "Tool"
                ],

                encoding,

                include_std=
                    config[
                        "target_std"
                    ]
            )
        )

        train_blocks.append(
            te_train
        )

        valid_blocks.append(
            te_valid
        )

    # --------------------------------------------------------
    # Tool anomaly
    # --------------------------------------------------------

    if config[
        "tool_anomaly"
    ]:

        train_blocks.append(
            tool_anomaly_train
        )

        valid_blocks.append(
            tool_anomaly_valid
        )

    # --------------------------------------------------------
    # Global anomaly
    # --------------------------------------------------------

    if config[
        "global_anomaly"
    ]:

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
    # 一次性合并
    # --------------------------------------------------------

    train_final = pd.concat(
        train_blocks,
        axis=1
    )

    valid_final = pd.concat(
        valid_blocks,
        axis=1
    )

    # --------------------------------------------------------
    # 去重
    # --------------------------------------------------------

    train_final, valid_final = (
        remove_duplicate_columns(

            train_final,

            valid_final
        )
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    train_final, valid_final = safe_fill(
        train_final,
        valid_final
    )

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

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
            f"Final Train长度错误："
            f"{train_final.shape}"
        )

    if len(
        valid_final
    ) != len(
        meta_valid
    ):

        raise RuntimeError(
            f"Final Valid长度错误："
            f"{valid_final.shape}"
        )

    if list(
        train_final.columns
    ) != list(
        valid_final.columns
    ):

        raise RuntimeError(
            "Train/Valid特征列不一致"
        )

    return {

        "train":
            train_final,

        "valid":
            valid_final,

        "base_features":
            base_features,

        "base_importance":
            base_importance,

        "tool_features":
            tool_columns,

        "tool_importance":
            tool_importance
    }


# ============================================================
# Tool-Z筛选
# ============================================================

def select_tool_z_features(
    tool_z,
    y,
    top_k
):

    if tool_z.empty:

        return (
            [],
            pd.Series(
                dtype=float
            )
        )

    if len(
        tool_z
    ) != len(
        y
    ):

        raise ValueError(
            "Tool-Z选择时长度不一致："
            f"X={len(tool_z)}, "
            f"y={len(y)}"
        )

    check_matrix(
        tool_z,
        y,
        "Tool-Z"
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
                top_k,
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
# 全局异常
# ============================================================

def build_global_anomaly(
    X_train,
    X_valid
):

    mean = (
        X_train.mean()
    )

    std = (
        X_train.std()
        .replace(
            0,
            np.nan
        )
        .fillna(
            1.0
        )
    )

    z = (
        X_valid
        -
        mean
    ) / std

    z = (
        z
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .fillna(0.0)
        .clip(
            -50,
            50
        )
    )

    values = np.abs(
        z.to_numpy(
            dtype=np.float64
        )
    )

    n_features = values.shape[1]

    k5 = min(
        5,
        n_features
    )

    k10 = min(
        10,
        n_features
    )

    top5 = np.partition(
        values,
        n_features - k5,
        axis=1
    )[
        :,
        n_features - k5:
    ]

    top10 = np.partition(
        values,
        n_features - k10,
        axis=1
    )[
        :,
        n_features - k10:
    ]

    return pd.DataFrame(
        {

            "AN_global_max_z":
                values.max(
                    axis=1
                ),

            "AN_global_top5_mean_z":
                top5.mean(
                    axis=1
                ),

            "AN_global_top10_mean_z":
                top10.mean(
                    axis=1
                ),

            "AN_global_count_z3":
                (
                    values > 3
                ).sum(axis=1),

            "AN_global_count_z5":
                (
                    values > 5
                ).sum(axis=1)
        }
    )


# ============================================================
# 单实验
# ============================================================

def run_experiment(
    name,
    config,
    X,
    y,
    meta,
    tool_col
):

    log(
        f"实验：{name}"
    )

    kfold = KFold(

        n_splits=
            N_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE
    )

    oof = np.full(
        len(y),
        np.nan,
        dtype=np.float64
    )

    fold_rows = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(

        kfold.split(X),

        start=1
    ):

        start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        X_train_raw = (
            X.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        X_valid_raw = (
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

        # ----------------------------------------------------
        # 缺失值
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 构造特征
        # ----------------------------------------------------

        feature_data = (
            build_final_features(

                X_train,

                X_valid,

                y_train,

                meta_train,

                meta_valid,

                config
            )
        )

        train_final = (
            feature_data[
                "train"
            ]
        )

        valid_final = (
            feature_data[
                "valid"
            ]
        )

        # ----------------------------------------------------
        # 模型
        # ----------------------------------------------------

        model = make_xgb()

        model.fit(

            train_final,

            y_train,

            eval_set=[
                (
                    valid_final,
                    y_valid
                )
            ],

            verbose=False
        )

        pred = model.predict(
            valid_final
        )

        if not np.isfinite(
            pred
        ).all():

            raise RuntimeError(
                f"{name} Fold {fold}预测出现NaN/Inf"
            )

        oof[
            valid_idx
        ] = pred

        # ----------------------------------------------------
        # 指标
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Tool O
        # ----------------------------------------------------

        tool_o_mask = (
            meta_valid[
                "Tool"
            ]
            .astype(str)
            .eq("O")
            .to_numpy()
        )

        if tool_o_mask.any():

            tool_o_mse = (
                mean_squared_error(
                    y_valid[
                        tool_o_mask
                    ],
                    pred[
                        tool_o_mask
                    ]
                )
            )

            tool_o_mae = (
                mean_absolute_error(
                    y_valid[
                        tool_o_mask
                    ],
                    pred[
                        tool_o_mask
                    ]
                )
            )

        else:

            tool_o_mse = np.nan

            tool_o_mae = np.nan

        fold_rows.append(
            {

                "experiment":
                    name,

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

                "tool_o_mse":
                    tool_o_mse,

                "tool_o_mae":
                    tool_o_mae,

                "base_features":
                    len(
                        feature_data[
                            "base_features"
                        ]
                    ),

                "tool_z_features":
                    len(
                        feature_data[
                            "tool_features"
                        ]
                    ),

                "operation_features":
                    sum(
                        col.startswith(
                            "OP_"
                        )
                        for col
                        in train_final.columns
                    ),

                "total_features":
                    train_final.shape[1],

                "time_sec":
                    (
                        time.perf_counter()
                        -
                        start_time
                    )
            }
        )

        log(
            f"{name} | "
            f"Fold {fold}/{N_SPLITS} | "
            f"MSE={mse:.8f} | "
            f"RMSE={rmse:.8f} | "
            f"ToolO={tool_o_mse:.8f} | "
            f"Features={train_final.shape[1]}"
        )

    # ========================================================
    # OOF检查
    # ========================================================

    if np.isnan(
        oof
    ).any():

        raise RuntimeError(
            f"{name} OOF预测存在NaN"
        )

    mse = (
        mean_squared_error(
            y,
            oof
        )
    )

    rmse = np.sqrt(
        mse
    )

    mae = (
        mean_absolute_error(
            y,
            oof
        )
    )

    r2 = (
        r2_score(
            y,
            oof
        )
    )

    tool_o_mask = (
        meta[
            "Tool"
        ]
        .astype(str)
        .eq("O")
        .to_numpy()
    )

    if tool_o_mask.any():

        tool_o_mse = (
            mean_squared_error(
                y.to_numpy()[
                    tool_o_mask
                ],
                oof[
                    tool_o_mask
                ]
            )
        )

        tool_o_mae = (
            mean_absolute_error(
                y.to_numpy()[
                    tool_o_mask
                ],
                oof[
                    tool_o_mask
                ]
            )
        )

    else:

        tool_o_mse = np.nan

        tool_o_mae = np.nan

    folds = pd.DataFrame(
        fold_rows
    )

    summary = {

        "experiment":
            name,

        "tool_onehot":
            config[
                "tool_onehot"
            ],

        "target_encoding":
            config[
                "target_encoding"
            ],

        "target_std":
            config[
                "target_std"
            ],

        "tool_anomaly":
            config[
                "tool_anomaly"
            ],

        "global_anomaly":
            config[
                "global_anomaly"
            ],

        "mse":
            mse,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2,

        "mse_std":
            folds[
                "mse"
            ].std(),

        "tool_o_mse":
            tool_o_mse,

        "tool_o_mae":
            tool_o_mae,

        "features_mean":
            folds[
                "total_features"
            ].mean()
    }

    return (
        summary,
        folds,
        oof
    )


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

    result = (
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

    return result


# ============================================================
# 图片
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
    summary_df,
    tool_df,
    sample_df
):

    section(
        "4. 生成可视化"
    )

    # --------------------------------------------------------
    # 实验MSE
    # --------------------------------------------------------

    temp = (
        summary_df
        .sort_values(
            "mse"
        )
    )

    plt.figure(
        figsize=(11, 9)
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
            "experiment"
        ]
    )

    plt.xlabel(
        "OOF MSE"
    )

    plt.title(
        "Tool Identity Experiments"
    )

    save_plot(
        "01_experiment_mse.png"
    )

    # --------------------------------------------------------
    # Tool MSE
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
            "02_tool_mse.png"
        )

    # --------------------------------------------------------
    # Y vs Error
    # --------------------------------------------------------

    if not sample_df.empty:

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "y_true"
            ],
            sample_df[
                "absolute_error"
            ],
            alpha=0.7
        )

        plt.xlabel(
            "Y"
        )

        plt.ylabel(
            "Absolute Error"
        )

        plt.title(
            "Y vs Absolute Error"
        )

        save_plot(
            "03_y_vs_error.png"
        )

    # --------------------------------------------------------
    # Tool误差
    # --------------------------------------------------------

    if not sample_df.empty:

        grouped = (
            sample_df
            .groupby(
                "Tool"
            )[
                "absolute_error"
            ]
            .mean()
            .sort_values()
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.barh(
            np.arange(
                len(grouped)
            ),
            grouped.values
        )

        plt.yticks(
            np.arange(
                len(grouped)
            ),
            grouped.index
            .astype(str)
        )

        plt.xlabel(
            "Mean Absolute Error"
        )

        plt.title(
            "Mean Absolute Error by Tool"
        )

        save_plot(
            "04_error_by_tool.png"
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
    # 实验配置
    # ========================================================

    experiments = [

        (
            "E0_baseline",

            {
                "tool_onehot":
                    False,

                "target_encoding":
                    False,

                "target_std":
                    False,

                "tool_anomaly":
                    False,

                "global_anomaly":
                    False
            }
        ),

        (
            "E1_tool_onehot",

            {
                "tool_onehot":
                    True,

                "target_encoding":
                    False,

                "target_std":
                    False,

                "tool_anomaly":
                    False,

                "global_anomaly":
                    False
            }
        ),

        (
            "E2_tool_target_mean",

            {
                "tool_onehot":
                    True,

                "target_encoding":
                    True,

                "target_std":
                    False,

                "tool_anomaly":
                    False,

                "global_anomaly":
                    False
            }
        ),

        (
            "E3_tool_target_mean_std",

            {
                "tool_onehot":
                    True,

                "target_encoding":
                    True,

                "target_std":
                    True,

                "tool_anomaly":
                    False,

                "global_anomaly":
                    False
            }
        ),

        (
            "E4_tool_target_anomaly",

            {
                "tool_onehot":
                    True,

                "target_encoding":
                    True,

                "target_std":
                    True,

                "tool_anomaly":
                    True,

                "global_anomaly":
                    False
            }
        ),

        (
            "E5_tool_global_anomaly",

            {
                "tool_onehot":
                    True,

                "target_encoding":
                    True,

                "target_std":
                    True,

                "tool_anomaly":
                    True,

                "global_anomaly":
                    True
            }
        )
    ]

    # ========================================================
    # 实验
    # ========================================================

    section(
        "2. Tool身份实验"
    )

    summaries = []

    fold_results = []

    oofs = {}

    for name, config in experiments:

        (
            summary,
            folds,
            oof
        ) = run_experiment(

            name,

            config,

            X,

            y,

            meta,

            tool_col
        )

        summaries.append(
            summary
        )

        fold_results.append(
            folds
        )

        oofs[
            name
        ] = oof

    # ========================================================
    # 排名
    # ========================================================

    summary_df = (
        pd.DataFrame(
            summaries
        )
        .sort_values(
            "mse"
        )
        .reset_index(
            drop=True
        )
    )

    folds_df = pd.concat(
        fold_results,
        ignore_index=True
    )

    # ========================================================
    # 当前最佳
    # ========================================================

    best = (
        summary_df
        .iloc[0]
    )

    baseline = (
        summary_df
        .loc[
            summary_df[
                "experiment"
            ]
            ==
            "E0_baseline"
        ]
        .iloc[0]
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

    best_name = (
        best[
            "experiment"
        ]
    )

    best_oof = (
        oofs[
            best_name
        ]
    )

    # ========================================================
    # Tool
    # ========================================================

    tool_df = analyze_tools(
        y,
        best_oof,
        meta
    )

    # ========================================================
    # Sample
    # ========================================================

    sample_df = (
        meta.copy()
    )

    sample_df[
        "y_true"
    ] = y.to_numpy()

    sample_df[
        "y_pred"
    ] = best_oof

    sample_df[
        "residual"
    ] = (

        sample_df[
            "y_true"
        ]

        -

        sample_df[
            "y_pred"
        ]
    )

    sample_df[
        "absolute_error"
    ] = (
        sample_df[
            "residual"
        ]
        .abs()
    )

    sample_df[
        "squared_error"
    ] = (
        sample_df[
            "residual"
        ]
        ** 2
    )

    sample_df[
        "error_percentile"
    ] = (
        sample_df[
            "absolute_error"
        ]
        .rank(
            pct=True
        )
    )

    # ========================================================
    # Hard
    # ========================================================

    hard_threshold = (
        sample_df[
            "absolute_error"
        ]
        .quantile(
            0.90
        )
    )

    sample_df[
        "hard_sample"
    ] = (
        sample_df[
            "absolute_error"
        ]
        >=
        hard_threshold
    )

    top_error = (
        sample_df
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(50)
    )

    # ========================================================
    # Target Encoding说明
    # ========================================================

    encoding_full = (
        fit_tool_target_encoding(

            y,

            meta[
                "Tool"
            ]
        )
    )

    encoding_df = (
        encoding_full[
            "table"
        ]
        .reset_index()
    )

    # ========================================================
    # 保存
    # ========================================================

    summary_df.to_csv(

        OUT_DIR
        /
        "experiment_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )

    folds_df.to_csv(

        OUT_DIR
        /
        "experiment_folds.csv",

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

    sample_df.to_csv(

        OUT_DIR
        /
        "sample_diagnosis.csv",

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

    encoding_df.to_csv(

        OUT_DIR
        /
        "tool_target_encoding.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # 可视化
    # ========================================================

    make_plots(

        summary_df,

        tool_df,

        sample_df
    )

    # ========================================================
    # 控制台
    # ========================================================

    section(
        "3. 实验排名"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    section(
        "4. 当前最优"
    )

    print(
        f"实验：{best_name}"
    )

    print(
        f"MSE："
        f"{best['mse']:.10f}"
    )

    print(
        f"RMSE："
        f"{best['rmse']:.10f}"
    )

    print(
        f"MAE："
        f"{best['mae']:.10f}"
    )

    print(
        f"R2："
        f"{best['r2']:.6f}"
    )

    print(
        f"Tool O MSE："
        f"{best['tool_o_mse']:.10f}"
    )

    print(
        f"相对Baseline："
        f"{improvement:.2f}%"
    )

    section(
        "5. Tool分析"
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
                "y_pred",
                "residual",
                "absolute_error",
                "missing_rate",
                "zero_rate",
                "hard_sample"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    section(
        "7. 完成"
    )

    print(
        f"结果目录："
        f"{OUT_DIR}"
    )

    print(
        f"最佳实验："
        f"{best_name}"
    )

    print(
        f"最佳MSE："
        f"{best['mse']:.10f}"
    )

    print(
        f"总耗时："
        f"{(time.perf_counter()-total_start)/60:.2f}分钟"
    )

    # ========================================================
    # Summary
    # ========================================================

    lines = [

        "工业AI Tool身份实验报告",

        "",

        "固定基础方案：",

        "缺失值 = operation_tool_mean",

        f"基础特征 = ExtraTrees Top{BASE_TOP_K}",

        f"Tool-Z = Top{TOOL_TOP_K}",

        "Operation = All",

        "模型 = XGBoost",

        "",

        "实验结果："
    ]

    for _, row in (
        summary_df
        .iterrows()
    ):

        lines.append(

            f"{row['experiment']} | "

            f"MSE={row['mse']:.10f} | "

            f"RMSE={row['rmse']:.10f} | "

            f"MAE={row['mae']:.10f} | "

            f"R2={row['r2']:.6f} | "

            f"ToolO_MSE="
            f"{row['tool_o_mse']:.10f} | "

            f"Features="
            f"{row['features_mean']:.1f}"
        )

    lines.extend([

        "",

        f"Baseline MSE="
        f"{baseline['mse']:.10f}",

        f"Best MSE="
        f"{best['mse']:.10f}",

        f"Relative improvement="
        f"{improvement:.2f}%",

        "",

        f"Hard Sample阈值="
        f"{hard_threshold:.10f}",

        f"Hard Sample数量="
        f"{int(sample_df['hard_sample'].sum())}",

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


if __name__ == "__main__":

    main()