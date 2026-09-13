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

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
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
    / "cv_grid_ensemble"
)

PLOT_DIR = OUT_DIR / "plots"

RANDOM_STATE = 42
N_SPLITS = 5

BASE_TOP_K = 300
TOOL_TOP_K = 100
TARGET_SMOOTH_ALPHA = 20.0


# ============================================================
# 网格搜索参数
# ============================================================

XGB_GRID = {
    "n_estimators": [800, 1200],
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.05],
    "min_child_weight": [1, 3],
}

RF_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 8],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}

ET_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 10],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
}


# ============================================================
# 日志
# ============================================================

def log(msg):
    print(
        f"[{time.strftime('%H:%M:%S')}] {msg}",
        flush=True
    )


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# 通用工具
# ============================================================

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_operation(feature):
    m = re.match(
        r"^\s*(\d+)\s*[Xx]",
        str(feature)
    )
    return m.group(1) if m else "UNKNOWN"


def to_numeric_df(df):
    return (
        df
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


def safe_fill(train, valid):

    train = to_numeric_df(train)
    valid = to_numeric_df(valid)

    medians = (
        train
        .median()
        .fillna(0.0)
    )

    train = (
        train
        .fillna(medians)
        .replace(
            [np.inf, -np.inf],
            0.0
        )
        .astype(np.float64)
    )

    valid = (
        valid
        .fillna(medians)
        .replace(
            [np.inf, -np.inf],
            0.0
        )
        .astype(np.float64)
    )

    return train, valid


def align_features(train, valid):

    mask = ~train.columns.duplicated()

    train = (
        train
        .loc[:, mask]
        .copy()
    )

    valid = (
        valid
        .reindex(
            columns=train.columns
        )
        .copy()
    )

    return train, valid


def check_X(X, y=None, name="X"):

    if y is not None and len(X) != len(y):
        raise ValueError(
            f"{name}长度不一致："
            f"{len(X)} vs {len(y)}"
        )

    if X.columns.duplicated().any():
        dup = (
            X.columns[
                X.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            f"{name}存在重复列："
            f"{dup[:20]}"
        )

    values = X.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name}存在NaN/Inf"
        )


def metric_dict(y_true, pred):

    mse = mean_squared_error(
        y_true,
        pred
    )

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(
            mean_absolute_error(
                y_true,
                pred
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                pred
            )
        ),
    }


# ============================================================
# 数据准备
# ============================================================

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
            "找不到目标列 Value"
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

    raw_count = len(features)

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
        .astype(float)
    )

    X = to_numeric_df(
        df[features]
    )

    meta = pd.DataFrame(
        {
            "row_index":
                np.arange(len(df))
        }
    )

    if id_col is not None:
        meta["ID"] = (
            df[id_col]
            .astype(str)
            .to_numpy()
        )
    else:
        meta["ID"] = (
            meta["row_index"]
            .astype(str)
        )

    if tool_col is not None:
        meta["Tool"] = (
            df[tool_col]
            .astype(str)
            .fillna("MISSING_TOOL")
            .to_numpy()
        )
    else:
        meta["Tool"] = "UNKNOWN_TOOL"

    meta["missing_rate"] = (
        X.isna()
        .mean(axis=1)
        .to_numpy()
    )

    meta["zero_rate"] = (
        (X == 0)
        .mean(axis=1)
        .to_numpy()
    )

    log(
        f"训练集 shape = {df.shape}"
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

    return (
        X,
        y,
        meta,
        tool_col
    )


# ============================================================
# Fold缺失值处理
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
                    meta_train["Tool"]
                    .to_numpy()
            }
        )

        valid_meta = pd.DataFrame(
            {
                tool_col:
                    meta_valid["Tool"]
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
# 基础特征选择
# ============================================================

def select_base_features(X, y):

    check_X(
        X,
        y,
        "Base X"
    )

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
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

    # 注意：
    # 这里明确返回两个对象，
    # 与调用端统一。
    return selected, importance


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

    temp["__TOOL__"] = tools

    grouped = (
        temp
        .groupby("__TOOL__")
    )

    return {
        "mean":
            grouped.mean(
                numeric_only=True
            ),

        "std":
            grouped.std(
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
            .fillna(1.0)
    }


# ============================================================
# Tool-Z
# ============================================================

def build_tool_z(
    X,
    tools,
    stats
):

    X = (
        X
        .reset_index(drop=True)
    )

    tools = np.asarray(
        tools,
        dtype=str
    )

    n = len(X)
    p = X.shape[1]

    if len(tools) != n:
        raise ValueError(
            f"Tool-Z行数错误："
            f"{len(tools)} vs {n}"
        )

    columns = list(
        X.columns
    )

    global_mean = np.array(
        stats["global_mean"]
        .reindex(columns)
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        ),
        dtype=np.float64,
        copy=True
    )

    global_std = np.array(
        stats["global_std"]
        .reindex(columns)
        .replace(
            0,
            np.nan
        )
        .fillna(1.0)
        .to_numpy(
            dtype=np.float64
        ),
        dtype=np.float64,
        copy=True
    )

    mean_table = (
        stats["mean"]
        .reindex(
            columns=columns
        )
    )

    std_table = (
        stats["std"]
        .reindex(
            columns=columns
        )
    )

    mean_matrix = np.empty(
        (n, p),
        dtype=np.float64
    )

    std_matrix = np.empty(
        (n, p),
        dtype=np.float64
    )

    for i, tool in enumerate(tools):

        if tool in mean_table.index:

            mean_values = np.array(
                mean_table.loc[tool]
                .to_numpy(
                    dtype=np.float64
                ),
                dtype=np.float64,
                copy=True
            )

            std_values = np.array(
                std_table.loc[tool]
                .to_numpy(
                    dtype=np.float64
                ),
                dtype=np.float64,
                copy=True
            )

            bad_mean = ~np.isfinite(
                mean_values
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

            if bad_mean.any():

                mean_values[
                    bad_mean
                ] = global_mean[
                    bad_mean
                ]

            if bad_std.any():

                std_values[
                    bad_std
                ] = global_std[
                    bad_std
                ]

            mean_matrix[i] = (
                mean_values
            )

            std_matrix[i] = (
                std_values
            )

        else:

            mean_matrix[i] = (
                global_mean
            )

            std_matrix[i] = (
                global_std
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

    z_array[
        ~np.isfinite(
            z_array
        )
    ] = 0.0

    z_array = np.clip(
        z_array,
        -50.0,
        50.0
    )

    return pd.DataFrame(
        z_array,
        columns=[
            f"TZ_{c}"
            for c in columns
        ]
    )


def select_tool_z(
    tool_z,
    y
):

    check_X(
        tool_z,
        y,
        "Tool-Z"
    )

    model = ExtraTreesRegressor(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
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

    return selected


# ============================================================
# Operation特征
# ============================================================

def build_operation_features(
    X,
    features
):

    groups = {}

    for feature in features:

        op = infer_operation(
            feature
        )

        groups.setdefault(
            op,
            []
        )

        groups[op].append(
            feature
        )

    blocks = []

    for op, cols in groups.items():

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
            f"OP_{op}_mean"
        ] = values.mean(
            axis=1
        ).to_numpy()

        block[
            f"OP_{op}_std"
        ] = values.std(
            axis=1
        ).to_numpy()

        block[
            f"OP_{op}_range"
        ] = (
            values.max(axis=1)
            -
            values.min(axis=1)
        ).to_numpy()

        block[
            f"OP_{op}_min"
        ] = values.min(
            axis=1
        ).to_numpy()

        block[
            f"OP_{op}_max"
        ] = values.max(
            axis=1
        ).to_numpy()

        block[
            f"OP_{op}_median"
        ] = values.median(
            axis=1
        ).to_numpy()

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
        pd.Series(train_tools)
        .astype(str)
        .reset_index(drop=True)
    )

    valid_tools = (
        pd.Series(valid_tools)
        .astype(str)
        .reset_index(drop=True)
    )

    categories = sorted(
        train_tools.unique()
    )

    train_data = {}
    valid_data = {}

    for tool in categories:

        col = f"TOOL_{tool}"

        train_data[col] = (
            train_tools
            .eq(tool)
            .astype(float)
            .to_numpy()
        )

        valid_data[col] = (
            valid_tools
            .eq(tool)
            .astype(float)
            .to_numpy()
        )

    return (
        pd.DataFrame(train_data),
        pd.DataFrame(valid_data)
    )


# ============================================================
# Tool Target Encoding
# ============================================================

def fit_target_encoding(
    y,
    tools
):

    y = (
        pd.Series(y)
        .astype(float)
        .reset_index(drop=True)
    )

    tools = (
        pd.Series(tools)
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
            "Tool": tools,
            "Y": y
        }
    )

    grouped = (
        data
        .groupby("Tool")["Y"]
        .agg(
            ["mean", "std", "count"]
        )
    )

    grouped[
        "smooth_mean"
    ] = (

        (
            grouped["count"]
            *
            grouped["mean"]
        )
        +
        TARGET_SMOOTH_ALPHA
        *
        global_mean

    ) / (

        grouped["count"]
        +
        TARGET_SMOOTH_ALPHA
    )

    grouped[
        "relative_mean"
    ] = (
        grouped["smooth_mean"]
        -
        global_mean
    )

    grouped[
        "mean_ratio"
    ] = (
        grouped["smooth_mean"]
        /
        global_mean
        if abs(global_mean) > 1e-12
        else 1.0
    )

    return {
        "table": grouped,
        "global_mean": global_mean
    }


def transform_target_encoding(
    tools,
    encoding
):

    tools = (
        pd.Series(tools)
        .astype(str)
        .reset_index(drop=True)
    )

    table = encoding["table"]

    global_mean = (
        encoding["global_mean"]
    )

    rows = []

    for tool in tools:

        if tool in table.index:

            row = table.loc[tool]

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
                                row["std"]
                            )
                            if pd.notna(
                                row["std"]
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

    return pd.DataFrame(
        rows
    )


# ============================================================
# 构建最终特征
# ============================================================

def build_features(
    X_train,
    X_valid,
    y_train,
    meta_train,
    meta_valid
):

    # --------------------------------------------------------
    # 1. Base Top300
    # --------------------------------------------------------

    base_features, _ = (
        select_base_features(
            X_train,
            y_train
        )
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
    # 2. Tool-Z Top100
    # --------------------------------------------------------

    stats = (
        fit_tool_statistics(
            base_train,
            meta_train[
                "Tool"
            ].to_numpy()
        )
    )

    tool_z_train = (
        build_tool_z(
            base_train,
            meta_train[
                "Tool"
            ].to_numpy(),
            stats
        )
    )

    tool_z_valid = (
        build_tool_z(
            base_valid,
            meta_valid[
                "Tool"
            ].to_numpy(),
            stats
        )
    )

    tool_features = (
        select_tool_z(
            tool_z_train,
            y_train
        )
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
    # 3. Operation All
    # --------------------------------------------------------

    op_train = (
        build_operation_features(
            base_train,
            base_features
        )
    )

    op_valid = (
        build_operation_features(
            base_valid,
            base_features
        )
    )

    if not op_train.empty:

        train_blocks.append(
            op_train
        )

        valid_blocks.append(
            op_valid
        )

    # --------------------------------------------------------
    # 4. Tool One-Hot
    # --------------------------------------------------------

    (
        onehot_train,
        onehot_valid
    ) = build_tool_onehot(

        meta_train["Tool"],

        meta_valid["Tool"]
    )

    train_blocks.append(
        onehot_train
    )

    valid_blocks.append(
        onehot_valid
    )

    # --------------------------------------------------------
    # 5. Tool Target Mean + Std
    # --------------------------------------------------------

    encoding = (
        fit_target_encoding(
            y_train,
            meta_train["Tool"]
        )
    )

    te_train = (
        transform_target_encoding(
            meta_train["Tool"],
            encoding
        )
    )

    te_valid = (
        transform_target_encoding(
            meta_valid["Tool"],
            encoding
        )
    )

    train_blocks.append(
        te_train
    )

    valid_blocks.append(
        te_valid
    )

    # --------------------------------------------------------
    # 合并
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
        align_features(
            train_final,
            valid_final
        )
    )

    train_final, valid_final = (
        safe_fill(
            train_final,
            valid_final
        )
    )

    check_X(
        train_final,
        y_train,
        "Final Train"
    )

    check_X(
        valid_final,
        None,
        "Final Valid"
    )

    if len(train_final) != len(y_train):
        raise RuntimeError(
            "最终Train行数异常"
        )

    if len(valid_final) != len(meta_valid):
        raise RuntimeError(
            "最终Valid行数异常"
        )

    if list(train_final.columns) != list(
        valid_final.columns
    ):

        raise RuntimeError(
            "Train/Valid列不一致"
        )

    return train_final, valid_final


# ============================================================
# 模型
# ============================================================

def make_model(
    model_type,
    params
):

    if model_type == "XGB":

        return XGBRegressor(

            objective=
                "reg:squarederror",

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1,

            subsample=
                0.8,

            colsample_bytree=
                0.6,

            reg_alpha=
                0.01,

            reg_lambda=
                1.0,

            gamma=
                0.0,

            tree_method=
                "hist",

            **params
        )

    if model_type == "RF":

        return RandomForestRegressor(

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1,

            **params
        )

    if model_type == "ET":

        return ExtraTreesRegressor(

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1,

            **params
        )

    raise ValueError(
        f"未知模型：{model_type}"
    )


# ============================================================
# 单组参数 + 5折CV
# ============================================================

def evaluate_config(
    model_type,
    params,
    X,
    y,
    meta,
    tool_col
):

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

        fold_start = (
            time.perf_counter()
        )

        X_train_raw = (
            X.iloc[
                train_idx
            ]
            .reset_index(drop=True)
        )

        X_valid_raw = (
            X.iloc[
                valid_idx
            ]
            .reset_index(drop=True)
        )

        y_train = (
            y.iloc[
                train_idx
            ]
            .reset_index(drop=True)
        )

        y_valid = (
            y.iloc[
                valid_idx
            ]
            .reset_index(drop=True)
        )

        meta_train = (
            meta.iloc[
                train_idx
            ]
            .reset_index(drop=True)
        )

        meta_valid = (
            meta.iloc[
                valid_idx
            ]
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # 缺失值
        # ----------------------------------------------------

        X_train, X_valid = (
            preprocess_fold(

                X_train_raw,

                X_valid_raw,

                meta_train,

                meta_valid,

                tool_col
            )
        )

        # ----------------------------------------------------
        # 特征
        # ----------------------------------------------------

        train_X, valid_X = (
            build_features(

                X_train,

                X_valid,

                y_train,

                meta_train,

                meta_valid
            )
        )

        # ----------------------------------------------------
        # 模型
        # ----------------------------------------------------

        model = make_model(
            model_type,
            params
        )

        # ====================================================
        # XGB和Forest分开fit
        # ====================================================

        if model_type == "XGB":

            model.fit(

                train_X,

                y_train,

                eval_set=[
                    (
                        valid_X,
                        y_valid
                    )
                ],

                verbose=False
            )

        else:

            model.fit(

                train_X,

                y_train
            )

        pred = np.asarray(
            model.predict(
                valid_X
            ),
            dtype=np.float64
        )

        if not np.isfinite(
            pred
        ).all():

            raise RuntimeError(
                f"{model_type} Fold "
                f"{fold}预测异常"
            )

        oof[
            valid_idx
        ] = pred

        metrics = metric_dict(
            y_valid,
            pred
        )

        fold_rows.append(

            {

                "model":
                    model_type,

                "params":
                    str(params),

                "fold":
                    fold,

                **metrics,

                "n_features":
                    train_X.shape[1],

                "time_sec":
                    (
                        time.perf_counter()
                        -
                        fold_start
                    )
            }
        )

        log(

            f"{model_type} | "
            f"Fold {fold}/{N_SPLITS} | "
            f"MSE={metrics['mse']:.8f} | "
            f"RMSE={metrics['rmse']:.8f} | "
            f"Features="
            f"{train_X.shape[1]}"
        )

    if np.isnan(oof).any():

        raise RuntimeError(
            "OOF存在NaN"
        )

    overall = metric_dict(
        y,
        oof
    )

    folds = pd.DataFrame(
        fold_rows
    )

    return {

        "model":
            model_type,

        "params":
            params,

        "oof":
            oof,

        **overall,

        "mse_std":
            folds["mse"].std(),

        "features_mean":
            folds[
                "n_features"
            ].mean(),

        "folds":
            folds
    }


# ============================================================
# 网格搜索
# ============================================================

def run_grid_search(
    X,
    y,
    meta,
    tool_col
):

    section(
        "2. 交叉验证 + 网格搜索"
    )

    grids = [

        (
            "XGB",
            XGB_GRID
        ),

        (
            "RF",
            RF_GRID
        ),

        (
            "ET",
            ET_GRID
        )
    ]

    summary_rows = []

    fold_list = []

    oofs = {}

    config_id = 0

    for model_type, grid in grids:

        combinations = list(
            ParameterGrid(
                grid
            )
        )

        section(
            f"{model_type} "
            f"参数组合 = "
            f"{len(combinations)}"
        )

        for params in combinations:

            config_id += 1

            log(
                f"配置 {config_id} | "
                f"{model_type} | "
                f"{params}"
            )

            result = (
                evaluate_config(

                    model_type,

                    params,

                    X,

                    y,

                    meta,

                    tool_col
                )
            )

            summary_rows.append(

                {

                    "config_id":
                        config_id,

                    "model":
                        model_type,

                    "params":
                        str(params),

                    "mse":
                        result["mse"],

                    "rmse":
                        result["rmse"],

                    "mae":
                        result["mae"],

                    "r2":
                        result["r2"],

                    "mse_std":
                        result["mse_std"],

                    "features_mean":
                        result[
                            "features_mean"
                        ]
                }
            )

            folds = (
                result["folds"]
                .copy()
            )

            folds[
                "config_id"
            ] = config_id

            fold_list.append(
                folds
            )

            oofs[
                config_id
            ] = result["oof"]

            log(

                f"配置完成 | "
                f"MSE="
                f"{result['mse']:.8f} | "
                f"R2="
                f"{result['r2']:.6f}"
            )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "mse"
        )
        .reset_index(
            drop=True
        )
    )

    folds_df = pd.concat(
        fold_list,
        ignore_index=True
    )

    return (
        summary_df,
        folds_df,
        oofs
    )


# ============================================================
# 集成
# ============================================================

def run_ensemble(
    summary_df,
    oofs,
    y
):

    section(
        "3. 集成学习"
    )

    best_models = {}

    for model_type in [
        "XGB",
        "RF",
        "ET"
    ]:

        subset = (
            summary_df
            .loc[
                summary_df[
                    "model"
                ]
                ==
                model_type
            ]
            .sort_values(
                "mse"
            )
        )

        if subset.empty:
            continue

        row = subset.iloc[0]

        best_models[
            model_type
        ] = {

            "row":
                row,

            "oof":
                oofs[
                    int(
                        row[
                            "config_id"
                        ]
                    )
                ]
        }

    rows = []

    # --------------------------------------------------------
    # 单模型
    # --------------------------------------------------------

    for model_type, info in (
        best_models.items()
    ):

        m = metric_dict(
            y,
            info["oof"]
        )

        rows.append(

            {

                "ensemble":
                    model_type,

                "weights":
                    f"{model_type}=1.00",

                **m
            }
        )

    # --------------------------------------------------------
    # 等权
    # --------------------------------------------------------

    preds = [
        info["oof"]
        for info in
        best_models.values()
    ]

    if len(preds) >= 2:

        equal_pred = np.mean(
            np.vstack(preds),
            axis=0
        )

        m = metric_dict(
            y,
            equal_pred
        )

        rows.append(

            {

                "ensemble":
                    "EqualWeight",

                "weights":
                    ", ".join(

                        [
                            f"{name}="
                            f"{1/len(preds):.2f}"

                            for name
                            in best_models
                        ]
                    ),

                **m
            }
        )

    else:

        equal_pred = preds[0]

    # --------------------------------------------------------
    # 三模型加权
    # --------------------------------------------------------

    weighted_pred = None
    best_weights = None
    best_mse = np.inf

    if all(
        x in best_models
        for x in [
            "XGB",
            "RF",
            "ET"
        ]
    ):

        px = (
            best_models[
                "XGB"
            ]["oof"]
        )

        pr = (
            best_models[
                "RF"
            ]["oof"]
        )

        pe = (
            best_models[
                "ET"
            ]["oof"]
        )

        # 0.05步长权重网格
        for wx in np.arange(
            0.0,
            1.0001,
            0.05
        ):

            for wr in np.arange(
                0.0,
                1.0001 - wx,
                0.05
            ):

                we = (
                    1.0
                    -
                    wx
                    -
                    wr
                )

                if we < -1e-9:
                    continue

                pred = (

                    wx * px
                    +
                    wr * pr
                    +
                    we * pe
                )

                mse = (
                    mean_squared_error(
                        y,
                        pred
                    )
                )

                if mse < best_mse:

                    best_mse = mse

                    weighted_pred = (
                        pred
                    )

                    best_weights = (
                        wx,
                        wr,
                        we
                    )

        wx, wr, we = best_weights

        m = metric_dict(
            y,
            weighted_pred
        )

        rows.append(

            {

                "ensemble":
                    "Weighted",

                "weights":
                    (
                        f"XGB={wx:.2f}, "
                        f"RF={wr:.2f}, "
                        f"ET={we:.2f}"
                    ),

                **m
            }
        )

    ensemble_df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "mse"
        )
        .reset_index(
            drop=True
        )
    )

    best = (
        ensemble_df
        .iloc[0]
    )

    if best["ensemble"] == "Weighted":

        best_pred = weighted_pred

    elif best["ensemble"] == "EqualWeight":

        best_pred = equal_pred

    else:

        best_pred = (
            best_models[
                best["ensemble"]
            ]["oof"]
        )

    return {
        "summary":
            ensemble_df,

        "best_models":
            best_models,

        "best":
            best,

        "best_pred":
            best_pred,

        "weights":
            best_weights
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

def save_plot(name):

    path = (
        PLOT_DIR
        /
        name
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
    ensemble_df,
    tool_df,
    y,
    pred
):

    section(
        "4. 生成可视化"
    )

    # --------------------------------------------------------
    # Model MSE
    # --------------------------------------------------------

    temp = (
        summary_df
        .head(20)
        .copy()
        .sort_values(
            "mse"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        np.arange(
            len(temp)
        ),
        temp["mse"]
    )

    labels = [
        f"{m}-{cid}"
        for m, cid
        in zip(
            temp["model"],
            temp["config_id"]
        )
    ]

    plt.yticks(
        np.arange(
            len(temp)
        ),
        labels
    )

    plt.xlabel(
        "OOF MSE"
    )

    plt.title(
        "Grid Search Model MSE"
    )

    save_plot(
        "01_model_mse.png"
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    temp = (
        summary_df
        .head(15)
        .sort_values(
            "rmse"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        np.arange(
            len(temp)
        ),
        temp["rmse"]
    )

    labels = [
        f"{m}-{cid}"
        for m, cid
        in zip(
            temp["model"],
            temp["config_id"]
        )
    ]

    plt.yticks(
        np.arange(
            len(temp)
        ),
        labels
    )

    plt.xlabel(
        "OOF RMSE"
    )

    plt.title(
        "Model RMSE"
    )

    save_plot(
        "02_model_rmse.png"
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    temp = (
        summary_df
        .head(15)
        .sort_values(
            "mae"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        np.arange(
            len(temp)
        ),
        temp["mae"]
    )

    labels = [
        f"{m}-{cid}"
        for m, cid
        in zip(
            temp["model"],
            temp["config_id"]
        )
    ]

    plt.yticks(
        np.arange(
            len(temp)
        ),
        labels
    )

    plt.xlabel(
        "OOF MAE"
    )

    plt.title(
        "Model MAE"
    )

    save_plot(
        "03_model_mae.png"
    )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    temp = (
        summary_df
        .head(15)
        .sort_values(
            "r2"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        np.arange(
            len(temp)
        ),
        temp["r2"]
    )

    labels = [
        f"{m}-{cid}"
        for m, cid
        in zip(
            temp["model"],
            temp["config_id"]
        )
    ]

    plt.yticks(
        np.arange(
            len(temp)
        ),
        labels
    )

    plt.xlabel(
        "OOF R2"
    )

    plt.title(
        "Model R2"
    )

    save_plot(
        "04_model_r2.png"
    )

    # --------------------------------------------------------
    # 集成
    # --------------------------------------------------------

    temp = (
        ensemble_df
        .sort_values(
            "mse"
        )
    )

    plt.figure(
        figsize=(9, 6)
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
        temp["ensemble"]
    )

    plt.xlabel(
        "OOF MSE"
    )

    plt.title(
        "Single Models vs Ensemble"
    )

    save_plot(
        "05_ensemble_comparison.png"
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
            temp["mse"]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp["Tool"]
            .astype(str)
        )

        plt.xlabel(
            "MSE"
        )

        plt.title(
            "Tool MSE"
        )

        save_plot(
            "06_tool_mse.png"
        )

    # --------------------------------------------------------
    # True vs Pred
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        y,
        pred,
        alpha=0.7
    )

    low = min(
        float(np.min(y)),
        float(np.min(pred))
    )

    high = max(
        float(np.max(y)),
        float(np.max(pred))
    )

    plt.plot(
        [low, high],
        [low, high],
        linestyle="--"
    )

    plt.xlabel(
        "True Value"
    )

    plt.ylabel(
        "Predicted Value"
    )

    plt.title(
        "True vs Predicted"
    )

    save_plot(
        "07_true_vs_pred.png"
    )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    residual = (
        y.to_numpy()
        -
        pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        residual,
        bins=40,
        alpha=0.8
    )

    plt.axvline(
        0,
        linestyle="--"
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

    save_plot(
        "08_residual_distribution.png"
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

    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------

    (
        X,
        y,
        meta,
        tool_col
    ) = prepare_data()

    # --------------------------------------------------------
    # Grid Search
    # --------------------------------------------------------

    (
        summary_df,
        folds_df,
        oofs
    ) = run_grid_search(

        X,

        y,

        meta,

        tool_col
    )

    # --------------------------------------------------------
    # 集成
    # --------------------------------------------------------

    ensemble = run_ensemble(

        summary_df,

        oofs,

        y
    )

    ensemble_df = (
        ensemble[
            "summary"
        ]
    )

    best_pred = (
        ensemble[
            "best_pred"
        ]
    )

    # --------------------------------------------------------
    # Tool
    # --------------------------------------------------------

    tool_df = analyze_tools(

        y,

        best_pred,

        meta
    )

    # --------------------------------------------------------
    # OOF结果
    # --------------------------------------------------------

    oof_df = (
        meta.copy()
    )

    oof_df[
        "y_true"
    ] = y.to_numpy()

    for model_type, info in (
        ensemble[
            "best_models"
        ].items()
    ):

        oof_df[
            f"{model_type}_pred"
        ] = info["oof"]

    oof_df[
        "ensemble_pred"
    ] = best_pred

    oof_df[
        "residual"
    ] = (
        oof_df[
            "y_true"
        ]
        -
        best_pred
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    summary_df.to_csv(

        OUT_DIR
        /
        "model_grid_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    folds_df.to_csv(

        OUT_DIR
        /
        "cv_fold_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    ensemble_df.to_csv(

        OUT_DIR
        /
        "ensemble_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    tool_df.to_csv(

        OUT_DIR
        /
        "tool_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    oof_df.to_csv(

        OUT_DIR
        /
        "oof_predictions.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 图片
    # --------------------------------------------------------

    make_plots(

        summary_df,

        ensemble_df,

        tool_df,

        y,

        best_pred
    )

    # ========================================================
    # 输出
    # ========================================================

    section(
        "5. 网格搜索最终排名"
    )

    print(
        summary_df
        .head(20)
        .to_string(
            index=False
        )
    )

    section(
        "6. 集成结果"
    )

    print(
        ensemble_df.to_string(
            index=False
        )
    )

    # ========================================================
    # 最佳单模型
    # ========================================================

    best_single = (
        summary_df.iloc[0]
    )

    best_ensemble = (
        ensemble_df.iloc[0]
    )

    improvement = (

        (
            best_single[
                "mse"
            ]
            -
            best_ensemble[
                "mse"
            ]
        )

        /

        best_single[
            "mse"
        ]

        *

        100.0
    )

    section(
        "7. 当前最优"
    )

    print(
        f"最佳单模型："
        f"{best_single['model']}"
    )

    print(
        f"最佳单模型参数："
        f"{best_single['params']}"
    )

    print(
        f"单模型 MSE："
        f"{best_single['mse']:.10f}"
    )

    print(
        f"单模型 RMSE："
        f"{best_single['rmse']:.10f}"
    )

    print(
        f"单模型 MAE："
        f"{best_single['mae']:.10f}"
    )

    print(
        f"单模型 R2："
        f"{best_single['r2']:.6f}"
    )

    print()

    print(
        f"最佳集成："
        f"{best_ensemble['ensemble']}"
    )

    print(
        f"集成权重："
        f"{best_ensemble['weights']}"
    )

    print(
        f"集成 MSE："
        f"{best_ensemble['mse']:.10f}"
    )

    print(
        f"集成 RMSE："
        f"{best_ensemble['rmse']:.10f}"
    )

    print(
        f"集成 MAE："
        f"{best_ensemble['mae']:.10f}"
    )

    print(
        f"集成 R2："
        f"{best_ensemble['r2']:.6f}"
    )

    print(
        f"相对最佳单模型："
        f"{improvement:.2f}%"
    )

    # ========================================================
    # Tool
    # ========================================================

    section(
        "8. Tool误差"
    )

    print(
        tool_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    elapsed = (
        time.perf_counter()
        -
        total_start
    )

    summary_text = "\n".join(

        [

            "工业AI：交叉验证 + 网格搜索 + 集成学习",

            "",

            "固定特征工程：",

            "缺失值 = operation_tool_mean",

            f"Base = ExtraTrees Top{BASE_TOP_K}",

            f"Tool-Z = Top{TOOL_TOP_K}",

            "Operation = All",

            "Tool One-Hot = True",

            "Tool Target Mean = True",

            "Tool Target Std = True",

            "",

            f"最佳单模型 = "
            f"{best_single['model']}",

            f"最佳单模型参数 = "
            f"{best_single['params']}",

            f"最佳单模型 MSE = "
            f"{best_single['mse']:.10f}",

            f"最佳单模型 RMSE = "
            f"{best_single['rmse']:.10f}",

            f"最佳单模型 MAE = "
            f"{best_single['mae']:.10f}",

            f"最佳单模型 R2 = "
            f"{best_single['r2']:.6f}",

            "",

            f"最佳集成 = "
            f"{best_ensemble['ensemble']}",

            f"集成权重 = "
            f"{best_ensemble['weights']}",

            f"集成 MSE = "
            f"{best_ensemble['mse']:.10f}",

            f"集成 RMSE = "
            f"{best_ensemble['rmse']:.10f}",

            f"集成 MAE = "
            f"{best_ensemble['mae']:.10f}",

            f"集成 R2 = "
            f"{best_ensemble['r2']:.6f}",

            "",

            f"相对最佳单模型 = "
            f"{improvement:.2f}%",

            "",

            f"输出目录 = "
            f"{OUT_DIR}",

            f"总耗时 = "
            f"{elapsed / 60:.2f}分钟"
        ]
    )

    (
        OUT_DIR
        /
        "best_models.txt"
    ).write_text(
        summary_text,
        encoding="utf-8"
    )

    section(
        "9. 完成"
    )

    print(
        f"结果目录："
        f"{OUT_DIR}"
    )

    print(
        f"最佳集成："
        f"{best_ensemble['ensemble']}"
    )

    print(
        f"最佳MSE："
        f"{best_ensemble['mse']:.10f}"
    )

    print(
        f"总耗时："
        f"{elapsed / 60:.2f}分钟"
    )


if __name__ == "__main__":
    main()