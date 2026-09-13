# -*- coding: utf-8 -*-

"""
tool_feature_experiment.py

工业AI：Tool条件特征 + Operation统计特征实验

固定：
    缺失值 = operation_tool_mean
    基础特征 = ExtraTrees Top300
    模型 = XGBoost

实验：
    T01  Baseline
    T02  Tool-Z Top50
    T03  Tool-Z Top100
    T04  Tool-Z Top150
    T05  Tool-Z Top200
    T06  Tool-Z Top300

    A01  Global Anomaly
    A02  Tool Anomaly
    A03  Global + Tool Anomaly

    O01  Operation Mean
    O02  Operation Mean + Std
    O03  Operation Mean + Std + Range
    O04  Operation All

    C01~C08  Tool + Operation + Tool Anomaly

所有Fold内部：
    缺失值拟合
    ExtraTrees选特征
    Tool统计量
    Operation统计量
    异常统计量

均只使用Train Fold。
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
    / "tool_feature_experiment"
)

PLOT_DIR = OUT_DIR / "plots"

RANDOM_STATE = 42
N_SPLITS = 5
BASE_TOP_K = 300


# =========================================================
# 日志
# =========================================================

def log(msg):
    print(
        f"[{time.strftime('%H:%M:%S')}] {msg}",
        flush=True
    )


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


# =========================================================
# 基础工具
# =========================================================

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

    return m.group(1) if m else "UNKNOWN"


def safe_numeric(X):

    return (
        X.copy()
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
        .astype(np.float64)
    )

    valid = (
        valid
        .fillna(medians)
        .astype(np.float64)
    )

    train = train.replace(
        [np.inf, -np.inf],
        0.0
    )

    valid = valid.replace(
        [np.inf, -np.inf],
        0.0
    )

    return train, valid


def remove_duplicate_columns(
    train,
    valid
):

    # 保留第一次出现的列
    mask = ~train.columns.duplicated()

    train = train.loc[:, mask].copy()

    # Valid完全按照Train列对齐
    valid = valid.reindex(
        columns=train.columns
    ).copy()

    return train, valid


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
# 数据
# =========================================================

def prepare_data():

    section("1. 数据准备")

    df = missing_value.load_train(
        DATA_DIR
    )

    target_col = find_col(
        df,
        ["Value", "value", "Y", "y"]
    )

    id_col = find_col(
        df,
        ["ID", "id", "Id"]
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
            "找不到Value/Y目标列。"
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

    log(
        f"原始特征 = {raw_count}"
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
        f"候选特征 = {len(features)}"
    )

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    valid_mask = y.notna()

    df = (
        df
        .loc[valid_mask]
        .reset_index(drop=True)
    )

    y = (
        y
        .loc[valid_mask]
        .reset_index(drop=True)
    )

    X = safe_numeric(
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

    return (
        X,
        y,
        meta,
        features,
        tool_col
    )


# =========================================================
# 缺失值
# =========================================================

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
                    ].to_numpy()
            }
        )

        valid_meta = pd.DataFrame(
            {
                tool_col:
                    meta_valid[
                        "Tool"
                    ].to_numpy()
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


# =========================================================
# 基础特征选择
# =========================================================

def select_base_features(
    X_train,
    y_train
):

    model = ExtraTreesRegressor(

        n_estimators=300,

        max_features="sqrt",

        min_samples_leaf=2,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    importance = (
        pd.Series(
            model.feature_importances_,
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
                BASE_TOP_K,
                len(importance)
            )
        )
        .index
        .tolist()
    )

    return selected


# =========================================================
# Tool统计
# =========================================================

def fit_tool_statistics(
    X,
    tools
):

    tools = (
        pd.Series(tools)
        .astype(str)
        .reset_index(drop=True)
    )

    temp = X.copy()

    temp["__TOOL__"] = (
        tools.to_numpy()
    )

    return {

        "mean":
            temp
            .groupby(
                "__TOOL__"
            )
            .mean(
                numeric_only=True
            ),

        "std":
            temp
            .groupby(
                "__TOOL__"
            )
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
            .fillna(1.0)
    }


# =========================================================
# Tool-Z
# =========================================================

def build_tool_z(
    X,
    tools,
    statistics
):

    tools = (
        pd.Series(tools)
        .astype(str)
        .reset_index(drop=True)
    )

    tool_mean = statistics["mean"]
    tool_std = statistics["std"]

    global_mean = statistics["global_mean"]
    global_std = statistics["global_std"]

    # -----------------------------------------------------
    # 构造Tool均值
    # -----------------------------------------------------

    mean_rows = []

    std_rows = []

    for tool in tools:

        if tool in tool_mean.index:

            mean_rows.append(
                tool_mean.loc[tool]
            )

            std_rows.append(
                tool_std.loc[tool]
            )

        else:

            mean_rows.append(
                global_mean
            )

            std_rows.append(
                global_std
            )

    mean_df = pd.DataFrame(
        mean_rows,
        columns=X.columns
    ).reset_index(drop=True)

    std_df = pd.DataFrame(
        std_rows,
        columns=X.columns
    ).reset_index(drop=True)

    fallback = (
        global_std
        .replace(0, 1.0)
        .fillna(1.0)
    )

    std_df = (
        std_df
        .replace(
            [
                np.inf,
                -np.inf,
                0
            ],
            np.nan
        )
    )

    # 不逐列插入DataFrame
    std_df = std_df.fillna(
        fallback
    )

    z = (
        (
            X.reset_index(drop=True)
            -
            mean_df
        )
        /
        std_df
    )

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

    # -----------------------------------------------------
    # 关键修复：
    # Tool-Z全部增加 TZ_ 前缀
    # -----------------------------------------------------

    z.columns = [
        f"TZ_{col}"
        for col in z.columns
    ]

    abs_z = z.abs()

    summary = pd.DataFrame(
        {
            "AN_tool_max_z":
                abs_z.max(axis=1),

            "AN_tool_top5_mean_z":
                abs_z.apply(
                    lambda row:
                    row.nlargest(
                        min(5, len(row))
                    ).mean(),
                    axis=1
                ),

            "AN_tool_top10_mean_z":
                abs_z.apply(
                    lambda row:
                    row.nlargest(
                        min(10, len(row))
                    ).mean(),
                    axis=1
                ),

            "AN_tool_count_z3":
                abs_z.gt(3).sum(axis=1),

            "AN_tool_count_z5":
                abs_z.gt(5).sum(axis=1)
        }
    )

    return (
        z,
        summary
    )


# =========================================================
# 全局异常
# =========================================================

def build_global_anomaly(
    X
):

    mean = X.mean()

    std = (
        X.std()
        .replace(
            0,
            np.nan
        )
        .fillna(1.0)
    )

    z = (
        (
            X - mean
        )
        /
        std
    )

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

    abs_z = z.abs()

    return pd.DataFrame(
        {

            "AN_global_max_z":
                abs_z.max(axis=1),

            "AN_global_top5_mean_z":
                abs_z.apply(
                    lambda row:
                    row.nlargest(
                        min(5, len(row))
                    ).mean(),
                    axis=1
                ),

            "AN_global_top10_mean_z":
                abs_z.apply(
                    lambda row:
                    row.nlargest(
                        min(10, len(row))
                    ).mean(),
                    axis=1
                ),

            "AN_global_count_z3":
                abs_z.gt(3).sum(axis=1),

            "AN_global_count_z5":
                abs_z.gt(5).sum(axis=1)
        }
    )


# =========================================================
# Operation
# =========================================================

def build_operation_statistics(
    X,
    selected,
    mode
):

    operations = {}

    for feature in selected:

        op = infer_operation(
            feature
        )

        operations.setdefault(
            op,
            []
        )

        operations[
            op
        ].append(feature)

    blocks = []

    for op, cols in operations.items():

        cols = [
            c
            for c in cols
            if c in X.columns
        ]

        if not cols:
            continue

        values = X[cols]

        block = pd.DataFrame(
            index=X.index
        )

        block[
            f"OP_{op}_mean"
        ] = values.mean(
            axis=1
        )

        if mode in {
            "mean_std",
            "mean_std_range",
            "all"
        }:

            block[
                f"OP_{op}_std"
            ] = values.std(
                axis=1
            )

        if mode in {
            "mean_std_range",
            "all"
        }:

            block[
                f"OP_{op}_range"
            ] = (
                values.max(axis=1)
                -
                values.min(axis=1)
            )

        if mode == "all":

            block[
                f"OP_{op}_min"
            ] = values.min(
                axis=1
            )

            block[
                f"OP_{op}_max"
            ] = values.max(
                axis=1
            )

            block[
                f"OP_{op}_median"
            ] = values.median(
                axis=1
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


# =========================================================
# Tool-Z选择
# =========================================================

def select_tool_features(
    tool_z_train,
    y_train,
    top_k
):

    if tool_z_train.empty:

        return (
            [],
            pd.Series(dtype=float)
        )

    model = ExtraTreesRegressor(

        n_estimators=200,

        max_features="sqrt",

        min_samples_leaf=2,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    model.fit(
        tool_z_train,
        y_train
    )

    importance = (
        pd.Series(
            model.feature_importances_,
            index=tool_z_train.columns
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


# =========================================================
# 单实验
# =========================================================

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

    kf = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    fold_rows = []

    oof = np.full(
        len(y),
        np.nan,
        dtype=np.float64
    )

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        kf.split(X),
        start=1
    ):

        # =================================================
        # Fold切分
        # =================================================

        Xtr_raw = (
            X.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        Xva_raw = (
            X.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        ytr = (
            y.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        yva = (
            y.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        mtr = (
            meta.iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        mva = (
            meta.iloc[
                valid_idx
            ]
            .reset_index(
                drop=True
            )
        )

        # =================================================
        # 缺失值
        # =================================================

        Xtr, Xva = preprocess_fold(

            Xtr_raw,

            Xva_raw,

            mtr,

            mva,

            tool_col
        )

        # =================================================
        # 基础Top300
        # =================================================

        base_features = (
            select_base_features(
                Xtr,
                ytr
            )
        )

        train_blocks = [
            Xtr[
                base_features
            ].copy()
        ]

        valid_blocks = [
            Xva[
                base_features
            ].copy()
        ]

        # =================================================
        # Tool-Z
        # =================================================

        tool_top_k = int(
            config.get(
                "tool_top_k",
                0
            )
        )

        selected_tool = []

        if (
            tool_top_k > 0
            and
            tool_col is not None
        ):

            tool_stats = (
                fit_tool_statistics(

                    Xtr[
                        base_features
                    ],

                    mtr[
                        "Tool"
                    ]
                    .to_numpy()
                )
            )

            (
                tool_z_tr,
                tool_anomaly_tr
            ) = build_tool_z(

                Xtr[
                    base_features
                ],

                mtr[
                    "Tool"
                ]
                .to_numpy(),

                tool_stats
            )

            (
                tool_z_va,
                tool_anomaly_va
            ) = build_tool_z(

                Xva[
                    base_features
                ],

                mva[
                    "Tool"
                ]
                .to_numpy(),

                tool_stats
            )

            (
                selected_tool,
                _
            ) = select_tool_features(

                tool_z_tr,

                ytr,

                tool_top_k
            )

            if selected_tool:

                train_blocks.append(
                    tool_z_tr[
                        selected_tool
                    ]
                )

                valid_blocks.append(
                    tool_z_va[
                        selected_tool
                    ]
                )

        # =================================================
        # Tool内部异常摘要
        # =================================================

        if (
            config.get(
                "use_tool_anomaly",
                False
            )
            and
            tool_col is not None
        ):

            if tool_top_k == 0:

                tool_stats = (
                    fit_tool_statistics(

                        Xtr[
                            base_features
                        ],

                        mtr[
                            "Tool"
                        ]
                        .to_numpy()
                    )
                )

                (
                    _,
                    tool_anomaly_tr
                ) = build_tool_z(

                    Xtr[
                        base_features
                    ],

                    mtr[
                        "Tool"
                    ]
                    .to_numpy(),

                    tool_stats
                )

                (
                    _,
                    tool_anomaly_va
                ) = build_tool_z(

                    Xva[
                        base_features
                    ],

                    mva[
                        "Tool"
                    ]
                    .to_numpy(),

                    tool_stats
                )

            train_blocks.append(
                tool_anomaly_tr
            )

            valid_blocks.append(
                tool_anomaly_va
            )

        # =================================================
        # Global异常
        # =================================================

        if config.get(
            "use_global_anomaly",
            False
        ):

            global_tr = (
                build_global_anomaly(
                    Xtr[
                        base_features
                    ]
                )
            )

            # Valid必须使用Train统计
            train_mean = (
                Xtr[
                    base_features
                ].mean()
            )

            train_std = (
                Xtr[
                    base_features
                ]
                .std()
                .replace(
                    0,
                    np.nan
                )
                .fillna(1.0)
            )

            z_valid = (
                (
                    Xva[
                        base_features
                    ]
                    -
                    train_mean
                )
                /
                train_std
            )

            z_valid = (
                z_valid
                .replace(
                    [
                        np.inf,
                        -np.inf
                    ],
                    np.nan
                )
                .fillna(0)
                .clip(
                    -50,
                    50
                )
            )

            abs_valid = z_valid.abs()

            global_va = pd.DataFrame(
                {

                    "AN_global_max_z":
                        abs_valid.max(
                            axis=1
                        ),

                    "AN_global_top5_mean_z":
                        abs_valid.apply(
                            lambda row:
                            row.nlargest(
                                min(
                                    5,
                                    len(row)
                                )
                            ).mean(),
                            axis=1
                        ),

                    "AN_global_top10_mean_z":
                        abs_valid.apply(
                            lambda row:
                            row.nlargest(
                                min(
                                    10,
                                    len(row)
                                )
                            ).mean(),
                            axis=1
                        ),

                    "AN_global_count_z3":
                        abs_valid.gt(
                            3
                        ).sum(axis=1),

                    "AN_global_count_z5":
                        abs_valid.gt(
                            5
                        ).sum(axis=1)
                }
            )

            train_blocks.append(
                global_tr
            )

            valid_blocks.append(
                global_va
            )

        # =================================================
        # Operation
        # =================================================

        operation_mode = (
            config.get(
                "operation_mode",
                "none"
            )
        )

        if operation_mode != "none":

            operation_tr = (
                build_operation_statistics(

                    Xtr[
                        base_features
                    ],

                    base_features,

                    operation_mode
                )
            )

            operation_va = (
                build_operation_statistics(

                    Xva[
                        base_features
                    ],

                    base_features,

                    operation_mode
                )
            )

            train_blocks.append(
                operation_tr
            )

            valid_blocks.append(
                operation_va
            )

        # =================================================
        # 一次性拼接
        # =================================================

        Xtr_final = pd.concat(
            train_blocks,
            axis=1
        )

        Xva_final = pd.concat(
            valid_blocks,
            axis=1
        )

        # =================================================
        # 去重
        # =================================================

        Xtr_final, Xva_final = (
            remove_duplicate_columns(
                Xtr_final,
                Xva_final
            )
        )

        # =================================================
        # 安全
        # =================================================

        Xtr_final, Xva_final = safe_fill(
            Xtr_final,
            Xva_final
        )

        # =================================================
        # 最终硬检查
        # =================================================

        duplicates = (
            Xtr_final.columns[
                Xtr_final.columns.duplicated()
            ]
            .tolist()
        )

        if duplicates:

            raise ValueError(
                f"{name} Fold {fold}仍有重复列："
                f"{duplicates[:20]}"
            )

        if (
            list(Xtr_final.columns)
            !=
            list(Xva_final.columns)
        ):

            raise ValueError(
                f"{name} Fold {fold} Train/Valid列不一致"
            )

        # =================================================
        # 模型
        # =================================================

        model = make_xgb()

        model.fit(

            Xtr_final,

            ytr,

            eval_set=[
                (
                    Xva_final,
                    yva
                )
            ],

            verbose=False
        )

        pred = model.predict(
            Xva_final
        )

        oof[
            valid_idx
        ] = pred

        # =================================================
        # Fold指标
        # =================================================

        mse = mean_squared_error(
            yva,
            pred
        )

        rmse = np.sqrt(
            mse
        )

        mae = mean_absolute_error(
            yva,
            pred
        )

        r2 = r2_score(
            yva,
            pred
        )

        fold_rows.append({

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

            "base_features":
                len(base_features),

            "tool_features":
                len(selected_tool),

            "operation_features":
                (
                    operation_tr.shape[1]
                    if operation_mode != "none"
                    else 0
                ),

            "total_features":
                Xtr_final.shape[1],

            "time_sec":
                time.perf_counter()
                -
                time.perf_counter()
                +
                0
        })

        log(
            f"{name} | "
            f"Fold {fold}/{N_SPLITS} | "
            f"MSE={mse:.8f} | "
            f"Features={Xtr_final.shape[1]}"
        )

    # =====================================================
    # OOF
    # =====================================================

    if np.isnan(oof).any():

        raise RuntimeError(
            f"{name} OOF预测存在NaN"
        )

    mse = mean_squared_error(
        y,
        oof
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y,
        oof
    )

    r2 = r2_score(
        y,
        oof
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    summary = {

        "experiment":
            name,

        "tool_top_k":
            tool_top_k,

        "operation_mode":
            operation_mode,

        "use_global_anomaly":
            config.get(
                "use_global_anomaly",
                False
            ),

        "use_tool_anomaly":
            config.get(
                "use_tool_anomaly",
                False
            ),

        "mse":
            mse,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2,

        "mse_mean_fold":
            fold_df[
                "mse"
            ].mean(),

        "mse_std_fold":
            fold_df[
                "mse"
            ].std(),

        "total_features_mean":
            fold_df[
                "total_features"
            ].mean()
    }

    return (
        summary,
        fold_df,
        oof
    )


# =========================================================
# Tool分布漂移
# =========================================================

def calculate_tool_shift(
    X,
    meta,
    features
):

    global_mean = (
        X[
            features
        ].mean()
    )

    global_std = (
        X[
            features
        ].std()
        .replace(
            0,
            np.nan
        )
    )

    blocks = []

    for tool, indices in (
        meta
        .groupby("Tool")
        .groups
        .items()
    ):

        if len(indices) < 3:
            continue

        tool_mean = (
            X.loc[
                indices,
                features
            ].mean()
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

        block = pd.DataFrame(
            {

                "Tool":
                    tool,

                "feature":
                    shift.index,

                "shift_z":
                    shift.values,

                "operation":
                    [
                        infer_operation(x)
                        for x
                        in shift.index
                    ],

                "n":
                    len(indices)
            }
        )

        block["abs_shift_z"] = (
            block[
                "shift_z"
            ]
            .abs()
        )

        blocks.append(
            block
        )

    if not blocks:

        return pd.DataFrame()

    return (
        pd.concat(
            blocks,
            ignore_index=True
        )
        .sort_values(
            "abs_shift_z",
            ascending=False
        )
        .reset_index(
            drop=True
        )
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
    summary_df,
    sample_df,
    tool_shift_df
):

    section(
        "4. 生成可视化"
    )

    # -----------------------------------------------------
    # 实验
    # -----------------------------------------------------

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
        "Tool Feature Experiments"
    )

    save_plot(
        "01_experiment_mse.png"
    )

    # -----------------------------------------------------
    # Tool Shift
    # -----------------------------------------------------

    if not tool_shift_df.empty:

        temp = (
            tool_shift_df
            .head(30)
            .sort_values(
                "abs_shift_z"
            )
        )

        labels = (
            temp["Tool"].astype(str)
            +
            ":"
            +
            temp["feature"].astype(str)
        )

        plt.figure(
            figsize=(11, 9)
        )

        plt.barh(
            np.arange(
                len(temp)
            ),
            temp[
                "abs_shift_z"
            ]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            labels
        )

        plt.xlabel(
            "|Tool Shift Z|"
        )

        plt.title(
            "Largest Tool Distribution Shifts"
        )

        save_plot(
            "02_tool_shift.png"
        )

    # -----------------------------------------------------
    # Tool anomaly
    # -----------------------------------------------------

    if (
        "tool_anomaly_max_z"
        in sample_df.columns
    ):

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "tool_anomaly_max_z"
            ],
            sample_df[
                "absolute_residual"
            ]
            if "absolute_residual"
            in sample_df.columns
            else sample_df[
                "best_abs_residual"
            ],
            alpha=0.7
        )

        plt.xlabel(
            "Tool Anomaly Max Z"
        )

        plt.ylabel(
            "Absolute Residual"
        )

        plt.title(
            "Tool Anomaly vs Error"
        )

        save_plot(
            "03_tool_anomaly_vs_error.png"
        )

    # -----------------------------------------------------
    # Missing
    # -----------------------------------------------------

    error_col = (
        "absolute_residual"
        if
        "absolute_residual"
        in sample_df.columns
        else "best_abs_residual"
    )

    if "missing_rate" in sample_df.columns:

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "missing_rate"
            ],
            sample_df[
                error_col
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
            "04_missing_vs_error.png"
        )

    # -----------------------------------------------------
    # Zero
    # -----------------------------------------------------

    if "zero_rate" in sample_df.columns:

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "zero_rate"
            ],
            sample_df[
                error_col
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
            "05_zero_vs_error.png"
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
    # Tool分布
    # =====================================================

    tool_shift_df = (
        calculate_tool_shift(
            X,
            meta,
            features
        )
    )

    # =====================================================
    # 实验列表
    # =====================================================

    experiments = [

        (
            "T01_baseline",
            {
                "tool_top_k": 0,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "T02_tool_top50",
            {
                "tool_top_k": 50,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "T03_tool_top100",
            {
                "tool_top_k": 100,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "T04_tool_top150",
            {
                "tool_top_k": 150,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "T05_tool_top200",
            {
                "tool_top_k": 200,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "T06_tool_top300",
            {
                "tool_top_k": 300,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "A01_global_anomaly",
            {
                "tool_top_k": 0,
                "operation_mode": "none",
                "use_global_anomaly": True,
                "use_tool_anomaly": False
            }
        ),

        (
            "A02_tool_anomaly",
            {
                "tool_top_k": 0,
                "operation_mode": "none",
                "use_global_anomaly": False,
                "use_tool_anomaly": True
            }
        ),

        (
            "A03_global_tool_anomaly",
            {
                "tool_top_k": 0,
                "operation_mode": "none",
                "use_global_anomaly": True,
                "use_tool_anomaly": True
            }
        ),

        (
            "O01_operation_mean",
            {
                "tool_top_k": 0,
                "operation_mode": "mean",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "O02_operation_mean_std",
            {
                "tool_top_k": 0,
                "operation_mode": "mean_std",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "O03_operation_mean_std_range",
            {
                "tool_top_k": 0,
                "operation_mode": "mean_std_range",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "O04_operation_all",
            {
                "tool_top_k": 0,
                "operation_mode": "all",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "C01_tool100_operation_mean",
            {
                "tool_top_k": 100,
                "operation_mode": "mean",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "C02_tool100_operation_mean_std",
            {
                "tool_top_k": 100,
                "operation_mode": "mean_std",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "C03_tool100_operation_mean_std_range",
            {
                "tool_top_k": 100,
                "operation_mode": "mean_std_range",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "C04_tool100_operation_all",
            {
                "tool_top_k": 100,
                "operation_mode": "all",
                "use_global_anomaly": False,
                "use_tool_anomaly": False
            }
        ),

        (
            "C05_tool100_anomaly_operation",
            {
                "tool_top_k": 100,
                "operation_mode": "mean_std_range",
                "use_global_anomaly": False,
                "use_tool_anomaly": True
            }
        ),

        (
            "C06_tool150_anomaly_operation",
            {
                "tool_top_k": 150,
                "operation_mode": "mean_std_range",
                "use_global_anomaly": False,
                "use_tool_anomaly": True
            }
        ),

        (
            "C07_tool200_anomaly_operation",
            {
                "tool_top_k": 200,
                "operation_mode": "mean_std_range",
                "use_global_anomaly": False,
                "use_tool_anomaly": True
            }
        ),

        (
            "C08_tool300_anomaly_operation",
            {
                "tool_top_k": 300,
                "operation_mode": "mean_std_range",
                "use_global_anomaly": False,
                "use_tool_anomaly": True
            }
        )
    ]

    # =====================================================
    # 实验
    # =====================================================

    section(
        "2. 开始实验"
    )

    summaries = []

    fold_results = []

    oof_dict = {}

    for name, config in experiments:

        (
            summary,
            fold_df,
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
            fold_df
        )

        oof_dict[
            name
        ] = oof

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

    fold_df = pd.concat(
        fold_results,
        ignore_index=True
    )

    # =====================================================
    # 最优
    # =====================================================

    best = (
        summary_df.iloc[0]
    )

    baseline = (
        summary_df
        .loc[
            summary_df[
                "experiment"
            ]
            ==
            "T01_baseline"
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
        100
    )

    # =====================================================
    # 最优OOF
    # =====================================================

    best_oof = (
        oof_dict[
            best[
                "experiment"
            ]
        ]
    )

    sample_result = (
        meta.copy()
    )

    sample_result["y_true"] = (
        y.to_numpy()
    )

    sample_result["y_pred"] = (
        best_oof
    )

    sample_result["residual"] = (
        sample_result[
            "y_true"
        ]
        -
        sample_result[
            "y_pred"
        ]
    )

    sample_result["absolute_residual"] = (
        sample_result[
            "residual"
        ]
        .abs()
    )

    sample_result["squared_error"] = (
        sample_result[
            "residual"
        ]
        ** 2
    )

    # =====================================================
    # Tool异常统计
    # =====================================================

    if tool_col is not None:

        tool_stats = (
            fit_tool_statistics(
                X,
                meta[
                    "Tool"
                ].to_numpy()
            )
        )

        (
            _,
            tool_anomaly
        ) = build_tool_z(

            X,

            meta[
                "Tool"
            ].to_numpy(),

            tool_stats
        )

        sample_result = pd.concat(
            [
                sample_result,
                tool_anomaly
            ],
            axis=1
        )

    # =====================================================
    # Top错误
    # =====================================================

    top_error = (
        sample_result
        .nlargest(
            50,
            "absolute_residual"
        )
    )

    # =====================================================
    # 保存
    # =====================================================

    summary_df.to_csv(

        OUT_DIR
        /
        "experiment_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )

    fold_df.to_csv(

        OUT_DIR
        /
        "experiment_folds.csv",

        index=False,

        encoding="utf-8-sig"
    )

    sample_result.to_csv(

        OUT_DIR
        /
        "sample_diagnosis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    tool_shift_df.to_csv(

        OUT_DIR
        /
        "tool_distribution_shift.csv",

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

    # =====================================================
    # 图片
    # =====================================================

    make_plots(

        summary_df,

        sample_result,

        tool_shift_df
    )

    # =====================================================
    # 输出
    # =====================================================

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
        f"实验：{best['experiment']}"
    )

    print(
        f"MSE：{best['mse']:.10f}"
    )

    print(
        f"RMSE：{best['rmse']:.10f}"
    )

    print(
        f"MAE：{best['mae']:.10f}"
    )

    print(
        f"R2：{best['r2']:.6f}"
    )

    print(
        f"相对Baseline："
        f"{improvement:.2f}%"
    )

    section(
        "5. Top 20异常样本"
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
                "absolute_residual",
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
    # Summary
    # =====================================================

    lines = [

        "工业AI Tool条件特征实验报告",
        "",

        "固定缺失值：operation_tool_mean",
        f"基础特征：ExtraTrees Top{BASE_TOP_K}",
        "模型：XGBoost",
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
            f"Features="
            f"{row['total_features_mean']:.1f}"
        )

    lines.extend([

        "",
        f"Baseline MSE={baseline['mse']:.10f}",
        f"Best MSE={best['mse']:.10f}",
        f"Relative change={improvement:.2f}%",
        "",
        f"输出目录：{OUT_DIR}",
        f"总耗时："
        f"{(time.perf_counter()-total_start)/60:.2f}分钟"
    ])

    (
        OUT_DIR
        /
        "summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    section(
        "6. 完成"
    )

    print(
        f"结果目录：{OUT_DIR}"
    )

    print(
        f"最佳实验："
        f"{best['experiment']}"
    )

    print(
        f"最佳MSE："
        f"{best['mse']:.10f}"
    )


if __name__ == "__main__":
    main()