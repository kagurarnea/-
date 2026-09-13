# -*- coding: utf-8 -*-

"""
hard_sample_experiment.py

============================================================
工业AI：难样本 / Tool专项模型实验
============================================================

当前最佳基础方案：
    缺失值：
        operation_tool_mean

    基础特征：
        ExtraTrees Top300

    Tool条件：
        Tool-Z Top100

    Operation：
        全部统计特征

    全局模型：
        XGBoost

============================================================
本阶段目标
============================================================

1. 重新验证当前最佳全局方案 C04
2. 单独分析各 Tool 的残差
3. 分析 Hard Sample 是否集中在某些 Tool
4. 对 Tool O 建立专家模型
5. 比较：

    E0  全局模型

    E1  Tool O 专家模型

    E2  Tool O:
        Global + Expert
        alpha = 0.25

    E3  alpha = 0.50

    E4  alpha = 0.75

    E5  alpha = 1.00

6. 分析不同Tool是否存在明显不同的误差机制

============================================================
核心思想
============================================================

正常样本：
    使用Global XGBoost

Tool O：
    Global模型预测
             +
    Tool O Expert模型预测

最终：

    pred =
        (1-alpha) * global_pred
        +
        alpha * expert_pred

这样避免Tool O专家模型完全替换全局模型
造成小样本过拟合。

============================================================
严格避免数据泄漏
============================================================

每个Fold：

Train Fold
    ↓
缺失值
    ↓
ExtraTrees Top300
    ↓
Tool-Z Top100
    ↓
Operation统计
    ↓
Global model

同时：

Train Fold中的Tool O
    ↓
Expert model

Valid Fold
    ↓
根据Tool决定是否使用Expert

============================================================
输出
============================================================

data/
└── preprocess_missing/
    └── hard_sample_experiment/
        ├── experiment_summary.csv
        ├── experiment_folds.csv
        ├── tool_o_analysis.csv
        ├── tool_analysis.csv
        ├── hard_samples.csv
        ├── oof_detail.csv
        ├── summary.txt
        └── plots/
            ├── 01_experiment_mse.png
            ├── 02_tool_mse.png
            ├── 03_tool_sample_count.png
            ├── 04_tool_o_global_vs_expert.png
            ├── 05_hard_sample_distribution.png
            ├── 06_error_by_y.png
            └── 07_error_by_missing.png
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
    / "hard_sample_experiment"
)

PLOT_DIR = (
    OUT_DIR
    / "plots"
)

RANDOM_STATE = 42

N_SPLITS = 5

BASE_TOP_K = 300

TOOL_TOP_K = 100

SPECIAL_TOOL = "O"

# Tool O专家模型至少需要的训练样本
MIN_SPECIAL_SAMPLES = 20


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
    print("=" * 100)
    print(title)
    print("=" * 100)


# =========================================================
# 基础工具
# =========================================================

def find_col(
    df,
    names
):

    for name in names:

        if name in df.columns:

            return name

    return None


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


def safe_numeric(X):

    return (
        X
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
    )

    valid = (
        valid
        .fillna(medians)
    )

    train = (
        train
        .replace(
            [
                np.inf,
                -np.inf
            ],
            0.0
        )
        .astype(np.float64)
    )

    valid = (
        valid
        .replace(
            [
                np.inf,
                -np.inf
            ],
            0.0
        )
        .astype(np.float64)
    )

    return train, valid


def remove_duplicate_columns(
    train,
    valid
):

    keep = ~train.columns.duplicated()

    train = (
        train
        .loc[:, keep]
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
        f"{len(features) + len(all_nan) + len(constant) + len(duplicate)}"
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

        meta["ID"] = (
            df[
                id_col
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        meta["ID"] = (
            meta[
                "row_index"
            ]
            .astype(str)
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
# ExtraTrees Top300
# =========================================================

def select_base_features(
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

    return (
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


# =========================================================
# Tool统计
# =========================================================

def fit_tool_statistics(
    X,
    tools
):

    tools = (
        pd.Series(
            tools
        )
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    temp = X.copy()

    temp["__TOOL__"] = (
        tools.to_numpy()
    )

    tool_mean = (
        temp
        .groupby(
            "__TOOL__"
        )
        .mean(
            numeric_only=True
        )
    )

    tool_std = (
        temp
        .groupby(
            "__TOOL__"
        )
        .std(
            numeric_only=True
        )
    )

    global_mean = (
        X.mean()
    )

    global_std = (
        X.std()
        .replace(
            0,
            np.nan
        )
        .fillna(
            1.0
        )
    )

    return {

        "tool_mean":
            tool_mean,

        "tool_std":
            tool_std,

        "global_mean":
            global_mean,

        "global_std":
            global_std
    }


# =========================================================
# Tool-Z
# =========================================================

def build_tool_z(
    X,
    tools,
    stats
):

    tools = (
        pd.Series(
            tools
        )
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    tool_mean = stats[
        "tool_mean"
    ]

    tool_std = stats[
        "tool_std"
    ]

    global_mean = stats[
        "global_mean"
    ]

    global_std = stats[
        "global_std"
    ]

    mean_rows = []

    std_rows = []

    for tool in tools:

        if tool in tool_mean.index:

            mean_rows.append(
                tool_mean.loc[
                    tool
                ]
            )

            std_rows.append(
                tool_std.loc[
                    tool
                ]
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
    ).reset_index(
        drop=True
    )

    std_df = pd.DataFrame(
        std_rows,
        columns=X.columns
    ).reset_index(
        drop=True
    )

    fallback = (
        global_std
        .replace(
            0,
            1.0
        )
        .fillna(
            1.0
        )
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
        .fillna(
            fallback
        )
    )

    z = (

        X.reset_index(
            drop=True
        )

        -
        mean_df

    ) / std_df

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

    # 关键：
    # 与原始特征使用不同名字
    z.columns = [
        f"TZ_{c}"
        for c in z.columns
    ]

    abs_z = z.abs()

    summary = pd.DataFrame(
        {

            "AN_tool_max_z":
                abs_z.max(
                    axis=1
                ),

            "AN_tool_top5_mean_z":
                abs_z.apply(
                    lambda row:
                    row.nlargest(
                        min(
                            5,
                            len(row)
                        )
                    ).mean(),
                    axis=1
                ),

            "AN_tool_top10_mean_z":
                abs_z.apply(
                    lambda row:
                    row.nlargest(
                        min(
                            10,
                            len(row)
                        )
                    ).mean(),
                    axis=1
                ),

            "AN_tool_count_z3":
                abs_z.gt(3)
                .sum(axis=1),

            "AN_tool_count_z5":
                abs_z.gt(5)
                .sum(axis=1)
        }
    )

    return (
        z,
        summary
    )


# =========================================================
# Operation
# =========================================================

def build_operation_statistics(
    X,
    selected,
    mode="all"
):

    mapping = {}

    for feature in selected:

        op = infer_operation(
            feature
        )

        mapping.setdefault(
            op,
            []
        )

        mapping[
            op
        ].append(
            feature
        )

    blocks = []

    for op, cols in mapping.items():

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

        # mean
        block[
            f"OP_{op}_mean"
        ] = values.mean(
            axis=1
        )

        # std
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

        # range
        if mode in {
            "mean_std_range",
            "all"
        }:

            block[
                f"OP_{op}_range"
            ] = (

                values.max(
                    axis=1
                )

                -

                values.min(
                    axis=1
                )
            )

        # all
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
# 构建最终特征
# =========================================================

def build_final_features(
    X_train,
    X_valid,
    y_train,
    meta_train,
    meta_valid,
    tool_col
):

    # -----------------------------------------------------
    # Top300
    # -----------------------------------------------------

    selected = (
        select_base_features(
            X_train,
            y_train
        )
    )

    base_train = (
        X_train[
            selected
        ]
        .copy()
    )

    base_valid = (
        X_valid[
            selected
        ]
        .copy()
    )

    train_blocks = [
        base_train
    ]

    valid_blocks = [
        base_valid
    ]

    # -----------------------------------------------------
    # Tool-Z Top100
    # -----------------------------------------------------

    if tool_col is not None:

        tool_stats = (
            fit_tool_statistics(

                base_train,

                meta_train[
                    "Tool"
                ].to_numpy()
            )
        )

        (
            tool_z_train,
            tool_summary_train
        ) = build_tool_z(

            base_train,

            meta_train[
                "Tool"
            ].to_numpy(),

            tool_stats
        )

        (
            tool_z_valid,
            tool_summary_valid
        ) = build_tool_z(

            base_valid,

            meta_valid[
                "Tool"
            ].to_numpy(),

            tool_stats
        )

        (
            selected_tool,
            tool_importance
        ) = select_tool_features(

            tool_z_train,

            y_train,

            TOOL_TOP_K
        )

        if selected_tool:

            train_blocks.append(
                tool_z_train[
                    selected_tool
                ]
            )

            valid_blocks.append(
                tool_z_valid[
                    selected_tool
                ]
            )

    # -----------------------------------------------------
    # Operation All
    # -----------------------------------------------------

    operation_train = (
        build_operation_statistics(
            base_train,
            selected,
            "all"
        )
    )

    operation_valid = (
        build_operation_statistics(
            base_valid,
            selected,
            "all"
        )
    )

    train_blocks.append(
        operation_train
    )

    valid_blocks.append(
        operation_valid
    )

    # -----------------------------------------------------
    # 拼接
    # -----------------------------------------------------

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

    return (
        train_final,
        valid_final,
        selected,
        selected_tool
        if tool_col is not None
        else []
    )


# =========================================================
# Tool-Z筛选
# =========================================================

def select_tool_features(
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


# =========================================================
# 一个Fold的全局模型
# =========================================================

def train_global_model(
    X_train,
    X_valid,
    y_train,
    y_valid,
    meta_train,
    meta_valid,
    tool_col
):

    (
        train_final,
        valid_final,
        selected,
        selected_tool
    ) = build_final_features(

        X_train,
        X_valid,
        y_train,
        meta_train,
        meta_valid,
        tool_col
    )

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

    return (
        pred,
        train_final,
        valid_final,
        selected,
        selected_tool
    )


# =========================================================
# Tool O专家模型
# =========================================================

def train_specialist_model(
    X_train,
    y_train,
    meta_train,
    tool_col,
    selected_features
):

    if tool_col is None:

        return None

    special_mask = (
        meta_train[
            "Tool"
        ]
        .astype(str)
        .eq(
            SPECIAL_TOOL
        )
        .to_numpy()
    )

    if special_mask.sum() < MIN_SPECIAL_SAMPLES:

        return None

    X_special = (
        X_train.loc[
            special_mask,
            selected_features
        ]
        .copy()
    )

    y_special = (
        y_train.loc[
            special_mask
        ]
        .copy()
    )

    # 专家模型使用全局Top300特征
    model = make_xgb()

    model.fit(
        X_special,
        y_special,
        verbose=False
    )

    return model


# =========================================================
# 注意：
# 专家模型需要和Global相同的特征工程
# 因此使用已经构造好的最终特征矩阵
# =========================================================

def train_specialist_from_final_features(
    X_final_train,
    y_train,
    meta_train
):

    special_mask = (
        meta_train[
            "Tool"
        ]
        .astype(str)
        .eq(
            SPECIAL_TOOL
        )
        .to_numpy()
    )

    if special_mask.sum() < MIN_SPECIAL_SAMPLES:

        return None

    X_special = (
        X_final_train
        .loc[
            special_mask
        ]
        .copy()
    )

    y_special = (
        y_train
        .loc[
            special_mask
        ]
        .copy()
    )

    model = make_xgb()

    # 专家模型适度降低复杂度
    model.set_params(
        max_depth=2,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.7,
        n_estimators=800
    )

    model.fit(
        X_special,
        y_special,
        verbose=False
    )

    return model


# =========================================================
# 一个Fold完整计算
# =========================================================

def run_fold(
    X,
    y,
    meta,
    train_idx,
    valid_idx,
    tool_col,
    alpha
):

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

    # -----------------------------------------------------
    # 缺失值
    # -----------------------------------------------------

    Xtr, Xva = preprocess_fold(

        Xtr_raw,
        Xva_raw,
        mtr,
        mva,
        tool_col
    )

    # -----------------------------------------------------
    # 最终特征
    # -----------------------------------------------------

    (
        train_final,
        valid_final,
        selected,
        selected_tool
    ) = build_final_features(

        Xtr,
        Xva,
        ytr,
        mtr,
        mva,
        tool_col
    )

    # -----------------------------------------------------
    # Global
    # -----------------------------------------------------

    global_model = make_xgb()

    global_model.fit(

        train_final,

        ytr,

        eval_set=[
            (
                valid_final,
                yva
            )
        ],

        verbose=False
    )

    global_pred = (
        global_model
        .predict(
            valid_final
        )
    )

    # -----------------------------------------------------
    # Tool O Expert
    # -----------------------------------------------------

    expert_model = (
        train_specialist_from_final_features(
            train_final,
            ytr,
            mtr
        )
    )

    final_pred = (
        global_pred.copy()
    )

    expert_pred = np.full(
        len(valid_idx),
        np.nan,
        dtype=np.float64
    )

    special_valid_mask = (
        mva[
            "Tool"
        ]
        .astype(str)
        .eq(
            SPECIAL_TOOL
        )
        .to_numpy()
    )

    if (
        expert_model is not None
        and
        special_valid_mask.any()
    ):

        expert_pred[
            special_valid_mask
        ] = (
            expert_model
            .predict(
                valid_final.loc[
                    special_valid_mask
                ]
            )
        )

        final_pred[
            special_valid_mask
        ] = (

            (
                1.0 - alpha
            )
            *
            global_pred[
                special_valid_mask
            ]

            +

            alpha
            *
            expert_pred[
                special_valid_mask
            ]
        )

    # -----------------------------------------------------
    # Tool O单独预测
    # -----------------------------------------------------

    return (
        global_pred,
        expert_pred,
        final_pred,
        yva.to_numpy(),
        mva
    )


# =========================================================
# Experiment
# =========================================================

def run_experiment(
    name,
    alpha,
    X,
    y,
    meta,
    tool_col
):

    section(
        f"实验：{name}"
    )

    kf = KFold(

        n_splits=
            N_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE
    )

    oof = np.full(
        len(y),
        np.nan
    )

    fold_rows = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(

        kf.split(X),

        start=1
    ):

        start = (
            time.perf_counter()
        )

        (
            global_pred,
            expert_pred,
            final_pred,
            y_valid,
            meta_valid
        ) = run_fold(

            X,
            y,
            meta,
            train_idx,
            valid_idx,
            tool_col,
            alpha
        )

        oof[
            valid_idx
        ] = final_pred

        mse = (
            mean_squared_error(
                y_valid,
                final_pred
            )
        )

        rmse = np.sqrt(
            mse
        )

        mae = (
            mean_absolute_error(
                y_valid,
                final_pred
            )
        )

        r2 = (
            r2_score(
                y_valid,
                final_pred
            )
        )

        # Tool O指标
        o_mask = (
            meta_valid[
                "Tool"
            ]
            .astype(str)
            .eq(
                SPECIAL_TOOL
            )
            .to_numpy()
        )

        if o_mask.sum() > 0:

            o_mse = (
                mean_squared_error(
                    y_valid[
                        o_mask
                    ],
                    final_pred[
                        o_mask
                    ]
                )
            )

            o_mae = (
                mean_absolute_error(
                    y_valid[
                        o_mask
                    ],
                    final_pred[
                        o_mask
                    ]
                )
            )

        else:

            o_mse = np.nan

            o_mae = np.nan

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

            "tool_o_n":
                int(
                    o_mask.sum()
                ),

            "tool_o_mse":
                o_mse,

            "tool_o_mae":
                o_mae,

            "time_sec":
                (
                    time.perf_counter()
                    -
                    start
                )
        })

        log(
            f"{name} | "
            f"Fold {fold}/{N_SPLITS} | "
            f"MSE={mse:.8f} | "
            f"ToolO_MSE="
            f"{o_mse:.8f}"
        )

    if np.isnan(
        oof
    ).any():

        raise RuntimeError(
            f"{name} OOF存在NaN"
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

    # -----------------------------------------------------
    # Tool O整体
    # -----------------------------------------------------

    o_mask_all = (
        meta[
            "Tool"
        ]
        .astype(str)
        .eq(
            SPECIAL_TOOL
        )
        .to_numpy()
    )

    if o_mask_all.sum() > 0:

        o_mse = (
            mean_squared_error(
                y.to_numpy()[
                    o_mask_all
                ],
                oof[
                    o_mask_all
                ]
            )
        )

        o_rmse = np.sqrt(
            o_mse
        )

        o_mae = (
            mean_absolute_error(
                y.to_numpy()[
                    o_mask_all
                ],
                oof[
                    o_mask_all
                ]
            )
        )

    else:

        o_mse = np.nan
        o_rmse = np.nan
        o_mae = np.nan

    folds_df = pd.DataFrame(
        fold_rows
    )

    summary = {

        "experiment":
            name,

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
                "mse"
            ].std(),

        "tool_o_mse":
            o_mse,

        "tool_o_rmse":
            o_rmse,

        "tool_o_mae":
            o_mae,

        "tool_o_n":
            int(
                o_mask_all.sum()
            )
    }

    return (
        summary,
        folds_df,
        oof
    )


# =========================================================
# Tool整体分析
# =========================================================

def analyze_tools(
    y,
    oof,
    meta
):

    result = pd.DataFrame(
        {

            "Tool":
                meta[
                    "Tool"
                ].astype(str),

            "y_true":
                y.to_numpy(),

            "y_pred":
                oof
        }
    )

    result["residual"] = (
        result[
            "y_true"
        ]
        -
        result[
            "y_pred"
        ]
    )

    result["abs_error"] = (
        result[
            "residual"
        ]
        .abs()
    )

    result["squared_error"] = (
        result[
            "residual"
        ]
        ** 2
    )

    return (

        result
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
                "abs_error",
                "mean"
            ),

            max_error=(
                "abs_error",
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


# =========================================================
# Hard Sample
# =========================================================

def make_hard_sample_table(
    y,
    oof,
    meta
):

    result = meta.copy()

    result["y_true"] = (
        y.to_numpy()
    )

    result["y_pred"] = oof

    result["residual"] = (
        result[
            "y_true"
        ]
        -
        result[
            "y_pred"
        ]
    )

    result["absolute_error"] = (
        result[
            "residual"
        ]
        .abs()
    )

    result["squared_error"] = (
        result[
            "residual"
        ]
        ** 2
    )

    threshold = (
        result[
            "absolute_error"
        ]
        .quantile(
            0.90
        )
    )

    result["hard_sample"] = (
        result[
            "absolute_error"
        ]
        >= threshold
    )

    result["error_percentile"] = (
        result[
            "absolute_error"
        ]
        .rank(
            pct=True
        )
    )

    return (
        result
        .sort_values(
            "absolute_error",
            ascending=False
        )
    )


# =========================================================
# 绘图
# =========================================================

def save_plot(
    name
):

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
    tool_df,
    hard_df
):

    section(
        "生成可视化"
    )

    # -----------------------------------------------------
    # 实验MSE
    # -----------------------------------------------------

    temp = (
        summary_df
        .sort_values(
            "mse"
        )
    )

    plt.figure(
        figsize=(11, 8)
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
        "Hard Sample / Tool Expert Experiments"
    )

    save_plot(
        "01_experiment_mse.png"
    )

    # -----------------------------------------------------
    # Tool MSE
    # -----------------------------------------------------

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
            ].astype(str)
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

    # -----------------------------------------------------
    # Tool样本数
    # -----------------------------------------------------

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
            temp[
                "n"
            ]
        )

        plt.yticks(
            np.arange(
                len(temp)
            ),
            temp[
                "Tool"
            ].astype(str)
        )

        plt.xlabel(
            "Sample Count"
        )

        plt.title(
            "Sample Count by Tool"
        )

        save_plot(
            "03_tool_sample_count.png"
        )

    # -----------------------------------------------------
    # Hard sample
    # -----------------------------------------------------

    if not hard_df.empty:

        count_df = (
            hard_df
            .groupby(
                "Tool"
            )["hard_sample"]
            .sum()
            .sort_values()
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.barh(
            np.arange(
                len(count_df)
            ),
            count_df.values
        )

        plt.yticks(
            np.arange(
                len(count_df)
            ),
            count_df.index.astype(str)
        )

        plt.xlabel(
            "Hard Sample Count"
        )

        plt.title(
            "Hard Samples by Tool"
        )

        save_plot(
            "05_hard_sample_distribution.png"
        )

        # Y vs Error

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            hard_df[
                "y_true"
            ],
            hard_df[
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
            "Y vs Prediction Error"
        )

        save_plot(
            "06_error_by_y.png"
        )

        # Missing vs Error

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            hard_df[
                "missing_rate"
            ],
            hard_df[
                "absolute_error"
            ],
            alpha=0.7
        )

        plt.xlabel(
            "Missing Rate"
        )

        plt.ylabel(
            "Absolute Error"
        )

        plt.title(
            "Missing Rate vs Error"
        )

        save_plot(
            "07_error_by_missing.png"
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
        tool_col
    ) = prepare_data()

    # =====================================================
    # 实验
    # =====================================================

    experiments = [

        (
            "E0_global_only",

            0.0
        ),

        (
            "E1_tool_o_expert_25",

            0.25
        ),

        (
            "E2_tool_o_expert_50",

            0.50
        ),

        (
            "E3_tool_o_expert_75",

            0.75
        ),

        (
            "E4_tool_o_expert_100",

            1.00
        )
    ]

    section(
        "2. Tool O专家模型实验"
    )

    summaries = []

    fold_results = []

    oof_dict = {}

    for name, alpha in experiments:

        (
            summary,
            folds,
            oof
        ) = run_experiment(

            name,

            alpha,

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
    # 最佳
    # =====================================================

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
            "E0_global_only"
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

    # =====================================================
    # Tool分析
    # =====================================================

    tool_df = (
        analyze_tools(
            y,
            best_oof,
            meta
        )
    )

    # =====================================================
    # Hard Samples
    # =====================================================

    hard_df = (
        make_hard_sample_table(
            y,
            best_oof,
            meta
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

    tool_df.to_csv(

        OUT_DIR
        /
        "tool_analysis.csv",

        index=False,

        encoding="utf-8-sig"
    )

    hard_df.to_csv(

        OUT_DIR
        /
        "hard_samples.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # =====================================================
    # OOF
    # =====================================================

    oof_detail = (
        meta.copy()
    )

    oof_detail["y_true"] = (
        y.to_numpy()
    )

    for name, pred in (
        oof_dict.items()
    ):

        oof_detail[
            f"pred_{name}"
        ] = pred

    oof_detail[
        "best_pred"
    ] = best_oof

    oof_detail[
        "best_residual"
    ] = (

        oof_detail[
            "y_true"
        ]

        -

        oof_detail[
            "best_pred"
        ]
    )

    oof_detail[
        "best_abs_error"
    ] = (
        oof_detail[
            "best_residual"
        ]
        .abs()
    )

    oof_detail.to_csv(

        OUT_DIR
        /
        "oof_detail.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # =====================================================
    # 图片
    # =====================================================

    make_plots(

        summary_df,

        tool_df,

        hard_df
    )

    # =====================================================
    # 输出
    # =====================================================

    section(
        "3. 实验排名"
    )

    print(
        summary_df
        .to_string(
            index=False
        )
    )

    # =====================================================
    # 最优
    # =====================================================

    section(
        "4. 当前最优"
    )

    print(
        f"实验："
        f"{best['experiment']}"
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
        f"相对Global："
        f"{improvement:.2f}%"
    )

    # =====================================================
    # Tool
    # =====================================================

    section(
        "5. Tool误差"
    )

    print(
        tool_df
        .to_string(
            index=False
        )
    )

    # =====================================================
    # Hard
    # =====================================================

    section(
        "6. Top 30 Hard Sample"
    )

    print(

        hard_df[
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
        .head(30)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # Tool O
    # =====================================================

    section(
        "7. Tool O重点"
    )

    o_rows = (
        hard_df[
            hard_df[
                "Tool"
            ]
            .astype(str)
            .eq(
                SPECIAL_TOOL
            )
        ]
    )

    if not o_rows.empty:

        threshold = (
            hard_df[
                "absolute_error"
            ]
            .quantile(
                0.90
            )
        )

        print(
            f"Tool O样本数："
            f"{len(o_rows)}"
        )

        print(
            f"Tool O平均绝对误差："
            f"{o_rows['absolute_error'].mean():.8f}"
        )

        print(
            f"Tool O Hard Sample："
            f"{int(o_rows['hard_sample'].sum())}"
        )

        print(
            f"总体Hard阈值："
            f"{threshold:.8f}"
        )

    # =====================================================
    # Summary
    # =====================================================

    lines = [

        "工业AI难样本 / Tool专家模型实验报告",

        "",

        "固定基础方案：",

        f"缺失值 = operation_tool_mean",

        f"基础特征 = ExtraTrees Top{BASE_TOP_K}",

        f"Tool-Z = Top{TOOL_TOP_K}",

        "Operation = 全部统计",

        "模型 = XGBoost",

        "",

        "==================================================",

        "实验结果",

        "=================================================="
    ]

    for _, row in (
        summary_df
        .iterrows()
    ):

        lines.append(

            f"{row['experiment']} | "
            f"alpha={row['alpha']:.2f} | "
            f"MSE={row['mse']:.10f} | "
            f"RMSE={row['rmse']:.10f} | "
            f"MAE={row['mae']:.10f} | "
            f"R2={row['r2']:.6f} | "
            f"ToolO_MSE="
            f"{row['tool_o_mse']:.10f}"
        )

    lines.extend([

        "",

        f"Baseline MSE = "
        f"{baseline['mse']:.10f}",

        f"Best MSE = "
        f"{best['mse']:.10f}",

        f"Relative improvement = "
        f"{improvement:.2f}%",

        "",

        "==================================================",

        "Hard Sample定义",

        "==================================================",

        "绝对残差大于等于全部样本绝对残差90分位数。",

        "",

        "==================================================",

        "输出目录",

        "==================================================",

        str(OUT_DIR),

        "",

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
        "8. 完成"
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

    print(
        f"Tool O MSE："
        f"{best['tool_o_mse']:.10f}"
    )


if __name__ == "__main__":

    main()