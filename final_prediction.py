# -*- coding: utf-8 -*-

"""
final_prediction.py

最终模型预测，不再做实验。

固定方案：
1. operation_tool_mean 缺失值处理
2. ExtraTrees Top300
3. Tool-Z Top100
4. Operation All
5. Tool One-Hot
6. Tool Target Mean + Std
7. XGBoost
8. Residual Correction alpha=0.30

最终输出：
ID,Prediction
但按照比赛提交模板：
ID,Prediction
不需要其他字段。

另外生成：
final_prediction.csv
prediction_distribution.png
tool_prediction_distribution.png
prediction_summary.txt
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
from xgboost import XGBRegressor

import missing_value


# ============================================================
# 配置
# ============================================================

BASE_DIR = Path(
    r"C:\Users\lenovo\Downloads\my method"
)

DATA_DIR = BASE_DIR / "data"

OUT_DIR = (
    DATA_DIR
    / "preprocess_missing"
    / "final_prediction"
)

PLOT_DIR = OUT_DIR / "plots"

RANDOM_STATE = 42

BASE_TOP_K = 300
TOOL_TOP_K = 100

TARGET_SMOOTH_ALPHA = 20.0

RESIDUAL_ALPHA = 0.30

TARGET_COL = "Value"

ID_COL = "ID"

TOOL_COL = "TOOL"


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
# 基础函数
# ============================================================

def infer_operation(feature):

    m = re.match(
        r"^\s*(\d+)\s*[Xx]",
        str(feature)
    )

    return (
        m.group(1)
        if m
        else "UNKNOWN"
    )


def find_col(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    return None


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


def align_columns(train, valid):

    mask = (
        ~train.columns.duplicated()
    )

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


# ============================================================
# 读取训练集
# ============================================================

def load_train():

    path = (
        DATA_DIR
        / "训练集.xlsx"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"找不到训练集：{path}"
        )

    df = pd.read_excel(
        path
    )

    return df


# ============================================================
# 读取测试集
# ============================================================

def load_test_files():

    candidates_a = [
        "测试集A.xlsx",
        "测试A.xlsx",
        "testA.xlsx",
        "TestA.xlsx"
    ]

    candidates_b = [
        "测试集B.xlsx",
        "测试B.xlsx",
        "testB.xlsx",
        "TestB.xlsx"
    ]

    test_a = None
    test_b = None

    for name in candidates_a:

        path = (
            DATA_DIR
            / name
        )

        if path.exists():

            test_a = pd.read_excel(
                path
            )

            log(
                f"找到测试A：{name}"
            )

            break

    for name in candidates_b:

        path = (
            DATA_DIR
            / name
        )

        if path.exists():

            test_b = pd.read_excel(
                path
            )

            log(
                f"找到测试B：{name}"
            )

            break

    if test_a is None and test_b is None:

        raise FileNotFoundError(
            "没有找到测试集A/B"
        )

    return test_a, test_b


# ============================================================
# 特征识别
# ============================================================

def detect_features(
    train,
    test
):

    target_col = find_col(
        train,
        [
            TARGET_COL,
            "value",
            "Y",
            "y"
        ]
    )

    id_col = find_col(
        train,
        [
            ID_COL,
            "id",
            "Id"
        ]
    )

    tool_col = find_col(
        train,
        [
            TOOL_COL,
            "Tool",
            "tool",
            "TOOL_ID",
            "Tool_ID"
        ]
    )

    if target_col is None:

        raise RuntimeError(
            "训练集找不到 Value"
        )

    numeric_cols = []

    for c in train.columns:

        if c in {
            target_col,
            id_col,
            tool_col
        }:

            continue

        s = pd.to_numeric(
            train[c],
            errors="coerce"
        )

        if s.notna().sum() > 0:

            numeric_cols.append(c)

    common = [
        c
        for c in numeric_cols
        if c in test.columns
    ]

    return (
        common,
        target_col,
        id_col,
        tool_col
    )


# ============================================================
# 结构清洗
# ============================================================

def structural_clean(
    train,
    features
):

    X = to_numeric_df(
        train[features]
    )

    # 全空
    all_nan = [
        c
        for c in X.columns
        if X[c].notna().sum() == 0
    ]

    keep = [
        c
        for c in X.columns
        if c not in all_nan
    ]

    X = X[
        keep
    ]

    # 常量
    nunique = (
        X
        .nunique(
            dropna=True
        )
    )

    constant = (
        nunique[
            nunique <= 1
        ]
        .index
        .tolist()
    )

    keep = [
        c
        for c in X.columns
        if c not in constant
    ]

    X = X[
        keep
    ]

    # 重复列
    duplicate_mask = (
        X.T
        .duplicated()
    )

    duplicate = (
        X.columns[
            duplicate_mask
        ]
        .tolist()
    )

    X = X.loc[
        :,
        ~duplicate_mask
    ]

    return (
        X,
        all_nan,
        constant,
        duplicate
    )


# ============================================================
# 缺失值
# ============================================================

def preprocess_final(
    train_x,
    test_x,
    train_meta,
    test_meta,
    tool_col
):

    train_x = (
        train_x
        .reset_index(drop=True)
    )

    test_x = (
        test_x
        .reset_index(drop=True)
    )

    train_meta = (
        train_meta
        .reset_index(drop=True)
    )

    test_meta = (
        test_meta
        .reset_index(drop=True)
    )

    if tool_col is not None:

        train_tool_meta = pd.DataFrame(
            {
                tool_col:
                    train_meta[
                        "Tool"
                    ]
                    .astype(str)
                    .to_numpy()
            }
        )

        test_tool_meta = pd.DataFrame(
            {
                tool_col:
                    test_meta[
                        "Tool"
                    ]
                    .astype(str)
                    .to_numpy()
            }
        )

    else:

        train_tool_meta = pd.DataFrame()
        test_tool_meta = pd.DataFrame()

    train_x, test_x = (
        missing_value
        .apply_best_missing_strategy(
            train_x,
            test_x,
            train_tool_meta,
            test_tool_meta,
            tool_col
        )
    )

    return safe_fill(
        train_x,
        test_x
    )


# ============================================================
# Base Top300
# ============================================================

def select_base_features(
    X,
    y
):

    model = ExtraTreesRegressor(

        n_estimators=400,

        max_features="sqrt",

        min_samples_leaf=2,

        random_state=
            RANDOM_STATE,

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

    return selected


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

    temp = X.copy()

    temp["__TOOL__"] = tools

    group = (
        temp
        .groupby(
            "__TOOL__"
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
        .fillna(1.0)
    )

    return {
        "mean":
            group.mean(
                numeric_only=True
            ),

        "std":
            group.std(
                numeric_only=True
            ),

        "global_mean":
            global_mean,

        "global_std":
            global_std
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

    columns = list(
        X.columns
    )

    n = len(X)
    p = len(columns)

    global_mean = np.array(
        stats[
            "global_mean"
        ]
        .reindex(columns)
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        ),
        copy=True
    )

    global_std = np.array(
        stats[
            "global_std"
        ]
        .reindex(columns)
        .fillna(1.0)
        .to_numpy(
            dtype=np.float64
        ),
        copy=True
    )

    mean_table = (
        stats[
            "mean"
        ]
        .reindex(
            columns=columns
        )
    )

    std_table = (
        stats[
            "std"
        ]
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
                mean_table.loc[
                    tool
                ]
                .to_numpy(
                    dtype=np.float64
                ),
                copy=True
            )

            std_values = np.array(
                std_table.loc[
                    tool
                ]
                .to_numpy(
                    dtype=np.float64
                ),
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

            mean_values[
                bad_mean
            ] = global_mean[
                bad_mean
            ]

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

    values = np.array(
        X.to_numpy(
            dtype=np.float64
        ),
        copy=True
    )

    z = (
        values
        -
        mean_matrix
    ) / std_matrix

    z[
        ~np.isfinite(z)
    ] = 0.0

    z = np.clip(
        z,
        -50,
        50
    )

    return pd.DataFrame(
        z,
        columns=[
            f"TZ_{c}"
            for c in columns
        ]
    )


# ============================================================
# Tool-Z Top100
# ============================================================

def select_tool_features(
    tool_z,
    y
):

    model = ExtraTreesRegressor(

        n_estimators=300,

        max_features="sqrt",

        min_samples_leaf=2,

        random_state=
            RANDOM_STATE,

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

    return (
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


# ============================================================
# Operation
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
        ).append(
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
        ] = (
            values.mean(
                axis=1
            )
            .to_numpy()
        )

        block[
            f"OP_{op}_std"
        ] = (
            values.std(
                axis=1
            )
            .to_numpy()
        )

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
        ).to_numpy()

        block[
            f"OP_{op}_min"
        ] = (
            values.min(
                axis=1
            )
            .to_numpy()
        )

        block[
            f"OP_{op}_max"
        ] = (
            values.max(
                axis=1
            )
            .to_numpy()
        )

        block[
            f"OP_{op}_median"
        ] = (
            values.median(
                axis=1
            )
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

def build_onehot(
    train_tools,
    test_tools
):

    train_tools = (
        pd.Series(
            train_tools
        )
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    test_tools = (
        pd.Series(
            test_tools
        )
        .astype(str)
        .reset_index(
            drop=True
        )
    )

    categories = sorted(
        train_tools.unique()
    )

    train = {}
    test = {}

    for tool in categories:

        name = (
            f"TOOL_{tool}"
        )

        train[name] = (
            train_tools
            .eq(tool)
            .astype(float)
            .to_numpy()
        )

        test[name] = (
            test_tools
            .eq(tool)
            .astype(float)
            .to_numpy()
        )

    return (
        pd.DataFrame(train),
        pd.DataFrame(test)
    )


# ============================================================
# Target Encoding
# ============================================================

def fit_target_encoding(
    y,
    tools
):

    y = (
        pd.Series(y)
        .astype(float)
        .reset_index(
            drop=True
        )
    )

    tools = (
        pd.Series(tools)
        .astype(str)
        .reset_index(
            drop=True
        )
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
        .groupby(
            "Tool"
        )["Y"]
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

        grouped["count"]
        *
        grouped["mean"]
        +
        TARGET_SMOOTH_ALPHA
        *
        global_mean

    ) / (

        grouped["count"]
        +
        TARGET_SMOOTH_ALPHA
    )

    return {
        "table":
            grouped,

        "global_mean":
            global_mean
    }


def transform_target_encoding(
    tools,
    encoding
):

    tools = (
        pd.Series(tools)
        .astype(str)
        .reset_index(
            drop=True
        )
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

            row = table.loc[
                tool
            ]

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
                        )
                }
            )

        else:

            rows.append(

                {
                    "TE_tool_mean":
                        global_mean,

                    "TE_tool_std":
                        0.0
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 构建最终特征
# ============================================================

def build_final_features(
    train_x,
    test_x,
    y,
    train_meta,
    test_meta
):

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    base_features = (
        select_base_features(
            train_x,
            y
        )
    )

    train_base = (
        train_x[
            base_features
        ]
        .copy()
    )

    test_base = (
        test_x[
            base_features
        ]
        .copy()
    )

    train_blocks = [
        train_base
    ]

    test_blocks = [
        test_base
    ]

    # --------------------------------------------------------
    # Tool-Z
    # --------------------------------------------------------

    tool_stats = (
        fit_tool_statistics(

            train_base,

            train_meta[
                "Tool"
            ].to_numpy()
        )
    )

    train_z = (
        build_tool_z(

            train_base,

            train_meta[
                "Tool"
            ].to_numpy(),

            tool_stats
        )
    )

    test_z = (
        build_tool_z(

            test_base,

            test_meta[
                "Tool"
            ].to_numpy(),

            tool_stats
        )
    )

    tool_features = (
        select_tool_features(
            train_z,
            y
        )
    )

    train_blocks.append(
        train_z[
            tool_features
        ]
    )

    test_blocks.append(
        test_z[
            tool_features
        ]
    )

    # --------------------------------------------------------
    # Operation All
    # --------------------------------------------------------

    train_op = (
        build_operation_features(
            train_base,
            base_features
        )
    )

    test_op = (
        build_operation_features(
            test_base,
            base_features
        )
    )

    train_blocks.append(
        train_op
    )

    test_blocks.append(
        test_op
    )

    # --------------------------------------------------------
    # Tool One-Hot
    # --------------------------------------------------------

    (
        train_onehot,
        test_onehot
    ) = build_onehot(

        train_meta[
            "Tool"
        ],

        test_meta[
            "Tool"
        ]
    )

    train_blocks.append(
        train_onehot
    )

    test_blocks.append(
        test_onehot
    )

    # --------------------------------------------------------
    # Target Mean + Std
    # --------------------------------------------------------

    encoding = (
        fit_target_encoding(

            y,

            train_meta[
                "Tool"
            ]
        )
    )

    train_te = (
        transform_target_encoding(

            train_meta[
                "Tool"
            ],

            encoding
        )
    )

    test_te = (
        transform_target_encoding(

            test_meta[
                "Tool"
            ],

            encoding
        )
    )

    train_blocks.append(
        train_te
    )

    test_blocks.append(
        test_te
    )

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    train_final = pd.concat(
        train_blocks,
        axis=1
    )

    test_final = pd.concat(
        test_blocks,
        axis=1
    )

    train_final, test_final = (
        align_columns(
            train_final,
            test_final
        )
    )

    train_final, test_final = (
        safe_fill(
            train_final,
            test_final
        )
    )

    return (
        train_final,
        test_final
    )


# ============================================================
# XGBoost
# ============================================================

def make_xgb():

    return XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=1200,

        max_depth=2,

        learning_rate=0.05,

        min_child_weight=1,

        subsample=0.8,

        colsample_bytree=0.6,

        reg_alpha=0.01,

        reg_lambda=1.0,

        gamma=0.0,

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )


# ============================================================
# Residual模型
# ============================================================

def train_residual_model(
    X,
    y,
    global_pred
):

    residual = (
        y.to_numpy()
        -
        global_pred
    )

    model = XGBRegressor(

        objective=
            "reg:squarederror",

        n_estimators=500,

        max_depth=2,

        learning_rate=0.03,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.6,

        reg_alpha=0.01,

        reg_lambda=1.0,

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )

    model.fit(
        X,
        residual
    )

    return model


# ============================================================
# 可视化
# ============================================================

def make_prediction_plots(
    prediction_df
):

    # --------------------------------------------------------
    # 预测分布
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        prediction_df[
            "prediction"
        ],
        bins=40,
        alpha=0.8
    )

    plt.xlabel(
        "Prediction"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Final Prediction Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        /
        "01_prediction_distribution.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # Tool预测分布
    # --------------------------------------------------------

    if (
        "Tool"
        in prediction_df.columns
    ):

        temp = (
            prediction_df
            .groupby(
                "Tool"
            )[
                "prediction"
            ]
            .mean()
            .sort_values()
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.bar(
            np.arange(
                len(temp)
            ),
            temp.values
        )

        plt.xticks(
            np.arange(
                len(temp)
            ),
            temp.index
            .astype(str)
        )

        plt.ylabel(
            "Mean Prediction"
        )

        plt.title(
            "Mean Prediction by Tool"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIR
            /
            "02_tool_prediction.png",
            dpi=180
        )

        plt.close()


# ============================================================
# Main
# ============================================================

def main():

    start = (
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
    # 1
    # ========================================================

    section(
        "1. 数据准备"
    )

    train = load_train()

    test_a, test_b = (
        load_test_files()
    )

    log(
        f"训练集："
        f"{train.shape}"
    )

    if test_a is not None:

        log(
            f"测试A："
            f"{test_a.shape}"
        )

    if test_b is not None:

        log(
            f"测试B："
            f"{test_b.shape}"
        )

    # ========================================================
    # 2
    # ========================================================

    section(
        "2. 构建训练集"
    )

    if (
        test_a is not None
        and test_b is not None
    ):

        test = pd.concat(
            [
                test_a,
                test_b
            ],
            axis=0,
            ignore_index=True
        )

    elif test_a is not None:

        test = (
            test_a
            .copy()
        )

    else:

        test = (
            test_b
            .copy()
        )

    (
        features,
        target_col,
        id_col,
        tool_col
    ) = detect_features(
        train,
        test
    )

    y = pd.to_numeric(
        train[
            target_col
        ],
        errors="coerce"
    )

    valid_y = y.notna()

    train = (
        train
        .loc[valid_y]
        .reset_index(
            drop=True
        )
    )

    y = (
        y
        .loc[valid_y]
        .reset_index(
            drop=True
        )
        .astype(float)
    )

    X_train_raw = (
        train[
            features
        ]
        .copy()
    )

    X_test_raw = (
        test[
            [
                c
                for c in features
                if c in test.columns
            ]
        ]
        .copy()
    )

    # 缺失列补齐
    X_test_raw = (
        X_test_raw
        .reindex(
            columns=features
        )
    )

    # 结构清洗
    (
        X_train_struct,
        all_nan,
        constant,
        duplicate
    ) = structural_clean(

        train,

        features
    )

    final_features = (
        X_train_struct
        .columns
        .tolist()
    )

    X_train_raw = (
        X_train_raw[
            final_features
        ]
        .copy()
    )

    X_test_raw = (
        X_test_raw[
            final_features
        ]
        .copy()
    )

    train_meta = pd.DataFrame()

    test_meta = pd.DataFrame()

    if id_col is not None:

        train_meta[
            "ID"
        ] = (
            train[
                id_col
            ]
            .astype(str)
            .to_numpy()
        )

        test_meta[
            "ID"
        ] = (
            test[
                id_col
            ]
            .astype(str)
            .to_numpy()
        )

    else:

        train_meta[
            "ID"
        ] = np.arange(
            len(train)
        ).astype(str)

        test_meta[
            "ID"
        ] = np.arange(
            len(test)
        ).astype(str)

    if tool_col is not None:

        train_meta[
            "Tool"
        ] = (
            train[
                tool_col
            ]
            .astype(str)
            .fillna(
                "UNKNOWN_TOOL"
            )
            .to_numpy()
        )

        test_meta[
            "Tool"
        ] = (
            test[
                tool_col
            ]
            .astype(str)
            .fillna(
                "UNKNOWN_TOOL"
            )
            .to_numpy()
        )

    else:

        train_meta[
            "Tool"
        ] = "UNKNOWN_TOOL"

        test_meta[
            "Tool"
        ] = "UNKNOWN_TOOL"

    log(
        f"原始特征 = {len(features)}"
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
        f"最终特征 = "
        f"{len(final_features)}"
    )

    # ========================================================
    # 3
    # ========================================================

    section(
        "3. 缺失值处理"
    )

    (
        X_train,
        X_test
    ) = preprocess_final(

        X_train_raw,

        X_test_raw,

        train_meta,

        test_meta,

        tool_col
    )

    # ========================================================
    # 4
    # ========================================================

    section(
        "4. 最终特征工程"
    )

    (
        train_final,
        test_final
    ) = build_final_features(

        X_train,

        X_test,

        y,

        train_meta,

        test_meta
    )

    log(
        f"最终训练特征："
        f"{train_final.shape}"
    )

    log(
        f"最终测试特征："
        f"{test_final.shape}"
    )

    # ========================================================
    # 5
    # ========================================================

    section(
        "5. XGBoost最终模型"
    )

    model = make_xgb()

    model.fit(
        train_final,
        y
    )

    global_train_pred = (
        model.predict(
            train_final
        )
    )

    global_test_pred = (
        model.predict(
            test_final
        )
    )

    global_test_pred = np.asarray(
        global_test_pred,
        dtype=np.float64
    )

    global_train_pred = np.asarray(
        global_train_pred,
        dtype=np.float64
    )

    # ========================================================
    # 6
    # ========================================================

    section(
        "6. Residual Correction"
    )

    residual_model = (
        train_residual_model(

            train_final,

            y,

            global_train_pred
        )
    )

    residual_test = (
        residual_model.predict(
            test_final
        )
    )

    residual_test = np.asarray(
        residual_test,
        dtype=np.float64
    )

    final_pred = (

        global_test_pred

        +

        RESIDUAL_ALPHA
        *
        residual_test
    )

    # 数值安全
    final_pred[
        ~np.isfinite(
            final_pred
        )
    ] = 0.0

    # --------------------------------------------------------
    # 防止极端预测
    # 使用训练目标范围扩展区间
    # --------------------------------------------------------

    y_min = float(
        y.min()
    )

    y_max = float(
        y.max()
    )

    y_std = float(
        y.std()
    )

    lower = (
        y_min
        -
        3.0
        *
        y_std
    )

    upper = (
        y_max
        +
        3.0
        *
        y_std
    )

    final_pred = np.clip(
        final_pred,
        lower,
        upper
    )

    # ========================================================
    # 7
    # ========================================================

    section(
        "7. 生成最终CSV"
    )

    prediction_df = pd.DataFrame(
        {
            "ID":
                test_meta[
                    "ID"
                ].to_numpy(),

            "prediction":
                final_pred,

            "Tool":
                test_meta[
                    "Tool"
                ].to_numpy()
        }
    )

    # --------------------------------------------------------
    # 真正提交文件
    # 无表头
    # --------------------------------------------------------

    submit_df = pd.DataFrame(
        {
            "ID":
                prediction_df[
                    "ID"
                ],

            "Prediction":
                prediction_df[
                    "prediction"
                ]
        }
    )

    submit_path = (
        OUT_DIR
        /
        "final_prediction.csv"
    )

    submit_df.to_csv(

        submit_path,

        index=False,

        header=False,

        encoding="utf-8-sig"
    )

    # 额外保存带信息版本
    full_path = (
        OUT_DIR
        /
        "final_prediction_with_tool.csv"
    )

    prediction_df.to_csv(

        full_path,

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # 8
    # ========================================================

    section(
        "8. 预测统计"
    )

    stats = {

        "n_samples":
            len(final_pred),

        "mean":
            float(
                np.mean(
                    final_pred
                )
            ),

        "std":
            float(
                np.std(
                    final_pred
                )
            ),

        "min":
            float(
                np.min(
                    final_pred
                )
            ),

        "max":
            float(
                np.max(
                    final_pred
                )
            ),

        "median":
            float(
                np.median(
                    final_pred
                )
            )
    }

    for k, v in stats.items():

        print(
            f"{k}: {v}"
        )

    # ========================================================
    # 9
    # ========================================================

    section(
        "9. Tool预测"
    )

    tool_summary = (
        prediction_df
        .groupby(
            "Tool"
        )
        .agg(
            n=(
                "prediction",
                "size"
            ),

            mean_prediction=(
                "prediction",
                "mean"
            ),

            std_prediction=(
                "prediction",
                "std"
            ),

            min_prediction=(
                "prediction",
                "min"
            ),

            max_prediction=(
                "prediction",
                "max"
            )
        )
        .reset_index()
    )

    print(
        tool_summary
        .to_string(
            index=False
        )
    )

    tool_summary.to_csv(

        OUT_DIR
        /
        "tool_prediction_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # ========================================================
    # 10
    # ========================================================

    section(
        "10. 可视化"
    )

    make_prediction_plots(
        prediction_df
    )

    # ========================================================
    # 11
    # ========================================================

    section(
        "11. 前20条预测"
    )

    print(
        submit_df
        .head(20)
        .to_string(
            index=False,
            header=False
        )
    )

    # ========================================================
    # 12
    # ========================================================

    elapsed = (
        time.perf_counter()
        -
        start
    )

    report = "\n".join(

        [

            "工业AI最终预测",

            "",

            f"训练样本 = {len(train)}",

            f"测试样本 = {len(test)}",

            f"结构清洗后特征 = "
            f"{len(final_features)}",

            f"最终特征数量 = "
            f"{train_final.shape[1]}",

            "",

            "固定方案：",

            "缺失值 = operation_tool_mean",

            f"ExtraTrees Top = "
            f"{BASE_TOP_K}",

            f"Tool-Z Top = "
            f"{TOOL_TOP_K}",

            "Operation = All",

            "Tool One-Hot = True",

            "Tool Target Mean = True",

            "Tool Target Std = True",

            "XGB learning_rate = 0.05",

            "XGB max_depth = 2",

            "XGB n_estimators = 1200",

            f"Residual alpha = "
            f"{RESIDUAL_ALPHA}",

            "",

            f"预测均值 = "
            f"{stats['mean']:.8f}",

            f"预测标准差 = "
            f"{stats['std']:.8f}",

            f"预测最小值 = "
            f"{stats['min']:.8f}",

            f"预测最大值 = "
            f"{stats['max']:.8f}",

            f"预测中位数 = "
            f"{stats['median']:.8f}",

            "",

            f"提交文件 = "
            f"{submit_path}",

            f"总耗时 = "
            f"{elapsed / 60:.2f}分钟"
        ]
    )

    (
        OUT_DIR
        /
        "prediction_summary.txt"
    ).write_text(

        report,

        encoding="utf-8"
    )

    section(
        "13. 完成"
    )

    print(
        f"最终提交文件："
        f"{submit_path}"
    )

    print(
        f"完整预测文件："
        f"{full_path}"
    )

    print(
        f"可视化目录："
        f"{PLOT_DIR}"
    )

    print(
        f"总耗时："
        f"{elapsed / 60:.2f}分钟"
    )


if __name__ == "__main__":
    main()