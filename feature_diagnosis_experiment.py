# -*- coding: utf-8 -*-

"""
feature_diagnosis_experiment.py

============================================================
工业AI：Tool / Operation 条件特征实验
============================================================

当前基线：
    缺失值：
        operation_tool_mean

    特征选择：
        ExtraTrees Top300

    模型：
        XGBoost

============================================================
本阶段研究的问题
============================================================

1. Tool O 为什么预测较差？
2. Operation 210 是否存在明显分布漂移？
3. 样本异常程度是否与预测误差有关？
4. Tool 条件标准化是否有帮助？
5. Operation 聚合统计是否有帮助？
6. 这些诊断信息加入模型后能否降低CV MSE？

============================================================
实验

E0:
    baseline

E1:
    baseline + anomaly score

E2:
    baseline + tool z-score

E3:
    baseline + operation statistics

E4:
    baseline + anomaly + tool z-score

E5:
    baseline + anomaly + operation statistics

E6:
    baseline + tool z-score + operation statistics

E7:
    baseline + anomaly + tool z-score + operation statistics

============================================================
原则

所有统计量：

    只使用Train Fold计算
    Valid Fold只做transform

避免：

    数据泄漏

============================================================
输出

feature_diagnosis_experiment/
├── experiment_summary.csv
├── experiment_folds.csv
├── sample_diagnosis.csv
├── feature_shift.csv
├── operation_shift.csv
├── tool_shift.csv
├── top_anomaly_samples.csv
├── summary.txt
└── plots/
    ├── 01_experiment_mse.png
    ├── 02_tool_mse.png
    ├── 03_operation_shift.png
    ├── 04_feature_shift.png
    ├── 05_anomaly_vs_error.png
    ├── 06_missing_vs_error.png
    ├── 07_zero_vs_error.png
    └── 08_y_vs_error.png
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
    / "feature_diagnosis_experiment"
)

PLOT_DIR = (
    OUT_DIR
    / "plots"
)

RANDOM_STATE = 42

N_SPLITS = 5

TOP_K = 300


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


def safe_float(
    value
):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return np.nan


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

    original_features = len(
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
        f"{original_features}"
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

    X = (
        df[
            features
        ]
        .copy()
        .astype(
            np.float64
        )
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
            .fillna(
                "MISSING_TOOL"
            )
            .to_numpy()
        )

    else:

        meta["Tool"] = (
            "UNKNOWN_TOOL"
        )

    # 原始缺失率
    meta["missing_rate"] = (
        X.isna()
        .mean(axis=1)
        .to_numpy()
    )

    # 原始0率
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
# Fold预处理
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

        tool_train = (
            meta_train["Tool"]
            .astype(str)
            .to_numpy()
        )

        tool_valid = (
            meta_valid["Tool"]
            .astype(str)
            .to_numpy()
        )

        meta_train_impute = pd.DataFrame(
            {
                tool_col:
                    tool_train
            }
        )

        meta_valid_impute = pd.DataFrame(
            {
                tool_col:
                    tool_valid
            }
        )

    else:

        meta_train_impute = pd.DataFrame(
            index=np.arange(
                len(X_train)
            )
        )

        meta_valid_impute = pd.DataFrame(
            index=np.arange(
                len(X_valid)
            )
        )

    X_train, X_valid = (
        missing_value
        .apply_best_missing_strategy(

            X_train,

            X_valid,

            meta_train_impute,

            meta_valid_impute,

            tool_col
        )
    )

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

    return (
        X_train,
        X_valid
    )


# =========================================================
# ExtraTrees Top300
# =========================================================

def select_features(
    X_train,
    y_train
):

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
        X_train,
        y_train
    )

    importance = (
        pd.Series(

            model.feature_importances_,

            index=
                X_train.columns
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
# 工具：计算Train统计量
# =========================================================

def fit_tool_statistics(
    X_train,
    features,
    tool_array
):

    tool_series = pd.Series(
        tool_array,
        index=X_train.index,
        dtype="object"
    )

    global_mean = (
        X_train[
            features
        ]
        .mean()
    )

    global_std = (
        X_train[
            features
        ]
        .std()
        .replace(
            0,
            np.nan
        )
    )

    global_median = (
        X_train[
            features
        ]
        .median()
    )

    grouped = (
        X_train[
            features
        ]
        .copy()
    )

    grouped["__TOOL__"] = (
        tool_series
        .to_numpy()
    )

    tool_mean = (
        grouped
        .groupby(
            "__TOOL__"
        )
        .mean()
    )

    tool_std = (
        grouped
        .groupby(
            "__TOOL__"
        )
        .std()
        .replace(
            0,
            np.nan
        )
    )

    return {

        "global_mean":
            global_mean,

        "global_std":
            global_std,

        "global_median":
            global_median,

        "tool_mean":
            tool_mean,

        "tool_std":
            tool_std
    }


# =========================================================
# Tool条件异常特征
# =========================================================

def build_tool_features(
    X,
    tool_array,
    statistics,
    prefix="tool"
):

    data = pd.DataFrame(
        index=X.index
    )

    tool_array = (
        pd.Series(
            tool_array,
            index=X.index
        )
        .astype(str)
        .to_numpy()
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
    )

    global_std = (
        statistics[
            "global_std"
        ]
    )

    for col in X.columns:

        values = X[col].to_numpy(
            dtype=np.float64
        )

        tool_means = np.array(

            [
                tool_mean.loc[
                    tool_name,
                    col
                ]
                if tool_name
                in tool_mean.index
                else global_mean[col]

                for tool_name
                in tool_array
            ],

            dtype=np.float64
        )

        tool_stds = np.array(

            [
                tool_std.loc[
                    tool_name,
                    col
                ]
                if (
                    tool_name
                    in tool_std.index
                    and
                    pd.notna(
                        tool_std.loc[
                            tool_name,
                            col
                        ]
                    )
                )
                else global_std[col]

                for tool_name
                in tool_array
            ],

            dtype=np.float64
        )

        fallback_std = (
            global_std[col]
        )

        if (
            not pd.notna(
                fallback_std
            )
            or
            fallback_std <= 1e-12
        ):

            fallback_std = 1.0

        tool_stds = np.where(
            (
                ~np.isfinite(
                    tool_stds
                )
                |
                (
                    np.abs(
                        tool_stds
                    )
                    < 1e-12
                )
            ),
            fallback_std,
            tool_stds
        )

        # Tool条件Z
        z = (
            values
            -
            tool_means
        ) / tool_stds

        data[
            f"{prefix}_z_{col}"
        ] = np.clip(
            z,
            -50,
            50
        )

    return data


# =========================================================
# Operation统计
# =========================================================

def build_operation_mapping(
    features
):

    mapping = {}

    for feature in features:

        operation = (
            infer_operation(
                feature
            )
        )

        mapping.setdefault(
            operation,
            []
        )

        mapping[
            operation
        ].append(
            feature
        )

    return mapping


def build_operation_features(
    X,
    operation_mapping,
    statistics=None
):

    data = pd.DataFrame(
        index=X.index
    )

    for operation, cols in (
        operation_mapping.items()
    ):

        valid_cols = [

            col

            for col in cols

            if col in X.columns
        ]

        if not valid_cols:

            continue

        values = (
            X[
                valid_cols
            ]
        )

        # -----------------------------------------------
        # 均值
        # -----------------------------------------------

        data[
            f"op_{operation}_mean"
        ] = (
            values
            .mean(axis=1)
        )

        # -----------------------------------------------
        # 标准差
        # -----------------------------------------------

        data[
            f"op_{operation}_std"
        ] = (
            values
            .std(
                axis=1
            )
        )

        # -----------------------------------------------
        # 最小
        # -----------------------------------------------

        data[
            f"op_{operation}_min"
        ] = (
            values
            .min(axis=1)
        )

        # -----------------------------------------------
        # 最大
        # -----------------------------------------------

        data[
            f"op_{operation}_max"
        ] = (
            values
            .max(axis=1)
        )

        # -----------------------------------------------
        # Range
        # -----------------------------------------------

        data[
            f"op_{operation}_range"
        ] = (
            data[
                f"op_{operation}_max"
            ]

            -

            data[
                f"op_{operation}_min"
            ]
        )

        # -----------------------------------------------
        # Median
        # -----------------------------------------------

        data[
            f"op_{operation}_median"
        ] = (
            values
            .median(axis=1)
        )

    return data


# =========================================================
# Anomaly Score
# =========================================================

def fit_global_statistics(
    X
):

    mean = X.mean()

    std = (
        X.std()
        .replace(
            0,
            np.nan
        )
    )

    std = std.fillna(
        1.0
    )

    return {
        "mean": mean,
        "std": std
    }


def build_anomaly_features(
    X,
    statistics
):

    mean = statistics[
        "mean"
    ]

    std = statistics[
        "std"
    ]

    z = (

        X

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
        .fillna(0)
    )

    abs_z = z.abs()

    result = pd.DataFrame(
        index=X.index
    )

    # 最大异常程度
    result[
        "anomaly_max_z"
    ] = (
        abs_z
        .max(axis=1)
        .clip(upper=50)
    )

    # Top 5
    result[
        "anomaly_top5_mean_z"
    ] = (
        abs_z
        .apply(
            lambda row:
            row.nlargest(5).mean(),
            axis=1
        )
        .clip(
            upper=50
        )
    )

    # Top10
    result[
        "anomaly_top10_mean_z"
    ] = (
        abs_z
        .apply(
            lambda row:
            row.nlargest(10).mean(),
            axis=1
        )
        .clip(
            upper=50
        )
    )

    # 超过2σ特征数
    result[
        "anomaly_count_z2"
    ] = (
        abs_z
        .gt(2)
        .sum(axis=1)
    )

    # 超过3σ特征数
    result[
        "anomaly_count_z3"
    ] = (
        abs_z
        .gt(3)
        .sum(axis=1)
    )

    # 超过5σ特征数
    result[
        "anomaly_count_z5"
    ] = (
        abs_z
        .gt(5)
        .sum(axis=1)
    )

    return result


# =========================================================
# 统计Tool漂移
# =========================================================

def calculate_tool_shift(
    X,
    meta,
    features
):

    rows = []

    global_mean = (
        X[
            features
        ]
        .mean()
    )

    global_std = (
        X[
            features
        ]
        .std()
        .replace(
            0,
            np.nan
        )
    )

    for tool, group_idx in (
        meta
        .groupby("Tool")
        .groups
        .items()
    ):

        if len(group_idx) < 3:

            continue

        values = (
            X.loc[
                group_idx,
                features
            ]
        )

        tool_mean = (
            values
            .mean()
        )

        z_shift = (

            (
                tool_mean
                -
                global_mean
            )

            /

            global_std
        )

        z_shift = (
            z_shift
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )

        for feature in features:

            value = z_shift[
                feature
            ]

            if pd.notna(value):

                rows.append({

                    "Tool":
                        tool,

                    "feature":
                        feature,

                    "shift_z":
                        float(value),

                    "abs_shift_z":
                        abs(float(value)),

                    "n":
                        len(group_idx),

                    "operation":
                        infer_operation(
                            feature
                        )
                })

    return pd.DataFrame(
        rows
    )


# =========================================================
# Operation漂移
# =========================================================

def calculate_operation_shift(
    X,
    features
):

    operations = (
        build_operation_mapping(
            features
        )
    )

    rows = []

    for operation, cols in (
        operations.items()
    ):

        cols = [
            c
            for c in cols
            if c in X.columns
        ]

        if not cols:

            continue

        values = (
            X[
                cols
            ]
        )

        mean_values = (
            values.mean(
                axis=0
            )
        )

        std_values = (
            values.std(
                axis=0
            )
        )

        op_mean = (
            float(
                mean_values.mean()
            )
        )

        op_std = (
            float(
                std_values.mean()
            )
        )

        rows.append({

            "operation":
                operation,

            "feature_count":
                len(cols),

            "mean_of_feature_means":
                op_mean,

            "mean_of_feature_stds":
                op_std
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# 单个实验
# =========================================================

def run_experiment(
    name,
    use_anomaly,
    use_tool,
    use_operation,
    X,
    y,
    meta,
    features,
    tool_col
):

    log(
        f"实验：{name}"
    )

    kf = KFold(

        n_splits=
            N_SPLITS,

        shuffle=True,

        random_state=
            RANDOM_STATE
    )

    fold_rows = []

    oof_pred = np.full(
        len(y),
        np.nan
    )

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

        # -------------------------------------------------
        # Fold切分
        # -------------------------------------------------

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

        # -------------------------------------------------
        # 缺失值
        # -------------------------------------------------

        X_train, X_valid = (
            preprocess_fold(

                X_train_raw,

                X_valid_raw,

                meta_train,

                meta_valid,

                tool_col
            )
        )

        # -------------------------------------------------
        # ExtraTrees Top300
        # -------------------------------------------------

        (
            selected,
            importance
        ) = select_features(

            X_train,

            y_train
        )

        Xtr = (
            X_train[
                selected
            ]
            .copy()
        )

        Xva = (
            X_valid[
                selected
            ]
            .copy()
        )

        # =================================================
        # 诊断特征全部基于Top300
        # =================================================

        selected_operations = (
            build_operation_mapping(
                selected
            )
        )

        # -------------------------------------------------
        # Anomaly
        # -------------------------------------------------

        if use_anomaly:

            anomaly_stats = (
                fit_global_statistics(
                    Xtr
                )
            )

            anomaly_train = (
                build_anomaly_features(
                    Xtr,
                    anomaly_stats
                )
            )

            anomaly_valid = (
                build_anomaly_features(
                    Xva,
                    anomaly_stats
                )
            )

            Xtr = pd.concat(
                [
                    Xtr,
                    anomaly_train
                ],
                axis=1
            )

            Xva = pd.concat(
                [
                    Xva,
                    anomaly_valid
                ],
                axis=1
            )

        # -------------------------------------------------
        # Tool Z
        # -------------------------------------------------

        if use_tool:

            if tool_col is not None:

                tool_statistics = (
                    fit_tool_statistics(

                        Xtr[
                            selected
                        ],

                        selected,

                        meta_train[
                            "Tool"
                        ]
                        .astype(str)
                        .to_numpy()
                    )
                )

                tool_train = (
                    build_tool_features(

                        Xtr[
                            selected
                        ],

                        meta_train[
                            "Tool"
                        ]
                        .astype(str)
                        .to_numpy(),

                        tool_statistics
                    )
                )

                tool_valid = (
                    build_tool_features(

                        Xva[
                            selected
                        ],

                        meta_valid[
                            "Tool"
                        ]
                        .astype(str)
                        .to_numpy(),

                        tool_statistics
                    )
                )

                Xtr = pd.concat(
                    [
                        Xtr,
                        tool_train
                    ],
                    axis=1
                )

                Xva = pd.concat(
                    [
                        Xva,
                        tool_valid
                    ],
                    axis=1
                )

        # -------------------------------------------------
        # Operation统计
        # -------------------------------------------------

        if use_operation:

            op_train = (
                build_operation_features(

                    Xtr[
                        selected
                    ],

                    selected_operations
                )
            )

            op_valid = (
                build_operation_features(

                    Xva[
                        selected
                    ],

                    selected_operations
                )
            )

            Xtr = pd.concat(
                [
                    Xtr,
                    op_train
                ],
                axis=1
            )

            Xva = pd.concat(
                [
                    Xva,
                    op_valid
                ],
                axis=1
            )

        # -------------------------------------------------
        # 数值安全
        # -------------------------------------------------

        Xtr = (
            Xtr
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )

        Xva = (
            Xva
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )

        medians = (
            Xtr
            .median()
            .fillna(0.0)
        )

        Xtr = (
            Xtr
            .fillna(
                medians
            )
            .astype(
                np.float64
            )
        )

        Xva = (
            Xva
            .fillna(
                medians
            )
            .astype(
                np.float64
            )
        )

        # -------------------------------------------------
        # XGBoost
        # -------------------------------------------------

        model = make_xgb()

        model.fit(

            Xtr,

            y_train,

            eval_set=[
                (
                    Xva,
                    y_valid
                )
            ],

            verbose=False
        )

        pred = (
            model
            .predict(
                Xva
            )
        )

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

            "experiment":
                name,

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
                int(Xtr.shape[1]),

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
            f"Features={Xtr.shape[1]}"
        )

    # =====================================================
    # OOF整体
    # =====================================================

    overall_mse = (
        mean_squared_error(
            y,
            oof_pred
        )
    )

    overall_rmse = (
        np.sqrt(
            overall_mse
        )
    )

    overall_mae = (
        mean_absolute_error(
            y,
            oof_pred
        )
    )

    overall_r2 = (
        r2_score(
            y,
            oof_pred
        )
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    summary = {

        "experiment":
            name,

        "use_anomaly":
            use_anomaly,

        "use_tool":
            use_tool,

        "use_operation":
            use_operation,

        "mse":
            float(
                overall_mse
            ),

        "rmse":
            float(
                overall_rmse
            ),

        "mae":
            float(
                overall_mae
            ),

        "r2":
            float(
                overall_r2
            ),

        "mse_mean_fold":
            float(
                fold_df[
                    "mse"
                ].mean()
            ),

        "mse_std_fold":
            float(
                fold_df[
                    "mse"
                ].std(
                    ddof=1
                )
            ),

        "features_mean":
            float(
                fold_df[
                    "n_features"
                ].mean()
            )
    }

    return (
        summary,
        fold_df,
        oof_pred
    )


# =========================================================
# 诊断样本信息
# =========================================================

def build_sample_diagnosis(
    X,
    y,
    meta
):

    section(
        "样本诊断"
    )

    # -----------------------------------------------------
    # 全局统计
    # -----------------------------------------------------

    stats_info = (
        fit_global_statistics(
            X
        )
    )

    anomaly = (
        build_anomaly_features(
            X,
            stats_info
        )
    )

    result = meta.copy()

    result["y_true"] = (
        y.to_numpy()
    )

    result = pd.concat(
        [
            result.reset_index(
                drop=True
            ),
            anomaly.reset_index(
                drop=True
            )
        ],
        axis=1
    )

    return result


# =========================================================
# 绘图
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
    tool_df,
    operation_df,
    feature_shift_df
):

    section(
        "生成可视化"
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # 1. Experiment MSE
    # =====================================================

    temp = (
        summary_df
        .sort_values(
            "mse"
        )
    )

    plt.figure(
        figsize=(10, 7)
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
        temp[
            "experiment"
        ]
    )

    plt.xlabel(
        "OOF MSE"
    )

    plt.title(
        "Feature Diagnosis Experiment"
    )

    save_plot(
        "01_experiment_mse.png"
    )

    # =====================================================
    # 2. Tool MSE
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
            temp[
                "Tool"
            ].astype(str)
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
    # 3. Operation shift
    # =====================================================

    if not operation_df.empty:

        temp = (
            operation_df
            .sort_values(
                "feature_count",
                ascending=False
            )
            .head(20)
        )

        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            np.arange(
                len(temp)
            ),
            temp[
                "feature_count"
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
            "Feature Count"
        )

        plt.title(
            "Operation Feature Count"
        )

        save_plot(
            "03_operation_shift.png"
        )

    # =====================================================
    # 4. Feature shift
    # =====================================================

    if not feature_shift_df.empty:

        temp = (
            feature_shift_df
            .nlargest(
                30,
                "abs_shift_z"
            )
            .sort_values(
                "abs_shift_z"
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
                "abs_shift_z"
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
            "Absolute Shift Z"
        )

        plt.title(
            "Largest Tool Feature Shifts"
        )

        save_plot(
            "04_feature_shift.png"
        )

    # =====================================================
    # 5. Anomaly vs error
    # =====================================================

    if (
        "absolute_residual"
        in sample_df.columns
    ):

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "anomaly_max_z"
            ],
            sample_df[
                "absolute_residual"
            ],
            alpha=0.7
        )

        plt.xlabel(
            "Max Absolute Z-score"
        )

        plt.ylabel(
            "Absolute Residual"
        )

        plt.title(
            "Anomaly Score vs Prediction Error"
        )

        save_plot(
            "05_anomaly_vs_error.png"
        )

    # =====================================================
    # 6. Missing vs error
    # =====================================================

    if (
        "absolute_residual"
        in sample_df.columns
    ):

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "missing_rate"
            ],
            sample_df[
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
            "06_missing_vs_error.png"
        )

    # =====================================================
    # 7. Zero vs error
    # =====================================================

    if (
        "absolute_residual"
        in sample_df.columns
    ):

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "zero_rate"
            ],
            sample_df[
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
            "07_zero_vs_error.png"
        )

    # =====================================================
    # 8. Y vs error
    # =====================================================

    if (
        "absolute_residual"
        in sample_df.columns
    ):

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            sample_df[
                "y_true"
            ],
            sample_df[
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
            "Y vs Error"
        )

        save_plot(
            "08_y_vs_error.png"
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
    # 样本基础诊断
    # =====================================================

    sample_base = (
        build_sample_diagnosis(
            X,
            y,
            meta
        )
    )

    # =====================================================
    # 实验设计
    # =====================================================

    experiments = [

        (
            "E0_baseline",
            False,
            False,
            False
        ),

        (
            "E1_anomaly",
            True,
            False,
            False
        ),

        (
            "E2_tool",
            False,
            True,
            False
        ),

        (
            "E3_operation",
            False,
            False,
            True
        ),

        (
            "E4_anomaly_tool",
            True,
            True,
            False
        ),

        (
            "E5_anomaly_operation",
            True,
            False,
            True
        ),

        (
            "E6_tool_operation",
            False,
            True,
            True
        ),

        (
            "E7_all",
            True,
            True,
            True
        )
    ]

    section(
        "2. 特征诊断实验"
    )

    summary_rows = []

    fold_results = []

    final_oof = None

    for (
        name,
        use_anomaly,
        use_tool,
        use_operation
    ) in experiments:

        (
            summary,
            folds,
            oof
        ) = run_experiment(

            name,

            use_anomaly,

            use_tool,

            use_operation,

            X,

            y,

            meta,

            features,

            tool_col
        )

        summary_rows.append(
            summary
        )

        fold_results.append(
            folds
        )

        # 保存最优实验OOF
        if (
            final_oof is None
            or
            summary["mse"]
            <
            min(
                x["mse"]
                for x
                in summary_rows
            )
        ):

            final_oof = (
                oof.copy()
            )

    # =====================================================
    # Summary
    # =====================================================

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

    fold_df = (
        pd.concat(
            fold_results,
            ignore_index=True
        )
    )

    # =====================================================
    # 基于E0重新获得OOF
    # =====================================================

    baseline_summary, baseline_folds, baseline_oof = (
        run_experiment(

            "BASELINE_FINAL_OOF",

            False,
            False,
            False,

            X,
            y,
            meta,
            features,
            tool_col
        )
    )

    sample_result = sample_base.copy()

    sample_result["y_pred"] = (
        baseline_oof
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
    # Tool误差
    # =====================================================

    tool_df = (

        sample_result

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

            mean_missing_rate=(
                "missing_rate",
                "mean"
            ),

            mean_zero_rate=(
                "zero_rate",
                "mean"
            ),

            mean_anomaly=(
                "anomaly_max_z",
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
    )

    # =====================================================
    # 特征漂移
    # =====================================================

    feature_shift_df = (
        calculate_tool_shift(
            X,
            meta,
            features
        )
    )

    # =====================================================
    # Operation
    # =====================================================

    operation_df = (
        calculate_operation_shift(
            X,
            features
        )
    )

    # =====================================================
    # Top异常样本
    # =====================================================

    top_anomaly = (
        sample_result
        .sort_values(
            "anomaly_max_z",
            ascending=False
        )
        .head(50)
    )

    # =====================================================
    # Top残差
    # =====================================================

    top_error = (
        sample_result
        .sort_values(
            "absolute_residual",
            ascending=False
        )
        .head(50)
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

    tool_df.to_csv(

        OUT_DIR
        /
        "tool_shift.csv",

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

    feature_shift_df.to_csv(

        OUT_DIR
        /
        "feature_shift.csv",

        index=False,

        encoding="utf-8-sig"
    )

    operation_df.to_csv(

        OUT_DIR
        /
        "operation_shift.csv",

        index=False,

        encoding="utf-8-sig"
    )

    top_anomaly.to_csv(

        OUT_DIR
        /
        "top_anomaly_samples.csv",

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

        tool_df,

        operation_df,

        feature_shift_df
    )

    # =====================================================
    # 打印排名
    # =====================================================

    section(
        "3. 实验结果"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # =====================================================
    # Tool
    # =====================================================

    section(
        "4. Tool分析"
    )

    print(
        tool_df.to_string(
            index=False
        )
    )

    # =====================================================
    # Top异常
    # =====================================================

    section(
        "5. Top异常样本"
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
                "zero_rate",
                "anomaly_max_z",
                "anomaly_top5_mean_z",
                "anomaly_count_z3"
            ]
        ]

        .head(20)

        .to_string(
            index=False
        )
    )

    # =====================================================
    # Operation
    # =====================================================

    section(
        "6. Operation"
    )

    print(
        operation_df
        .head(30)
        .to_string(
            index=False
        )
    )

    # =====================================================
    # 最优实验
    # =====================================================

    best = (
        summary_df
        .iloc[0]
    )

    baseline_mse = (
        summary_df
        .loc[
            summary_df[
                "experiment"
            ]
            ==
            "E0_baseline",
            "mse"
        ]
        .iloc[0]
    )

    improvement = (
        baseline_mse
        -
        best["mse"]
    )

    improvement_pct = (
        improvement
        /
        baseline_mse
        *
        100
    )

    # =====================================================
    # Summary
    # =====================================================

    lines = [

        "工业AI特征诊断实验",

        "",

        "==================================================",

        "实验目标",

        "==================================================",

        "验证异常程度、Tool条件信息、Operation统计信息是否有预测价值。",

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

            f"MSE={row['mse']:.10f} | "

            f"RMSE={row['rmse']:.10f} | "

            f"MAE={row['mae']:.10f} | "

            f"R2={row['r2']:.6f}"
        )

    lines.extend([

        "",

        f"Baseline MSE = "
        f"{baseline_mse:.10f}",

        f"Best MSE = "
        f"{best['mse']:.10f}",

        f"MSE improvement = "
        f"{improvement:.10f}",

        f"Relative improvement = "
        f"{improvement_pct:.2f}%",

        "",

        "==================================================",

        "最优方案",

        "==================================================",

        f"{best['experiment']}",

        "",

        "==================================================",

        "最差Tool",

        "=================================================="
    ])

    if not tool_df.empty:

        worst_tool = (
            tool_df.iloc[0]
        )

        lines.extend([

            f"Tool = "
            f"{worst_tool['Tool']}",

            f"n = "
            f"{int(worst_tool['n'])}",

            f"MSE = "
            f"{worst_tool['mse']:.10f}",

            f"MAE = "
            f"{worst_tool['mae']:.10f}",

            f"Mean missing rate = "
            f"{worst_tool['mean_missing_rate']:.6f}",

            f"Mean zero rate = "
            f"{worst_tool['mean_zero_rate']:.6f}",

            f"Mean anomaly = "
            f"{worst_tool['mean_anomaly']:.6f}"
        ])

    lines.extend([

        "",

        "==================================================",

        "最大残差样本",

        "=================================================="
    ])

    for _, row in (
        top_error
        .head(10)
        .iterrows()
    ):

        lines.append(

            f"ID={row['ID']} | "

            f"Tool={row['Tool']} | "

            f"Y={row['y_true']:.6f} | "

            f"Pred={row['y_pred']:.6f} | "

            f"Residual={row['residual']:.6f} | "

            f"AbsResidual="
            f"{row['absolute_residual']:.6f}"
        )

    lines.extend([

        "",

        "==================================================",

        "输出",

        "==================================================",

        str(OUT_DIR),

        "",

        f"总耗时："
        f"{(time.perf_counter()-total_start)/60:.2f} 分钟"
    ])

    (OUT_DIR / "summary.txt").write_text(

        "\n".join(lines),

        encoding="utf-8"
    )

    # =====================================================
    # 完成
    # =====================================================

    section(
        "7. 完成"
    )

    print(
        f"结果目录：{OUT_DIR}"
    )

    print(
        f"最佳实验："
        f"{best['experiment']}"
    )

    print(
        f"最佳OOF MSE："
        f"{best['mse']:.10f}"
    )

    print(
        f"相对于Baseline："
        f"{improvement_pct:.2f}%"
    )


if __name__ == "__main__":

    main()