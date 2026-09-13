# -*- coding: utf-8 -*-

"""
missing_value.py

缺失值模块

当前已经确定：
    operation_tool_mean
    CV MSE = 0.0135389961

后续特征工程全部固定使用该方法。

处理逻辑：
    Tool Mean
        ↓
    Global Mean 回退
        ↓
    训练Fold Median最终兜底

重要：
    fit只使用训练Fold，避免数据泄漏。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(
    r"C:\Users\lenovo\Downloads\my method\data"
)


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
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


# =========================================================
# 找列
# =========================================================

def first_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


# =========================================================
# 工序
# =========================================================

def infer_operation(feature: str) -> str:
    match = re.match(
        r"^\s*(\d+)\s*[Xx]",
        str(feature)
    )

    return (
        match.group(1)
        if match
        else "UNKNOWN"
    )


# =========================================================
# 读取训练集
# =========================================================

def load_train(
    data_dir: Path = DEFAULT_DATA_DIR
):

    files = sorted(
        data_dir.glob("*.xlsx")
    )

    if not files:
        raise FileNotFoundError(
            f"没有找到Excel文件：{data_dir}"
        )

    for path in files:

        df = pd.read_excel(path)

        target = first_col(
            df,
            [
                "Value",
                "value",
                "Y",
                "y"
            ]
        )

        if target is not None:

            log(
                f"训练集：{path.name}"
            )

            log(
                f"shape = {df.shape}"
            )

            return df

    raise RuntimeError(
        "未找到包含 Value/Y 的训练集。"
    )


# =========================================================
# 识别特征
# =========================================================

def detect_feature_cols(
    df,
    target,
    id_col,
    tool_col
):

    exclude = {
        x for x in
        [
            target,
            id_col,
            tool_col
        ]
        if x is not None
    }

    features = []

    for col in df.columns:

        if col in exclude:
            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            features.append(col)

    return features


# =========================================================
# 固定结构清洗
# =========================================================

def structural_clean(
    df,
    features
):

    x = df[features]

    # -----------------------------------------------------
    # 全空
    # -----------------------------------------------------

    all_nan = [
        col
        for col in features
        if x[col].isna().all()
    ]

    # -----------------------------------------------------
    # 常量
    # -----------------------------------------------------

    constant = []

    for col in features:

        if col in all_nan:
            continue

        valid = x[col].dropna()

        if len(valid) == 0:
            continue

        if valid.nunique() <= 1:
            constant.append(col)

    # -----------------------------------------------------
    # 重复
    # -----------------------------------------------------

    removed = (
        set(all_nan)
        |
        set(constant)
    )

    candidates = [
        col
        for col in features
        if col not in removed
    ]

    seen = {}
    duplicate = []

    for col in candidates:

        key = tuple(
            pd.util.hash_pandas_object(
                x[col],
                index=False
            ).values
        )

        if key in seen:
            duplicate.append(col)
        else:
            seen[key] = col

    remove_all = (
        set(all_nan)
        |
        set(constant)
        |
        set(duplicate)
    )

    keep = [
        col
        for col in features
        if col not in remove_all
    ]

    return (
        keep,
        all_nan,
        constant,
        duplicate
    )


# =========================================================
# 0值处理
# =========================================================

def apply_zero_mode(
    df,
    features,
    mode="keep"
):

    out = df.copy()

    if mode == "keep":
        return out

    if mode == "all_as_missing":

        out.loc[:, features] = (
            out[features]
            .mask(
                out[features] == 0
            )
        )

        return out

    raise ValueError(
        f"未知 zero_mode：{mode}"
    )


# =========================================================
# 缺失值报告
# =========================================================

def make_feature_report(
    df,
    features
):

    rows = []

    for col in features:

        s = df[col]

        valid = s.dropna()

        rows.append({

            "feature":
                col,

            "operation":
                infer_operation(col),

            "n_total":
                len(s),

            "n_missing":
                int(
                    s.isna().sum()
                ),

            "missing_rate":
                float(
                    s.isna().mean()
                ),

            "n_zero":
                int(
                    (s == 0).sum()
                ),

            "zero_rate":
                float(
                    (s == 0).mean()
                ),

            "n_unique":
                int(
                    valid.nunique()
                ),

            "mean":
                float(
                    valid.mean()
                )
                if len(valid)
                else np.nan,

            "median":
                float(
                    valid.median()
                )
                if len(valid)
                else np.nan,

            "std":
                float(
                    valid.std()
                )
                if len(valid) >= 2
                else np.nan,

            "skewness":
                float(
                    valid.skew()
                )
                if len(valid) >= 3
                else np.nan
        })

    return pd.DataFrame(rows)


# =========================================================
# 行缺失报告
# =========================================================

def make_row_report(
    df,
    features,
    id_col=None
):

    x = df[features]

    missing = (
        x.isna()
        .sum(axis=1)
    )

    zero = (
        (x == 0)
        .sum(axis=1)
    )

    report = pd.DataFrame({

        "row_index":
            np.arange(
                len(df)
            ),

        "missing_count":
            missing,

        "missing_rate":
            missing
            /
            max(
                len(features),
                1
            ),

        "zero_count":
            zero,

        "zero_rate":
            zero
            /
            max(
                len(features),
                1
            )
    })

    if id_col is not None:

        report.insert(
            1,
            "ID",
            df[id_col]
            .astype(str)
            .values
        )

    return report.sort_values(
        "missing_rate",
        ascending=False
    )


# =========================================================
# Operation × Tool Mean
# =========================================================

class OperationToolMeanImputer:

    """
    实际实现：

        Tool Mean
            ↓
        Global Mean

    Operation来自特征名，例如：
        311X53 -> 311

    在当前数据结构里Operation是特征级信息，
    对同一个特征的所有样本固定，因此样本级条件
    主要来自Tool。
    """

    def __init__(
        self,
        tool_col: Optional[str],
        min_group_count=3
    ):

        self.tool_col = tool_col

        self.min_group_count = (
            min_group_count
        )

        self.global_mean = {}

        self.tool_mean = {}

    # -----------------------------------------------------
    # fit
    # -----------------------------------------------------

    def fit(
        self,
        X,
        meta
    ):

        # Global Mean
        for col in X.columns:

            valid = (
                X[col]
                .dropna()
            )

            if len(valid):

                self.global_mean[col] = (
                    float(
                        valid.mean()
                    )
                )

            else:

                self.global_mean[col] = (
                    np.nan
                )

        # 没有Tool
        if (
            self.tool_col is None
            or
            self.tool_col not in meta.columns
        ):

            return self

        tools = (
            meta[
                self.tool_col
            ]
            .astype(str)
            .fillna(
                "MISSING_TOOL"
            )
        )

        temp = X.copy()

        temp["__TOOL__"] = (
            tools.to_numpy()
        )

        for col in X.columns:

            self.tool_mean[col] = {}

            grouped = (
                temp
                .groupby(
                    "__TOOL__"
                )[col]
            )

            for tool_name, values in grouped:

                valid = (
                    values
                    .dropna()
                )

                if (
                    len(valid)
                    >=
                    self.min_group_count
                ):

                    self.tool_mean[
                        col
                    ][
                        str(tool_name)
                    ] = float(
                        valid.mean()
                    )

        return self

    # -----------------------------------------------------
    # transform
    # -----------------------------------------------------

    def transform(
        self,
        X,
        meta
    ):

        result = X.copy()

        # 没有Tool
        if (
            self.tool_col is None
            or
            self.tool_col not in meta.columns
        ):

            for col, value in (
                self.global_mean.items()
            ):

                if not pd.isna(value):

                    result[col] = (
                        result[col]
                        .fillna(value)
                    )

            return result

        tools = (
            meta[
                self.tool_col
            ]
            .astype(str)
            .fillna(
                "MISSING_TOOL"
            )
        )

        for col in result.columns:

            tool_values = (
                self.tool_mean
                .get(
                    col,
                    {}
                )
            )

            global_value = (
                self.global_mean
                .get(
                    col,
                    np.nan
                )
            )

            missing_rows = (
                result.index[
                    result[col]
                    .isna()
                ]
            )

            for idx in missing_rows:

                tool_name = str(
                    tools.loc[idx]
                )

                value = (
                    tool_values.get(
                        tool_name,
                        global_value
                    )
                )

                if not pd.isna(value):

                    result.at[
                        idx,
                        col
                    ] = value

        return result


# =========================================================
# 固定最佳缺失策略
# =========================================================

def apply_best_missing_strategy(
    X_train,
    X_valid,
    meta_train,
    meta_valid,
    tool_col
):

    imputer = (
        OperationToolMeanImputer(
            tool_col
        )
    )

    # Train Fit
    imputer.fit(
        X_train,
        meta_train
    )

    # Transform
    train_out = (
        imputer.transform(
            X_train,
            meta_train
        )
    )

    valid_out = (
        imputer.transform(
            X_valid,
            meta_valid
        )
    )

    # Inf -> NaN
    train_out = (
        train_out
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    valid_out = (
        valid_out
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # Training fold median fallback
    medians = (
        train_out
        .median()
    )

    train_out = (
        train_out
        .fillna(
            medians
        )
    )

    valid_out = (
        valid_out
        .fillna(
            medians
        )
    )

    # float64
    train_out = (
        train_out
        .astype(
            np.float64
        )
    )

    valid_out = (
        valid_out
        .astype(
            np.float64
        )
    )

    return (
        train_out,
        valid_out
    )