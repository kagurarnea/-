# -*- coding: utf-8 -*-

"""
feature_engineering.py

工业AI特征工程模块

============================================================
功能
============================================================

1. 方差分析
2. 数据分布统计
3. 理论分布拟合
   - Normal
   - LogNormal
   - Gamma
   - Weibull
4. 实际分布 vs 理论分布可视化
5. Log1p
6. Yeo-Johnson
7. RandomForest 特征重要性
8. ExtraTrees 特征重要性
9. XGBoost 特征重要性
10. Top-K 特征选择
11. StandardScaler
12. MinMaxScaler
13. PCA90
14. PCA95
15. PCA50
16. TruncatedSVD50
17. NaN / Inf / 极端值安全处理

============================================================
原则
============================================================

方差：
    只用于低信息特征过滤
    不等同于预测贡献度

模型重要性：
    RF / ExtraTrees / XGB
    用于预测贡献度筛选

树模型：
    不要求标准化

PCA：
    StandardScaler
        ↓
    PCA

CV：
    所有 fit 都应该发生在 Train Fold
    Valid Fold 只能 transform

============================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer
)

from sklearn.decomposition import (
    PCA,
    TruncatedSVD
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor


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
# 数值安全处理
# =========================================================

def sanitize_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    max_abs_value: float = 1e30
):
    """
    处理：

        NaN
        +Inf
        -Inf
        极端大值

    所有填充统计量只来自Train。
    """

    train = (
        X_train
        .copy()
        .astype(np.float64)
    )

    valid = (
        X_valid
        .copy()
        .astype(np.float64)
    )

    # -----------------------------------------------------
    # Inf -> NaN
    # -----------------------------------------------------

    train = train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # 极端值 -> NaN
    # -----------------------------------------------------

    train = train.mask(
        np.abs(train) > max_abs_value
    )

    valid = valid.mask(
        np.abs(valid) > max_abs_value
    )

    # -----------------------------------------------------
    # 使用训练集Median
    # -----------------------------------------------------

    medians = train.median()

    train = train.fillna(
        medians
    )

    valid = valid.fillna(
        medians
    )

    # -----------------------------------------------------
    # 仍然完全为空的列直接删除
    # -----------------------------------------------------

    keep_columns = []

    for col in train.columns:

        if train[col].notna().any():

            keep_columns.append(
                col
            )

    train = train[
        keep_columns
    ]

    valid = valid[
        keep_columns
    ]

    # -----------------------------------------------------
    # 最终安全检查
    # -----------------------------------------------------

    if not np.isfinite(
        train.to_numpy()
    ).all():

        raise ValueError(
            "Train中仍然存在NaN或Inf。"
        )

    if not np.isfinite(
        valid.to_numpy()
    ).all():

        raise ValueError(
            "Valid中仍然存在NaN或Inf。"
        )

    return (
        train,
        valid
    )


# =========================================================
# 方差过滤
# =========================================================

def variance_filter(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    threshold: float = 0.0
):
    """
    根据训练集方差进行低信息特征过滤。
    """

    train = (
        X_train
        .copy()
        .astype(np.float64)
    )

    valid = (
        X_valid
        .copy()
        .astype(np.float64)
    )

    variance = (
        train
        .var(
            axis=0,
            ddof=0
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    keep = (
        variance[
            variance > threshold
        ]
        .index
        .tolist()
    )

    if not keep:

        raise ValueError(
            f"variance_threshold={threshold} "
            "导致所有特征被删除。"
        )

    return (
        train[
            keep
        ].copy(),

        valid[
            keep
        ].copy(),

        keep,

        variance
    )


# =========================================================
# 数据分布报告
# =========================================================

def distribution_report(
    X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    统计每个特征：

        n
        mean
        median
        std
        min
        max
        q01
        q25
        q75
        q99
        skewness
        cv
    """

    rows = []

    for col in X_train.columns:

        s = (
            pd.to_numeric(
                X_train[col],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(s) == 0:

            continue

        mean = float(
            s.mean()
        )

        std = (
            float(
                s.std()
            )
            if len(s) >= 2
            else np.nan
        )

        skewness = (
            float(
                s.skew()
            )
            if len(s) >= 3
            else np.nan
        )

        cv = (

            std
            /
            (
                abs(mean)
                + 1e-12
            )

        ) if pd.notna(
            std
        ) else np.nan

        rows.append({

            "feature":
                col,

            "n":
                int(len(s)),

            "mean":
                mean,

            "median":
                float(
                    s.median()
                ),

            "std":
                std,

            "min":
                float(
                    s.min()
                ),

            "max":
                float(
                    s.max()
                ),

            "q01":
                float(
                    s.quantile(0.01)
                ),

            "q25":
                float(
                    s.quantile(0.25)
                ),

            "q75":
                float(
                    s.quantile(0.75)
                ),

            "q99":
                float(
                    s.quantile(0.99)
                ),

            "skewness":
                skewness,

            "cv":
                cv
        })

    return pd.DataFrame(
        rows
    )


# =========================================================
# 理论分布比较
# =========================================================

def compare_theoretical_distributions(
    X_train: pd.DataFrame,
    output_dir,
    top_n: int = 30
) -> pd.DataFrame:
    """
    对偏度较大的特征进行理论分布拟合。

    比较：

        Normal
        LogNormal
        Gamma
        Weibull

    指标：

        KS statistic
        KS p-value
        AIC

    注意：

    这里是EDA分析。

    它不会自动决定最终特征变换。
    最终是否使用Log1p/Yeo-Johnson，
    仍由CV结果决定。
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report = (
        distribution_report(
            X_train
        )
    )

    if report.empty:

        return pd.DataFrame()

    # -----------------------------------------------------
    # 按绝对偏度排序
    # -----------------------------------------------------

    report["abs_skew"] = (
        report[
            "skewness"
        ]
        .abs()
    )

    features = (
        report
        .sort_values(
            "abs_skew",
            ascending=False
        )
        .head(top_n)
        [
            "feature"
        ]
        .tolist()
    )

    rows = []

    for feature in features:

        x = (
            pd.to_numeric(
                X_train[feature],
                errors="coerce"
            )
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        if len(x) < 20:

            continue

        skewness = float(
            pd.Series(
                x
            ).skew()
        )

        # =================================================
        # Normal
        # =================================================

        try:

            mu = float(
                np.mean(x)
            )

            sigma = float(
                np.std(
                    x,
                    ddof=0
                )
            )

            if (
                sigma
                >
                1e-12
                and
                np.isfinite(
                    sigma
                )
            ):

                # -----------------------------------------
                # Normal CDF
                # -----------------------------------------

                z = (
                    x - mu
                ) / sigma

                empirical = np.sort(
                    x
                )

                z_sorted = (
                    empirical - mu
                ) / sigma

                normal_cdf = (
                    stats.norm.cdf(
                        z_sorted
                    )
                )

                n = len(
                    empirical
                )

                empirical_upper = (
                    np.arange(
                        1,
                        n + 1
                    )
                    /
                    n
                )

                empirical_lower = (
                    np.arange(
                        0,
                        n
                    )
                    /
                    n
                )

                ks_stat = max(

                    np.max(
                        np.abs(
                            empirical_upper
                            -
                            normal_cdf
                        )
                    ),

                    np.max(
                        np.abs(
                            normal_cdf
                            -
                            empirical_lower
                        )
                    )
                )

                # scipy 只接收自定义CDF
                # 不再给norm传参数
                ks_result = stats.kstest(

                    x,

                    lambda value:
                    stats.norm.cdf(
                        value,
                        loc=mu,
                        scale=sigma
                    )
                )

                log_likelihood = float(
                    np.sum(
                        stats.norm.logpdf(
                            x,
                            loc=mu,
                            scale=sigma
                        )
                    )
                )

                aic = (
                    2.0 * 2.0
                    -
                    2.0
                    *
                    log_likelihood
                )

                rows.append({

                    "feature":
                        feature,

                    "distribution":
                        "normal",

                    "n":
                        len(x),

                    "ks_stat":
                        float(
                            ks_stat
                        ),

                    "ks_pvalue":
                        float(
                            ks_result.pvalue
                        ),

                    "aic":
                        float(
                            aic
                        ),

                    "skewness":
                        skewness
                })

        except Exception as error:

            log(
                f"Normal拟合失败："
                f"{feature} | "
                f"{error}"
            )

        # =================================================
        # LogNormal
        # =================================================

        x_pos = x[
            x > 0
        ]

        if len(x_pos) >= 20:

            try:

                log_x = np.log(
                    x_pos
                )

                log_mu = float(
                    np.mean(
                        log_x
                    )
                )

                log_sigma = float(
                    np.std(
                        log_x,
                        ddof=0
                    )
                )

                if (
                    log_sigma
                    >
                    1e-12
                ):

                    def lognormal_cdf(
                        values
                    ):

                        values = np.asarray(
                            values,
                            dtype=np.float64
                        )

                        result = np.zeros_like(
                            values
                        )

                        mask = (
                            values > 0
                        )

                        if np.any(mask):

                            result[mask] = (
                                stats.norm.cdf(
                                    (
                                        np.log(
                                            values[
                                                mask
                                            ]
                                        )
                                        -
                                        log_mu
                                    )
                                    /
                                    log_sigma
                                )
                            )

                        return result

                    ks_result = (
                        stats.kstest(
                            x_pos,
                            lognormal_cdf
                        )
                    )

                    log_likelihood = float(
                        np.sum(

                            -np.log(

                                x_pos
                                *
                                log_sigma
                                *
                                np.sqrt(
                                    2.0
                                    *
                                    np.pi
                                )
                            )

                            -

                            (
                                np.log(
                                    x_pos
                                )
                                -
                                log_mu
                            )
                            ** 2
                            /
                            (
                                2.0
                                *
                                log_sigma
                                ** 2
                            )
                        )
                    )

                    aic = (
                        2.0 * 2.0
                        -
                        2.0
                        *
                        log_likelihood
                    )

                    rows.append({

                        "feature":
                            feature,

                        "distribution":
                            "lognormal",

                        "n":
                            len(x_pos),

                        "ks_stat":
                            float(
                                ks_result.statistic
                            ),

                        "ks_pvalue":
                            float(
                                ks_result.pvalue
                            ),

                        "aic":
                            float(
                                aic
                            ),

                        "skewness":
                            skewness
                    })

            except Exception as error:

                log(
                    f"LogNormal拟合失败："
                    f"{feature} | "
                    f"{error}"
                )

        # =================================================
        # 近似常量判断
        # =================================================

        near_constant = False

        if len(x_pos) >= 20:

            pos_mean = float(
                np.mean(
                    x_pos
                )
            )

            pos_std = float(
                np.std(
                    x_pos,
                    ddof=0
                )
            )

            relative_std = (
                pos_std
                /
                (
                    abs(
                        pos_mean
                    )
                    +
                    1e-12
                )
            )

            near_constant = (
                relative_std < 1e-10
            )

        if near_constant:

            log(
                f"跳过Gamma/Weibull："
                f"{feature} "
                f"接近常量"
            )

        # =================================================
        # Gamma
        # =================================================

        if (
            len(x_pos) >= 20
            and
            not near_constant
        ):

            try:

                shape, loc, scale = (
                    stats.gamma.fit(
                        x_pos
                    )
                )

                if (
                    np.isfinite(shape)
                    and
                    np.isfinite(loc)
                    and
                    np.isfinite(scale)
                    and
                    shape > 0
                    and
                    scale > 0
                ):

                    ks_result = (
                        stats.kstest(
                            x_pos,
                            "gamma",
                            args=(
                                shape,
                                loc,
                                scale
                            )
                        )
                    )

                    log_likelihood = float(
                        np.sum(
                            stats.gamma.logpdf(
                                x_pos,
                                shape,
                                loc=loc,
                                scale=scale
                            )
                        )
                    )

                    if np.isfinite(
                        log_likelihood
                    ):

                        aic = (
                            2.0 * 3.0
                            -
                            2.0
                            *
                            log_likelihood
                        )

                        rows.append({

                            "feature":
                                feature,

                            "distribution":
                                "gamma",

                            "n":
                                len(x_pos),

                            "ks_stat":
                                float(
                                    ks_result.statistic
                                ),

                            "ks_pvalue":
                                float(
                                    ks_result.pvalue
                                ),

                            "aic":
                                float(
                                    aic
                                ),

                            "skewness":
                                skewness
                        })

            except Exception as error:

                log(
                    f"Gamma跳过："
                    f"{feature} | "
                    f"{error}"
                )

        # =================================================
        # Weibull
        # =================================================

        if (
            len(x_pos) >= 20
            and
            not near_constant
        ):

            try:

                shape, loc, scale = (
                    stats.weibull_min.fit(
                        x_pos
                    )
                )

                if (
                    np.isfinite(shape)
                    and
                    np.isfinite(loc)
                    and
                    np.isfinite(scale)
                    and
                    shape > 0
                    and
                    scale > 0
                ):

                    ks_result = (
                        stats.kstest(
                            x_pos,
                            "weibull_min",
                            args=(
                                shape,
                                loc,
                                scale
                            )
                        )
                    )

                    log_likelihood = float(
                        np.sum(
                            stats.weibull_min.logpdf(
                                x_pos,
                                shape,
                                loc=loc,
                                scale=scale
                            )
                        )
                    )

                    if np.isfinite(
                        log_likelihood
                    ):

                        aic = (
                            2.0 * 3.0
                            -
                            2.0
                            *
                            log_likelihood
                        )

                        rows.append({

                            "feature":
                                feature,

                            "distribution":
                                "weibull",

                            "n":
                                len(x_pos),

                            "ks_stat":
                                float(
                                    ks_result.statistic
                                ),

                            "ks_pvalue":
                                float(
                                    ks_result.pvalue
                                ),

                            "aic":
                                float(
                                    aic
                                ),

                            "skewness":
                                skewness
                        })

            except Exception as error:

                log(
                    f"Weibull跳过："
                    f"{feature} | "
                    f"{error}"
                )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    # =====================================================
    # 每个特征的最佳理论分布
    # =====================================================

    best_rows = []

    for feature, group in (
        result.groupby(
            "feature"
        )
    ):

        best = (
            group
            .sort_values(
                [
                    "aic",
                    "ks_stat"
                ]
            )
            .iloc[0]
        )

        best_rows.append({

            "feature":
                feature,

            "best_distribution":
                best[
                    "distribution"
                ],

            "best_aic":
                best[
                    "aic"
                ],

            "best_ks_stat":
                best[
                    "ks_stat"
                ],

            "best_ks_pvalue":
                best[
                    "ks_pvalue"
                ]
        })

    best_df = pd.DataFrame(
        best_rows
    )

    result = result.merge(
        best_df,
        on="feature",
        how="left"
    )

    # =====================================================
    # 保存
    # =====================================================

    result.to_csv(

        output_dir
        /
        "distribution_comparison.csv",

        index=False,

        encoding="utf-8-sig"
    )

    return result


# =========================================================
# 分布可视化
# =========================================================

def plot_distribution_comparison(
    X_train,
    comparison_df,
    output_dir,
    top_n=10
):

    """
    生成：

        实际直方图
        +
        最优理论分布PDF

    输出PNG。
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        comparison_df is None
        or
        comparison_df.empty
    ):

        return []

    features = (
        comparison_df[
            "feature"
        ]
        .drop_duplicates()
        .tolist()
    )

    features = features[
        :top_n
    ]

    saved = []

    for feature in features:

        row = (
            comparison_df[
                comparison_df[
                    "feature"
                ]
                ==
                feature
            ]
            .iloc[0]
        )

        dist_name = (
            row[
                "best_distribution"
            ]
        )

        x = (
            pd.to_numeric(
                X_train[feature],
                errors="coerce"
            )
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        if len(x) < 20:

            continue

        # -------------------------------------------------
        # 正值分布
        # -------------------------------------------------

        if dist_name in {

            "lognormal",

            "gamma",

            "weibull"

        }:

            x_fit = x[
                x > 0
            ]

        else:

            x_fit = x

        if len(x_fit) < 20:

            continue

        try:

            # -------------------------------------------------
            # Normal
            # -------------------------------------------------

            if dist_name == "normal":

                dist = stats.norm

                mu = float(
                    np.mean(
                        x_fit
                    )
                )

                sigma = float(
                    np.std(
                        x_fit,
                        ddof=0
                    )
                )

                if sigma <= 1e-12:

                    continue

                params = (
                    mu,
                    sigma
                )

            # -------------------------------------------------
            # LogNormal
            # -------------------------------------------------

            elif dist_name == "lognormal":

                dist = stats.lognorm

                shape, loc, scale = (
                    stats.lognorm.fit(
                        x_fit,
                        floc=0
                    )
                )

                params = (
                    shape,
                    loc,
                    scale
                )

            # -------------------------------------------------
            # Gamma
            # -------------------------------------------------

            elif dist_name == "gamma":

                dist = stats.gamma

                params = (
                    dist.fit(
                        x_fit
                    )
                )

            # -------------------------------------------------
            # Weibull
            # -------------------------------------------------

            elif dist_name == "weibull":

                dist = stats.weibull_min

                params = (
                    dist.fit(
                        x_fit
                    )
                )

            else:

                continue

            xmin = float(
                np.min(
                    x_fit
                )
            )

            xmax = float(
                np.max(
                    x_fit
                )
            )

            if np.isclose(
                xmin,
                xmax
            ):

                continue

            grid = np.linspace(
                xmin,
                xmax,
                500
            )

            pdf = (
                dist.pdf(
                    grid,
                    *params
                )
            )

            # -------------------------------------------------
            # Figure
            # -------------------------------------------------

            fig = plt.figure(
                figsize=(
                    10,
                    6
                )
            )

            plt.hist(

                x_fit,

                bins=30,

                density=True,

                alpha=0.55,

                label="Observed"
            )

            plt.plot(

                grid,

                pdf,

                linewidth=2,

                label=(
                    "Theoretical: "
                    f"{dist_name}"
                )
            )

            plt.title(

                f"{feature}: "
                f"Observed vs "
                f"{dist_name}"
            )

            plt.xlabel(
                str(feature)
            )

            plt.ylabel(
                "Density"
            )

            plt.legend()

            plt.tight_layout()

            output_file = (

                output_dir
                /
                f"{feature}_distribution.png"
            )

            fig.savefig(

                output_file,

                dpi=180,

                bbox_inches="tight"
            )

            plt.close(
                fig
            )

            saved.append(
                output_file
            )

        except Exception as error:

            log(
                f"理论分布绘图失败："
                f"{feature} | "
                f"{error}"
            )

    return saved


# =========================================================
# 特征变换
# =========================================================

def transform_features(
    X_train,
    X_valid,
    method="none"
):

    """
    特征变换：

        none
        log1p
        yeo_johnson

    统一使用float64，
    避免Pandas整数列写入浮点结果的问题。
    """

    train = (
        X_train
        .copy()
        .astype(
            np.float64
        )
    )

    valid = (
        X_valid
        .copy()
        .astype(
            np.float64
        )
    )

    # -----------------------------------------------------
    # None
    # -----------------------------------------------------

    if method == "none":

        return (
            train,
            valid,
            None
        )

    # -----------------------------------------------------
    # 变换前安全处理
    # -----------------------------------------------------

    train, valid = (
        sanitize_features(
            train,
            valid
        )
    )

    # =====================================================
    # Log1p
    # =====================================================

    if method == "log1p":

        transformed = []

        for col in train.columns:

            train_min = (
                train[col].min()
            )

            valid_min = (
                valid[col].min()
            )

            if (
                pd.notna(
                    train_min
                )
                and
                pd.notna(
                    valid_min
                )
                and
                train_min >= 0
                and
                valid_min >= 0
            ):

                train[col] = (
                    np.log1p(
                        train[col]
                    )
                )

                valid[col] = (
                    np.log1p(
                        valid[col]
                    )
                )

                transformed.append(
                    col
                )

        train, valid = (
            sanitize_features(
                train,
                valid
            )
        )

        return (
            train,
            valid,
            transformed
        )

    # =====================================================
    # Yeo-Johnson
    # =====================================================

    if method == "yeo_johnson":

        columns = [

            col

            for col in train.columns

            if train[col]
            .nunique(
                dropna=True
            ) > 1
        ]

        if not columns:

            return (
                train,
                valid,
                None
            )

        transformer = (
            PowerTransformer(

                method=
                    "yeo-johnson",

                standardize=False
            )
        )

        # -------------------------------------------------
        # Train fit
        # -------------------------------------------------

        train_values = (
            transformer
            .fit_transform(
                train[
                    columns
                ]
            )
        )

        # -------------------------------------------------
        # Valid transform
        # -------------------------------------------------

        valid_values = (
            transformer
            .transform(
                valid[
                    columns
                ]
            )
        )

        # -------------------------------------------------
        # 重新构造成float64
        # -------------------------------------------------

        train = pd.DataFrame(

            train_values,

            index=train.index,

            columns=columns
        ).astype(
            np.float64
        )

        valid = pd.DataFrame(

            valid_values,

            index=valid.index,

            columns=columns
        ).astype(
            np.float64
        )

        # -------------------------------------------------
        # 再做安全处理
        # -------------------------------------------------

        train, valid = (
            sanitize_features(
                train,
                valid
            )
        )

        return (
            train,
            valid,
            transformer
        )

    raise ValueError(
        f"未知特征变换：{method}"
    )


# =========================================================
# 模型特征重要性
# =========================================================

def model_feature_importance(
    X_train,
    y_train,
    method="extra_trees",
    top_k=300,
    seed=42
):

    """
    支持：

        rf
        extra_trees
        xgb
    """

    # -----------------------------------------------------
    # RF
    # -----------------------------------------------------

    if method == "rf":

        model = (
            RandomForestRegressor(

                n_estimators=
                    300,

                max_features=
                    "sqrt",

                min_samples_leaf=
                    2,

                random_state=
                    seed,

                n_jobs=
                    -1
            )
        )

    # -----------------------------------------------------
    # ExtraTrees
    # -----------------------------------------------------

    elif method == "extra_trees":

        model = (
            ExtraTreesRegressor(

                n_estimators=
                    300,

                max_features=
                    "sqrt",

                min_samples_leaf=
                    2,

                random_state=
                    seed,

                n_jobs=
                    -1
            )
        )

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------

    elif method == "xgb":

        model = (
            XGBRegressor(

                objective=
                    "reg:squarederror",

                n_estimators=
                    500,

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

                random_state=
                    seed,

                n_jobs=
                    -1,

                tree_method=
                    "hist"
            )
        )

    else:

        raise ValueError(
            f"未知模型：{method}"
        )

    # -----------------------------------------------------
    # fit
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # importance
    # -----------------------------------------------------

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
                int(top_k),
                len(importance)
            )
        )

        .index

        .tolist()
    )

    return (
        selected,
        importance,
        model
    )


# =========================================================
# 综合特征工程
# =========================================================

def apply_feature_engineering(
    X_train,
    X_valid,
    y_train,
    selector="baseline",
    variance_threshold=0.0,
    transform="none",
    top_k=300,
    seed=42
):

    # -----------------------------------------------------
    # 1. 数值安全
    # -----------------------------------------------------

    train, valid = (
        sanitize_features(
            X_train,
            X_valid
        )
    )

    # -----------------------------------------------------
    # 2. 方差
    # -----------------------------------------------------

    (
        train,
        valid,
        variance_columns,
        variance
    ) = variance_filter(

        train,

        valid,

        threshold=
            variance_threshold
    )

    # -----------------------------------------------------
    # 3. 特征变换
    # -----------------------------------------------------

    (
        train,
        valid,
        transform_info
    ) = transform_features(

        train,

        valid,

        method=
            transform
    )

    # -----------------------------------------------------
    # 4. 特征贡献度选择
    # -----------------------------------------------------

    if selector == "baseline":

        selected_columns = (
            list(
                train.columns
            )
        )

        importance = None

    else:

        (
            selected_columns,
            importance,
            _model
        ) = model_feature_importance(

            train,

            y_train,

            method=
                selector,

            top_k=
                top_k,

            seed=
                seed
        )

        train = (
            train[
                selected_columns
            ]
        )

        valid = (
            valid[
                selected_columns
            ]
        )

    # -----------------------------------------------------
    # 5. 最终安全检查
    # -----------------------------------------------------

    train, valid = (
        sanitize_features(
            train,
            valid
        )
    )

    return (

        train,

        valid,

        {

            "variance_columns":
                variance_columns,

            "variance":
                variance,

            "selected_columns":
                selected_columns,

            "importance":
                importance,

            "transform":
                transform_info
        }
    )


# =========================================================
# Scaling
# =========================================================

def scale_features(
    X_train,
    X_valid,
    scaler="none"
):

    """
    支持：

        none
        standard
        minmax
    """

    if scaler == "none":

        return (

            X_train.copy(),

            X_valid.copy(),

            None
        )

    # -----------------------------------------------------
    # 安全
    # -----------------------------------------------------

    X_train, X_valid = (
        sanitize_features(
            X_train,
            X_valid
        )
    )

    if scaler == "standard":

        transformer = (
            StandardScaler()
        )

    elif scaler == "minmax":

        transformer = (
            MinMaxScaler()
        )

    else:

        raise ValueError(
            f"未知Scaler：{scaler}"
        )

    # -----------------------------------------------------
    # Train fit
    # -----------------------------------------------------

    train_values = (
        transformer
        .fit_transform(
            X_train
        )
    )

    # -----------------------------------------------------
    # Valid transform
    # -----------------------------------------------------

    valid_values = (
        transformer
        .transform(
            X_valid
        )
    )

    train = pd.DataFrame(

        train_values,

        index=X_train.index,

        columns=X_train.columns
    )

    valid = pd.DataFrame(

        valid_values,

        index=X_valid.index,

        columns=X_valid.columns
    )

    return (
        train,
        valid,
        transformer
    )


# =========================================================
# PCA / SVD
# =========================================================

def apply_pca(
    X_train,
    X_valid,
    reducer="none"
):

    """
    降维流程：

        原始特征
            ↓
        StandardScaler
            ↓
        PCA / SVD

    支持：

        none
        pca90
        pca95
        pca50
        svd50
    """

    # -----------------------------------------------------
    # 不降维
    # -----------------------------------------------------

    if reducer == "none":

        return (

            X_train,

            X_valid,

            {

                "method":
                    "none",

                "n_components":
                    X_train.shape[1],

                "explained_variance":
                    1.0,

                "scaler":
                    None,

                "model":
                    None
            }
        )

    # -----------------------------------------------------
    # StandardScaler
    # -----------------------------------------------------

    (
        train_scaled,
        valid_scaled,
        scaler
    ) = scale_features(

        X_train,

        X_valid,

        scaler="standard"
    )

    # -----------------------------------------------------
    # PCA90
    # -----------------------------------------------------

    if reducer == "pca90":

        model = PCA(

            n_components=
                0.90,

            svd_solver=
                "full"
        )

    # -----------------------------------------------------
    # PCA95
    # -----------------------------------------------------

    elif reducer == "pca95":

        model = PCA(

            n_components=
                0.95,

            svd_solver=
                "full"
        )

    # -----------------------------------------------------
    # PCA50
    # -----------------------------------------------------

    elif reducer == "pca50":

        n_components = min(

            50,

            train_scaled.shape[1],

            train_scaled.shape[0] - 1
        )

        if n_components < 1:

            raise ValueError(
                "PCA50没有可用成分。"
            )

        model = PCA(

            n_components=
                n_components
        )

    # -----------------------------------------------------
    # SVD50
    # -----------------------------------------------------

    elif reducer == "svd50":

        n_components = min(

            50,

            train_scaled.shape[1] - 1
        )

        if n_components < 1:

            raise ValueError(
                "SVD50没有可用成分。"
            )

        model = TruncatedSVD(

            n_components=
                n_components,

            random_state=
                42
        )

    else:

        raise ValueError(
            f"未知降维方法：{reducer}"
        )

    # -----------------------------------------------------
    # Train fit
    # -----------------------------------------------------

    train_values = (
        model
        .fit_transform(
            train_scaled
        )
    )

    # -----------------------------------------------------
    # Valid transform
    # -----------------------------------------------------

    valid_values = (
        model
        .transform(
            valid_scaled
        )
    )

    train = pd.DataFrame(
        train_values,
        index=X_train.index
    )

    valid = pd.DataFrame(
        valid_values,
        index=X_valid.index
    )

    # -----------------------------------------------------
    # 解释方差
    # -----------------------------------------------------

    if hasattr(
        model,
        "explained_variance_ratio_"
    ):

        explained_variance = float(

            model
            .explained_variance_ratio_
            .sum()
        )

    else:

        explained_variance = np.nan

    return (

        train,

        valid,

        {

            "method":
                reducer,

            "n_components":
                train.shape[1],

            "explained_variance":
                explained_variance,

            "scaler":
                scaler,

            "model":
                model
        }
    )


# =========================================================
# main.py兼容别名
# =========================================================

def reduce_dimension(
    X_train,
    X_valid,
    reducer="none"
):

    return apply_pca(
        X_train,
        X_valid,
        reducer
    )