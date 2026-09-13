from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .config import PipelineConfig


TARGET_CANDIDATES = ("Value", "value", "Y", "y")
ID_CANDIDATES = ("ID", "id", "Id")
TIMESTAMP_LENGTHS = (8, 14, 16)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def infer_operation(feature: str) -> str:
    match = re.match(r"^\s*(\d+[A-Za-z]?)\s*[Xx]", str(feature))
    return match.group(1) if match else "META"


def operation_sort_key(operation: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)([A-Za-z]?)$", str(operation))
    if not match:
        return (10**9, str(operation))
    return (int(match.group(1)), match.group(2))


def id_number(value: object) -> float:
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def metric_row(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, prediction))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


class RunTrace:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(
        self,
        stage: str,
        started: float,
        rows: int | None = None,
        columns: int | None = None,
        note: str = "",
    ) -> None:
        self.rows.append(
            {
                "stage": stage,
                "elapsed_sec": time.perf_counter() - started,
                "rows": rows,
                "columns": columns,
                "note": note,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def read_excel_cached(path: Path, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{path.stem}.pkl"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        log(f"读取缓存：{cache_path.name}")
        return pd.read_pickle(cache_path)

    log(f"读取Excel：{path.name}")
    frame = pd.read_excel(path)
    frame.to_pickle(cache_path)
    return frame


def timestamp_ratio(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values.loc[values.abs() < np.iinfo(np.int64).max]
    if values.empty:
        return 0.0
    sample = values.iloc[: min(300, len(values))].round().astype("int64").astype(str)
    length_ok = sample.str.len().isin(TIMESTAMP_LENGTHS)
    year_ok = sample.str.startswith(("2016", "2017", "2018", "2019", "2020"))
    return float((length_ok & year_ok).mean())


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.where(numeric.abs() < np.iinfo(np.int64).max)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = numeric.notna()
    if not valid.any():
        return result

    text = numeric.loc[valid].round().astype("int64").astype(str)
    for length, fmt in ((8, "%Y%m%d"), (14, "%Y%m%d%H%M%S"), (16, "%Y%m%d%H%M%S")):
        mask = text.str.len() == length
        if not mask.any():
            continue
        part = text.loc[mask]
        core = part.str[:14] if length == 16 else part
        parsed = pd.to_datetime(core, format=fmt, errors="coerce")
        seconds = parsed.astype("int64").astype(float) / 1e9
        seconds.loc[parsed.isna()] = np.nan
        if length == 16:
            seconds = seconds + pd.to_numeric(part.str[14:16], errors="coerce") / 100.0
        result.loc[part.index] = seconds
    return result


def duplicate_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    seen: dict[bytes, str] = {}
    duplicates: list[str] = []
    for column in columns:
        signature = pd.util.hash_pandas_object(
            frame[column], index=False
        ).to_numpy().tobytes()
        if signature in seen and frame[column].equals(frame[seen[signature]]):
            duplicates.append(column)
        else:
            seen[signature] = column
    return duplicates


@dataclass
class PreparedData:
    train_numeric: pd.DataFrame
    test_a_numeric: pd.DataFrame
    test_b_numeric: pd.DataFrame
    train_categorical: pd.DataFrame
    test_a_categorical: pd.DataFrame
    test_b_categorical: pd.DataFrame
    train_time: pd.DataFrame
    test_a_time: pd.DataFrame
    test_b_time: pd.DataFrame
    y: pd.Series
    train_ids: pd.Series
    test_a_ids: pd.Series
    test_b_ids: pd.Series
    order_key: pd.Series
    feature_operation: dict[str, str]
    time_operation: dict[str, str]
    operation_tool: dict[str, str | None]
    feature_audit: pd.DataFrame
    operation_audit: pd.DataFrame
    primary_tool_column: str | None


def validation_groups(data: PreparedData) -> np.ndarray:
    """Group repeated IDs and exact predictor rows without looking at targets.

    The transitive closure matters when one row shares an ID with a second row
    and that second row shares all predictors with a third row.
    """
    frame = pd.concat([data.train_numeric.reset_index(drop=True),
                       data.train_categorical.reset_index(drop=True)], axis=1)
    parents = np.arange(len(frame))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def unite(left: int, right: int) -> None:
        parents[root(right)] = root(left)

    seen_ids: dict[str, int] = {}
    for index, value in enumerate(data.train_ids):
        if pd.isna(value):
            continue
        key = str(value)
        if key in seen_ids:
            unite(seen_ids[key], index)
        else:
            seen_ids[key] = index
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    seen_rows: dict[int, list[int]] = {}
    for index, signature in enumerate(hashes):
        bucket = seen_rows.setdefault(int(signature), [])
        for previous in bucket:
            # Hashes accelerate lookup; equality guards against hash collisions.
            if frame.iloc[index].equals(frame.iloc[previous]):
                unite(previous, index)
                break
        bucket.append(index)
    return np.asarray([root(index) for index in range(len(frame))], dtype=int)


def purge_training_overlap(
    train_index: np.ndarray, valid_index: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Keep all evaluation rows; exclude their entities from the training side."""
    train_index = np.asarray(train_index, dtype=int)
    valid_index = np.asarray(valid_index, dtype=int)
    keep = ~np.isin(groups[train_index], groups[valid_index])
    result = train_index[keep]
    if not len(result) or not len(valid_index):
        raise ValueError("Validation split has an empty training or evaluation set after purging")
    return result


def prepare_data(config: PipelineConfig, trace: RunTrace) -> PreparedData:
    started = time.perf_counter()
    train = read_excel_cached(config.data_dir / config.train_file, config.cache_dir)
    test_a = read_excel_cached(config.data_dir / config.test_a_file, config.cache_dir)
    test_b = read_excel_cached(config.data_dir / config.test_b_file, config.cache_dir)

    target = find_column(train, TARGET_CANDIDATES)
    id_col = find_column(train, ID_CANDIDATES)
    if target is None or id_col is None:
        raise RuntimeError("训练集缺少ID或Value/Y列")

    y = pd.to_numeric(train[target], errors="coerce")
    valid_target = y.notna()
    train = train.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    original_columns = list(train.columns)
    categorical = [
        column
        for column in train.columns
        if column not in {target, id_col}
        and (
            not pd.api.types.is_numeric_dtype(train[column])
            or any(
                token in str(column).lower()
                for token in ("tool", "chamber", "operation")
            )
        )
    ]
    numeric = [
        column
        for column in train.columns
        if column not in {target, id_col, *categorical}
        and pd.api.types.is_numeric_dtype(train[column])
    ]

    all_nan = [column for column in numeric if train[column].isna().all()]
    constant = [
        column
        for column in numeric
        if column not in all_nan and train[column].nunique(dropna=True) <= 1
    ]
    duplicate_candidates = [
        column for column in numeric if column not in set(all_nan + constant)
    ]
    duplicates = duplicate_columns(train, duplicate_candidates)
    removed = set(all_nan + constant + duplicates)
    structurally_clean = [column for column in numeric if column not in removed]

    date_columns = [
        column
        for column in structurally_clean
        if timestamp_ratio(train[column]) >= 0.75
    ]
    model_numeric = [column for column in structurally_clean if column not in date_columns]

    audit_rows: list[dict[str, object]] = []
    for position, column in enumerate(numeric):
        if column in all_nan:
            status = "removed_all_nan"
        elif column in constant:
            status = "removed_constant"
        elif column in duplicates:
            status = "removed_duplicate"
        elif column in date_columns:
            status = "converted_timestamp"
        else:
            status = "kept_numeric"
        series = train[column]
        audit_rows.append(
            {
                "feature": column,
                "position": position,
                "operation": infer_operation(column),
                "status": status,
                "missing_rate": float(series.isna().mean()),
                "zero_rate": float(series.eq(0).mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "timestamp_ratio": timestamp_ratio(series),
            }
        )
    feature_audit = pd.DataFrame(audit_rows)

    feature_operation = {column: infer_operation(column) for column in model_numeric}
    time_operation = {column: infer_operation(column) for column in date_columns}

    operation_positions: dict[str, int] = {}
    for column in numeric:
        operation = infer_operation(column)
        if operation != "META":
            operation_positions.setdefault(operation, original_columns.index(column))

    categorical_positions = {
        column: original_columns.index(column)
        for column in categorical
        if column in original_columns
    }
    operation_tool: dict[str, str | None] = {}
    for operation, position in operation_positions.items():
        previous = [
            (cat_position, column)
            for column, cat_position in categorical_positions.items()
            if cat_position < position
        ]
        operation_tool[operation] = max(previous)[1] if previous else None

    # Full-data structural statistics above are descriptive audit output only.
    # Every fit must receive the original numeric schema, including fields that
    # are constant/duplicated only in a particular training subset.
    feature_operation = {column: infer_operation(column) for column in numeric}
    train_numeric = train.reindex(columns=numeric).apply(
        pd.to_numeric, errors="coerce"
    )
    test_a_numeric = test_a.reindex(columns=numeric).apply(
        pd.to_numeric, errors="coerce"
    )
    test_b_numeric = test_b.reindex(columns=numeric).apply(
        pd.to_numeric, errors="coerce"
    )

    def categorical_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.reindex(columns=categorical)
            .astype("string")
            .fillna("__MISSING__")
        )

    train_categorical = categorical_frame(train)
    test_a_categorical = categorical_frame(test_a)
    test_b_categorical = categorical_frame(test_b)

    def time_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {column: parse_timestamp_series(frame[column]) for column in date_columns},
            index=frame.index,
        )

    train_time = time_frame(train)
    test_a_time = time_frame(test_a)
    test_b_time = time_frame(test_b)

    order_key = train_time.median(axis=1, skipna=True)
    fallback = train[id_col].map(id_number)
    if order_key.notna().any():
        base = float(order_key.dropna().median())
        fallback_scaled = base + fallback.fillna(fallback.median())
        order_key = order_key.fillna(fallback_scaled)
    else:
        order_key = fallback.fillna(pd.Series(np.arange(len(train)), index=train.index))

    operation_audit = (
        feature_audit.groupby("operation", as_index=False)
        .agg(
            original_features=("feature", "size"),
            kept_numeric=("status", lambda values: int((values == "kept_numeric").sum())),
            time_features=(
                "status",
                lambda values: int((values == "converted_timestamp").sum()),
            ),
            mean_missing_rate=("missing_rate", "mean"),
            mean_zero_rate=("zero_rate", "mean"),
        )
        .sort_values("operation", key=lambda values: values.map(operation_sort_key))
    )

    primary_tool = "TOOL" if "TOOL" in categorical else (categorical[0] if categorical else None)
    trace.add(
        "数据读取与结构审计",
        started,
        rows=len(train),
        columns=len(model_numeric) + len(categorical) + len(date_columns),
        note=(
            f"全量描述性审计：全空{len(all_nan)}、常量{len(constant)}、重复{len(duplicates)}；"
            f"候选时间列{len(date_columns)}；实际筛选仅在每次训练子集内拟合"
        ),
    )
    log(
        f"数据准备完成：train={train.shape}，清洗数值={len(model_numeric)}，"
        f"类别={len(categorical)}，时间={len(date_columns)}"
    )
    return PreparedData(
        train_numeric=train_numeric,
        test_a_numeric=test_a_numeric,
        test_b_numeric=test_b_numeric,
        train_categorical=train_categorical,
        test_a_categorical=test_a_categorical,
        test_b_categorical=test_b_categorical,
        train_time=train_time,
        test_a_time=test_a_time,
        test_b_time=test_b_time,
        y=y,
        train_ids=train[id_col].reset_index(drop=True),
        test_a_ids=test_a[id_col].reset_index(drop=True),
        test_b_ids=test_b[id_col].reset_index(drop=True),
        order_key=order_key.reset_index(drop=True),
        feature_operation=feature_operation,
        time_operation=time_operation,
        operation_tool=operation_tool,
        feature_audit=feature_audit,
        operation_audit=operation_audit,
        primary_tool_column=primary_tool,
    )


class FoldFeatureBuilder:
    """Fold-local imputation, stable selection and process-aware features."""

    def __init__(self, data: PreparedData, config: PipelineConfig, seed: int):
        self.data = data
        self.config = config
        self.seed = seed
        self.global_median: pd.Series | None = None
        self.group_means: dict[str, pd.DataFrame] = {}
        self.group_columns: dict[str, list[str]] = {}
        self.robust_median: pd.Series | None = None
        self.robust_iqr: pd.Series | None = None
        self.time_origin: float = 0.0
        self.encoder: OneHotEncoder | None = None
        self.selected_features: list[str] = []
        self.missing_indicator_features: list[str] = []
        self.selection_report: pd.DataFrame = pd.DataFrame()
        self.feature_names_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.raw_time_columns_: list[str] = []
        self.time_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.structural_report_: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _cat_values(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.astype("string").fillna("__MISSING__")

    def fit(
        self,
        numeric: pd.DataFrame,
        categorical: pd.DataFrame,
        time_frame: pd.DataFrame,
        y: pd.Series,
        order_key: pd.Series,
    ) -> "FoldFeatureBuilder":
        # Reset fitted state so reusing a builder never retains prior-fold state.
        self.group_means = {}
        self.feature_names_ = []
        raw = numeric.replace([np.inf, -np.inf], np.nan)
        all_nan = [column for column in raw if raw[column].isna().all()]
        # A constant observed value plus missingness can still carry information.
        constant = [column for column in raw if column not in all_nan
                    and raw[column].nunique(dropna=False) <= 1]
        candidates = [column for column in raw if column not in set(all_nan + constant)]
        duplicates = duplicate_columns(raw, candidates)
        kept = [column for column in candidates if column not in set(duplicates)]
        self.raw_time_columns_ = [column for column in kept
                                  if timestamp_ratio(raw[column]) >= 0.75]
        self.numeric_columns_ = [column for column in kept
                                 if column not in self.raw_time_columns_]
        # Legacy callers may supply parsed time fields not present in numeric.
        legacy_times = [column for column in time_frame if column not in raw.columns
                        and time_frame[column].nunique(dropna=False) > 1]
        self.time_columns_ = self.raw_time_columns_ + legacy_times
        self.categorical_columns_ = [column for column in categorical
                                     if categorical[column].nunique(dropna=False) > 1]
        self.structural_report_ = pd.DataFrame({
            "feature": list(raw.columns),
            "status": ["removed_all_nan" if c in all_nan else
                       "removed_constant" if c in constant else
                       "removed_duplicate" if c in duplicates else
                       "converted_timestamp" if c in self.raw_time_columns_ else
                       "kept_numeric" for c in raw],
            "fit_rows": len(raw),
        })
        time_frame = self._time_input(raw, time_frame)
        numeric = raw.reindex(columns=self.numeric_columns_)
        categorical = categorical.reindex(columns=self.categorical_columns_)
        self.global_median = numeric.median().fillna(0.0)

        by_tool: dict[str, list[str]] = {}
        for column in numeric.columns:
            operation = self.data.feature_operation.get(column, "META")
            tool = self.data.operation_tool.get(operation)
            if tool is not None and tool in categorical.columns:
                by_tool.setdefault(tool, []).append(column)
        self.group_columns = by_tool

        cat = self._cat_values(categorical)
        for tool, columns in by_tool.items():
            grouped = numeric[columns].copy()
            grouped["__group__"] = cat[tool].to_numpy()
            self.group_means[tool] = grouped.groupby("__group__", observed=True).mean()

        imputed = self.impute(numeric, categorical)
        self.robust_median = imputed.median().fillna(0.0)
        q75 = imputed.quantile(0.75)
        q25 = imputed.quantile(0.25)
        self.robust_iqr = (q75 - q25).replace(0, 1.0).fillna(1.0)

        finite_time = time_frame.to_numpy(dtype=float)
        finite_time = finite_time[np.isfinite(finite_time)]
        self.time_origin = float(np.min(finite_time)) if finite_time.size else 0.0

        self.encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        if len(cat.columns):
            self.encoder.fit(cat)

        selector = ExtraTreesRegressor(
            n_estimators=self.config.selector_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=self.seed,
            n_jobs=-1,
        )
        if imputed.shape[1]:
            selector.fit(imputed, y)
            importance = pd.Series(selector.feature_importances_, index=imputed.columns)
        else:
            importance = pd.Series(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = imputed.corrwith(
                pd.Series(y.to_numpy(), index=imputed.index)
            ).abs()
        correlation = correlation.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        order = np.argsort(order_key.to_numpy(dtype=float))
        split = max(1, len(order) // 2)
        early = imputed.iloc[order[:split]]
        late = imputed.iloc[order[split:]]
        drift = (late.median() - early.median()).abs() / self.robust_iqr
        drift = drift.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 20)

        importance_rank = importance.rank(pct=True)
        correlation_rank = correlation.rank(pct=True)
        stability = 1.0 / (1.0 + drift)
        score = (0.70 * importance_rank + 0.30 * correlation_rank) * np.power(
            stability, 0.25
        )

        top_k = min(self.config.top_k_raw_features, len(score))
        self.selected_features = score.nlargest(top_k).index.tolist()
        missing_rate = numeric[self.selected_features].isna().mean()
        self.missing_indicator_features = missing_rate.loc[
            missing_rate >= self.config.missing_indicator_threshold
        ].index.tolist()
        selected_set = set(self.selected_features)
        self.selection_report = pd.DataFrame(
            {
                "feature": imputed.columns,
                "operation": [
                    self.data.feature_operation.get(column, "META")
                    for column in imputed.columns
                ],
                "importance": importance.reindex(imputed.columns).to_numpy(),
                "abs_correlation": correlation.reindex(imputed.columns).to_numpy(),
                "temporal_drift": drift.reindex(imputed.columns).to_numpy(),
                "stable_score": score.reindex(imputed.columns).to_numpy(),
                "selected": [column in selected_set for column in imputed.columns],
            }
        )
        return self

    def _time_input(self, numeric: pd.DataFrame, time_frame: pd.DataFrame) -> pd.DataFrame:
        parsed = {column: parse_timestamp_series(numeric[column])
                  for column in self.raw_time_columns_}
        for column in self.time_columns_:
            if column not in parsed:
                parsed[column] = time_frame[column]
        return pd.DataFrame(parsed, index=numeric.index)

    def impute(
        self, numeric: pd.DataFrame, categorical: pd.DataFrame
    ) -> pd.DataFrame:
        if self.global_median is None:
            raise RuntimeError("FoldFeatureBuilder尚未拟合")
        result = numeric.reindex(columns=self.numeric_columns_).replace(
            [np.inf, -np.inf], np.nan
        )
        cat = self._cat_values(categorical)
        for tool, columns in self.group_columns.items():
            if tool not in cat.columns or tool not in self.group_means:
                continue
            labels = cat[tool]
            fill_values = self.group_means[tool].reindex(labels.to_numpy())
            fill_values.index = result.index
            fill_values.columns = columns
            result.loc[:, columns] = result[columns].where(
                result[columns].notna(), fill_values
            )
        return (
            result.fillna(self.global_median)
            .replace([np.inf, -np.inf], 0.0)
            .astype(np.float64)
        )

    def _process_features(
        self, original: pd.DataFrame, imputed: pd.DataFrame
    ) -> pd.DataFrame:
        assert self.robust_median is not None and self.robust_iqr is not None
        robust = ((imputed - self.robust_median) / self.robust_iqr).clip(-8, 8)
        output: dict[str, pd.Series] = {}
        operations = sorted(
            set(self.data.feature_operation.values()), key=operation_sort_key
        )
        for operation in operations:
            if operation == "META":
                continue
            columns = [
                column
                for column in imputed.columns
                if self.data.feature_operation.get(column) == operation
            ]
            if not columns:
                continue
            values = robust[columns]
            prefix = f"proc__{operation}__"
            output[prefix + "mean_z"] = values.mean(axis=1)
            output[prefix + "std_z"] = values.std(axis=1).fillna(0.0)
            output[prefix + "min_z"] = values.min(axis=1)
            output[prefix + "max_z"] = values.max(axis=1)
            output[prefix + "median_z"] = values.median(axis=1)
            output[prefix + "q25_z"] = values.quantile(0.25, axis=1)
            output[prefix + "q75_z"] = values.quantile(0.75, axis=1)
            output[prefix + "range_z"] = values.max(axis=1) - values.min(axis=1)
            output[prefix + "outlier_rate"] = values.abs().gt(3).mean(axis=1)
            output[prefix + "missing_rate"] = original[columns].isna().mean(axis=1)
            output[prefix + "zero_rate"] = original[columns].eq(0).mean(axis=1)
        return pd.DataFrame(output, index=original.index)

    def _time_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        output: dict[str, pd.Series] = {}
        starts: dict[str, pd.Series] = {}
        ends: dict[str, pd.Series] = {}
        operations = sorted(
            {infer_operation(column) for column in frame.columns}, key=operation_sort_key
        )
        for operation in operations:
            if operation == "META":
                continue
            columns = [
                column
                for column in frame.columns
                if infer_operation(column) == operation
            ]
            if not columns:
                continue
            values = frame[columns]
            start = values.min(axis=1, skipna=True)
            end = values.max(axis=1, skipna=True)
            median = values.median(axis=1, skipna=True)
            starts[operation] = start
            ends[operation] = end
            prefix = f"time__{operation}__"
            output[prefix + "start_day"] = (start - self.time_origin) / 86400.0
            output[prefix + "end_day"] = (end - self.time_origin) / 86400.0
            output[prefix + "median_day"] = (median - self.time_origin) / 86400.0
            output[prefix + "span_min"] = ((end - start) / 60.0).clip(-1e4, 1e4)
            output[prefix + "valid_rate"] = values.notna().mean(axis=1)

        ordered = [operation for operation in operations if operation in starts]
        for previous, current in zip(ordered, ordered[1:]):
            gap = ((starts[current] - ends[previous]) / 3600.0).clip(-720, 720)
            output[f"route__{previous}_to_{current}__gap_hours"] = gap
            output[f"route__{previous}_to_{current}__overlap"] = gap.lt(0).astype(float)
        return pd.DataFrame(output, index=frame.index)

    def transform(
        self,
        numeric: pd.DataFrame,
        categorical: pd.DataFrame,
        time_frame: pd.DataFrame,
        feature_view: str | None = None,
    ) -> pd.DataFrame:
        if self.encoder is None:
            raise RuntimeError("FoldFeatureBuilder尚未拟合")
        time_frame = self._time_input(numeric, time_frame)
        numeric = numeric.reindex(columns=self.numeric_columns_).replace(
            [np.inf, -np.inf], np.nan
        )
        categorical = categorical.reindex(columns=self.categorical_columns_)
        imputed = self.impute(numeric, categorical)
        raw = imputed[self.selected_features].copy()
        raw.columns = [f"raw__{column}" for column in raw.columns]

        missing = numeric[self.missing_indicator_features].isna().astype(np.float32)
        missing.columns = [f"missing__{column}" for column in missing.columns]
        process = self._process_features(numeric, imputed)
        time_features = self._time_features(time_frame)

        cat = self._cat_values(categorical)
        encoded = self.encoder.transform(cat) if len(cat.columns) else np.empty((len(cat), 0))
        categorical_features = pd.DataFrame(
            encoded,
            index=numeric.index,
            columns=([f"cat__{name}" for name in self.encoder.get_feature_names_out()]
                     if len(cat.columns) else []),
        )
        result = pd.concat(
            [raw, missing, process, time_features, categorical_features], axis=1
        )
        view = feature_view or self.config.feature_view
        view_prefixes = {
            "raw": ("raw__",),
            "raw_missing": ("raw__", "missing__"),
            "process": ("raw__", "missing__", "proc__"),
            "process_time": ("raw__", "missing__", "proc__", "time__", "route__"),
            "full": ("raw__", "missing__", "proc__", "time__", "route__", "cat__"),
        }
        if view not in view_prefixes:
            raise ValueError(
                f"未知特征视图：{view}；可用值：{sorted(view_prefixes)}"
            )
        result = result.loc[
            :, [column.startswith(view_prefixes[view]) for column in result.columns]
        ]
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result = result.loc[:, ~result.columns.duplicated()].astype(np.float32)

        if not self.feature_names_:
            self.feature_names_ = result.columns.tolist()
        else:
            result = result.reindex(columns=self.feature_names_, fill_value=0.0)
        return result


def build_fold_features(
    data: PreparedData,
    config: PipelineConfig,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    seed: int | None = None,
    feature_view: str | None = None,
) -> tuple[FoldFeatureBuilder, pd.DataFrame, pd.DataFrame]:
    """Fit the entire feature builder on exactly the supplied training rows.

    Call purge_training_overlap first when repeated entities are present. This
    helper never silently changes indices, so targets remain correctly aligned.
    """
    train_index, valid_index = np.asarray(train_index), np.asarray(valid_index)
    if np.intersect1d(train_index, valid_index).size:
        raise ValueError("Training and evaluation row indices must be disjoint")
    builder = FoldFeatureBuilder(data, config, config.random_state if seed is None else seed)
    builder.fit(data.train_numeric.iloc[train_index], data.train_categorical.iloc[train_index],
                data.train_time.iloc[train_index], data.y.iloc[train_index],
                data.order_key.iloc[train_index])
    X_train = builder.transform(data.train_numeric.iloc[train_index],
                                data.train_categorical.iloc[train_index],
                                data.train_time.iloc[train_index], feature_view=feature_view)
    X_valid = builder.transform(data.train_numeric.iloc[valid_index],
                                data.train_categorical.iloc[valid_index],
                                data.train_time.iloc[valid_index], feature_view=feature_view)
    return builder, X_train, X_valid


class ScaledRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alphas: tuple[float, ...]):
        self.alphas = alphas
        self.scaler = StandardScaler()
        # Alpha is fixed a priori. CV over a matrix already selected using y
        # would leak inner-validation labels into the upstream feature selector.
        self.model = Ridge(alpha=100.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScaledRidgeRegressor":
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(X)), dtype=float)


class MLPEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Small deep ensemble inspired by recent parameter-efficient MLP ensembles."""

    def __init__(self, members: int, seed: int):
        self.members = members
        self.seed = seed
        self.scaler = StandardScaler()
        self.models: list[MLPRegressor] = []
        self.target_mean = 0.0
        self.target_std = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPEnsembleRegressor":
        transformed = self.scaler.fit_transform(X)
        target = np.asarray(y, dtype=float)
        self.target_mean = float(target.mean())
        self.target_std = float(target.std()) or 1.0
        target_scaled = (target - self.target_mean) / self.target_std
        self.models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for member in range(self.members):
                model = MLPRegressor(
                    hidden_layer_sizes=(96, 48),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=350,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=self.seed + member * 101,
                )
                model.fit(transformed, target_scaled)
                self.models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.scaler.transform(X)
        predictions = np.column_stack(
            [model.predict(transformed) for model in self.models]
        )
        return predictions.mean(axis=1) * self.target_std + self.target_mean


def make_models(config: PipelineConfig, seed: int) -> dict[str, BaseEstimator]:
    xgb_parameters = {
        "objective": "reg:squarederror",
        "n_estimators": config.xgb_estimators,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.55,
        "reg_alpha": 0.01,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "Ridge": ScaledRidgeRegressor(config.ridge_alphas),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=config.extra_trees_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(**xgb_parameters),
        "XGB-Recency": XGBRegressor(**xgb_parameters),
        "MLP-Ensemble": MLPEnsembleRegressor(config.mlp_members, seed),
    }


def recency_weights(order_key: pd.Series, half_life_fraction: float = 0.35) -> np.ndarray:
    """Return smooth time-decay weights with mean one."""

    order = np.argsort(order_key.to_numpy(dtype=float))
    rank = np.empty(len(order), dtype=float)
    rank[order] = np.arange(len(order), dtype=float)
    age_fraction = (len(order) - 1 - rank) / max(len(order) - 1, 1)
    weights = np.power(0.5, age_fraction / half_life_fraction)
    return weights / weights.mean()


def fit_candidate(
    model_name: str,
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    order_key: pd.Series,
) -> BaseEstimator:
    if model_name == "XGB-Recency":
        model.fit(X, y, sample_weight=recency_weights(order_key))
    else:
        model.fit(X, y)
    return model


def optimize_blend_weights(y_true: np.ndarray, predictions: pd.DataFrame) -> pd.Series:
    matrix = predictions.to_numpy(dtype=float)
    individual_mse = np.mean(np.square(matrix - y_true[:, None]), axis=0)
    inverse = 1.0 / np.maximum(individual_mse, 1e-12)
    initial = inverse / inverse.sum()

    result = minimize(
        lambda weights: float(np.mean(np.square(y_true - matrix @ weights))),
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = result.x if result.success else initial
    weights[np.abs(weights) < 1e-6] = 0.0
    weights = weights / weights.sum()
    return pd.Series(weights, index=predictions.columns, name="weight")


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    values = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration residuals must be a nonempty finite vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    # Include the +infinity atom in the finite-sample split-conformal rule.
    # This order statistic alone does not establish exchangeability under drift.
    return float(values[rank - 1]) if rank <= len(values) else float("inf")


def engineered_operation(feature: str) -> str:
    for prefix in ("raw__", "missing__"):
        if feature.startswith(prefix):
            return infer_operation(feature[len(prefix) :])
    for prefix in ("proc__", "time__"):
        if feature.startswith(prefix):
            parts = feature.split("__")
            return parts[1] if len(parts) > 1 else "META"
    if feature.startswith("route__"):
        parts = feature.split("__")
        return parts[1] if len(parts) > 1 else "ROUTE"
    return "META"


def aggregate_selection(reports: list[pd.DataFrame], n_folds: int) -> pd.DataFrame:
    frame = pd.concat(reports, ignore_index=True)
    return (
        frame.groupby(["feature", "operation"], as_index=False)
        .agg(
            selected_folds=("selected", "sum"),
            mean_stable_score=("stable_score", "mean"),
            std_stable_score=("stable_score", "std"),
            mean_importance=("importance", "mean"),
            mean_abs_correlation=("abs_correlation", "mean"),
            mean_temporal_drift=("temporal_drift", "mean"),
        )
        .assign(selection_rate=lambda value: value["selected_folds"] / n_folds)
        .sort_values(
            ["selection_rate", "mean_stable_score"], ascending=[False, False]
        )
    )


def drift_report(
    builder: FoldFeatureBuilder,
    train_numeric: pd.DataFrame,
    train_cat: pd.DataFrame,
    test_numeric: pd.DataFrame,
    test_cat: pd.DataFrame,
    feature_operation: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert builder.robust_iqr is not None
    train = builder.impute(train_numeric, train_cat)
    test = builder.impute(test_numeric, test_cat)
    rows = []
    for feature in builder.selected_features:
        scale = float(builder.robust_iqr[feature]) or 1.0
        median_shift = abs(float(test[feature].median() - train[feature].median())) / scale
        missing_shift = abs(
            float(test_numeric[feature].isna().mean() - train_numeric[feature].isna().mean())
        )
        rows.append(
            {
                "feature": feature,
                "operation": feature_operation.get(feature, "META"),
                "robust_median_shift": median_shift,
                "missing_rate_shift": missing_shift,
                "combined_shift": median_shift + 2.0 * missing_shift,
            }
        )
    feature_frame = pd.DataFrame(rows).sort_values("combined_shift", ascending=False)
    operation_frame = (
        feature_frame.groupby("operation", as_index=False)
        .agg(
            feature_count=("feature", "size"),
            mean_shift=("combined_shift", "mean"),
            max_shift=("combined_shift", "max"),
            mean_missing_shift=("missing_rate_shift", "mean"),
        )
        .sort_values("mean_shift", ascending=False)
    )
    return feature_frame, operation_frame


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trace = RunTrace()
    total_started = time.perf_counter()
    data = prepare_data(config, trace)

    order = np.argsort(data.order_key.to_numpy(dtype=float), kind="stable")
    holdout_size = max(1, int(round(len(order) * config.temporal_holdout_fraction)))
    holdout_index = order[-holdout_size:]
    groups = validation_groups(data)
    development_index = purge_training_overlap(order[:-holdout_size], holdout_index, groups)
    log(
        f"锁定时间外推集：development={len(development_index)}，"
        f"holdout={len(holdout_index)}"
    )

    model_names = list(make_models(config, config.random_state).keys())
    oof = pd.DataFrame(index=development_index, columns=model_names, dtype=float)
    fold_rows: list[dict[str, object]] = []
    selection_reports: list[pd.DataFrame] = []
    engineered_importance_rows: list[pd.DataFrame] = []

    cv_started = time.perf_counter()
    splitter = KFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_state
    )
    for fold, (train_position, valid_position) in enumerate(
        splitter.split(development_index), start=1
    ):
        fold_started = time.perf_counter()
        train_index = development_index[train_position]
        valid_index = development_index[valid_position]
        train_index = purge_training_overlap(train_index, valid_index, groups)
        log(f"新方案CV Fold {fold}/{config.n_splits}：特征构建")

        builder = FoldFeatureBuilder(data, config, seed=config.random_state + fold)
        builder.fit(
            data.train_numeric.iloc[train_index],
            data.train_categorical.iloc[train_index],
            data.train_time.iloc[train_index],
            data.y.iloc[train_index],
            data.order_key.iloc[train_index],
        )
        X_train = builder.transform(
            data.train_numeric.iloc[train_index],
            data.train_categorical.iloc[train_index],
            data.train_time.iloc[train_index],
        )
        X_valid = builder.transform(
            data.train_numeric.iloc[valid_index],
            data.train_categorical.iloc[valid_index],
            data.train_time.iloc[valid_index],
        )
        report = builder.selection_report.copy()
        report["fold"] = fold
        selection_reports.append(report)

        models = make_models(config, config.random_state + fold)
        for model_name, model in models.items():
            model_started = time.perf_counter()
            fit_candidate(
                model_name,
                model,
                X_train,
                data.y.iloc[train_index],
                data.order_key.iloc[train_index],
            )
            prediction = np.asarray(model.predict(X_valid), dtype=float)
            oof.loc[valid_index, model_name] = prediction
            metrics = metric_row(data.y.iloc[valid_index].to_numpy(), prediction)
            fold_rows.append(
                {
                    "validation_scheme": "random_cv",
                    "fold": fold,
                    "model": model_name,
                    **metrics,
                    "n_train": len(train_index),
                    "n_valid": len(valid_index),
                    "n_features": X_train.shape[1],
                    "fit_predict_sec": time.perf_counter() - model_started,
                }
            )
            log(
                f"Fold {fold} | {model_name} | MSE={metrics['mse']:.6f} | "
                f"RMSE={metrics['rmse']:.6f}"
            )
            if model_name == "XGBoost":
                importance = pd.DataFrame(
                    {
                        "feature": builder.feature_names_,
                        "importance": model.feature_importances_,
                        "fold": fold,
                    }
                )
                importance["operation"] = importance["feature"].map(
                    engineered_operation
                )
                engineered_importance_rows.append(importance)
        trace.add(
            f"CV Fold {fold}",
            fold_started,
            rows=len(train_index) + len(valid_index),
            columns=X_train.shape[1],
            note="Fold内填充、稳定筛选、工序特征、时间特征与候选模型",
        )

    trace.add(
        "开发集五折交叉验证",
        cv_started,
        rows=len(development_index),
        columns=int(fold_rows[-1]["n_features"]),
        note=f"{len(model_names)}类模型",
    )
    oof = oof.sort_index()
    development_y = data.y.loc[oof.index].to_numpy(dtype=float)

    # Rolling validation deliberately evaluates future blocks and is the
    # source used for blend weights and conformal calibration.
    rolling_started = time.perf_counter()
    rolling_test_size = max(
        20,
        len(development_index) // (config.rolling_n_splits + 4),
    )
    rolling_splitter = TimeSeriesSplit(
        n_splits=config.rolling_n_splits,
        test_size=rolling_test_size,
    )
    rolling_oof = pd.DataFrame(
        index=development_index, columns=model_names, dtype=float
    )
    rolling_rows: list[dict[str, object]] = []
    for fold, (train_position, valid_position) in enumerate(
        rolling_splitter.split(development_index), start=1
    ):
        fold_started = time.perf_counter()
        train_index = development_index[train_position]
        valid_index = development_index[valid_position]
        train_index = purge_training_overlap(train_index, valid_index, groups)
        log(
            f"滚动时间CV Fold {fold}/{config.rolling_n_splits}："
            f"train={len(train_index)}，future={len(valid_index)}"
        )

        builder = FoldFeatureBuilder(
            data, config, seed=config.random_state + 500 + fold
        )
        builder.fit(
            data.train_numeric.iloc[train_index],
            data.train_categorical.iloc[train_index],
            data.train_time.iloc[train_index],
            data.y.iloc[train_index],
            data.order_key.iloc[train_index],
        )
        X_train = builder.transform(
            data.train_numeric.iloc[train_index],
            data.train_categorical.iloc[train_index],
            data.train_time.iloc[train_index],
        )
        X_valid = builder.transform(
            data.train_numeric.iloc[valid_index],
            data.train_categorical.iloc[valid_index],
            data.train_time.iloc[valid_index],
        )
        models = make_models(config, config.random_state + 500 + fold)
        for model_name, model in models.items():
            model_started = time.perf_counter()
            fit_candidate(
                model_name,
                model,
                X_train,
                data.y.iloc[train_index],
                data.order_key.iloc[train_index],
            )
            prediction = np.asarray(model.predict(X_valid), dtype=float)
            rolling_oof.loc[valid_index, model_name] = prediction
            metrics = metric_row(data.y.iloc[valid_index].to_numpy(), prediction)
            rolling_rows.append(
                {
                    "validation_scheme": "rolling_cv",
                    "fold": fold,
                    "model": model_name,
                    **metrics,
                    "n_train": len(train_index),
                    "n_valid": len(valid_index),
                    "n_features": X_train.shape[1],
                    "fit_predict_sec": time.perf_counter() - model_started,
                }
            )
            log(
                f"Rolling {fold} | {model_name} | MSE={metrics['mse']:.6f}"
            )
        trace.add(
            f"滚动时间CV Fold {fold}",
            fold_started,
            rows=len(train_index) + len(valid_index),
            columns=X_train.shape[1],
            note="仅用过去批次预测未来批次",
        )

    rolling_oof = rolling_oof.dropna(how="any").sort_index()
    rolling_y = data.y.loc[rolling_oof.index].to_numpy(dtype=float)
    trace.add(
        "滚动时间交叉验证",
        rolling_started,
        rows=len(rolling_oof),
        columns=int(rolling_rows[-1]["n_features"]),
        note="用于融合权重、时间泛化选择与预测区间校准",
    )

    candidate_names = [name for name in model_names if name != "Dummy"]
    weights = optimize_blend_weights(rolling_y, rolling_oof[candidate_names])
    oof["RobustBlend"] = oof[candidate_names].to_numpy() @ weights.to_numpy()
    rolling_oof["RobustBlend"] = (
        rolling_oof[candidate_names].to_numpy() @ weights.to_numpy()
    )

    summary_rows: list[dict[str, object]] = []
    for model_name in model_names + ["RobustBlend"]:
        metrics = metric_row(development_y, oof[model_name].to_numpy(dtype=float))
        model_folds = pd.DataFrame(fold_rows)
        std = (
            float(model_folds.loc[model_folds.model == model_name, "mse"].std(ddof=1))
            if model_name != "RobustBlend"
            else np.nan
        )
        summary_rows.append(
            {
                "scope": "random_oof",
                "model": model_name,
                **metrics,
                "mse_fold_std": std,
            }
        )

    rolling_fold_frame = pd.DataFrame(rolling_rows)
    for model_name in model_names + ["RobustBlend"]:
        metrics = metric_row(
            rolling_y, rolling_oof[model_name].to_numpy(dtype=float)
        )
        std = (
            float(
                rolling_fold_frame.loc[
                    rolling_fold_frame.model == model_name, "mse"
                ].std(ddof=1)
            )
            if model_name != "RobustBlend"
            else np.nan
        )
        summary_rows.append(
            {
                "scope": "rolling_oof",
                "model": model_name,
                **metrics,
                "mse_fold_std": std,
            }
        )

    holdout_started = time.perf_counter()
    log("拟合开发集并评估锁定时间外推集")
    holdout_builder = FoldFeatureBuilder(data, config, seed=config.random_state + 100)
    holdout_builder.fit(
        data.train_numeric.iloc[development_index],
        data.train_categorical.iloc[development_index],
        data.train_time.iloc[development_index],
        data.y.iloc[development_index],
        data.order_key.iloc[development_index],
    )
    X_development = holdout_builder.transform(
        data.train_numeric.iloc[development_index],
        data.train_categorical.iloc[development_index],
        data.train_time.iloc[development_index],
    )
    X_holdout = holdout_builder.transform(
        data.train_numeric.iloc[holdout_index],
        data.train_categorical.iloc[holdout_index],
        data.train_time.iloc[holdout_index],
    )
    holdout_predictions = pd.DataFrame(index=holdout_index)
    holdout_models = make_models(config, config.random_state + 100)
    holdout_fit_times: dict[str, float] = {}
    for model_name, model in holdout_models.items():
        model_started = time.perf_counter()
        fit_candidate(
            model_name,
            model,
            X_development,
            data.y.iloc[development_index],
            data.order_key.iloc[development_index],
        )
        holdout_predictions[model_name] = model.predict(X_holdout)
        holdout_fit_times[model_name] = time.perf_counter() - model_started
        summary_rows.append(
            {
                "scope": "temporal_holdout",
                "model": model_name,
                **metric_row(
                    data.y.iloc[holdout_index].to_numpy(),
                    holdout_predictions[model_name].to_numpy(),
                ),
                "mse_fold_std": np.nan,
            }
        )
    holdout_predictions["RobustBlend"] = (
        holdout_predictions[candidate_names].to_numpy() @ weights.to_numpy()
    )
    blend_holdout_metrics = metric_row(
        data.y.iloc[holdout_index].to_numpy(),
        holdout_predictions["RobustBlend"].to_numpy(),
    )
    summary_rows.append(
        {
            "scope": "temporal_holdout",
            "model": "RobustBlend",
            **blend_holdout_metrics,
            "mse_fold_std": np.nan,
        }
    )

    conformal_radius = conformal_quantile(
        rolling_y - rolling_oof["RobustBlend"].to_numpy(), config.conformal_alpha
    )
    holdout_true = data.y.iloc[holdout_index].to_numpy(dtype=float)
    holdout_blend = holdout_predictions["RobustBlend"].to_numpy(dtype=float)
    holdout_lower = holdout_blend - conformal_radius
    holdout_upper = holdout_blend + conformal_radius
    coverage = float(np.mean((holdout_true >= holdout_lower) & (holdout_true <= holdout_upper)))
    conformal_metrics = pd.DataFrame(
        [
            {
                "alpha": config.conformal_alpha,
                "nominal_coverage": 1.0 - config.conformal_alpha,
                "empirical_holdout_coverage": coverage,
                "interval_radius": conformal_radius,
                "mean_interval_width": 2.0 * conformal_radius,
                "n_calibration": len(rolling_y),
                "n_holdout": len(holdout_index),
                "inference_status": "exploratory_rolling_residual_interval_no_distribution_free_guarantee",
                "calibration_limitation": "blend_weights_fit_on_same_residual_pool; use independent calibration for confirmatory analysis",
            }
        ]
    )
    trace.add(
        "锁定时间外推集评估",
        holdout_started,
        rows=len(holdout_index),
        columns=X_development.shape[1],
        note=f"RobustBlend MSE={blend_holdout_metrics['mse']:.6f}",
    )

    final_started = time.perf_counter()
    log("全量训练并生成A/B预测")
    final_builder = FoldFeatureBuilder(data, config, seed=config.random_state + 200)
    final_builder.fit(
        data.train_numeric,
        data.train_categorical,
        data.train_time,
        data.y,
        data.order_key,
    )
    X_full = final_builder.transform(
        data.train_numeric, data.train_categorical, data.train_time
    )
    X_test_a = final_builder.transform(
        data.test_a_numeric, data.test_a_categorical, data.test_a_time
    )
    X_test_b = final_builder.transform(
        data.test_b_numeric, data.test_b_categorical, data.test_b_time
    )
    final_models = make_models(config, config.random_state + 200)
    test_a_base = pd.DataFrame(index=data.test_a_numeric.index)
    test_b_base = pd.DataFrame(index=data.test_b_numeric.index)
    final_fit_times: dict[str, float] = {}
    for model_name, model in final_models.items():
        model_started = time.perf_counter()
        fit_candidate(
            model_name,
            model,
            X_full,
            data.y,
            data.order_key,
        )
        test_a_base[model_name] = model.predict(X_test_a)
        test_b_base[model_name] = model.predict(X_test_b)
        final_fit_times[model_name] = time.perf_counter() - model_started
    test_a_prediction = test_a_base[candidate_names].to_numpy() @ weights.to_numpy()
    test_b_prediction = test_b_base[candidate_names].to_numpy() @ weights.to_numpy()
    trace.add(
        "全量重训与测试集推理",
        final_started,
        rows=len(data.y) + len(test_a_prediction) + len(test_b_prediction),
        columns=X_full.shape[1],
        note="生成A/B提交文件与90%预测区间",
    )

    selection_stability = aggregate_selection(selection_reports, config.n_splits)
    engineered_importance = pd.concat(engineered_importance_rows, ignore_index=True)
    feature_importance = (
        engineered_importance.groupby(["feature", "operation"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            selected_folds=("importance", lambda values: int((values > 0).sum())),
        )
        .sort_values("mean_importance", ascending=False)
    )
    operation_importance = (
        engineered_importance.groupby(["fold", "operation"], as_index=False)[
            "importance"
        ]
        .sum()
        .groupby("operation", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
        .sort_values("mean_importance", ascending=False)
    )

    combined_test_numeric = pd.concat(
        [data.test_a_numeric, data.test_b_numeric], ignore_index=True
    )
    combined_test_cat = pd.concat(
        [data.test_a_categorical, data.test_b_categorical], ignore_index=True
    )
    feature_drift, operation_drift = drift_report(
        final_builder,
        data.train_numeric,
        data.train_categorical,
        combined_test_numeric,
        combined_test_cat,
        data.feature_operation,
    )

    fold_metrics = pd.concat(
        [pd.DataFrame(fold_rows), pd.DataFrame(rolling_rows)], ignore_index=True
    )
    model_summary = pd.DataFrame(summary_rows)
    model_summary["assessment_role"] = np.where(
        model_summary["model"].eq("RobustBlend")
        & model_summary["scope"].isin(["random_oof", "rolling_oof"]),
        "exploratory_blend_weights_reuse_development_targets",
        "historical_validation_requires_new_run_after_preprocessing_revision",
    )
    blend_weights = weights.rename_axis("model").reset_index()

    oof_detail = pd.DataFrame(
        {
            "row_index": oof.index,
            "ID": data.train_ids.loc[oof.index].to_numpy(),
            "order_key": data.order_key.loc[oof.index].to_numpy(),
            "y_true": development_y,
        }
    )
    for name in oof.columns:
        oof_detail[f"pred_{name}"] = oof[name].to_numpy()
    oof_detail["residual_RobustBlend"] = (
        oof_detail["y_true"] - oof_detail["pred_RobustBlend"]
    )
    if data.primary_tool_column:
        oof_detail["Tool"] = data.train_categorical.loc[
            oof.index, data.primary_tool_column
        ].to_numpy()

    rolling_oof_detail = pd.DataFrame(
        {
            "row_index": rolling_oof.index,
            "ID": data.train_ids.loc[rolling_oof.index].to_numpy(),
            "order_key": data.order_key.loc[rolling_oof.index].to_numpy(),
            "y_true": rolling_y,
        }
    )
    for name in rolling_oof.columns:
        rolling_oof_detail[f"pred_{name}"] = rolling_oof[name].to_numpy()
    rolling_oof_detail["residual_RobustBlend"] = (
        rolling_oof_detail["y_true"]
        - rolling_oof_detail["pred_RobustBlend"]
    )
    if data.primary_tool_column:
        rolling_oof_detail["Tool"] = data.train_categorical.loc[
            rolling_oof.index, data.primary_tool_column
        ].to_numpy()

    holdout_detail = pd.DataFrame(
        {
            "row_index": holdout_index,
            "ID": data.train_ids.iloc[holdout_index].to_numpy(),
            "order_key": data.order_key.iloc[holdout_index].to_numpy(),
            "y_true": holdout_true,
        }
    )
    for name in holdout_predictions.columns:
        holdout_detail[f"pred_{name}"] = holdout_predictions[name].to_numpy()
    holdout_detail["lower_90"] = holdout_lower
    holdout_detail["upper_90"] = holdout_upper
    holdout_detail["covered_90"] = (
        (holdout_true >= holdout_lower) & (holdout_true <= holdout_upper)
    )
    holdout_detail["residual_RobustBlend"] = holdout_true - holdout_blend
    if data.primary_tool_column:
        holdout_detail["Tool"] = data.train_categorical.iloc[holdout_index][
            data.primary_tool_column
        ].to_numpy()

    prediction_a_detail = pd.DataFrame(
        {
            "ID": data.test_a_ids,
            "prediction": test_a_prediction,
            "lower_90": test_a_prediction - conformal_radius,
            "upper_90": test_a_prediction + conformal_radius,
        }
    )
    prediction_b_detail = pd.DataFrame(
        {
            "ID": data.test_b_ids,
            "prediction": test_b_prediction,
            "lower_90": test_b_prediction - conformal_radius,
            "upper_90": test_b_prediction + conformal_radius,
        }
    )

    data.feature_audit.to_csv(
        config.output_dir / "feature_audit.csv", index=False, encoding="utf-8-sig"
    )
    data.operation_audit.to_csv(
        config.output_dir / "operation_audit.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(
        config.output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    model_summary.to_csv(
        config.output_dir / "model_summary.csv", index=False, encoding="utf-8-sig"
    )
    blend_weights.to_csv(
        config.output_dir / "blend_weights.csv", index=False, encoding="utf-8-sig"
    )
    oof_detail.to_csv(
        config.output_dir / "random_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_oof_detail.to_csv(
        config.output_dir / "rolling_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_oof_detail.to_csv(
        config.output_dir / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_detail.to_csv(
        config.output_dir / "temporal_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection_stability.to_csv(
        config.output_dir / "feature_selection_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_importance.to_csv(
        config.output_dir / "engineered_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    operation_importance.to_csv(
        config.output_dir / "operation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_drift.to_csv(
        config.output_dir / "feature_drift.csv", index=False, encoding="utf-8-sig"
    )
    operation_drift.to_csv(
        config.output_dir / "operation_drift.csv", index=False, encoding="utf-8-sig"
    )
    conformal_metrics.to_csv(
        config.output_dir / "conformal_metrics.csv", index=False, encoding="utf-8-sig"
    )
    prediction_a_detail.to_csv(
        config.output_dir / "predictions_A_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_b_detail.to_csv(
        config.output_dir / "predictions_B_detailed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_a_detail[["ID", "prediction"]].to_csv(
        config.output_dir / "submission_A.csv", index=False, header=False
    )
    prediction_b_detail[["ID", "prediction"]].to_csv(
        config.output_dir / "submission_B.csv", index=False, header=False
    )

    trace.add(
        "完整新方案",
        total_started,
        rows=len(data.y),
        columns=X_full.shape[1],
        note="从原始Excel到CV、外推验证、解释、漂移与提交",
    )
    trace_frame = trace.frame()
    trace_frame.to_csv(
        config.output_dir / "run_trace.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "train_rows": len(data.y),
        "test_a_rows": len(data.test_a_ids),
        "test_b_rows": len(data.test_b_ids),
        "development_rows": len(development_index),
        "rolling_calibration_rows": len(rolling_oof),
        "temporal_holdout_rows": len(holdout_index),
        "clean_numeric_features": len(data.train_numeric.columns),
        "timestamp_features": len(data.train_time.columns),
        "categorical_features": len(data.train_categorical.columns),
        "engineered_features": X_full.shape[1],
        "feature_view": config.feature_view,
        "blend_weights": {key: float(value) for key, value in weights.items()},
        "conformal_radius": conformal_radius,
        "holdout_coverage": coverage,
        "holdout_blend_metrics": blend_holdout_metrics,
        "total_elapsed_sec": time.perf_counter() - total_started,
        "holdout_fit_times": holdout_fit_times,
        "final_fit_times": final_fit_times,
        "random_state": config.random_state,
        "validation_schemes": ["random_cv", "rolling_cv", "temporal_holdout"],
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(
        f"新方案完成：Holdout MSE={blend_holdout_metrics['mse']:.6f}，"
        f"90%区间覆盖率={coverage:.1%}"
    )
    return {
        "data": data,
        "fold_metrics": fold_metrics,
        "model_summary": model_summary,
        "blend_weights": blend_weights,
        "random_oof_detail": oof_detail,
        "rolling_oof_detail": rolling_oof_detail,
        "oof_detail": rolling_oof_detail,
        "holdout_detail": holdout_detail,
        "selection_stability": selection_stability,
        "feature_importance": feature_importance,
        "operation_importance": operation_importance,
        "operation_drift": operation_drift,
        "conformal_metrics": conformal_metrics,
        "trace": trace_frame,
        "manifest": manifest,
    }
