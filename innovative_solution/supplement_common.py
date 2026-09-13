"""Fixed supplementary experiments for a single quality-prediction study."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .pipeline import FoldFeatureBuilder, metric_row

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / ".research_deps"))
OUT = PACKAGE / "outputs" / "supplement"
BASE = PACKAGE / "outputs" / "review"
SEEDS = [42, 143, 244]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SupportedGroupBuilder(FoldFeatureBuilder):
    """Require observed cells per device/field; optionally shrink to train median.

    Counts exclude missing cells. No training labels or held-out observations enter
    these means. Options are fixed before comparisons, never selected on test A.
    """
    def __init__(self, data, config, seed, min_count=1, prior_count=0, median_only=False):
        super().__init__(data, config, seed)
        self.min_count = min_count
        self.prior_count = prior_count
        self.median_only = median_only

    def fit(self, numeric, categorical, time_frame, y, order_key):
        self._support_numeric = numeric.replace([np.inf, -np.inf], np.nan)
        self._support_cat = self._cat_values(categorical)
        self._support_applied = False
        self.support_audit_ = []
        result = super().fit(numeric, categorical, time_frame, y, order_key)
        del self._support_numeric, self._support_cat
        return result

    def impute(self, numeric, categorical):
        if not self._support_applied:
            for tool, columns in self.group_columns.items():
                values = self._support_numeric[columns].copy()
                values["__group__"] = self._support_cat[tool].to_numpy()
                counts = values.groupby("__group__", observed=True).count()
                means = self.group_means[tool]
                self.support_audit_.append({"tool": tool, "n_groups": len(counts),
                    "n_group_field_cells": int(counts.size), "singleton_cells": int((counts == 1).sum().sum()),
                    "below_5_cells": int((counts < 5).sum().sum()),
                    "minimum_positive_count": int(counts.where(counts.gt(0)).min().min())})
                if self.median_only:
                    self.group_means[tool] = means * np.nan
                else:
                    smoothed = ((means * counts + self.global_median[columns] * self.prior_count) / (counts + self.prior_count)
                                if self.prior_count else means)
                    self.group_means[tool] = smoothed.where(counts >= self.min_count)
            self._support_applied = True
        return super().impute(numeric, categorical)


def rerank(builder, numeric, importance_weight=.7, drift_power=.25, top_k=320, missing_threshold=.005):
    report = builder.selection_report
    correlation_weight = .30 if importance_weight == .70 else 1-importance_weight
    scores = (importance_weight * report.importance.rank(pct=True) +
              correlation_weight * report.abs_correlation.rank(pct=True)) * np.power(1/(1+report.temporal_drift),drift_power)
    builder.selected_features = report.loc[scores.nlargest(min(top_k, len(scores))).index, "feature"].tolist()
    rates = numeric[builder.selected_features].isna().mean()
    builder.missing_indicator_features = rates[rates >= missing_threshold].index.tolist()
    builder.feature_names_ = []
    return builder


def columns_for(X, view):
    basic = ("raw__", "cat__")
    if view == "full":
        return list(X)
    if view == "basic":
        return [c for c in X if c.startswith(basic + ("missing__",))]
    if view == "no_mask":
        return [c for c in X if not c.startswith("missing__") and not c.endswith("__missing_rate") and not c.endswith("__valid_rate")]
    if view == "process_only":
        return [c for c in X if c.startswith(basic + ("missing__", "proc__"))]
    if view == "time_only":
        return [c for c in X if c.startswith(basic + ("missing__", "time__", "route__"))]
    if view == "measurements":
        return [c for c in X if c.startswith(basic)]
    raise ValueError(view)


def matrix(data, train, valid, seed=42, **options):
    builder = SupportedGroupBuilder(data, PipelineConfig(), seed,
        min_count=options.get("min_count", 1), prior_count=options.get("prior_count", 0),
        median_only=options.get("median_only", False))
    builder.fit(data.train_numeric.iloc[train], data.train_categorical.iloc[train], data.train_time.iloc[train],
                data.y.iloc[train], data.order_key.iloc[train])
    rank_options={k: v for k, v in options.items() if k in ["importance_weight", "drift_power", "top_k", "missing_threshold"]}
    if rank_options:
        rerank(builder,data.train_numeric.iloc[train],**rank_options)
    X = builder.transform(data.train_numeric.iloc[train], data.train_categorical.iloc[train], data.train_time.iloc[train])
    V = builder.transform(data.train_numeric.iloc[valid], data.train_categorical.iloc[valid], data.train_time.iloc[valid])
    return builder, X, V


def prediction_rows(data, groups, valid, pred, name, fold, scope):
    return pd.DataFrame({"scope": scope, "fold": fold, "model": name, "row_index": valid,
        "ID": data.train_ids.iloc[valid].to_numpy(), "group": groups[valid],
        "time_proxy": data.order_key.iloc[valid].to_numpy(), "actual": data.y.iloc[valid].to_numpy(),
        "prediction": pred})


def summary(frame, by=("scope", "model")):
    return pd.DataFrame([{**dict(zip(by, key if isinstance(key, tuple) else (key,))), "n": len(part),
        **metric_row(part.actual.to_numpy(), part.prediction.to_numpy())}
        for key, part in frame.groupby(list(by))])
