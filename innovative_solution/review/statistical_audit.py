"""Read-only statistical audit of archived experiments.

Run from the project root:
    python innovative_solution/review/statistical_audit.py

The script never fits or selects a model and never changes archived outputs.
All resampling is exploratory: it does not undo strategy selection, repeated
access to A labels, or unavailable production-batch identifiers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "innovative_solution" / "outputs"
DEST = OUT / "review"
SEED = 20260913
REPLICATES = 5000


def load_data(name: str) -> pd.DataFrame:
    print(f"Reading {name}", flush=True)
    return pd.read_excel(ROOT / "data" / name)


def time_parse(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    text = numeric.dropna().round().astype("int64").astype(str)
    for length, fmt in ((8, "%Y%m%d"), (14, "%Y%m%d%H%M%S"), (16, "%Y%m%d%H%M%S")):
        part = text.loc[text.str.len() == length]
        parsed = pd.to_datetime(part.str[:14] if length == 16 else part, format=fmt, errors="coerce")
        seconds = parsed.astype("int64").astype(float) / 1e9
        seconds.loc[parsed.isna()] = np.nan
        if length == 16:
            seconds += pd.to_numeric(part.str[14:16], errors="coerce") / 100
        result.loc[part.index] = seconds
    return result


def date(value: float) -> str | None:
    return None if pd.isna(value) else str(pd.to_datetime(value, unit="s"))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    mse = float(np.mean((pred - y) ** 2))
    return {"rows": len(y), "mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(pred-y))),
            "r2": float(1 - np.sum((pred-y)**2) / np.sum((y-y.mean())**2)),
            "bias_prediction_minus_y": float(np.mean(pred-y))}


def bootstrap(name: str, groups: list[np.ndarray], block: int) -> dict:
    """Stratified circular moving blocks in original row order, paired loss."""
    rng = np.random.default_rng(SEED + block)
    means = np.zeros(REPLICATES)
    n = sum(len(group) for group in groups)
    for group in groups:
        starts = rng.integers(len(group), size=(REPLICATES, int(np.ceil(len(group)/block))))
        index = ((starts[:, :, None] + np.arange(block)) % len(group)).reshape(REPLICATES, -1)[:, :len(group)]
        means += group[index].sum(axis=1) / n
    low, high = np.quantile(means, [0.025, 0.975])
    return {"comparison": name, "block_length_rows": block, "n_rows": n,
            "fold_stratified": len(groups) > 1,
            "mse_difference_baseline_minus_candidate": float(np.concatenate(groups).mean()),
            "exploratory_percentile_95_lower": float(low),
            "exploratory_percentile_95_upper": float(high),
            "replicates": REPLICATES, "random_seed": SEED + block,
            "interpretation": "exploratory sensitivity only; no post-selection or batch-valid significance claim"}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    files = {"train": "训练集.xlsx", "A": "测试集A.xlsx", "B": "测试集B.xlsx"}
    frames = {name: load_data(filename) for name, filename in files.items()}
    audit = pd.read_csv(OUT / "feature_audit.csv")
    time_columns = audit.loc[audit.status.eq("converted_timestamp"), "feature"].tolist()
    answer = pd.read_csv(ROOT / "data" / "测试集A_答案.csv", header=None)
    assert answer.iloc[:, 0].astype(str).equals(frames["A"].iloc[:, 0].astype(str))
    target_column = next(c for c in ("Value", "Y", "value", "y") if c in frames["train"].columns)
    id_column = next(c for c in ("ID", "Id", "id") if c in frames["train"].columns)
    labels = {"train": pd.to_numeric(frames["train"][target_column]), "A": answer.iloc[:, 1], "B": pd.Series(np.nan, index=frames["B"].index)}
    proxies = {}
    temporal_rows = []
    all_rows = []
    raw_feature_rows = []
    duplicate_rows = []
    duplicate_id_groups = []
    common_features = [c for c in frames["train"].columns if c not in (id_column, target_column)]
    for name, frame in frames.items():
        parsed = pd.DataFrame({c: time_parse(frame[c]) for c in time_columns})
        proxy = parsed.median(axis=1, skipna=True)
        proxies[name] = proxy
        ids = frame[id_column].astype(str)
        id_number = pd.to_numeric(ids.str.extract(r"(\d+)")[0])
        temporal_rows.append({"dataset": name, "rows": len(frame), "columns": len(frame.columns),
                              "duplicate_id_extra_rows": int(ids.duplicated().sum()),
                              "proxy_definition": "median of archived 72 timestamp fields; physical time meaning unverified",
                              "proxy_missing_rows": int(proxy.isna().sum()),
                              "proxy_min": date(proxy.min()), "proxy_max": date(proxy.max()),
                              "row_order_proxy_spearman": float(spearmanr(np.arange(len(frame)), proxy, nan_policy="omit").statistic),
                              "adjacent_time_decreases": int(proxy.diff().lt(0).sum()),
                              "row_order_id_spearman": float(spearmanr(np.arange(len(frame)), id_number).statistic),
                              "all_parsed_timestamp_min": date(parsed.min().min()),
                              "all_parsed_timestamp_max": date(parsed.max().max())})
        meta = pd.DataFrame({"dataset": name, "source_row_zero_based": np.arange(len(frame)), "ID": ids, "known_y": labels[name].to_numpy(), "proxy_seconds": proxy})
        all_rows.append(meta)
        raw_feature_rows.append(frame[common_features].reset_index(drop=True))
        for _, group in meta.loc[ids.duplicated(keep=False)].groupby("ID"):
            positions = group.source_row_zero_based.to_numpy()
            x_group = frame[common_features].iloc[positions]
            same = x_group.eq(x_group.iloc[0]) | (x_group.isna() & x_group.iloc[0].isna())
            duplicate_id_groups.append({"dataset": name, "ID": str(group.ID.iloc[0]), "rows":len(group),
                                        "known_y_unique":int(group.known_y.nunique()),
                                        "differing_X_columns":int((~same.all(axis=0)).sum()),
                                        "total_X_columns":len(common_features)})
            for row in group.to_dict("records"):
                duplicate_rows.append({"duplicate_type": "ID_within_dataset", "group_key": row["ID"], **row})
    meta_all = pd.concat(all_rows, ignore_index=True)
    features_all = pd.concat(raw_feature_rows, ignore_index=True)
    hashes = pd.util.hash_pandas_object(features_all, index=False)
    exact_groups = []
    for key, indices in hashes.loc[hashes.duplicated(keep=False)].groupby(hashes).groups.items():
        positions = list(indices)
        first = features_all.iloc[positions[0]]
        assert all(first.equals(features_all.iloc[i]) for i in positions), "Hash collision: stop audit"
        group = meta_all.loc[positions]
        exact_groups.append({"group_key": str(key), "rows": len(group), "datasets": sorted(group.dataset.unique().tolist()),
                             "known_y_unique": int(group.known_y.dropna().nunique()),
                             "IDs": group.ID.tolist()})
        for row in group.to_dict("records"):
            duplicate_rows.append({"duplicate_type": "exact_all_X", "group_key": str(key), **row})
    pd.DataFrame(duplicate_rows).to_csv(DEST / "duplicate_records.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_id_groups).to_csv(DEST / "duplicate_id_groups.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(temporal_rows).to_csv(DEST / "dataset_temporal_audit.csv", index=False, encoding="utf-8-sig")
    # Reconstruct the archived, unpurged splits (not later corrected source code).
    train_ids = frames["train"][id_column].astype(str)
    development = np.argsort(proxies["train"].to_numpy())[:-160]
    holdout_index = np.argsort(proxies["train"].to_numpy())[-160:]
    split_overlap = []
    split_specs = [("archived_holdout", 0, development, holdout_index)]
    for fold, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(development), 1):
        split_specs.append(("archived_random_CV",fold,development[tr],development[va]))
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=4,test_size=80).split(development),1):
        split_specs.append(("archived_rolling_CV",fold,development[tr],development[va]))
    for scheme, fold, tr, va in split_specs:
        common_ids = sorted(set(train_ids.iloc[tr]) & set(train_ids.iloc[va]))
        split_overlap.append({"scheme":scheme,"fold":fold,"train_rows":len(tr),"valid_rows":len(va),
                              "shared_ID_groups":len(common_ids),"validation_rows_with_ID_in_training":int(train_ids.iloc[va].isin(common_ids).sum()),
                              "shared_IDs":";".join(common_ids)})
    pd.DataFrame(split_overlap).to_csv(DEST/"archived_split_ID_overlap.csv",index=False,encoding="utf-8-sig")
    overlap_rows = []
    for name in ("A", "B"):
        proxy = proxies[name]
        overlap_rows.append({"dataset": name, "rows": len(proxy),
                             "rows_in_train_proxy_range": int(proxy.between(proxies["train"].min(), proxies["train"].max()).sum()),
                             "rows_after_train_proxy_max": int(proxy.gt(proxies["train"].max()).sum()),
                             "rows_before_train_proxy_min": int(proxy.lt(proxies["train"].min()).sum()),
                             "id_overlap_with_train": int(frames[name][id_column].isin(frames["train"][id_column]).sum())})
    pd.DataFrame(overlap_rows).to_csv(DEST / "dataset_overlap.csv", index=False, encoding="utf-8-sig")
    window_rows = []
    for stage, windows in (("A", [(500,100),(600,100),(700,100)]), ("B", [(100,50),(150,50),(200,100)])):
        for fold, (prefix, size) in enumerate(windows, 1):
            if stage == "A":
                train_time, valid_time = proxies["train"].iloc[:prefix], proxies["train"].iloc[prefix:prefix+size]
            else:
                train_time = pd.concat([proxies["train"], proxies["A"].iloc[:prefix]])
                valid_time = proxies["A"].iloc[prefix:prefix+size]
            window_rows.append({"stage": stage, "fold": fold, "prefix_rows": prefix, "valid_rows": size,
                                "train_proxy_min": date(train_time.min()), "train_proxy_max": date(train_time.max()),
                                "valid_proxy_min": date(valid_time.min()), "valid_proxy_max": date(valid_time.max()),
                                "valid_rows_strictly_after_training_max": int(valid_time.gt(train_time.max()).sum()),
                                "strict_future_split_by_proxy": bool(valid_time.min() > train_time.max())})
    pd.DataFrame(window_rows).to_csv(DEST / "phase_window_chronology.csv", index=False, encoding="utf-8-sig")
    current = pd.read_csv(OUT / "predictions_A_detailed.csv")
    baseline = pd.read_csv(OUT / "predictions_A_blend_baseline.csv")
    y = answer.iloc[:, 1].to_numpy(float)
    interval_rows = []
    for name, detail in (("current_A",current),("original_blend_A",baseline)):
        assert detail.ID.equals(frames["A"][id_column].astype(str))
        covered = (y >= detail.lower_90) & (y <= detail.upper_90)
        interval_rows.append({"evaluation": name, "nominal_coverage": .9, "rows":len(y),
                              "covered_rows": int(covered.sum()), "observed_coverage": float(covered.mean()),
                              "mean_width": float((detail.upper_90-detail.lower_90).mean()),
                              "interpretation": "retrospective empirical coverage; no exchangeability or independent calibration guarantee"})
    holdout = pd.read_csv(OUT / "temporal_holdout_predictions.csv")
    interval_rows.append({"evaluation":"archived_holdout_blend", "nominal_coverage":.9, "rows":len(holdout),
                          "covered_rows": int(((holdout.y_true>=holdout.lower_90)&(holdout.y_true<=holdout.upper_90)).sum()),
                          "observed_coverage":float(((holdout.y_true>=holdout.lower_90)&(holdout.y_true<=holdout.upper_90)).mean()),
                          "mean_width":float((holdout.upper_90-holdout.lower_90).mean()),
                          "interpretation":"empirical coverage only; repeated development and calibration reuse limit inference"})
    pd.DataFrame(interval_rows).to_csv(DEST / "interval_coverage.csv",index=False,encoding="utf-8-sig")
    metric_rows = [{"evaluation":"retrospective_A_current",**metrics(y,current.prediction.to_numpy())},
                   {"evaluation":"retrospective_A_original_blend",**metrics(y,baseline.prediction.to_numpy())}]
    pd.DataFrame(metric_rows).to_csv(DEST/"retrospective_metrics.csv",index=False,encoding="utf-8-sig")
    backtest = pd.read_csv(OUT / "phase_b_backtest_predictions.csv")
    strategies = backtest.strategy.unique()
    fold_rows = []
    aligned = {}
    for strategy in strategies:
        piece = backtest.loc[backtest.strategy.eq(strategy)].copy().reset_index(drop=True)
        aligned[strategy] = piece
        for fold, group in piece.groupby("fold",sort=False):
            fold_rows.append({"strategy":strategy,"fold":int(fold),**metrics(group.y_true.to_numpy(),group.prediction.to_numpy())})
    for strategy, piece in aligned.items():
        assert piece[["fold","ID","y_true"]].equals(aligned[strategies[0]][["fold","ID","y_true"]])
    pd.DataFrame(fold_rows).to_csv(DEST/"phase_b_fold_comparison.csv",index=False,encoding="utf-8-sig")
    bootstrap_rows = []
    comparisons = [("train_plus_released_a_x1", "train_plus_released_a_x2"),
                   ("train_only_robust_blend", "train_plus_released_a_x1"),
                   ("train_only_robust_blend", "train_plus_released_a_x2")]
    for base_name, candidate_name in comparisons:
        base, candidate = aligned[base_name], aligned[candidate_name]
        loss_difference = (base.prediction-base.y_true)**2 - (candidate.prediction-candidate.y_true)**2
        groups = [loss_difference.loc[base.fold.eq(f)].to_numpy() for f in base.fold.unique()]
        for block in (1,5,10,20):
            bootstrap_rows.append(bootstrap(f"B_backtest: {base_name} minus {candidate_name}",groups,block))
    a_difference = (baseline.prediction.to_numpy()-y)**2 - (current.prediction.to_numpy()-y)**2
    for block in (1,5,10,20):
        bootstrap_rows.append(bootstrap("retrospective_A: original_blend minus current",[a_difference],block))
    pd.DataFrame(bootstrap_rows).to_csv(DEST/"paired_bootstrap_sensitivity.csv",index=False,encoding="utf-8-sig")
    b_base = aligned["train_only_robust_blend"]
    val_positions = np.concatenate([np.arange(100,150),np.arange(150,200),np.arange(200,300)])
    provenance = {"baseline_label_in_backtest":"train_only_robust_blend",
                  "max_abs_difference_from_current_A_on_same_positions":float(np.max(np.abs(b_base.prediction.to_numpy()-current.prediction.to_numpy()[val_positions]))),
                  "max_abs_difference_from_archived_blend_on_same_positions":float(np.max(np.abs(b_base.prediction.to_numpy()-baseline.prediction.to_numpy()[val_positions])))}
    referenced_outputs = ["feature_audit.csv","predictions_A_detailed.csv","predictions_A_blend_baseline.csv",
                          "temporal_holdout_predictions.csv","phase_b_backtest_predictions.csv"]
    summary = {"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_sha256":{name:hashlib.sha256((ROOT/"data"/filename).read_bytes()).hexdigest() for name,filename in files.items()},
               "archived_output_sha256":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in referenced_outputs},
               "timestamp_columns_used":len(time_columns),"dataset_temporal_audit":temporal_rows,"dataset_overlap":overlap_rows,
               "duplicate_id_groups":duplicate_id_groups,"archived_split_ID_overlap":split_overlap,
               "exact_feature_duplicate_groups":exact_groups,"interval_coverage":interval_rows,
               "retrospective_metrics":metric_rows,"B_baseline_provenance":provenance,
               "inference_limits":["A labels were reviewed in development history; current A is retrospective, not an untouched blind test.",
                                   "File row order is not certified manufacturing chronology; timestamp median is itself a heuristic.",
                                   "Feature schema was inferred from all original 800 covariate rows before CV.",
                                   "Blend weights and phase strategies were chosen on the same validation predictions whose minimum error is reported.",
                                   "Residual quantile intervals reuse selection residuals and lack verified exchangeability or independent calibration.",
                                   "Block bootstrap is exploratory row-order sensitivity, not post-selection-corrected or physical-batch-confirmatory inference."]}
    (DEST/"statistical_audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


def audit_missing_denominators() -> None:
    """Append-only supplementary audit; existing audit results are untouched.

    Run with --missing-only. A fresh, existing project cache can avoid rereading
    Excel; the source and cache hashes are recorded for provenance.
    """
    source = ROOT / "data" / "训练集.xlsx"
    cache = ROOT / "innovative_solution" / ".cache" / "训练集.pkl"
    cached = cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime
    frame = pd.read_pickle(cache) if cached else pd.read_excel(source)
    feature_audit = pd.read_csv(OUT / "feature_audit.csv")
    target = next(c for c in ("Value", "Y", "value", "y") if c in frame.columns)
    identifier = next(c for c in ("ID", "Id", "id") if c in frame.columns)
    views = {
        "raw_all_X_excluding_ID_and_Y": [c for c in frame.columns if c not in (target, identifier)],
        "archived_cleaned_kept_numeric": feature_audit.loc[feature_audit.status.eq("kept_numeric"), "feature"].tolist(),
    }
    results = []
    for name, columns in views.items():
        missing = frame[columns].isna()
        row_counts = missing.sum(axis=1)
        rates = row_counts / len(columns)
        maximum_rows = np.flatnonzero(rates.eq(rates.max()).to_numpy())
        results.append({"view": name, "sample_count": len(frame), "feature_denominator": len(columns),
                        "rows_with_at_least_one_missing": int(row_counts.gt(0).sum()),
                        "fraction_rows_with_at_least_one_missing": float(row_counts.gt(0).mean()),
                        "all_empty_feature_count": int(missing.all(axis=0).sum()),
                        "total_missing_cells": int(missing.to_numpy().sum()),
                        "total_cells": int(missing.size),
                        "overall_missing_cell_rate": float(missing.to_numpy().mean()),
                        "maximum_row_missing_count": int(row_counts.max()),
                        "maximum_row_missing_rate": float(rates.max()),
                        "maximum_row_source_indices_zero_based": maximum_rows.tolist(),
                        "maximum_row_IDs": frame[identifier].iloc[maximum_rows].astype(str).tolist()})
    result = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "read_from": str(cache if cached else source),
              "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest() if cached else None,
              "definition": "pandas isna; raw X excludes ID and Y; cleaned view reproduces archived kept_numeric columns",
              "views": results,
              "interpretation": "Raw and structurally cleaned missingness use different feature denominators. Do not attribute cleaned-view 609/800 to all raw fields."}
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "missing_denominator_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if "--missing-only" in sys.argv:
        audit_missing_denominators()
    else:
        main()
