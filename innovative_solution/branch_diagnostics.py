"""Diagnostics tied to the actual selected eight-candidate model."""
import json
import time
import platform
import joblib
import numpy as np
import pandas as pd
from .branch_selection import OUT
from .pipeline import metric_row
from .review_validation import json_write
from .supplement_tail import rolling_radii, wilson


def predict(bundle, frames):
    values = []
    for member in bundle["members"]:
        X = member["builder"].transform(*frames)
        values.append(member["model"].predict(X[member["columns"]]).astype(float))
    return np.mean(values, axis=0)


def main():
    manifest = json.loads((OUT / "review_manifest.json").read_text(encoding="utf-8"))
    frozen = joblib.load(OUT / "frozen_historical_model.joblib")
    data = frozen["members"][0]["builder"].data
    splits = pd.read_csv(OUT / "split_audit.csv")
    row = splits[splits.stage.eq("calibration")].iloc[0]
    train, cal = np.array(json.loads(row.train_indices)), np.array(json.loads(row.valid_indices))
    tail = pd.read_csv(OUT / "frozen_tail_predictions.csv")
    valid = tail.row_index.to_numpy()
    cp = predict(frozen, (data.train_numeric.iloc[cal], data.train_categorical.iloc[cal], data.train_time.iloc[cal]))
    cal_error = data.y.iloc[cal].to_numpy() - cp
    expected = pd.read_csv(OUT / "calibration_residuals.csv")
    np.testing.assert_allclose(np.abs(cal_error), expected.absolute_error, rtol=0, atol=1e-12)
    y, vp = tail.actual.to_numpy(), tail.prediction.to_numpy()
    rolling = rolling_radii(cal_error, data.order_key.iloc[cal].to_numpy(), y, vp, data.order_key.iloc[valid].to_numpy())
    q1, q2 = np.quantile(cp, [1/3, 2/3])
    bins = np.where(vp < q1, "lower_predicted", np.where(vp < q2, "middle_predicted", "upper_predicted"))
    categories = [("overall", "all", np.ones(len(valid), bool))]
    categories += [("predicted_tercile", b, bins == b) for b in np.unique(bins)]
    for col in data.train_categorical:
        labels = data.train_categorical.iloc[valid][col].astype("string").fillna("__MISSING__").to_numpy()
        categories += [(str(col), str(label), labels == label) for label in pd.unique(labels)]
    coverage, predictions = [], []
    for strategy, radii in [("fixed", np.full(len(valid), frozen["radius"])), ("rolling80_immediate_labels", rolling)]:
        covered = np.abs(y-vp) <= radii
        predictions.append(tail.assign(strategy=strategy, radius=radii, covered=covered))
        for variable, label, mask in categories:
            n, k = int(mask.sum()), int(covered[mask].sum()); low, high = wilson(k, n)
            coverage.append(dict(strategy=strategy, stratifier=variable, level=label, n=n, covered=k,
                coverage=k/n, mean_width=float(np.mean(2*radii[mask])), wilson_low=low, wilson_high=high, small_group=n<20))
    pd.DataFrame(coverage).to_csv(OUT / "conditional_coverage.csv", index=False)
    pd.concat(predictions).to_csv(OUT / "conditional_interval_predictions.csv", index=False)
    cutoff = float(data.y.iloc[train].quantile(.1)); low = y <= cutoff; flag = vp <= cutoff
    metrics = dict(model=manifest["selected_candidate"], **metric_row(y, vp),
        low_cutoff_from_development=cutoff, n_low=int(low.sum()), low_tail_mse=float(np.mean((y[low]-vp[low])**2)),
        low_value_recall=float(np.mean(flag[low])), median_pinball_loss=float(np.mean(abs(y-vp))/2))
    pd.DataFrame([metrics]).to_csv(OUT / "selected_tail_metrics.csv", index=False)
    tail["residual"] = y-vp; tail["squared_error"] = (y-vp)**2
    tail["fraction_total_squared_error"] = tail.squared_error / tail.squared_error.sum()
    tail.sort_values("squared_error", ascending=False).to_csv(OUT / "tail_error_influence.csv", index=False)
    bundle = joblib.load(OUT / "model_A.joblib")
    times = []
    for n in [1, 100]:
        for repeat in range(4):
            stamp = time.perf_counter()
            predict(bundle, (data.test_a_numeric.iloc[:n], data.test_a_categorical.iloc[:n], data.test_a_time.iloc[:n]))
            if repeat: times.append(dict(batch_size=n, repeat=repeat, seconds=time.perf_counter()-stamp))
    json_write(OUT / "inference_latency.json", dict(candidate=manifest["selected_candidate"], platform=platform.platform(),
        medians={str(n):float(np.median([r["seconds"] for r in times if r["batch_size"]==n])) for n in [1,100]},
        measurements=times, scope="warm model and input; preprocessing plus three members; excludes I/O and production delays"))
    checks = []
    for file, ids in [("submission_A.csv", data.test_a_ids), ("submission_B.csv", data.test_b_ids), ("submission_B_train_only.csv", data.test_b_ids)]:
        frame = pd.read_csv(OUT / file, header=None)
        assert frame.shape == (len(ids), 2)
        assert frame.iloc[:, 0].tolist() == ids.tolist()
        assert np.isfinite(frame.iloc[:, 1]).all()
        checks.append(dict(file=file, rows=len(frame), columns=2, header=False, exact_ID_order=True, finite=True))
    json_write(OUT / "submission_format_check.json", checks)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
