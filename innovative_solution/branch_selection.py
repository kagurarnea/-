"""Four feature views compete in one retrospective nested selection protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .review_validation import PROTOCOL, run_review_validation, json_write, sha256
from .supplement_statistics import paired_cluster

OUT = Path(__file__).resolve().parent / "outputs" / "branch_selection"
CANDIDATES = [f"{view}_d{depth}" for view in ["raw", "process", "time", "full"] for depth in [2, 3]]
BRANCH_PROTOCOL = {**PROTOCOL, "version": "feature_branch_selection_v1",
    "candidates": CANDIDATES, "outer_fixed_candidates": CANDIDATES,
    "candidate_scope": "basic, basic+process, basic+time, combined; each depth 2/3",
    "candidate_scope_informed_by_prior_ablation": True,
    "independent_confirmation": False,
    "interpretation": "retrospective_adaptive_candidate_expansion_not_pristine_test",
    "final_selection": "pooled development-inner MSE only; never outer, calibration, tail or A score",
    "time_view": "basic plus time__ and route__; excludes proc__",
    "process_view": "basic plus proc__; excludes time__ and route__"}


def summarize():
    scores = pd.read_csv(OUT / "inner_selection.csv")
    scores["loss"] = scores.mse * scores.n_valid
    pooled = scores.groupby(["stage", "candidate"], sort=False).agg(
        loss=("loss", "sum"), n=("n_valid", "sum"),
        fold_min=("mse", "min"), fold_max=("mse", "max")).reset_index()
    pooled["pooled_mse"] = pooled.loss / pooled.n
    pooled["rank"] = pooled.groupby("stage").pooled_mse.rank(method="first")
    pooled.to_csv(OUT / "selection_score_distribution.csv", index=False)
    selected = pooled[pooled["rank"].eq(1)]
    selected.to_csv(OUT / "selected_configs.csv", index=False)
    frequency = selected[selected.stage.str.startswith("outer_")].candidate.value_counts()
    frequency.reindex(CANDIDATES, fill_value=0).rename_axis("candidate").reset_index(name="outer_count").to_csv(
        OUT / "selection_frequency.csv", index=False)
    pred = pd.read_csv(OUT / "nested_predictions.csv")
    comparisons = [("fixed_raw_d2", "fixed_time_d2"), ("fixed_full_d2", "fixed_time_d2"),
                   ("fixed_full_d2", "nested_selected"), ("fixed_time_d2", "nested_selected")]
    stats = pd.DataFrame([paired_cluster(pred, a, b) for a, b in comparisons])
    p = stats.group_sign_flip_p.to_numpy(); order = np.argsort(p); adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    stats["holm_p_within_four_reported_comparisons"] = adjusted
    stats.to_csv(OUT / "paired_group_inference.csv", index=False)
    # Compare restricted and expanded selection on exactly the same outer records.
    restricted = pd.read_csv(OUT.parent / "review" / "nested_predictions.csv")
    restricted = restricted[restricted.model.eq("nested_selected")].assign(model="restricted_selected")
    both = pd.concat([pred[pred.model.eq("nested_selected")], restricted], ignore_index=True)
    pd.DataFrame([paired_cluster(both, "restricted_selected", "nested_selected")]).to_csv(
        OUT / "search_scope_comparison.csv", index=False)
    audit = pd.read_csv(OUT / "split_audit.csv")
    reference = pd.read_csv(OUT.parent / "review" / "split_audit.csv")
    keys = ["stage", "fold", "train_indices", "valid_indices"]
    aligned = reference[keys].merge(audit[keys], on=["stage", "fold"], suffixes=("_ref", "_run"), validate="one_to_one")
    if (len(aligned) != len(reference) or
        not (aligned.train_indices_ref == aligned.train_indices_run).all() or
        not (aligned.valid_indices_ref == aligned.valid_indices_run).all()):
        raise AssertionError("Branch search must preserve the exact entity/time-isolated splits")
    # The reference file predates a separate calibration-vs-tail audit row.
    extra = audit.merge(reference[["stage", "fold"]], on=["stage", "fold"], how="left", indicator=True)
    assert set(extra.loc[extra["_merge"].eq("left_only"), "stage"]) <= {"calibration_vs_tail"}
    manifest = json.loads((OUT / "review_manifest.json").read_text(encoding="utf-8"))
    reference_manifest = json.loads((OUT.parent / "review" / "review_manifest.json").read_text(encoding="utf-8"))
    if manifest["input_sha256"] != reference_manifest["input_sha256"]:
        raise AssertionError("Search-scope comparison requires the same source files and labels")
    dev = selected[selected.stage.eq("final_development")].iloc[0]
    assert manifest["selected_candidate"] == dev.candidate
    manifest["code_sha256"][Path(__file__).name] = sha256(Path(__file__))
    manifest["split_indices_equal_restricted_search"] = True
    json_write(OUT / "review_manifest.json", manifest)


def main():
    run_review_validation(protocol=BRANCH_PROTOCOL, output=OUT)
    summarize()


if __name__ == "__main__":
    main()
