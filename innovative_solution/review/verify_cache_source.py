"""Compare existing Excel caches against source workbooks without updating them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "innovative_solution" / "outputs" / "review"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((DEST / "review_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for name in ("训练集", "测试集A", "测试集B"):
        source = ROOT / "data" / f"{name}.xlsx"
        cache = ROOT / "innovative_solution" / ".cache" / f"{name}.pkl"
        print(f"Checking {name}", flush=True)
        source_hash = sha256(source)
        cache_hash = sha256(cache)
        excel_frame = pd.read_excel(source)
        cache_frame = pd.read_pickle(cache)
        comparison_error = None
        try:
            assert_frame_equal(excel_frame, cache_frame, check_dtype=False, check_exact=True,
                               check_index_type=False, check_column_type=False)
            equal = True
        except AssertionError as exc:
            equal = False
            comparison_error = str(exc)
        rows.append({"dataset": name, "source_path": str(source), "cache_path": str(cache),
                     "source_sha256": source_hash, "cache_sha256": cache_hash,
                     "source_matches_training_manifest_sha256": source_hash == manifest["input_sha256"][source.name],
                     "source_shape": list(excel_frame.shape), "cache_shape": list(cache_frame.shape),
                     "row_index_equal": bool(excel_frame.index.equals(cache_frame.index)),
                     "column_names_and_order_equal": bool(excel_frame.columns.equals(cache_frame.columns)),
                     "all_cell_values_exactly_equal": equal,
                     "comparison_settings": "assert_frame_equal(check_dtype=False, check_exact=True, check_index_type=False, check_column_type=False)",
                     "comparison_error": comparison_error,
                     "source_unchanged_during_read": sha256(source) == source_hash,
                     "cache_unchanged_during_read": sha256(cache) == cache_hash})
    result = {"verification": "direct source Excel versus existing pickle cache; caches and models not modified",
              "comparison_numeric_tolerance": "exact equality; no floating point tolerance was needed or applied",
              "all_sources_and_caches_match": all(row["all_cell_values_exactly_equal"]
                                                  and row["source_matches_training_manifest_sha256"]
                                                  and row["source_unchanged_during_read"]
                                                  and row["cache_unchanged_during_read"] for row in rows),
              "datasets": rows}
    (DEST / "cache_source_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["all_sources_and_caches_match"]:
        raise SystemExit("One or more source/cache checks failed; inspect the JSON")


if __name__ == "__main__":
    main()
