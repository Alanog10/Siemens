#!/usr/bin/env python3
"""
Single-file hierarchical scorer.

Use case
--------
You have one Excel file with:
- hierarchy columns
- an ACTUAL column
- a PREDICTION column

The script reads that file and applies one function:
    calculate_hierarchical_error(...)

Metric
------
1) Score at subsegment / segment / BU levels
2) Within each level, weight rows by abs(actual)^alpha
3) Compute weighted RMSE in raw units
4) Normalize each level by its own weighted average abs(actual)
5) Combine:
      0.50 * subsegment
    + 0.25 * segment
    + 0.25 * BU

Lower is better.
0.0 is perfect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data_scoring.xlsx"   # one Excel file, one default sheet
OUTPUT_JSON = ""                    # e.g. "scoring_results.json" or "" to skip

# Hierarchy columns
PERIOD_COL = "Anon Period"
BU_COL = "TGL Business Unit"
SEG_COL = "TGL Business Segment"
SUBSEG_COL = "TGL Business Subsegment"

# Example target columns in the SAME table
# Edit these to match your file
ACTUAL_COL = "Revenue Actual"
PRED_COL = "Revenue Prediction"

# Final hierarchy weights AFTER normalization
LEVEL_WEIGHTS = {
    "subsegment": 0.50,
    "segment": 0.25,
    "bu": 0.25,
}

# Weighting power within each level
ACTUAL_WEIGHT_POWER = 1.0

# Small fallback so zero-actual rows still count a little
ZERO_WEIGHT_FLOOR = 1.0

# Numerical tolerance
PERFECT_TOL = 1e-12


def calculate_hierarchical_error(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    period_col: str = "Anon Period",
    bu_col: str = "TGL Business Unit",
    seg_col: str = "TGL Business Segment",
    subseg_col: str = "TGL Business Subsegment",
    level_weights: Dict[str, float] | None = None,
    actual_weight_power: float = 1.0,
    zero_weight_floor: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate normalized hierarchical error from a single table containing
    both actuals and predictions.

    Returns a dictionary with:
    - final_error
    - per-level normalized errors
    - per-level raw weighted RMSE / MAE
    - per-level scales
    """

    if level_weights is None:
        level_weights = {"subsegment": 0.50, "segment": 0.25, "bu": 0.25}

    required = [period_col, bu_col, seg_col, subseg_col, actual_col, pred_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[required].copy()
    work.columns = [str(c).strip() for c in work.columns]

    # Convert numeric columns
    work[period_col] = pd.to_numeric(work[period_col], errors="coerce")
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")

    if work[actual_col].isna().all():
        raise ValueError(f"Column '{actual_col}' is fully empty.")
    if work[pred_col].isna().all():
        raise ValueError(f"Column '{pred_col}' is fully empty.")

    # Treat blanks as 0 if needed
    work[actual_col] = work[actual_col].fillna(0.0)
    work[pred_col] = work[pred_col].fillna(0.0)

    # Aggregate duplicate leaf rows first
    leaf_keys = [period_col, bu_col, seg_col, subseg_col]
    leaf = (
        work.groupby(leaf_keys, dropna=False, as_index=False)[[actual_col, pred_col]]
        .sum()
        .sort_values(leaf_keys)
        .reset_index(drop=True)
    )

    def aggregate_level(frame: pd.DataFrame, level: str) -> pd.DataFrame:
        if level == "subsegment":
            keys = [period_col, bu_col, seg_col, subseg_col]
        elif level == "segment":
            keys = [period_col, bu_col, seg_col]
        elif level == "bu":
            keys = [period_col, bu_col]
        else:
            raise ValueError(f"Unknown level: {level}")

        return (
            frame.groupby(keys, dropna=False, as_index=False)[[actual_col, pred_col]]
            .sum()
            .sort_values(keys)
            .reset_index(drop=True)
        )

    def make_weights(actual: pd.Series) -> pd.Series:
        w = np.power(np.abs(actual.astype(float)), actual_weight_power)
        w = np.where(np.isfinite(w), w, zero_weight_floor)
        w = np.where(w <= 0, zero_weight_floor, w)
        return pd.Series(w, index=actual.index, dtype="float64")

    results = {}
    final_error = 0.0

    for level in ["subsegment", "segment", "bu"]:
        agg = aggregate_level(leaf, level)

        agg["err"] = agg[pred_col] - agg[actual_col]
        agg["sq_err"] = np.square(agg["err"])
        agg["abs_err"] = np.abs(agg["err"])
        agg["abs_actual"] = np.abs(agg[actual_col])
        agg["w"] = make_weights(agg[actual_col])

        wrmse = float(np.sqrt(np.average(agg["sq_err"], weights=agg["w"])))
        wmae = float(np.average(agg["abs_err"], weights=agg["w"]))

        # Normalize by the weighted average absolute actual within the SAME level
        level_scale = float(np.average(agg["abs_actual"], weights=agg["w"]))
        if not np.isfinite(level_scale) or level_scale <= 0:
            level_scale = float(zero_weight_floor)

        normalized_error = float(wrmse / level_scale)

        results[level] = {
            "normalized_error": normalized_error,
            "wrmse_raw_units": wrmse,
            "wmae_raw_units": wmae,
            "scale": level_scale,
            "n_rows_scored": int(len(agg)),
            "total_weight": float(agg["w"].sum()),
            "perfect_prediction": bool(wrmse <= PERFECT_TOL),
        }

        final_error += level_weights[level] * normalized_error

    out = {
        "final_error": float(final_error),
        "lower_is_better": True,
        "perfect_prediction": bool(final_error <= PERFECT_TOL),
        "actual_weight_power": actual_weight_power,
        "zero_weight_floor": zero_weight_floor,
        "level_weights": level_weights,
        "subsegment_normalized_error": results["subsegment"]["normalized_error"],
        "segment_normalized_error": results["segment"]["normalized_error"],
        "bu_normalized_error": results["bu"]["normalized_error"],
        "subsegment_wrmse_raw_units": results["subsegment"]["wrmse_raw_units"],
        "segment_wrmse_raw_units": results["segment"]["wrmse_raw_units"],
        "bu_wrmse_raw_units": results["bu"]["wrmse_raw_units"],
        "subsegment_scale": results["subsegment"]["scale"],
        "segment_scale": results["segment"]["scale"],
        "bu_scale": results["bu"]["scale"],
        "subsegment_n_rows_scored": results["subsegment"]["n_rows_scored"],
        "segment_n_rows_scored": results["segment"]["n_rows_scored"],
        "bu_n_rows_scored": results["bu"]["n_rows_scored"],
    }
    return out


def main() -> None:
    df = pd.read_excel(INPUT_FILE)

    result = calculate_hierarchical_error(
        df,
        actual_col=ACTUAL_COL,
        pred_col=PRED_COL,
        period_col=PERIOD_COL,
        bu_col=BU_COL,
        seg_col=SEG_COL,
        subseg_col=SUBSEG_COL,
        level_weights=LEVEL_WEIGHTS,
        actual_weight_power=ACTUAL_WEIGHT_POWER,
        zero_weight_floor=ZERO_WEIGHT_FLOOR,
    )

    print(json.dumps(result, indent=2))

    if OUTPUT_JSON:
        Path(OUTPUT_JSON).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
