"""
Build ML-ready matrix from NeuroSignatures.

Input:
  data/neurosignatures/neurosignatures.csv

Outputs:
  data/ml/ml_matrix_unscaled.csv   (features + session_id + is_baseline, unscaled)
  data/ml/ml_matrix.csv            (features + session_id + is_baseline, robust-scaled)
  data/ml/feature_manifest.json    (columns kept/dropped + rules)

This script is intentionally conservative:
- Drops sessions with too little usable data
- Drops unstable/noisy columns (std, qc, fps, clamp, calibration, counts)
- Keeps primarily biomarker MEANS (plus a few key behavior rates)
- Robust-scales features (median/IQR) for anomaly detection stability
- Adds is_baseline as metadata (NOT a feature) for downstream steps (profiles, filtering)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


INPUT_CSV = Path("data/neurosignatures/neurosignatures.csv")
OUT_DIR = Path("data/ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_UNSCALED = OUT_DIR / "ml_matrix_unscaled.csv"
OUT_SCALED = OUT_DIR / "ml_matrix.csv"
OUT_MANIFEST = OUT_DIR / "feature_manifest.json"


# --- Hard gating rules (defensible + prevents garbage training) ---
# NOTE: if your sessions are full multi-task runs, set this to 6.
MIN_VALID_TASKS = 1


# --- Column drop rules (conservative, research-safe) ---
DROP_IF_CONTAINS = [
    "_std",  # unstable if n small; keep means first
    "fps_", "fps",  # device/compute artifact, not biomarker
    "clamp_", "clamp",  # tracking artifact; use for QC, not ML
    "calibration_", "calib",  # session setup artifact
    "qc_", "invalid_reason",  # QC labels, not biomarkers
]

DROP_IF_ENDS_WITH = [
    "_n_trials", "_n_valid", "_valid_rate",
    "_pass_min_points", "_valid_point_count",
    "_warn_rate_raw", "_fail_rate_raw",
    "_rate_wrong_direction", "_rate_no_meaningful_movement", "_rate_clamp_rate_high",

    # IMPORTANT:
    # We KEEP key behavior/biomarker rates for Profile #1 (executive/inhibitory control),
    # so we do NOT drop:
    #   - *_anticipatory_rate
    #   - *_inhibition_success_rate
    #
    # We still drop clamp-high-rate because it is primarily a tracking artifact.
    "_clamp_high_rate",
]


def load_df() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError(f"Input file exists but is empty: {INPUT_CSV}")
    if "session_id" not in df.columns:
        raise ValueError("Expected 'session_id' column not found in neurosignatures.csv")
    return df


def add_is_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds explicit baseline label for downstream steps (profiles, ML filtering).
    Rule: session_id contains 'Baseline' => baseline.
    """
    out = df.copy()
    out["is_baseline"] = out["session_id"].astype(str).str.contains("Baseline", case=False, na=False).astype(int)
    return out


def gate_sessions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Drop sessions with insufficient usable task coverage."""
    meta: Dict = {}

    if "session_valid_task_count" not in df.columns:
        raise ValueError(
            "Expected column 'session_valid_task_count' not found. "
            "Your NeuroSignature builder should have produced it."
        )

    before = len(df)

    # Dataset max coverage
    max_tasks = int(df["session_valid_task_count"].fillna(0).max())

    # If the dataset cannot meet MIN_VALID_TASKS, fallback so we don't drop everything.
    if max_tasks < MIN_VALID_TASKS:
        min_req = max(1, max_tasks)
        df2 = df[df["session_valid_task_count"].fillna(0) >= min_req].copy()
        meta["session_gate"] = {
            "rule": f"session_valid_task_count >= {min_req} (fallback, dataset max was {max_tasks})",
            "before": before,
            "after": len(df2),
            "dropped": before - len(df2),
        }
        return df2, meta

    df2 = df[df["session_valid_task_count"].fillna(0) >= MIN_VALID_TASKS].copy()
    after = len(df2)

    meta["session_gate"] = {
        "rule": f"session_valid_task_count >= {MIN_VALID_TASKS}",
        "before": before,
        "after": after,
        "dropped": before - after,
    }
    return df2, meta


def should_drop(col: str) -> bool:
    # Keep metadata columns
    if col in ("session_id", "is_baseline"):
        return False

    for s in DROP_IF_CONTAINS:
        if s in col:
            return True

    for suf in DROP_IF_ENDS_WITH:
        if col.endswith(suf):
            return True

    # Drop obvious metadata / bookkeeping if present
    if col in {"session_total_task_count", "session_quality_score_mean"}:
        # session_quality_score_mean is mostly tracking quality, not a biomarker.
        return True

    return False


def select_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Pick a conservative biomarker feature set."""
    # Exclude metadata columns from features
    candidates = [c for c in df.columns if c not in ("session_id", "is_baseline")]

    dropped: List[str] = []
    kept: List[str] = []

    for c in candidates:
        if should_drop(c):
            dropped.append(c)
            continue

        # Keep only numeric columns
        if not pd.api.types.is_numeric_dtype(df[c]):
            dropped.append(c)
            continue

        kept.append(c)

    # Extra guard: remove columns that are entirely NaN after gating
    kept2: List[str] = []
    for c in kept:
        if df[c].isna().all():
            dropped.append(c)
        else:
            kept2.append(c)

    return kept2, sorted(set(dropped))


def build_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Build unscaled and scaled matrices.

    NaN policy:
      - If a feature is missing in a session, impute with column median
        (after gating, this is usually from occasional invalid trials).

    NOTE:
      - is_baseline is metadata only (NOT scaled, NOT used as a feature).
    """
    if len(df) == 0:
        raise ValueError(
            "After gating, 0 sessions remain. "
            "Lower gating thresholds or collect more sessions."
        )

    if len(feature_cols) == 0:
        raise ValueError(
            "After feature selection, 0 features remain. "
            "Your drop rules are too aggressive for your current dataset."
        )

    X = df[feature_cols].copy()

    # Impute with median (robust + judge-defensible)
    medians = X.median(numeric_only=True)
    X_imputed = X.fillna(medians)

    # Robust scaling
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    X_scaled_arr = scaler.fit_transform(X_imputed.values)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=feature_cols)

    # Add metadata back for traceability
    meta_cols = ["session_id", "is_baseline"]

    unscaled_out = pd.concat(
        [df[meta_cols].reset_index(drop=True), X_imputed.reset_index(drop=True)],
        axis=1
    )
    scaled_out = pd.concat(
        [df[meta_cols].reset_index(drop=True), X_scaled.reset_index(drop=True)],
        axis=1
    )

    meta = {
        "n_sessions": int(len(df)),
        "n_features": int(len(feature_cols)),
        "nan_imputation": "column_median",
        "scaling": {
            "method": "RobustScaler",
            "quantile_range": [25.0, 75.0],
        },
        "feature_medians": {k: float(v) if np.isfinite(v) else None for k, v in medians.to_dict().items()},
    }

    return unscaled_out, scaled_out, meta


def main():
    df = load_df()

    # Gate sessions
    df_gated, meta_gate = gate_sessions(df)
    if len(df_gated) == 0:
        raise ValueError(
            "All sessions were dropped by gating. "
            "Collect more valid sessions or reduce gating."
        )

    # Add baseline flag (metadata)
    df_gated = add_is_baseline(df_gated)

    # Select features (excluding metadata)
    feature_cols, dropped_cols = select_features(df_gated)

    # Build matrices
    unscaled_out, scaled_out, meta_matrix = build_matrix(df_gated, feature_cols)

    # Save
    unscaled_out.to_csv(OUT_UNSCALED, index=False)
    scaled_out.to_csv(OUT_SCALED, index=False)

    manifest = {
        "input": str(INPUT_CSV),
        "outputs": {
            "unscaled": str(OUT_UNSCALED),
            "scaled": str(OUT_SCALED),
            "manifest": str(OUT_MANIFEST),
        },
        **meta_gate,
        **meta_matrix,
        "kept_features": feature_cols,
        "dropped_features": dropped_cols,
        "rules": {
            "min_valid_tasks": MIN_VALID_TASKS,
            "drop_if_contains": DROP_IF_CONTAINS,
            "drop_if_ends_with": DROP_IF_ENDS_WITH,
            "baseline_label_rule": "session_id contains 'Baseline' (case-insensitive) => is_baseline=1 else 0",
            "notes": [
                "This is a conservative feature set intended for unsupervised anomaly detection.",
                "QC/device artifacts are excluded from ML features and used only for gating/monitoring.",
                "Sessions may currently be single-task sessions; full multi-task models require combined sessions.",
                "Key executive/inhibitory control behavior rates are retained (e.g., antisaccade_anticipatory_rate, antisaccade_inhibition_success_rate).",
                "is_baseline is included as metadata only and is NOT scaled or used as an ML feature.",
            ],
        },
    }

    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))

    # Print preview
    print(f"✅ Wrote unscaled ML matrix: {OUT_UNSCALED}")
    print(f"✅ Wrote scaled ML matrix:   {OUT_SCALED}")
    print(f"✅ Wrote manifest:           {OUT_MANIFEST}")
    print(f"Sessions kept: {len(df_gated)} / {len(df)}")
    print(f"Features kept: {len(feature_cols)}")
    print("\nPreview (scaled):")
    print(scaled_out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
