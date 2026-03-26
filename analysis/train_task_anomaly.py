"""
Train a per-task unsupervised anomaly model from neurosignatures.csv.

Why per-task?
- Your current dataset contains many single-task sessions.
- A combined matrix forces heavy imputation -> lots of zeros after scaling.
- Per-task training uses only sessions where that task exists.

Outputs:
  data/models/<task>_isoforest.joblib
  data/models/<task>_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib


INPUT = Path("data/neurosignatures/neurosignatures.csv")
OUT_DIR = Path("data/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Conservative drop rules (same spirit as build_ml_matrix)
DROP_IF_CONTAINS = ["_std", "fps", "clamp", "calibration", "qc_", "invalid_reason"]
DROP_IF_ENDS_WITH = [
    "_n_trials", "_n_valid", "_valid_rate",
    "_pass_min_points", "_valid_point_count",
    "_warn_rate_raw", "_fail_rate_raw",
    "_rate_wrong_direction", "_rate_no_meaningful_movement", "_rate_clamp_rate_high",
    "_anticipatory_rate", "_clamp_high_rate", "_inhibition_success_rate",
]

# Minimal requirement so training isn't nonsense
MIN_SESSIONS = 3


def should_drop(col: str) -> bool:
    for s in DROP_IF_CONTAINS:
        if s in col:
            return True
    for suf in DROP_IF_ENDS_WITH:
        if col.endswith(suf):
            return True
    # avoid session-level bookkeeping as features
    if col in {"session_valid_task_count", "session_total_task_count", "session_quality_score_mean"}:
        return True
    return False


def select_task_features(df: pd.DataFrame, task: str) -> List[str]:
    prefix = f"{task}_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    kept = []
    for c in cols:
        if should_drop(c):
            continue
        if c == "session_id":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].isna().all():
            continue
        kept.append(c)
    return kept


def filter_sessions_with_task(df: pd.DataFrame, task_features: List[str]) -> pd.DataFrame:
    # Keep rows that have at least one non-NaN task feature
    mask = df[task_features].notna().any(axis=1)
    return df.loc[mask].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="One of: fixation, pursuit, saccade, antisaccade, grid9, visual_search")
    ap.add_argument("--contamination", type=float, default=0.15,
                    help="Expected anomaly fraction for IsolationForest (default 0.15)")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing: {INPUT}")

    df = pd.read_csv(INPUT)
    if "session_id" not in df.columns:
        raise ValueError("neurosignatures.csv missing 'session_id' column")

    feat_cols = select_task_features(df, args.task)
    if len(feat_cols) == 0:
        raise ValueError(
            f"No usable features found for task='{args.task}'. "
            f"Check neurosignatures.csv columns or drop rules."
        )

    df_task = filter_sessions_with_task(df, feat_cols)

    if len(df_task) < MIN_SESSIONS:
        raise ValueError(
            f"Not enough sessions containing task='{args.task}' to train safely "
            f"({len(df_task)} found, need >= {MIN_SESSIONS})."
        )

    X = df_task[feat_cols].copy()

    # Impute medians (robust)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X.values)

    # Robust scale
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    X_scaled = scaler.fit_transform(X_imp)

    # Isolation Forest
    model = IsolationForest(
        n_estimators=400,
        contamination=float(args.contamination),
        random_state=int(args.random_state),
        n_jobs=-1
    )
    model.fit(X_scaled)

    out_model = OUT_DIR / f"{args.task}_isoforest.joblib"
    out_manifest = OUT_DIR / f"{args.task}_manifest.json"

    bundle = {
        "task": args.task,
        "feature_cols": feat_cols,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
    }
    joblib.dump(bundle, out_model)

    manifest = {
        "task": args.task,
        "input": str(INPUT),
        "n_sessions_used": int(len(df_task)),
        "n_features_used": int(len(feat_cols)),
        "features": feat_cols,
        "model": {
            "type": "IsolationForest",
            "n_estimators": 400,
            "contamination": float(args.contamination),
            "random_state": int(args.random_state),
        },
        "preprocess": {
            "imputer": "median",
            "scaler": "RobustScaler(25-75)",
        }
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"✅ Trained {args.task} IsolationForest")
    print(f"Sessions used: {len(df_task)}")
    print(f"Features used: {len(feat_cols)}")
    print(f"Saved model: {out_model}")
    print(f"Saved manifest: {out_manifest}")

    # Quick sanity: show top 5 most anomalous sessions (lower = more anomalous)
    scores = model.decision_function(X_scaled)  # higher = more normal
    tmp = df_task[["session_id"]].copy()
    tmp["score"] = scores
    tmp = tmp.sort_values("score", ascending=True).head(5)
    print("\nMost anomalous (lowest scores):")
    print(tmp.to_string(index=False))


if __name__ == "__main__":
    main()
n