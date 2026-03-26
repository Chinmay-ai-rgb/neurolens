#!/usr/bin/env python3
"""
train_iforest.py

Loads ml_matrix.csv, trains an Isolation Forest on BASELINE sessions only,
scores all sessions, and writes:
- models/iforest.pkl
- data/results/session_scores.csv

Assumptions:
- ml_matrix.csv has a 'session_id' column.
- Baseline sessions are identifiable by session_id containing 'baseline' (case-insensitive).
  (Works with names like "Baseline 6(1F)", "Baseline3(2F)", etc.)
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# -----------------------------
# Config (minimal + sensible)
# -----------------------------
PROJECT_ROOT = Path(".")
DEFAULT_INPUT = PROJECT_ROOT / "data" /"ml" / "ml_matrix.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "iforest.pkl"
OUT_PATH = PROJECT_ROOT / "data" / "results" / "session_scores.csv"
META_PATH = PROJECT_ROOT / "data" / "results" / "iforest_metadata.json"

RANDOM_STATE = 42

# Keep this stable; no tuning frenzy.
IFOREST_PARAMS = dict(
    n_estimators=300,
    contamination="auto",  # good default for novelty-ish setup
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def is_baseline_session(session_id: str) -> bool:
    """Return True if session_id looks like baseline."""
    if session_id is None:
        return False
    return "baseline" in str(session_id).lower()


def load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}.")
    df = pd.read_csv(path)
    if "session_id" not in df.columns:
        raise ValueError("ml_matrix.csv must contain a 'session_id' column.")
    if df.shape[0] < 5:
        raise ValueError(f"ml_matrix.csv has too few rows ({df.shape[0]}).")
    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Choose numeric feature columns, excluding identifiers and obvious labels.
    Keeps it generic so it works even if you add/remove features.
    """
    exclude = {
        "session_id",
        "label",
        "condition",
        "is_baseline",
        "is_fatigue",
        "session_valid_task_count",
        "session_total_task_count",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    if len(feature_cols) < 5:
        raise ValueError(
            f"Too few numeric feature columns found ({len(feature_cols)}). "
            "Check ml_matrix.csv formatting."
        )
    return feature_cols


def main(
    input_path: Path = DEFAULT_INPUT,
    model_path: Path = MODEL_PATH,
    out_path: Path = OUT_PATH,
    meta_path: Path = META_PATH,
) -> None:
    df = load_matrix(input_path)

    # Flag baseline/fatigue based on session_id string
    df["is_baseline"] = df["session_id"].apply(is_baseline_session)

    n_baseline = int(df["is_baseline"].sum())
    n_total = df.shape[0]
    if n_baseline < 3:
        raise ValueError(
            f"Found only {n_baseline} baseline sessions. "
            "Need at least 3 to train a stable baseline model."
        )

    feature_cols = select_feature_columns(df)

    X_all = df[feature_cols].copy()
    X_base = df.loc[df["is_baseline"], feature_cols].copy()

    # Pipeline: impute missing -> IsolationForest
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("iforest", IsolationForest(**IFOREST_PARAMS)),
        ]
    )

    # Train only on baseline
    pipe.fit(X_base)

    # Score everything
    # decision_function: higher = more normal, lower = more anomalous
    decision = pipe.decision_function(X_all)
    raw_score = -decision  # higher = more anomalous (intuitive)

    # Normalize anomaly to 0–1 for easier display (min-max across THIS dataset)
    # This is for visualization/reporting only.
    eps = 1e-9
    s_min, s_max = float(np.min(raw_score)), float(np.max(raw_score))
    anomaly_0_1 = (raw_score - s_min) / (s_max - s_min + eps)

    # Predict labels: -1 anomaly, 1 normal (based on iForest internal threshold)
    pred = pipe.predict(X_all)

    out_df = pd.DataFrame(
        {
            "session_id": df["session_id"].astype(str),
            "is_baseline": df["is_baseline"].astype(int),
            "iforest_pred": pred,  # -1 anomalous, 1 normal
            "iforest_decision_function": decision,
            "anomaly_score_raw": raw_score,
            "anomaly_score_0_1": anomaly_0_1,
        }
    )

    # Create dirs and save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipe,
            "feature_cols": feature_cols,
            "iforest_params": IFOREST_PARAMS,
            "baseline_rule": "session_id contains 'baseline' (case-insensitive)",
        },
        model_path,
    )

    out_df.to_csv(out_path, index=False)

    meta = {
        "input_path": str(input_path),
        "n_total_sessions": n_total,
        "n_baseline_sessions": n_baseline,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "iforest_params": IFOREST_PARAMS,
        "outputs": {
            "model_path": str(model_path),
            "scores_path": str(out_path),
        },
        "notes": [
            "Model trained on baseline only.",
            "anomaly_score_raw = -decision_function (higher = more anomalous).",
            "anomaly_score_0_1 is min-max normalized across this dataset for display.",
        ],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Isolation Forest training complete.")
    print(f"Baseline sessions: {n_baseline}/{n_total}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Saved model: {model_path}")
    print(f"Saved scores: {out_path}")
    print(f"Saved metadata: {meta_path}")

    # Quick sanity summary
    base_mean = float(out_df.loc[out_df["is_baseline"] == 1, "anomaly_score_raw"].mean())
    fat_mean = float(out_df.loc[out_df["is_baseline"] == 0, "anomaly_score_raw"].mean())
    print("\nSanity summary (raw anomaly, higher = more anomalous):")
    print(f"  baseline mean: {base_mean:.4f}")
    print(f"  non-baseline mean: {fat_mean:.4f}")
    if fat_mean > base_mean:
        print("  ✅ Non-baseline is more anomalous on average (expected if fatigue deviates).")
    else:
        print("  ⚠️ Non-baseline is NOT more anomalous on average — check labels / fatigue protocol.")


if __name__ == "__main__":
    main()
