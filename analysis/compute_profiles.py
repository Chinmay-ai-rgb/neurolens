# analysis/compute_profiles.py
"""
Rule-based Biomarker Profiles for NeuroLens+

Input:
  - data/ml/ml_matrix_unscaled.csv   (preferred)
    or data/ml_matrix_unscaled.csv   (fallback)

Output:
  - data/results/profile_scores.csv
  - data/results/profile_top_biomarkers.json
  - data/results/profile_metadata.json

What it does:
  - Uses BASELINE sessions to compute per-feature baseline mean/std
  - Converts each session's features to abs(z) deviations from baseline
  - Aggregates deviations into 6 neurologically grounded profiles
  - Extracts top contributing biomarkers per profile per session

NOTE:
  - Rule-based, not ML
  - Non-diagnostic: "elevated profiles" not disease labels
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

DEFAULT_INPUT_CANDIDATES = [
    Path("data/ml/ml_matrix_unscaled.csv"),
    Path("data/ml_matrix_unscaled.csv"),
]

OUTPUT_DIR = Path("data/results")
OUTPUT_SCORES = OUTPUT_DIR / "profile_scores.csv"
OUTPUT_TOP = OUTPUT_DIR / "profile_top_biomarkers.json"
OUTPUT_META = OUTPUT_DIR / "profile_metadata.json"

SESSION_ID_COL = "session_id"
IS_BASELINE_COL = "is_baseline"

# If std is ~0 for a feature in baseline, avoid divide-by-zero explosions
STD_EPS = 1e-8

# How many top biomarkers to report per profile per session
TOP_K = 3


# -----------------------------
# Profile Definitions
# -----------------------------

@dataclass(frozen=True)
class ProfileDef:
    name: str
    description: str
    # list of (include_patterns, exclude_patterns)
    # patterns are matched as substring against column names
    include: Tuple[str, ...]
    exclude: Tuple[str, ...] = ()


PROFILES: List[ProfileDef] = [
    ProfileDef(
        name="Inhibitory Control / Executive Function",
        description="Response inhibition and error control (anti-saccade behavior).",
        include=(
            "antisaccade_direction_error",
            "antisaccade_inhibition_success",
            "antisaccade_anticipatory_rate",
            "antisaccade_latency",
            "antisaccade_correction_time",
            "antisaccade_quality_score",
        ),
    ),
    ProfileDef(
        name="Oculomotor Speed & Timing",
        description="Saccadic timing/kinematics (latency, peak velocity, duration, gain).",
        include=(
            "saccade_saccade_latency",
            "saccade_saccade_duration",
            "saccade_peak_velocity",
            "saccade_gain",
            "saccade_landing_error",
            "saccade_corrective_saccade_count",
            "saccade_quality_score",
        ),
    ),
    ProfileDef(
        name="Smooth Pursuit & Sensorimotor Integration",
        description="Pursuit tracking fidelity (gain, lag, position error, catch-up saccades).",
        include=(
            "pursuit_pursuit_gain",
            "pursuit_pursuit_latency",
            "pursuit_phase_lag",
            "pursuit_position_error",
            "pursuit_catch_up_saccade",
            "pursuit_quality_score",
        ),
    ),
    ProfileDef(
        name="Fixation Stability & Gaze Noise",
        description="Stability during fixation (drift, RMS, BCEA, percent on target).",
        include=(
            "fixation_fixation_stability_rms",
            "fixation_fixation_bcea",
            "fixation_drift_velocity",
            "fixation_percent_time_on_target",
            "fixation_microsaccade_rate",
            "fixation_quality_score",
        ),
    ),
    ProfileDef(
        name="Spatial Gaze Accuracy & Coordination",
        description="Spatial accuracy across gaze targets (grid9 errors & dwell stability).",
        include=(
            "grid9_mean_gaze_error",
            "grid9_rmse_gaze_error",
            "grid9_dwell_stability",
            "grid9_quality_score",
        ),
    ),
    ProfileDef(
        name="Blink Regulation & Attentional Arousal",
        description="Blink behavior and attentional stability (blink rate, IBI, gaze presence).",
        include=(
            "visual_search_blink_rate",
            "visual_search_blink_count",
            "visual_search_interblink_interval",
            "visual_search_blink_burstiness",
            "visual_search_gaze_presence",
            "visual_search_quality_score",
            # optionally capture fixation blink too if present
            "fixation_blink_rate",
        ),
    ),
]


# -----------------------------
# Helpers
# -----------------------------

def find_input_path() -> Path:
    for p in DEFAULT_INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find ml_matrix_unscaled.csv. Tried:\n"
        + "\n".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    )


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if SESSION_ID_COL not in df.columns:
        raise ValueError(f"Missing required column: {SESSION_ID_COL}")
    if IS_BASELINE_COL not in df.columns:
        # Allow older matrices that used session_id naming conventions only
        # but strongly prefer explicit is_baseline
        raise ValueError(
            f"Missing required column: {IS_BASELINE_COL}. "
            "Your build_ML_matrix.py should include is_baseline."
        )
    return df


def numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    # Keep only numeric columns that aren't metadata
    excluded = {SESSION_ID_COL, IS_BASELINE_COL}
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def baseline_stats(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    base = df[df[IS_BASELINE_COL] == 1]
    if len(base) < 3:
        raise ValueError(f"Not enough baseline rows to compute stable stats: {len(base)}")
    mu = base[feat_cols].mean(axis=0)
    sd = base[feat_cols].std(axis=0, ddof=0).replace(0.0, STD_EPS)
    sd = sd.fillna(STD_EPS)
    return mu, sd


def abs_zscores(df: pd.DataFrame, feat_cols: List[str], mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    z = (df[feat_cols] - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.abs()


def match_profile_columns(all_cols: List[str], prof: ProfileDef) -> List[str]:
    matched = []
    for c in all_cols:
        inc_ok = any(pat in c for pat in prof.include)
        exc_ok = not any(pat in c for pat in prof.exclude)
        if inc_ok and exc_ok:
            matched.append(c)
    return matched


def compute_profile_score(absz_row: pd.Series, cols: List[str]) -> float:
    if not cols:
        return float("nan")
    # Mean absolute deviation across selected biomarkers
    return float(np.mean(absz_row[cols].values))


def top_contributors(absz_row: pd.Series, cols: List[str], k: int = TOP_K) -> List[Dict]:
    if not cols:
        return []
    sub = absz_row[cols].sort_values(ascending=False)
    out = []
    for name, val in sub.head(k).items():
        out.append({"biomarker": name, "abs_z": float(val)})
    return out


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    in_path = find_input_path()
    df = load_matrix(in_path)

    feat_cols = numeric_feature_cols(df)
    if len(feat_cols) < 10:
        raise ValueError(f"Too few numeric features found ({len(feat_cols)}). Check your matrix output.")

    mu, sd = baseline_stats(df, feat_cols)
    absz = abs_zscores(df, feat_cols, mu, sd)

    # Build profile column mapping
    profile_to_cols: Dict[str, List[str]] = {}
    missing_profiles: List[str] = []
    for prof in PROFILES:
        cols = match_profile_columns(feat_cols, prof)
        profile_to_cols[prof.name] = cols
        if len(cols) == 0:
            missing_profiles.append(prof.name)

    # Compute per-session profile scores + top biomarkers
    rows = []
    tops: Dict[str, Dict[str, List[Dict]]] = {}  # session_id -> profile_name -> contributors

    for i, r in df.iterrows():
        sid = str(r[SESSION_ID_COL])
        is_base = int(r[IS_BASELINE_COL])

        absz_row = absz.loc[i]
        session_profile_scores = {}

        tops[sid] = {}
        for prof in PROFILES:
            cols = profile_to_cols[prof.name]
            score = compute_profile_score(absz_row, cols)
            session_profile_scores[prof.name] = score
            tops[sid][prof.name] = top_contributors(absz_row, cols, TOP_K)

        # overall deviation index: mean of available profile scores
        prof_vals = [v for v in session_profile_scores.values() if np.isfinite(v)]
        overall = float(np.mean(prof_vals)) if prof_vals else float("nan")

        row = {
            "session_id": sid,
            "is_baseline": is_base,
            "overall_deviation_index": overall,
        }
        # add profile columns (snake-ish but readable)
        for prof in PROFILES:
            col_name = (
                "profile__"
                + prof.name.lower()
                .replace(" / ", "_")
                .replace("&", "and")
                .replace(" ", "_")
                .replace("__", "_")
            )
            row[col_name] = session_profile_scores[prof.name]

        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Ensure output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(OUTPUT_SCORES, index=False)
    with open(OUTPUT_TOP, "w", encoding="utf-8") as f:
        json.dump(tops, f, indent=2)

    meta = {
        "input_path": str(in_path),
        "n_sessions": int(len(df)),
        "n_baseline": int((df[IS_BASELINE_COL] == 1).sum()),
        "n_features_total": int(len(feat_cols)),
        "profiles": [
            {
                "name": p.name,
                "description": p.description,
                "n_biomarkers_matched": int(len(profile_to_cols[p.name])),
                "biomarkers": profile_to_cols[p.name],
            }
            for p in PROFILES
        ],
        "missing_profiles": missing_profiles,
        "notes": [
            "Scores are mean absolute z-deviation from baseline across selected biomarkers.",
            "Top biomarkers are those with highest absolute z per profile per session.",
            "Rule-based profiles are interpretive aids; not diagnoses.",
        ],
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Console summary
    print("✅ compute_profiles.py complete.")
    print(f"Loaded: {in_path}")
    print(f"Saved scores: {OUTPUT_SCORES}")
    print(f"Saved top biomarkers: {OUTPUT_TOP}")
    print(f"Saved metadata: {OUTPUT_META}")
    if missing_profiles:
        print("\n⚠️ Warning: Some profiles matched 0 biomarkers (check your column names):")
        for name in missing_profiles:
            print(f"  - {name}")

    # Quick sanity: fatigue vs baseline overall deviation (if both exist)
    base_vals = out_df[out_df["is_baseline"] == 1]["overall_deviation_index"].dropna()
    non_vals = out_df[out_df["is_baseline"] == 0]["overall_deviation_index"].dropna()
    if len(base_vals) and len(non_vals):
        print("\nSanity (overall_deviation_index):")
        print(f"  baseline mean: {base_vals.mean():.4f}")
        print(f"  non-baseline mean: {non_vals.mean():.4f}")
        if non_vals.mean() > base_vals.mean():
            print("  ✅ Non-baseline higher on average (expected if fatigue deviates).")
        else:
            print("  ⚠️ Non-baseline not higher on average (could be overlap, check sessions/QC).")


if __name__ == "__main__":
    main()
