#!/usr/bin/env python3
"""
NeuroLens+ NeuroSignature Builder
- Reads per-trial summary_*.csv files under data/sessions/<session_id>/
- Applies QC gates
- Aggregates valid trials only
- Outputs one row per session to data/neurosignatures/neurosignatures.csv
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# QC / Gating configuration
# -----------------------------

@dataclass
class Gates:
    min_valid_fraction: float = 0.85
    max_clamp_rate: float = 0.25
    min_fps_median: float = 15.0


@dataclass
class VisualSearchGates:
    # You will adjust these later; keep strict now
    require_high_confidence: bool = True
    min_trial_duration_s: float = 10.0
    min_gaze_presence_pct: float = 50.0
    min_blinks: int = 3


@dataclass
class Grid9Gates:
    # Allow partial aggregation as long as enough points are valid
    min_valid_points: int = 6


DEFAULT_GATES = Gates()
VISUAL_GATES = VisualSearchGates()
GRID9_GATES = Grid9Gates()


# -----------------------------
# Helpers
# -----------------------------

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def col_exists(df: pd.DataFrame, col: str) -> bool:
    return df is not None and (col in df.columns)


def apply_generic_gates(df: pd.DataFrame, gates: Gates) -> pd.DataFrame:
    """Generic gates used across tasks (when columns exist)."""
    if df is None or df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    # prefer explicit 'valid' column when present
    if "valid" in df.columns:
        mask &= (df["valid"].astype(int) == 1)

    if "valid_fraction" in df.columns:
        mask &= (df["valid_fraction"].astype(float) >= gates.min_valid_fraction)

    if "clamp_rate" in df.columns:
        mask &= (df["clamp_rate"].astype(float) <= gates.max_clamp_rate)

    if "fps_median" in df.columns:
        mask &= (df["fps_median"].astype(float) >= gates.min_fps_median)

    return df.loc[mask].copy()


def numeric_agg(df: pd.DataFrame, cols: List[str], prefix: str) -> Dict[str, float]:
    """Mean/Std aggregation for numeric columns that exist."""
    out: Dict[str, float] = {}
    if df is None or df.empty:
        for c in cols:
            out[f"{prefix}_{c}_mean"] = np.nan
            out[f"{prefix}_{c}_std"] = np.nan
        return out

    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out[f"{prefix}_{c}_mean"] = float(np.nanmean(s)) if s.notna().any() else np.nan
            out[f"{prefix}_{c}_std"] = float(np.nanstd(s)) if s.notna().any() else np.nan
        else:
            out[f"{prefix}_{c}_mean"] = np.nan
            out[f"{prefix}_{c}_std"] = np.nan
    return out


def basic_counts(df_all: pd.DataFrame, df_valid: pd.DataFrame, prefix: str) -> Dict[str, float]:
    n_all = int(len(df_all)) if df_all is not None else 0
    n_valid = int(len(df_valid)) if df_valid is not None else 0
    valid_rate = (n_valid / n_all) if n_all > 0 else np.nan
    return {
        f"{prefix}_n_trials": n_all,
        f"{prefix}_n_valid": n_valid,
        f"{prefix}_valid_rate": float(valid_rate) if valid_rate == valid_rate else np.nan,
    }


# -----------------------------
# Task-specific aggregation
# -----------------------------

def aggregate_fixation(df_all: pd.DataFrame) -> Dict[str, float]:
    df_valid = apply_generic_gates(df_all, DEFAULT_GATES)
    out = {}
    out.update(basic_counts(df_all, df_valid, "fixation"))

    cols = [
        "fixation_stability_rms_px",
        "fixation_bcea_px2",
        "microsaccade_rate_per_min",
        "drift_velocity_px_s",
        "percent_time_on_target",
        # blink_rate_per_min exists in fixation summary but may be 0; keep it
        "blink_rate_per_min",
        "quality_score",
        "calibration_quality",
        "fps_median",
        "clamp_rate",
    ]
    out.update(numeric_agg(df_valid, cols, "fixation"))
    return out


def aggregate_pursuit(df_all: pd.DataFrame) -> Dict[str, float]:
    df_valid = apply_generic_gates(df_all, DEFAULT_GATES)
    out = {}
    out.update(basic_counts(df_all, df_valid, "pursuit"))

    cols = [
        "pursuit_gain",
        "pursuit_latency_ms",
        "catch_up_saccade_count",
        "catch_up_saccade_rate_per_s",
        "position_error_mean_px",
        "position_error_rmse_px",
        "phase_lag_ms",
        "quality_score",
        "calibration_quality",
        "fps_median",
        "clamp_rate",
    ]
    out.update(numeric_agg(df_valid, cols, "pursuit"))

    # Direction balance (optional but useful)
    if df_valid is not None and not df_valid.empty and "direction" in df_valid.columns:
        left = (df_valid["direction"] == "left").sum()
        right = (df_valid["direction"] == "right").sum()
        total = len(df_valid)
        out["pursuit_left_frac"] = float(left / total) if total else np.nan
        out["pursuit_right_frac"] = float(right / total) if total else np.nan
    else:
        out["pursuit_left_frac"] = np.nan
        out["pursuit_right_frac"] = np.nan

    return out


def aggregate_saccade(df_all: pd.DataFrame) -> Dict[str, float]:
    # Keep all for invalid-reason rates, but aggregate valid only for metrics
    df_valid = apply_generic_gates(df_all, DEFAULT_GATES)
    out = {}
    out.update(basic_counts(df_all, df_valid, "saccade"))

    cols = [
        "saccade_latency_ms",
        "saccade_duration_ms",
        "peak_velocity_px_s",
        "saccade_amplitude_px",
        "gain",
        "landing_error_px",
        "overshoot_px",
        "undershoot_px",
        "corrective_saccade_count",
        "quality_score",
        "calibration_quality",
        "fps_median",
        "clamp_rate",
    ]
    out.update(numeric_agg(df_valid, cols, "saccade"))

    # Invalid reason rates (ISEF-friendly)
    if df_all is not None and not df_all.empty and "invalid_reason" in df_all.columns and "valid" in df_all.columns:
        invalid = df_all[df_all["valid"].astype(int) == 0]
        denom = len(df_all)
        for reason in ["wrong_direction", "no_meaningful_movement", "clamp_rate_high"]:
            out[f"saccade_rate_{reason}"] = float((invalid["invalid_reason"] == reason).sum() / denom) if denom else np.nan
    else:
        out["saccade_rate_wrong_direction"] = np.nan
        out["saccade_rate_no_meaningful_movement"] = np.nan
        out["saccade_rate_clamp_rate_high"] = np.nan

    return out


def aggregate_antisaccade(df_all: pd.DataFrame) -> Dict[str, float]:
    df_valid = apply_generic_gates(df_all, DEFAULT_GATES)
    out = {}
    out.update(basic_counts(df_all, df_valid, "antisaccade"))

    cols = [
        "antisaccade_latency_ms",
        "direction_error",
        "correction_time_ms",
        "inhibition_success",
        "peak_velocity_px_s",
        "amplitude_px",
        "landing_error_px",
        "quality_score",
        "calibration_quality",
        "fps_median",
        "clamp_rate",
    ]
    out.update(numeric_agg(df_valid, cols, "antisaccade"))

    # Rates of key failure modes (very valuable biomarkers)
    if df_all is not None and not df_all.empty and "invalid_reason" in df_all.columns and "valid" in df_all.columns:
        denom = len(df_all)
        invalid = df_all[df_all["valid"].astype(int) == 0]
        out["antisaccade_anticipatory_rate"] = float((invalid["invalid_reason"] == "anticipatory_response").sum() / denom) if denom else np.nan
        out["antisaccade_clamp_high_rate"] = float((invalid["invalid_reason"] == "clamp_rate_high").sum() / denom) if denom else np.nan
    else:
        out["antisaccade_anticipatory_rate"] = np.nan
        out["antisaccade_clamp_high_rate"] = np.nan

    # Inhibition success rate among valid trials
    if df_valid is not None and not df_valid.empty and "inhibition_success" in df_valid.columns:
        out["antisaccade_inhibition_success_rate"] = float(np.nanmean(pd.to_numeric(df_valid["inhibition_success"], errors="coerce")))
    else:
        out["antisaccade_inhibition_success_rate"] = np.nan

    return out


def aggregate_grid9(df_all: pd.DataFrame) -> Dict[str, float]:
    # generic gates filter valid==1 and qc-ish columns if present
    df_valid = apply_generic_gates(df_all, DEFAULT_GATES)

    out = {}
    out.update(basic_counts(df_all, df_valid, "grid9"))

    # partial aggregation rule
    valid_points = len(df_valid) if df_valid is not None else 0
    out["grid9_valid_point_count"] = float(valid_points)

    # If not enough valid points, blank out grid biomarker aggregates but keep counts.
    if valid_points < GRID9_GATES.min_valid_points:
        for c in ["mean_gaze_error_px", "rmse_gaze_error_px", "dwell_stability_rms_px", "quality_score", "calibration_quality", "fps_median", "clamp_rate"]:
            out[f"grid9_{c}_mean"] = np.nan
            out[f"grid9_{c}_std"] = np.nan
        out["grid9_pass_min_points"] = 0.0
        return out

    out["grid9_pass_min_points"] = 1.0

    cols = [
        "mean_gaze_error_px",
        "rmse_gaze_error_px",
        "dwell_stability_rms_px",
        "quality_score",
        "calibration_quality",
        "fps_median",
        "clamp_rate",
    ]
    out.update(numeric_agg(df_valid, cols, "grid9"))
    return out


def aggregate_visual_search(df_all: pd.DataFrame) -> Dict[str, float]:
    # Visual search has its own QC fields; we gate more intentionally.
    if df_all is None or df_all.empty:
        return {
            "visual_search_n_trials": 0,
            "visual_search_n_valid": 0,
            "visual_search_valid_rate": np.nan,
        }

    df = df_all.copy()

    # Start with explicit valid==1
    if "valid" in df.columns:
        df = df[df["valid"].astype(int) == 1]

    # Require high confidence blink-rate if configured
    if VISUAL_GATES.require_high_confidence and "blink_rate_confidence" in df.columns:
        df = df[df["blink_rate_confidence"].astype(str).str.upper() == "HIGH"]

    # Ensure duration sufficient
    if "search_duration_s" in df.columns:
        df = df[pd.to_numeric(df["search_duration_s"], errors="coerce") >= VISUAL_GATES.min_trial_duration_s]

    # Gaze presence minimum
    if "gaze_presence_pct" in df.columns:
        df = df[pd.to_numeric(df["gaze_presence_pct"], errors="coerce") >= VISUAL_GATES.min_gaze_presence_pct]

    # Blink count minimum
    if "blink_count" in df.columns:
        df = df[pd.to_numeric(df["blink_count"], errors="coerce") >= VISUAL_GATES.min_blinks]

    # Also apply generic gates when possible (valid_fraction/clamp_rate/fps)
    df_valid = apply_generic_gates(df, DEFAULT_GATES)

    out = {}
    out.update(basic_counts(df_all, df_valid, "visual_search"))

    cols = [
        "blink_count",
        "blink_rate_per_min",
        "blink_duration_proxy_mean_ms",
        "blink_duration_proxy_std_ms",
        "blink_duration_valid_count",
        "interblink_interval_mean_s",
        "interblink_interval_std_s",
        "interblink_interval_cv",
        "blink_burstiness",
        "gaze_presence_pct",
        "search_duration_s",
        "search_time_ms",
        "quality_score",
        "calibration_quality",
        "fps_median",
        "clamp_rate",
    ]
    out.update(numeric_agg(df_valid, cols, "visual_search"))

    # Additional ISEF-friendly: proportion of WARN qc_status in original (even if filtered later)
    if "qc_status" in df_all.columns:
        denom = len(df_all)
        out["visual_search_warn_rate_raw"] = float((df_all["qc_status"].astype(str).str.upper() == "WARN").sum() / denom) if denom else np.nan
        out["visual_search_fail_rate_raw"] = float((df_all["qc_status"].astype(str).str.upper() == "FAIL").sum() / denom) if denom else np.nan
    else:
        out["visual_search_warn_rate_raw"] = np.nan
        out["visual_search_fail_rate_raw"] = np.nan

    return out


# -----------------------------
# Main session walker
# -----------------------------

def load_summary_csv(session_dir: str, task_name: str) -> Optional[pd.DataFrame]:
    """
    Tries to load a summary CSV for a task:
    - summary_<task>.csv
    - summary_<task_name>.csv
    Returns None if not found.
    """
    patterns = [
        os.path.join(session_dir, f"summary_{task_name}.csv"),
        os.path.join(session_dir, f"summary_{task_name.lower()}.csv"),
    ]
    for p in patterns:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                return None
    # fallback: any summary file that contains task_name
    alt = glob.glob(os.path.join(session_dir, "summary_*.csv"))
    for p in alt:
        if task_name.lower() in os.path.basename(p).lower():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
    return None


def build_session_row(session_dir: str) -> Dict[str, float]:
    session_id = os.path.basename(os.path.normpath(session_dir))

    row: Dict[str, float] = {
        "session_id": session_id,
    }

    # Load each task summary
    df_fix = load_summary_csv(session_dir, "fixation")
    df_pur = load_summary_csv(session_dir, "pursuit")
    df_sac = load_summary_csv(session_dir, "saccade")
    df_as  = load_summary_csv(session_dir, "antisaccade")
    df_g9  = load_summary_csv(session_dir, "grid9")
    df_vs  = load_summary_csv(session_dir, "visual_search")

    # Aggregate per task
    if df_fix is not None: row.update(aggregate_fixation(df_fix))
    else: row.update({"fixation_n_trials": 0, "fixation_n_valid": 0, "fixation_valid_rate": np.nan})

    if df_pur is not None: row.update(aggregate_pursuit(df_pur))
    else: row.update({"pursuit_n_trials": 0, "pursuit_n_valid": 0, "pursuit_valid_rate": np.nan})

    if df_sac is not None: row.update(aggregate_saccade(df_sac))
    else: row.update({"saccade_n_trials": 0, "saccade_n_valid": 0, "saccade_valid_rate": np.nan})

    if df_as is not None: row.update(aggregate_antisaccade(df_as))
    else: row.update({"antisaccade_n_trials": 0, "antisaccade_n_valid": 0, "antisaccade_valid_rate": np.nan})

    if df_g9 is not None: row.update(aggregate_grid9(df_g9))
    else:
        row.update({
            "grid9_n_trials": 0,
            "grid9_n_valid": 0,
            "grid9_valid_rate": np.nan,
            "grid9_valid_point_count": 0.0,
            "grid9_pass_min_points": 0.0,
        })

    if df_vs is not None: row.update(aggregate_visual_search(df_vs))
    else: row.update({"visual_search_n_trials": 0, "visual_search_n_valid": 0, "visual_search_valid_rate": np.nan})

    # Session-level overall quality (simple, interpretable)
    # Use average of per-task mean quality_score where available
    qs = []
    for k in [
        "fixation_quality_score_mean",
        "pursuit_quality_score_mean",
        "saccade_quality_score_mean",
        "antisaccade_quality_score_mean",
        "grid9_quality_score_mean",
        "visual_search_quality_score_mean",
    ]:
        if k in row and row[k] == row[k]:
            qs.append(row[k])
    row["session_quality_score_mean"] = float(np.mean(qs)) if qs else np.nan

    # Overall valid coverage (how many tasks produced at least 1 valid trial)
    valid_tasks = 0
    for k in [
        "fixation_n_valid", "pursuit_n_valid", "saccade_n_valid",
        "antisaccade_n_valid", "grid9_n_valid", "visual_search_n_valid"
    ]:
        if k in row and row[k] and row[k] > 0:
            valid_tasks += 1
    row["session_valid_task_count"] = float(valid_tasks)
    row["session_total_task_count"] = 6.0

    return row


def main():
    sessions_root = os.path.join("data", "sessions")
    out_dir = os.path.join("data", "neurosignatures")
    os.makedirs(out_dir, exist_ok=True)

    session_dirs = sorted([p for p in glob.glob(os.path.join(sessions_root, "*")) if os.path.isdir(p)])

    if not session_dirs:
        raise SystemExit(f"No session folders found under {sessions_root}/")

    rows = []
    for sd in session_dirs:
        rows.append(build_session_row(sd))

    df_out = pd.DataFrame(rows)

    # Stable column ordering: session_id first
    cols = ["session_id"] + [c for c in df_out.columns if c != "session_id"]
    df_out = df_out[cols]

    out_path = os.path.join(out_dir, "neurosignatures.csv")
    df_out.to_csv(out_path, index=False)

    print(f"✅ Wrote NeuroSignatures: {out_path}")
    print(f"Sessions processed: {len(df_out)}")
    print("Preview:")
    print(df_out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
