# interface/app.py
# NeuroLens+ — Session Runner + Clean, non-technical Results screen
# Run:
#   streamlit run interface/app.py

from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Ensure project root is importable (Streamlit-safe)
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../neurolens
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from interface.run_pipeline import run_analysis_pipeline  # noqa: E402


# -------------------------
# Paths (absolute, project-root relative)
# -------------------------
SCORES_CSV = ROOT_DIR / "data" / "results" / "session_scores.csv"
PROFILES_CSV = ROOT_DIR / "data" / "results" / "profile_scores.csv"
TOP_BIOMARKERS_JSON = ROOT_DIR / "data" / "results" / "profile_top_biomarkers.json"
ML_UNSCALED_CSV = ROOT_DIR / "data" / "ml" / "ml_matrix_unscaled.csv"


# -------------------------
# Friendly labeling
# -------------------------
PROFILE_FRIENDLY = {
    "profile__inhibitory_control__executive_function": "Focus control & impulse control",
    "profile__oculomotor_speed_and_timing": "Eye-movement speed & timing",
    "profile__smooth_pursuit__sensorimotor_integration": "Tracking moving things smoothly",
    "profile__fixation_stability__gaze_noise": "Steadiness (how still your gaze is)",
    "profile__spatial_gaze_accuracy__and_coordination": "Accuracy looking at targets",
    "profile__blink_regulation__attentional_arousal": "Blink patterns & alertness",
}

PROFILE_ONE_LINERS = {
    "Focus control & impulse control": "How well your eyes resisted “auto-reacting” and stayed controlled.",
    "Eye-movement speed & timing": "How quickly and consistently your eyes moved when they needed to.",
    "Tracking moving things smoothly": "How smoothly your eyes followed a moving target.",
    "Steadiness (how still your gaze is)": "How steady your eyes were when you tried to hold still.",
    "Accuracy looking at targets": "How accurately your eyes landed on targets around the screen.",
    "Blink patterns & alertness": "Changes in blink rhythm that often shift with tiredness/attention.",
}

FEATURE_FRIENDLY = {
    # Pursuit
    "pursuit_pursuit_gain_mean": "Smooth tracking strength",
    "pursuit_pursuit_latency_ms_mean": "Delay before tracking starts",
    "pursuit_position_error_mean_px_mean": "How far off the moving target you were (average)",
    "pursuit_position_error_rmse_px_mean": "Tracking error (overall)",

    # Saccade
    "saccade_saccade_latency_ms_mean": "Delay before quick eye-jumps",
    "saccade_peak_velocity_px_s_mean": "Fastest speed during eye-jumps",
    "saccade_gain_mean": "How accurately eye-jumps hit the target",
    "saccade_landing_error_px_mean": "How far off target your eye-jumps landed",

    # Antisaccade
    "antisaccade_antisaccade_latency_ms_mean": "Delay before controlled eye-response",
    "antisaccade_direction_error_mean": "Wrong-direction mistakes (average)",
    "antisaccade_inhibition_success_mean": "Success rate resisting the wrong move (average)",
    "antisaccade_anticipatory_rate": "Too-fast reactions (jumping early)",
    "antisaccade_inhibition_success_rate": "Resisting wrong move (rate)",

    # Fixation
    "fixation_fixation_stability_rms_px_mean": "Steadiness while staring (wobble)",
    "fixation_percent_time_on_target_mean": "Time stayed on the target",
    "fixation_blink_rate_per_min_mean": "Blink rate while staring",

    # Grid9
    "grid9_mean_gaze_error_px_mean": "Average accuracy in the 9-point grid",
    "grid9_rmse_gaze_error_px_mean": "Overall grid accuracy",

    # Visual search / blink
    "visual_search_blink_rate_per_min_mean": "Blink rate during the task",
    "visual_search_interblink_interval_mean_s_mean": "Time between blinks (average)",
    "visual_search_gaze_presence_pct_mean": "How consistently your eyes stayed tracked",
}


def pretty_feature_name(col: str) -> str:
    col = str(col)
    if col in FEATURE_FRIENDLY:
        return FEATURE_FRIENDLY[col]
    return col.replace("_", " ").strip().capitalize()


def deviation_label(score_0_1: float) -> str:
    if score_0_1 >= 0.75:
        return "High"
    if score_0_1 >= 0.55:
        return "Moderate"
    return "Low"


def confidence_label(valid_tasks: Optional[float], qc_ok: bool = True) -> Tuple[str, str]:
    if valid_tasks is None:
        return "Unknown", "Missing data-quality info"
    try:
        vt = float(valid_tasks)
    except Exception:
        return "Unknown", "Missing data-quality info"

    if vt >= 6 and qc_ok:
        return "High", f"Valid tasks: {int(vt)}"
    if vt >= 4:
        return "Medium", f"Valid tasks: {int(vt)}"
    return "Low", f"Valid tasks: {int(vt)}"


# -------------------------
# Robust loader helpers
# -------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_session_ids(scores_df: pd.DataFrame) -> List[str]:
    if scores_df.empty or "session_id" not in scores_df.columns:
        return []
    return scores_df["session_id"].astype(str).tolist()


def get_top_profiles(profiles_df: pd.DataFrame, session_id: str) -> List[Tuple[str, float]]:
    if profiles_df.empty or "session_id" not in profiles_df.columns:
        return []

    row = profiles_df[profiles_df["session_id"].astype(str) == str(session_id)]
    if row.empty:
        return []

    prof_cols = [c for c in profiles_df.columns if c.startswith("profile__")]
    if not prof_cols:
        return []

    vals = row[prof_cols].iloc[0].to_dict()
    items: List[Tuple[str, float]] = []

    for k, v in vals.items():
        try:
            fv = float(v)
        except Exception:
            continue
        friendly = PROFILE_FRIENDLY.get(k, k.replace("profile__", "").replace("_", " ").strip().title())
        items.append((friendly, fv))

    items.sort(key=lambda x: x[1], reverse=True)
    top = [(n, s) for (n, s) in items if np.isfinite(s)]
    return top[:2]


def get_top_signals(
    top_biomarkers_json: dict,
    ml_unscaled: pd.DataFrame,
    scores_df: pd.DataFrame,
    session_id: str,
) -> List[str]:
    def coerce_to_feature_list(obj) -> List[str]:
        feats: List[str] = []
        if obj is None:
            return feats

        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, str):
                    feats.append(it)
                elif isinstance(it, dict):
                    for key in ("feature", "biomarker", "name"):
                        if key in it:
                            feats.append(str(it[key]))
                            break
            return feats

        if isinstance(obj, dict):
            for key in ("top_biomarkers", "top_signals", "signals", "biomarkers", "features"):
                if key in obj:
                    return coerce_to_feature_list(obj[key])

            if "profiles" in obj and isinstance(obj["profiles"], dict):
                for _, v in obj["profiles"].items():
                    feats.extend(coerce_to_feature_list(v))
                return feats

            for _, v in obj.items():
                feats.extend(coerce_to_feature_list(v))
            return feats

        return feats

    # 1) JSON path
    if isinstance(top_biomarkers_json, dict) and str(session_id) in top_biomarkers_json:
        raw = top_biomarkers_json.get(str(session_id))
        feats = coerce_to_feature_list(raw)

        seen = set()
        cleaned = []
        for f in feats:
            if f and f not in seen:
                cleaned.append(f)
                seen.add(f)

        if cleaned:
            return [pretty_feature_name(f) for f in cleaned[:3]]

    # 2) fallback: baseline-median difference
    if ml_unscaled.empty or "session_id" not in ml_unscaled.columns:
        return []

    row = ml_unscaled[ml_unscaled["session_id"].astype(str) == str(session_id)]
    if row.empty:
        return []

    if scores_df.empty or "is_baseline" not in scores_df.columns:
        return []

    baseline_ids = scores_df.loc[scores_df["is_baseline"] == 1, "session_id"].astype(str).tolist()
    base = ml_unscaled[ml_unscaled["session_id"].astype(str).isin(baseline_ids)]
    if base.empty:
        return []

    feat_cols = [
        c for c in ml_unscaled.columns
        if c != "session_id" and pd.api.types.is_numeric_dtype(ml_unscaled[c])
    ]
    if not feat_cols:
        return []

    base_median = base[feat_cols].median(numeric_only=True)
    base_iqr = (base[feat_cols].quantile(0.75) - base[feat_cols].quantile(0.25)).replace(0, np.nan)

    x = row[feat_cols].iloc[0].astype(float)
    robust_z = ((x - base_median) / base_iqr).abs().sort_values(ascending=False)

    top3 = robust_z.head(3).index.tolist()
    return [pretty_feature_name(c) for c in top3]


def build_report_text(
    session_id: str,
    status_text: str,
    deviation_text: str,
    deviation_score: float,
    top_profiles: List[Tuple[str, float]],
    top_signals: List[str],
    confidence_text: str,
    confidence_detail: str,
) -> str:
    lines = []
    lines.append("NeuroLens+ Summary Report")
    lines.append(f"Session: {session_id}")
    lines.append("")
    lines.append(f"Status: {status_text}")
    if np.isfinite(deviation_score):
        lines.append(f"Overall change vs your usual: {deviation_text} (score: {deviation_score:.3f})")
    else:
        lines.append(f"Overall change vs your usual: {deviation_text}")
    lines.append("")
    if top_profiles:
        lines.append("What changed the most (top areas):")
        for name, _ in top_profiles[:2]:
            one_liner = PROFILE_ONE_LINERS.get(name, "")
            if one_liner:
                lines.append(f"- {name}: {one_liner}")
            else:
                lines.append(f"- {name}")
        lines.append("")
    if top_signals:
        lines.append("Top signals we used (plain-English):")
        for i, s in enumerate(top_signals[:3], 1):
            lines.append(f"{i}. {s}")
        lines.append("")
    lines.append(f"Confidence / data quality: {confidence_text} ({confidence_detail})")
    lines.append("")
    lines.append("Note: This tool does not diagnose conditions. It flags changes from your own baseline.")
    return "\n".join(lines)


def run_pipeline_ui_safe() -> Tuple[bool, str]:
    """
    Runs the pipeline and returns (ok, log_text) no matter what your underlying pipeline returns.
    Supports these common shapes:
      - None
      - dict
      - (ok: bool, log: str)
      - (result_dict, log: str)
    """
    try:
        out = run_analysis_pipeline()
    except TypeError:
        return False, (
            "run_analysis_pipeline() signature mismatch. "
            "Update interface/run_pipeline.py to accept no args, or call it with required args here."
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if out is None:
        return True, "Pipeline ran."
    if isinstance(out, tuple) and len(out) == 2:
        a, b = out
        if isinstance(a, bool):
            return a, str(b)
        return True, str(b)
    if isinstance(out, dict):
        return True, "Pipeline ran."
    return True, "Pipeline ran."


def run_cmd(cmd: List[str]) -> Tuple[bool, str]:
    """Run a command in project root and capture stdout/stderr."""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        log = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return (p.returncode == 0), log.strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="NeuroLens+ Results", layout="wide")

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .soft-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px;
        background: rgba(255,255,255,0.03);
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.95rem;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        margin-right: 8px;
    }
    .big {
        font-size: 2.4rem;
        font-weight: 750;
        line-height: 1.05;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .muted { opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)

scores_df = load_csv(SCORES_CSV)
profiles_df = load_csv(PROFILES_CSV)
ml_unscaled = load_csv(ML_UNSCALED_CSV)
top_biomarkers_json = load_json(TOP_BIOMARKERS_JSON)

if "page" not in st.session_state:
    st.session_state.page = "Run Tasks"

# Sidebar navigation + actions
with st.sidebar:
    st.header("NeuroLens+")
    st.session_state.page = st.radio("Go to", ["Run Tasks", "Results"], index=0)

    st.divider()

    if st.session_state.page == "Run Tasks":
        st.header("Session Runner")
        st.caption("Step 1: Run the 6 tasks to create a new session.")
        if st.button("Run 6 Tasks (python main.py -t all)"):
            with st.spinner("Running 6 tasks…"):
                ok, log = run_cmd(["python", "main.py", "-t", "all"])
            if ok:
                st.success("Tasks finished. Now run analysis.")
                st.cache_data.clear()
                st.session_state["tasks_ran"] = True
            else:
                st.error("Task run failed.")
            if log:
                st.code(log)

        st.caption("Step 2: Run analysis (ML + Profiles).")
        if st.button("Analyze Session (ML + Profiles)"):
            with st.spinner("Running analysis pipeline…"):
                ok, log_text = run_pipeline_ui_safe()
            if ok:
                st.success("Analysis complete. Opening Results…")
                st.cache_data.clear()
                st.session_state["jump_to_results"] = True
                st.session_state.page = "Results"
                st.rerun()
            else:
                st.error("Pipeline failed.")
                st.code(log_text)

    else:
        st.header("Results")
        if scores_df.empty:
            st.warning("No results found yet. Run tasks + analysis first.")
        else:
            session_ids = get_session_ids(scores_df)
            # Default to newest session (last row order)
            st.session_state.selected_session = st.selectbox(
                "Choose a session",
                options=session_ids,
                index=max(0, len(session_ids) - 1),
            )

# Main page routing
if st.session_state.page == "Run Tasks":
    st.title("Run a New Session")
    st.caption("This is the in-app version of running: `python main.py -t all`")

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Step 1 — Run the 6 tasks")
    st.write("This will run all tasks and save a new session to your data folder.")
    st.code("python main.py -t all")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Step 2 — Analyze the session")
    st.write("This computes ML anomaly scoring + functional profile elevations.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Use the sidebar buttons to run tasks, then analyze, then switch to Results.")
    st.stop()

# -------------------------
# Results page
# -------------------------
st.title("Your Session Results")

if scores_df.empty:
    st.error(
        "I can’t find your results yet.\n\n"
        "Make sure these files exist:\n"
        "- data/results/session_scores.csv\n"
        "- data/results/profile_scores.csv\n"
        "- data/ml/ml_matrix_unscaled.csv\n"
        "- data/results/profile_top_biomarkers.json (optional)\n"
    )
    st.stop()

session_id = st.session_state.get("selected_session")
if not session_id:
    session_ids = get_session_ids(scores_df)
    session_id = session_ids[-1] if session_ids else None

if not session_id:
    st.warning("No sessions available yet. Run tasks + analysis first.")
    st.stop()

# Pull current row
row = scores_df[scores_df["session_id"].astype(str) == str(session_id)]
if row.empty:
    st.error("Session not found in session_scores.csv")
    st.stop()

r = row.iloc[0].to_dict()

is_baseline = int(r.get("is_baseline", 0)) == 1
status_text = "Baseline (your usual)" if is_baseline else "Fatigue / Non-baseline"

# Accept either "anomaly_score_0_1" or "anomaly_score"
score_col = "anomaly_score_0_1" if "anomaly_score_0_1" in scores_df.columns else "anomaly_score"
score_0_1 = float(r.get(score_col, np.nan))
deviation_text = deviation_label(score_0_1) if np.isfinite(score_0_1) else "Unknown"

# Confidence: use ml_unscaled session_valid_task_count if present
valid_tasks = None
if (
    not ml_unscaled.empty
    and "session_id" in ml_unscaled.columns
    and "session_valid_task_count" in ml_unscaled.columns
):
    vt_row = ml_unscaled[ml_unscaled["session_id"].astype(str) == str(session_id)]
    if not vt_row.empty:
        valid_tasks = vt_row["session_valid_task_count"].iloc[0]

conf_text, conf_detail = confidence_label(valid_tasks)

top_profiles = get_top_profiles(profiles_df, session_id)
top_signals = get_top_signals(top_biomarkers_json, ml_unscaled, scores_df, session_id)

# -------------------------
# Top summary row
# -------------------------
c1, c2, c3 = st.columns([1.25, 1.0, 0.9], gap="large")

with c1:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="muted">Session</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{session_id}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pill">{status_text}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="muted">Overall change vs your usual</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{deviation_text}</div>', unsafe_allow_html=True)

    baseline_scores = scores_df.loc[scores_df["is_baseline"] == 1, score_col].dropna().astype(float)

    if len(baseline_scores) >= 3 and np.isfinite(score_0_1):
        lo = float(baseline_scores.quantile(0.25))
        hi = float(baseline_scores.quantile(0.75))
        med = float(baseline_scores.median())

        st.caption("Where you land compared to your baseline (middle band = typical range)")
        st.progress(min(max(float(score_0_1), 0.0), 1.0))
        st.caption(f"Baseline typical range: {lo:.2f}–{hi:.2f} (median {med:.2f}) • Today: {score_0_1:.2f}")
    else:
        st.caption("Not enough baseline sessions to show your typical range yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="muted">Confidence / data quality</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{conf_text}</div>', unsafe_allow_html=True)
    st.caption(conf_detail)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# -------------------------
# “What changed” + “Top signals”
# -------------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("What changed the most")
    st.caption("These are the 1–2 areas that shifted most compared to your baseline.")

    if not top_profiles:
        st.info("No profile scores found for this session yet.")
    else:
        for name, _score in top_profiles[:2]:
            st.markdown(f"**• {name}**")
            one_liner = PROFILE_ONE_LINERS.get(name, "")
            if one_liner:
                st.caption(one_liner)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Top signals (the big reasons)")
    st.caption("These are the 3 strongest signals the system used for this result.")

    if not top_signals:
        st.info("Top signals not available yet (missing/empty profile_top_biomarkers.json).")
    else:
        for i, s in enumerate(top_signals[:3], 1):
            st.markdown(f"**{i}. {s}**")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# -------------------------
# Export report
# -------------------------
report_text = build_report_text(
    session_id=session_id,
    status_text=status_text,
    deviation_text=deviation_text,
    deviation_score=float(score_0_1) if np.isfinite(score_0_1) else float("nan"),
    top_profiles=top_profiles,
    top_signals=top_signals,
    confidence_text=conf_text,
    confidence_detail=conf_detail,
)

st.markdown('<div class="soft-card">', unsafe_allow_html=True)
st.subheader("Export")
st.caption("Download a simple report you can share or save.")

st.download_button(
    label="Export Report",
    data=report_text,
    file_name=f"neurolens_report_{session_id}.txt",
    mime="text/plain",
    use_container_width=True,
)
with st.expander("Preview report text"):
    st.text(report_text)
st.markdown("</div>", unsafe_allow_html=True)

st.caption("Reminder: NeuroLens+ flags changes from your personal baseline — it does not diagnose medical conditions.")
