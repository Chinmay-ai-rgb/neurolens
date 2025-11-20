import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ipd_calibration import IPDCalibration
from tasks.fixation import FixationTask
from tasks.saccade import SaccadeTask
from tasks.smooth_pursuit import SmoothPursuitTask
from tasks.anti_saccade import AntiSaccadeTask
from extraction.extract_features import extract_all_features, combine_session_features
from models.infer import InferenceEngine
from export.pdf_report import generate_pdf_report
from export.json_export import export_json
from interface.ui_helpers import (
    plot_fixation_scatter, plot_saccade_velocity, plot_pursuit_overlay,
    plot_anti_saccade_performance, plot_risk_scores, format_risk_indicator
)


st.set_page_config(
    page_title="NeuroLens+",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroLens+ - Neurological Biomarker Assessment")

if 'ipd_value' not in st.session_state:
    st.session_state.ipd_value = None
if 'latest_features' not in st.session_state:
    st.session_state.latest_features = None
if 'latest_prediction' not in st.session_state:
    st.session_state.latest_prediction = None
if 'session_files' not in st.session_state:
    st.session_state.session_files = []


sidebar = st.sidebar
sidebar.header("Navigation")

if sidebar.button("🔬 Run IPD Calibration"):
    with st.spinner("Running IPD calibration..."):
        try:
            calibrator = IPDCalibration()
            ipd = calibrator.run_calibration(duration=5.0)
            st.session_state.ipd_value = ipd
            st.success(f"IPD calibrated: {ipd:.2f} pixels")
        except Exception as e:
            st.error(f"Calibration failed: {str(e)}")

if sidebar.button("🔵 Run Fixation Test"):
    if st.session_state.ipd_value is None:
        st.warning("Please run IPD calibration first")
    else:
        with st.spinner("Running fixation task..."):
            try:
                task = FixationTask()
                output_file = task.run_task(st.session_state.ipd_value, fixation_duration=10.0)
                st.session_state.session_files.append(output_file)
                st.success(f"Fixation test completed: {output_file}")
            except Exception as e:
                st.error(f"Fixation test failed: {str(e)}")

if sidebar.button("🟠 Run Saccade Test"):
    if st.session_state.ipd_value is None:
        st.warning("Please run IPD calibration first")
    else:
        with st.spinner("Running saccade task..."):
            try:
                task = SaccadeTask()
                output_file = task.run_task(st.session_state.ipd_value, num_trials=10)
                st.session_state.session_files.append(output_file)
                st.success(f"Saccade test completed: {output_file}")
            except Exception as e:
                st.error(f"Saccade test failed: {str(e)}")

if sidebar.button("🟢 Run Smooth Pursuit Test"):
    if st.session_state.ipd_value is None:
        st.warning("Please run IPD calibration first")
    else:
        with st.spinner("Running smooth pursuit task..."):
            try:
                task = SmoothPursuitTask()
                output_file = task.run_task(st.session_state.ipd_value, duration=10.0)
                st.session_state.session_files.append(output_file)
                st.success(f"Smooth pursuit test completed: {output_file}")
            except Exception as e:
                st.error(f"Smooth pursuit test failed: {str(e)}")

if sidebar.button("🔴 Run Anti-Saccade Test"):
    if st.session_state.ipd_value is None:
        st.warning("Please run IPD calibration first")
    else:
        with st.spinner("Running anti-saccade task..."):
            try:
                task = AntiSaccadeTask()
                output_file = task.run_task(st.session_state.ipd_value, num_trials=15)
                st.session_state.session_files.append(output_file)
                st.success(f"Anti-saccade test completed: {output_file}")
            except Exception as e:
                st.error(f"Anti-saccade test failed: {str(e)}")

sidebar.divider()

if sidebar.button("⚙️ Process Latest Features"):
    if not st.session_state.session_files:
        st.warning("No task files to process")
    else:
        with st.spinner("Processing features..."):
            try:
                combined_features = combine_session_features(st.session_state.session_files)
                st.session_state.latest_features = combined_features
                st.success("Features processed successfully")
            except Exception as e:
                st.error(f"Feature extraction failed: {str(e)}")

if sidebar.button("🤖 Run AI Prediction"):
    if st.session_state.latest_features is None:
        st.warning("Please process features first")
    else:
        with st.spinner("Running inference..."):
            try:
                engine = InferenceEngine()
                prediction = engine.predict(st.session_state.latest_features)
                st.session_state.latest_prediction = prediction
                st.success("Prediction completed")
            except Exception as e:
                st.warning(f"Model inference not available (using dummy data): {str(e)}")
                st.session_state.latest_prediction = {
                    "MS_risk": 0.25,
                    "PD_risk": 0.15,
                    "PSP_risk": 0.10,
                    "CN6_risk": 0.05,
                    "confidence": 0.75,
                    "top_features": ["MS", "PD"]
                }

sidebar.divider()

if sidebar.button("📄 Export PDF Report"):
    if st.session_state.latest_features is None or st.session_state.latest_prediction is None:
        st.warning("Please run prediction first")
    else:
        try:
            pdf_path = generate_pdf_report(
                st.session_state.latest_features,
                st.session_state.latest_prediction
            )
            st.success(f"PDF report generated: {pdf_path}")
        except Exception as e:
            st.error(f"PDF export failed: {str(e)}")

if sidebar.button("📊 Export JSON"):
    if st.session_state.latest_features is None:
        st.warning("Please process features first")
    else:
        try:
            json_path = export_json(
                st.session_state.latest_features,
                st.session_state.latest_prediction
            )
            st.success(f"JSON exported: {json_path}")
        except Exception as e:
            st.error(f"JSON export failed: {str(e)}")


main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    st.header("Instructions")
    st.markdown("""
    1. **Start with IPD Calibration** - Measures your inter-pupillary distance for accurate tracking
    2. **Run Eye Movement Tests** - Complete fixation, saccade, smooth pursuit, and anti-saccade tasks
    3. **Process Features** - Extract neurological biomarkers from recorded data
    4. **Run AI Prediction** - Get risk assessment using machine learning model
    5. **Export Results** - Generate PDF report or JSON export
    
    Ensure good lighting and keep your face centered in the camera frame during tests.
    """)

with main_col2:
    if st.session_state.ipd_value:
        st.metric("IPD Value", f"{st.session_state.ipd_value:.2f} px")
    else:
        st.info("IPD not calibrated")

st.divider()

if st.session_state.latest_features:
    st.header("Feature Metrics")
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    with feat_col1:
        st.subheader("Fixation")
        if not np.isnan(st.session_state.latest_features.get('fix_std_x', np.nan)):
            st.metric("Stability X", f"{st.session_state.latest_features['fix_std_x']:.3f}")
            st.metric("Stability Y", f"{st.session_state.latest_features['fix_std_y']:.3f}")
            st.metric("Drift", f"{st.session_state.latest_features['fix_drift']:.3f}")
            st.metric("Microsaccades", int(st.session_state.latest_features['fix_microsaccades']))
    
    with feat_col2:
        st.subheader("Saccade")
        if not np.isnan(st.session_state.latest_features.get('sac_latency', np.nan)):
            st.metric("Latency (ms)", f"{st.session_state.latest_features['sac_latency']*1000:.1f}")
            st.metric("Peak Velocity", f"{st.session_state.latest_features['sac_peak_vel']:.1f}")
            st.metric("Amplitude Error", f"{st.session_state.latest_features['sac_amp_error']:.2f}")
    
    with feat_col3:
        st.subheader("Smooth Pursuit")
        if not np.isnan(st.session_state.latest_features.get('pursuit_gain', np.nan)):
            st.metric("Gain", f"{st.session_state.latest_features['pursuit_gain']:.3f}")
            st.metric("Phase Lag (ms)", f"{st.session_state.latest_features['pursuit_phase_lag']*1000:.1f}")
            st.metric("Catch-up Saccades", int(st.session_state.latest_features['pursuit_catchups']))
    
    with feat_col4:
        st.subheader("Anti-Saccade")
        if not np.isnan(st.session_state.latest_features.get('anti_error_rate', np.nan)):
            st.metric("Error Rate", f"{st.session_state.latest_features['anti_error_rate']:.2%}")
            st.metric("Correction Latency (ms)", f"{st.session_state.latest_features['anti_corr_latency']*1000:.1f}")

st.divider()

if st.session_state.latest_prediction:
    st.header("AI Risk Assessment")
    
    pred = st.session_state.latest_prediction
    
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    
    with risk_col1:
        st.metric("MS Risk", f"{pred['MS_risk']:.2%}", delta=format_risk_indicator(pred['MS_risk']))
    with risk_col2:
        st.metric("PD Risk", f"{pred['PD_risk']:.2%}", delta=format_risk_indicator(pred['PD_risk']))
    with risk_col3:
        st.metric("PSP Risk", f"{pred['PSP_risk']:.2%}", delta=format_risk_indicator(pred['PSP_risk']))
    with risk_col4:
        st.metric("CN6 Risk", f"{pred['CN6_risk']:.2%}", delta=format_risk_indicator(pred['CN6_risk']))
    
    st.metric("Confidence", f"{pred['confidence']:.2%}")
    
    if pred['top_features']:
        st.info(f"Top risk indicators: {', '.join(pred['top_features'])}")
    
    risk_fig = plot_risk_scores({
        "MS": pred['MS_risk'],
        "PD": pred['PD_risk'],
        "PSP": pred['PSP_risk'],
        "CN6": pred['CN6_risk']
    })
    st.pyplot(risk_fig)

st.divider()

if st.session_state.session_files:
    st.header("Visualizations")
    
    viz_files = st.session_state.session_files[-4:]
    
    for file_path in viz_files:
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)
                
                if 'fixation' in file_path.lower():
                    fig = plot_fixation_scatter(df)
                    st.pyplot(fig)
                elif 'saccade' in file_path.lower():
                    fig = plot_saccade_velocity(df)
                    st.pyplot(fig)
                elif 'smooth' in file_path.lower() or 'pursuit' in file_path.lower():
                    fig = plot_pursuit_overlay(df)
                    st.pyplot(fig)
                elif 'anti' in file_path.lower():
                    if st.session_state.latest_features and not np.isnan(st.session_state.latest_features.get('anti_error_rate', np.nan)):
                        fig = plot_anti_saccade_performance(st.session_state.latest_features['anti_error_rate'])
                        st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not visualize {file_path}: {str(e)}")

