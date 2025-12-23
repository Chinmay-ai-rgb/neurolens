#!/usr/bin/env python3
"""
NeuroLens+ Dataset Builder

Builds a unified ML-ready dataset from session data.
Combines per-task summary biomarkers with QC flags into a single CSV.

Usage:
    python tools/build_dataset.py --input data/sessions/ --output dataset.csv
    python tools/build_dataset.py --input data/sessions/ --output dataset.csv --min-quality 0.7
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


# Task names and their summary file patterns
TASKS = {
    'fixation': 'summary_fixation.csv',
    'pursuit': 'summary_pursuit.csv',
    'saccade': 'summary_saccade.csv',
    'antisaccade': 'summary_antisaccade.csv',
    'grid9': 'summary_grid9.csv',
    'visual_search': 'summary_visual_search.csv',
}

# Key biomarkers to extract per task (for session-level summary)
TASK_BIOMARKERS = {
    'fixation': [
        'fixation_stability_rms_px',
        'fixation_bcea_px2',
        'microsaccade_rate_per_min',
        'drift_velocity_px_s',
        'percent_time_on_target',
    ],
    'pursuit': [
        'pursuit_gain',
        'pursuit_latency_ms',
        'catch_up_saccade_count',
        'position_error_rmse_px',
        'phase_lag_ms',
    ],
    'saccade': [
        'saccade_latency_ms',
        'saccade_duration_ms',
        'peak_velocity_px_s',
        'saccade_amplitude_px',
        'gain',
        'landing_error_px',
    ],
    'antisaccade': [
        'antisaccade_latency_ms',
        'direction_error',
        'correction_time_ms',
        'inhibition_success',
    ],
    'grid9': [
        'mean_gaze_error_px',
        'rmse_gaze_error_px',
        'dwell_stability_rms_px',
    ],
    'visual_search': [
        'blink_rate_per_min',
        'blink_duration_proxy_mean_ms',  # Updated: proxy measurement
        'blink_duration_valid_count',
        'interblink_interval_mean_s',
        'interblink_interval_cv',
        'blink_burstiness',
        'gaze_presence_pct',
        'qc_status',
        'blink_rate_confidence',
    ],
}


def load_session_metadata(session_dir: Path) -> Optional[Dict[str, Any]]:
    """Load session metadata from meta.json."""
    meta_path = session_dir / 'meta.json'
    if not meta_path.exists():
        return None
    
    with open(meta_path, 'r') as f:
        return json.load(f)


def load_task_summary(session_dir: Path, task_name: str) -> Optional[pd.DataFrame]:
    """Load task summary CSV if it exists."""
    summary_file = TASKS.get(task_name)
    if not summary_file:
        return None
    
    summary_path = session_dir / summary_file
    if not summary_path.exists():
        return None
    
    try:
        return pd.read_csv(summary_path)
    except Exception as e:
        print(f"Warning: Could not load {summary_path}: {e}")
        return None


def compute_session_biomarkers(
    session_dir: Path,
    task_name: str,
    min_valid_fraction: float = 0.5
) -> Dict[str, Any]:
    """
    Compute session-level biomarkers for a task.
    
    Returns mean of valid trials for each biomarker.
    """
    df = load_task_summary(session_dir, task_name)
    if df is None or len(df) == 0:
        return {}
    
    # Filter to valid trials
    if 'valid' in df.columns:
        valid_df = df[df['valid'] == 1]
    else:
        valid_df = df
    
    # Check if enough valid trials
    valid_rate = len(valid_df) / len(df) if len(df) > 0 else 0
    if valid_rate < min_valid_fraction:
        return {'_valid_rate': valid_rate, '_status': 'insufficient_valid_trials'}
    
    # Compute mean biomarkers
    biomarkers = {'_valid_rate': valid_rate, '_n_trials': len(df), '_n_valid': len(valid_df)}
    
    for col in TASK_BIOMARKERS.get(task_name, []):
        if col in valid_df.columns:
            values = valid_df[col].dropna()
            if len(values) > 0:
                biomarkers[col] = values.mean()
                biomarkers[f'{col}_std'] = values.std() if len(values) > 1 else np.nan
    
    return biomarkers


def compute_derived_features(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived features that combine biomarkers across tasks.
    
    These features may be more diagnostically relevant than individual biomarkers.
    """
    derived = {}
    
    # Saccade main sequence validation (peak velocity vs amplitude)
    if 'saccade_peak_velocity_px_s' in session_data and 'saccade_saccade_amplitude_px' in session_data:
        pv = session_data['saccade_peak_velocity_px_s']
        amp = session_data['saccade_saccade_amplitude_px']
        if pv > 0 and amp > 0:
            # Main sequence: PV should scale with amplitude
            derived['main_sequence_ratio'] = pv / amp
    
    # Inhibitory control index (anti-saccade performance)
    if 'antisaccade_direction_error' in session_data:
        error_rate = session_data['antisaccade_direction_error']
        derived['inhibitory_control_index'] = 1.0 - error_rate
    
    # Pursuit quality index
    if 'pursuit_pursuit_gain' in session_data:
        gain = session_data['pursuit_pursuit_gain']
        derived['pursuit_quality'] = min(gain, 2.0 - gain) if gain <= 2.0 else 0
    
    # Fixation quality index
    if 'fixation_fixation_stability_rms_px' in session_data:
        rms = session_data['fixation_fixation_stability_rms_px']
        # Normalize: lower RMS = better quality
        derived['fixation_quality'] = max(0, 1.0 - rms / 300.0)
    
    # Blink regularity index
    if 'visual_search_interblink_interval_cv' in session_data:
        cv = session_data['visual_search_interblink_interval_cv']
        # Lower CV = more regular blinking
        derived['blink_regularity'] = max(0, 1.0 - cv) if cv < 2.0 else 0
    
    # Overall oculomotor health score (composite)
    scores = []
    if 'inhibitory_control_index' in derived:
        scores.append(derived['inhibitory_control_index'])
    if 'pursuit_quality' in derived:
        scores.append(derived['pursuit_quality'])
    if 'fixation_quality' in derived:
        scores.append(derived['fixation_quality'])
    
    if scores:
        derived['oculomotor_health_score'] = np.mean(scores)
    
    return derived


def process_session(
    session_dir: Path,
    min_quality: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    Process a single session and extract all biomarkers.
    
    Returns a dictionary with session-level features.
    """
    # Load metadata
    meta = load_session_metadata(session_dir)
    if meta is None:
        return None
    
    session_data = {
        'session_id': meta.get('session_id', session_dir.name),
        'timestamp': meta.get('timestamp', ''),
        'screen_width': meta.get('screen_width', 0),
        'screen_height': meta.get('screen_height', 0),
        'calibration_accepted': meta.get('calibration_accepted', False),
        'calibration_error_mean_px': meta.get('calibration_error_mean_px', np.nan),
        'fps_mean': meta.get('fps_mean', np.nan),
    }
    
    # Process each task
    tasks_completed = 0
    for task_name in TASKS.keys():
        biomarkers = compute_session_biomarkers(session_dir, task_name, min_quality)
        
        if biomarkers:
            tasks_completed += 1
            # Prefix biomarkers with task name
            for key, value in biomarkers.items():
                session_data[f'{task_name}_{key}'] = value
    
    session_data['tasks_completed'] = tasks_completed
    
    # Compute derived features
    derived = compute_derived_features(session_data)
    session_data.update(derived)
    
    return session_data


def build_dataset(
    input_dir: str,
    output_file: str,
    min_quality: float = 0.5,
    min_tasks: int = 1
) -> pd.DataFrame:
    """
    Build dataset from all sessions in input directory.
    
    Args:
        input_dir: Path to sessions directory
        output_file: Path to output CSV
        min_quality: Minimum valid trial fraction per task
        min_tasks: Minimum number of tasks required per session
    
    Returns:
        DataFrame with all session data
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Find all session directories
    session_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not session_dirs:
        print(f"Warning: No session directories found in {input_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(session_dirs)} session directories")
    
    # Process each session
    all_sessions = []
    for session_dir in session_dirs:
        session_data = process_session(session_dir, min_quality)
        
        if session_data is None:
            print(f"  Skipping {session_dir.name}: No metadata found")
            continue
        
        if session_data.get('tasks_completed', 0) < min_tasks:
            print(f"  Skipping {session_dir.name}: Only {session_data.get('tasks_completed', 0)} tasks completed")
            continue
        
        all_sessions.append(session_data)
        print(f"  Processed {session_dir.name}: {session_data.get('tasks_completed', 0)} tasks")
    
    if not all_sessions:
        print("Warning: No valid sessions found")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_sessions)
    
    # Reorder columns: metadata first, then tasks, then derived
    meta_cols = ['session_id', 'timestamp', 'screen_width', 'screen_height', 
                 'calibration_accepted', 'calibration_error_mean_px', 'fps_mean', 'tasks_completed']
    derived_cols = ['main_sequence_ratio', 'inhibitory_control_index', 'pursuit_quality',
                    'fixation_quality', 'blink_regularity', 'oculomotor_health_score']
    
    # Get all columns in order
    all_cols = [c for c in meta_cols if c in df.columns]
    task_cols = [c for c in df.columns if c not in meta_cols and c not in derived_cols]
    all_cols.extend(sorted(task_cols))
    all_cols.extend([c for c in derived_cols if c in df.columns])
    
    df = df[all_cols]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to {output_file}")
    print(f"  Sessions: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build ML-ready dataset from NeuroLens+ sessions"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/sessions',
        help='Input directory containing session folders'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='dataset.csv',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--min-quality', '-q',
        type=float,
        default=0.5,
        help='Minimum valid trial fraction per task (default: 0.5)'
    )
    
    parser.add_argument(
        '--min-tasks', '-t',
        type=int,
        default=1,
        help='Minimum number of tasks required per session (default: 1)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics of the dataset'
    )
    
    args = parser.parse_args()
    
    df = build_dataset(
        args.input,
        args.output,
        args.min_quality,
        args.min_tasks
    )
    
    if args.summary and len(df) > 0:
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(df.describe().to_string())


if __name__ == '__main__':
    main()
