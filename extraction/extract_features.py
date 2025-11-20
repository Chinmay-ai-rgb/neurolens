import pandas as pd
import numpy as np
from pathlib import Path
from extraction.fixation_features import extract_fixation_features
from extraction.saccade_features import extract_saccade_features
from extraction.smooth_features import extract_smooth_features
from extraction.anti_saccade_features import extract_anti_saccade_features


def detect_task_type(filename: str) -> str:
    filename_lower = filename.lower()
    
    if 'fixation' in filename_lower:
        return 'fixation'
    elif 'saccade' in filename_lower:
        return 'saccade'
    elif 'smooth' in filename_lower or 'pursuit' in filename_lower:
        return 'smooth_pursuit'
    elif 'anti' in filename_lower:
        return 'anti_saccade'
    else:
        raise ValueError(f"Unknown task type for file: {filename}")


def extract_all_features(csv_path: str, output_dir: str = "data/processed_features") -> dict:
    csv_path_obj = Path(csv_path)
    filename = csv_path_obj.name
    task_type = detect_task_type(filename)
    
    all_features = {}
    
    if task_type == 'fixation':
        features = extract_fixation_features(csv_path)
        all_features.update(features)
        
        for key in ['sac_latency', 'sac_peak_vel', 'sac_amp_error', 'sac_direction_error', 'sac_dysmetria_type',
                   'pursuit_gain', 'pursuit_phase_lag', 'pursuit_catchups', 'pursuit_smoothness_r2',
                   'anti_error_rate', 'anti_corr_latency', 'anti_reflexive_count']:
            all_features[key] = np.nan
    
    elif task_type == 'saccade':
        features = extract_saccade_features(csv_path)
        all_features.update(features)
        
        for key in ['fix_std_x', 'fix_std_y', 'fix_drift', 'fix_microsaccades', 'fix_swj', 'fix_nystagmus_hz',
                   'pursuit_gain', 'pursuit_phase_lag', 'pursuit_catchups', 'pursuit_smoothness_r2',
                   'anti_error_rate', 'anti_corr_latency', 'anti_reflexive_count']:
            all_features[key] = np.nan
    
    elif task_type == 'smooth_pursuit':
        features = extract_smooth_features(csv_path)
        all_features.update(features)
        
        for key in ['fix_std_x', 'fix_std_y', 'fix_drift', 'fix_microsaccades', 'fix_swj', 'fix_nystagmus_hz',
                   'sac_latency', 'sac_peak_vel', 'sac_amp_error', 'sac_direction_error', 'sac_dysmetria_type',
                   'anti_error_rate', 'anti_corr_latency', 'anti_reflexive_count']:
            all_features[key] = np.nan
    
    elif task_type == 'anti_saccade':
        features = extract_anti_saccade_features(csv_path)
        all_features.update(features)
        
        for key in ['fix_std_x', 'fix_std_y', 'fix_drift', 'fix_microsaccades', 'fix_swj', 'fix_nystagmus_hz',
                   'sac_latency', 'sac_peak_vel', 'sac_amp_error', 'sac_direction_error', 'sac_dysmetria_type',
                   'pursuit_gain', 'pursuit_phase_lag', 'pursuit_catchups', 'pursuit_smoothness_r2']:
            all_features[key] = np.nan
    
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir_obj / f"features_{csv_path_obj.stem}.csv"
    
    feature_df = pd.DataFrame([all_features])
    feature_df.to_csv(output_file, index=False)
    
    return all_features


def combine_session_features(session_files: list, output_file: str = None) -> dict:
    combined_features = {
        'fix_std_x': [],
        'fix_std_y': [],
        'fix_drift': [],
        'fix_microsaccades': [],
        'fix_swj': [],
        'fix_nystagmus_hz': [],
        'sac_latency': [],
        'sac_peak_vel': [],
        'sac_amp_error': [],
        'sac_direction_error': [],
        'sac_dysmetria_type': [],
        'pursuit_gain': [],
        'pursuit_phase_lag': [],
        'pursuit_catchups': [],
        'pursuit_smoothness_r2': [],
        'anti_error_rate': [],
        'anti_corr_latency': [],
        'anti_reflexive_count': []
    }
    
    for csv_path in session_files:
        features = extract_all_features(csv_path)
        for key in combined_features:
            if key in features:
                combined_features[key].append(features[key])
    
    final_features = {}
    for key, values in combined_features.items():
        valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        if valid_values:
            final_features[key] = np.mean(valid_values)
        else:
            final_features[key] = np.nan
    
    if output_file:
        feature_df = pd.DataFrame([final_features])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_file, index=False)
    
    return final_features

