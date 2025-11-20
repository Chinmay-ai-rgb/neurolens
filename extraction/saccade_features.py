import numpy as np
import pandas as pd


def detect_blinks(df: pd.DataFrame, threshold: float = 0.15) -> np.ndarray:
    if 'left_y' not in df.columns or 'right_y' not in df.columns:
        return np.ones(len(df), dtype=bool)
    
    left_valid = ~df['left_y'].isna()
    right_valid = ~df['right_y'].isna()
    
    left_median = df['left_y'].median()
    right_median = df['right_y'].median()
    
    left_diff = np.abs(df['left_y'] - left_median)
    right_diff = np.abs(df['right_y'] - right_median)
    
    left_blink = left_diff > (threshold * left_median) if left_median > 0 else np.zeros(len(df), dtype=bool)
    right_blink = right_diff > (threshold * right_median) if right_median > 0 else np.zeros(len(df), dtype=bool)
    
    blinks = left_blink | right_blink
    return ~blinks


def compute_saccade_latency(df: pd.DataFrame) -> float:
    if 'stimulus_onset' not in df.columns:
        return np.nan
    
    stimulus_onsets = df[~df['stimulus_onset'].isna()]['stimulus_onset'].unique()
    
    if len(stimulus_onsets) == 0:
        return np.nan
    
    latencies = []
    
    for onset_time in stimulus_onsets:
        trial_df = df[df['timestamp'] >= onset_time].copy()
        trial_df = trial_df[trial_df['timestamp'] <= onset_time + 1.0]
        
        if len(trial_df) < 2:
            continue
        
        trial_df = trial_df[detect_blinks(trial_df)]
        
        if len(trial_df) < 2:
            continue
        
        timestamps = trial_df['timestamp'].values
        left_x = trial_df['left_x'].values
        
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 1/30)
        
        dx = np.diff(left_x)
        velocity = np.abs(dx) / dt
        
        for i in range(len(velocity)):
            if velocity[i] > 50:
                saccade_onset = timestamps[i]
                latency = saccade_onset - onset_time
                latencies.append(latency)
                break
    
    return np.mean(latencies) if latencies else np.nan


def compute_peak_velocity(df: pd.DataFrame) -> float:
    df_clean = df[detect_blinks(df)].copy()
    
    if len(df_clean) < 2:
        return np.nan
    
    timestamps = df_clean['timestamp'].values
    left_x = df_clean['left_x'].values
    left_y = df_clean['left_y'].values
    
    dt = np.diff(timestamps)
    dt = np.where(dt > 0, dt, 1/30)
    
    dx = np.diff(left_x)
    dy = np.diff(left_y)
    velocity = np.sqrt(dx**2 + dy**2) / dt
    
    peak_velocity = np.nanmax(velocity) if len(velocity) > 0 else np.nan
    
    return peak_velocity


def compute_amplitude_error(df: pd.DataFrame) -> float:
    if 'target_amplitude_px' not in df.columns:
        return np.nan
    
    stimulus_onsets = df[~df['stimulus_onset'].isna()]['stimulus_onset'].unique()
    target_amplitudes = df[~df['target_amplitude_px'].isna()]['target_amplitude_px'].unique()
    
    if len(stimulus_onsets) == 0 or len(target_amplitudes) == 0:
        return np.nan
    
    errors = []
    
    for onset_time, target_amp in zip(stimulus_onsets, target_amplitudes):
        trial_df = df[df['timestamp'] >= onset_time].copy()
        trial_df = trial_df[trial_df['timestamp'] <= onset_time + 1.0]
        
        if len(trial_df) < 2:
            continue
        
        trial_df = trial_df[detect_blinks(trial_df)]
        
        if len(trial_df) < 2:
            continue
        
        initial_x = trial_df.iloc[0]['left_x']
        final_x = trial_df.iloc[-1]['left_x']
        
        if np.isnan(initial_x) or np.isnan(final_x):
            continue
        
        actual_amplitude = abs(final_x - initial_x)
        error = abs(actual_amplitude - target_amp)
        errors.append(error)
    
    return np.mean(errors) if errors else np.nan


def compute_directional_accuracy(df: pd.DataFrame) -> float:
    if 'target_direction' not in df.columns:
        return np.nan
    
    stimulus_onsets = df[~df['stimulus_onset'].isna()]['stimulus_onset'].unique()
    target_directions = df[~df['target_direction'].isna()]['target_direction'].unique()
    
    if len(stimulus_onsets) == 0 or len(target_directions) == 0:
        return np.nan
    
    correct_count = 0
    total_count = 0
    
    for onset_time, direction in zip(stimulus_onsets, target_directions):
        trial_df = df[df['timestamp'] >= onset_time].copy()
        trial_df = trial_df[trial_df['timestamp'] <= onset_time + 1.0]
        
        if len(trial_df) < 2:
            continue
        
        trial_df = trial_df[detect_blinks(trial_df)]
        
        if len(trial_df) < 2:
            continue
        
        initial_x = trial_df.iloc[0]['left_x']
        final_x = trial_df.iloc[-1]['left_x']
        
        if np.isnan(initial_x) or np.isnan(final_x):
            continue
        
        movement = final_x - initial_x
        
        if direction == "right" and movement > 0:
            correct_count += 1
        elif direction == "left" and movement < 0:
            correct_count += 1
        
        total_count += 1
    
    return correct_count / total_count if total_count > 0 else np.nan


def classify_dysmetria(df: pd.DataFrame) -> str:
    if 'target_amplitude_px' not in df.columns:
        return "normal"
    
    stimulus_onsets = df[~df['stimulus_onset'].isna()]['stimulus_onset'].unique()
    target_amplitudes = df[~df['target_amplitude_px'].isna()]['target_amplitude_px'].unique()
    
    if len(stimulus_onsets) == 0 or len(target_amplitudes) == 0:
        return "normal"
    
    ratios = []
    
    for onset_time, target_amp in zip(stimulus_onsets, target_amplitudes):
        trial_df = df[df['timestamp'] >= onset_time].copy()
        trial_df = trial_df[trial_df['timestamp'] <= onset_time + 1.0]
        
        if len(trial_df) < 2:
            continue
        
        trial_df = trial_df[detect_blinks(trial_df)]
        
        if len(trial_df) < 2:
            continue
        
        initial_x = trial_df.iloc[0]['left_x']
        final_x = trial_df.iloc[-1]['left_x']
        
        if np.isnan(initial_x) or np.isnan(final_x) or target_amp == 0:
            continue
        
        actual_amplitude = abs(final_x - initial_x)
        ratio = actual_amplitude / target_amp
        ratios.append(ratio)
    
    if len(ratios) == 0:
        return "normal"
    
    mean_ratio = np.mean(ratios)
    
    if mean_ratio < 0.9:
        return "hypometric"
    elif mean_ratio > 1.1:
        return "hypermetric"
    else:
        return "normal"


def extract_saccade_features(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    latency = compute_saccade_latency(df)
    peak_velocity = compute_peak_velocity(df)
    amplitude_error = compute_amplitude_error(df)
    directional_accuracy = compute_directional_accuracy(df)
    dysmetria = classify_dysmetria(df)
    
    dysmetria_encoding = {"normal": 0, "hypometric": 1, "hypermetric": 2}
    direction_error = 1.0 - directional_accuracy if not np.isnan(directional_accuracy) else np.nan
    
    features = {
        "sac_latency": latency,
        "sac_peak_vel": peak_velocity,
        "sac_amp_error": amplitude_error,
        "sac_direction_error": direction_error,
        "sac_dysmetria_type": dysmetria_encoding.get(dysmetria, 0)
    }
    
    return features

