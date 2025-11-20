import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression


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


def compute_pursuit_gain(df: pd.DataFrame) -> float:
    df_clean = df[detect_blinks(df)].copy()
    
    if 'eye_velocity_x' not in df_clean.columns or 'target_velocity_x' not in df_clean.columns:
        return np.nan
    
    eye_vel = df_clean['eye_velocity_x'].dropna()
    target_vel = df_clean['target_velocity_x'].dropna()
    
    if len(eye_vel) == 0 or len(target_vel) == 0:
        return np.nan
    
    min_len = min(len(eye_vel), len(target_vel))
    eye_vel = eye_vel.iloc[:min_len]
    target_vel = target_vel.iloc[:min_len]
    
    target_vel_abs = np.abs(target_vel)
    valid_mask = target_vel_abs > 0.1
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    eye_vel_valid = eye_vel[valid_mask]
    target_vel_valid = target_vel[valid_mask]
    
    gains = eye_vel_valid / target_vel_valid
    mean_gain = np.mean(gains)
    
    return mean_gain


def compute_phase_lag(df: pd.DataFrame) -> float:
    df_clean = df[detect_blinks(df)].copy()
    
    if 'left_x' not in df_clean.columns or 'target_x' not in df_clean.columns:
        return np.nan
    
    eye_pos = df_clean['left_x'].ffill().bfill().values
    target_pos = df_clean['target_x'].ffill().bfill().values
    
    if len(eye_pos) < 10 or len(target_pos) < 10:
        return np.nan
    
    eye_pos_centered = eye_pos - np.mean(eye_pos)
    target_pos_centered = target_pos - np.mean(target_pos)
    
    correlation = signal.correlate(eye_pos_centered, target_pos_centered, mode='full')
    lags = signal.correlation_lags(len(eye_pos_centered), len(target_pos_centered), mode='full')
    
    max_corr_idx = np.argmax(np.abs(correlation))
    lag_samples = lags[max_corr_idx]
    
    if 'timestamp' in df_clean.columns:
        timestamps = df_clean['timestamp'].values
        if len(timestamps) > 1:
            dt = np.mean(np.diff(timestamps))
            lag_seconds = lag_samples * dt
            return lag_seconds
    
    return np.nan


def detect_catchup_saccades(df: pd.DataFrame, velocity_threshold: float = 100.0) -> int:
    df_clean = df[detect_blinks(df)].copy()
    
    if 'eye_velocity_x' not in df_clean.columns or 'target_velocity_x' not in df_clean.columns:
        return 0
    
    if len(df_clean) < 2:
        return 0
    
    eye_vel = df_clean['eye_velocity_x'].fillna(0).values
    target_vel = df_clean['target_velocity_x'].fillna(0).values
    
    target_moving = np.abs(target_vel) > 5.0
    
    if np.sum(target_moving) == 0:
        return 0
    
    eye_vel_abs = np.abs(eye_vel)
    
    catchup_count = 0
    i = 0
    
    while i < len(eye_vel_abs) - 2:
        if target_moving[i] and eye_vel_abs[i] > velocity_threshold:
            start = i
            while i < len(eye_vel_abs) and eye_vel_abs[i] > velocity_threshold:
                i += 1
            
            duration = i - start
            if duration >= 2:
                catchup_count += 1
        else:
            i += 1
    
    return catchup_count


def compute_smoothness_r2(df: pd.DataFrame) -> float:
    df_clean = df[detect_blinks(df)].copy()
    
    if 'left_x' not in df_clean.columns or 'target_x' not in df_clean.columns:
        return np.nan
    
    eye_pos = df_clean['left_x'].dropna().values
    target_pos = df_clean['target_x'].dropna().values
    
    if len(eye_pos) < 10 or len(target_pos) < 10:
        return np.nan
    
    min_len = min(len(eye_pos), len(target_pos))
    eye_pos = eye_pos[:min_len]
    target_pos = target_pos[:min_len]
    
    eye_pos = eye_pos.reshape(-1, 1)
    target_pos = target_pos.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(target_pos, eye_pos)
    
    y_pred = model.predict(target_pos)
    ss_res = np.sum((eye_pos - y_pred) ** 2)
    ss_tot = np.sum((eye_pos - np.mean(eye_pos)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    r2 = 1 - (ss_res / ss_tot)
    
    return r2


def extract_smooth_features(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    gain = compute_pursuit_gain(df)
    phase_lag = compute_phase_lag(df)
    catchups = detect_catchup_saccades(df)
    smoothness_r2 = compute_smoothness_r2(df)
    
    features = {
        "pursuit_gain": gain,
        "pursuit_phase_lag": phase_lag,
        "pursuit_catchups": catchups,
        "pursuit_smoothness_r2": smoothness_r2
    }
    
    return features

