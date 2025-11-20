import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq


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


def smooth_signal(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def compute_fixation_stability(df: pd.DataFrame, ipd: float) -> dict:
    df_clean = df[detect_blinks(df)].copy()
    
    if len(df_clean) == 0 or df_clean['left_x'].isna().all():
        return {
            "fix_std_x": np.nan,
            "fix_std_y": np.nan
        }
    
    left_x_norm = df_clean['left_x'] / ipd if ipd > 0 else df_clean['left_x']
    left_y_norm = df_clean['left_y'] / ipd if ipd > 0 else df_clean['left_y']
    
    left_x_clean = left_x_norm.dropna()
    left_y_clean = left_y_norm.dropna()
    
    std_x = np.std(left_x_clean) if len(left_x_clean) > 0 else np.nan
    std_y = np.std(left_y_clean) if len(left_y_clean) > 0 else np.nan
    
    return {
        "fix_std_x": std_x,
        "fix_std_y": std_y
    }


def compute_drift_magnitude(df: pd.DataFrame, ipd: float) -> float:
    df_clean = df[detect_blinks(df)].copy()
    
    if len(df_clean) == 0 or df_clean['left_x'].isna().all():
        return np.nan
    
    left_x_norm = df_clean['left_x'] / ipd if ipd > 0 else df_clean['left_x']
    left_y_norm = df_clean['left_y'] / ipd if ipd > 0 else df_clean['left_y']
    
    mean_x = np.nanmean(left_x_norm)
    mean_y = np.nanmean(left_y_norm)
    
    distances = np.sqrt((left_x_norm - mean_x)**2 + (left_y_norm - mean_y)**2)
    max_drift = np.nanmax(distances)
    
    return max_drift


def detect_microsaccades(df: pd.DataFrame, velocity_threshold: float = 5.0, 
                        min_duration: int = 2, max_duration: int = 6) -> int:
    df_clean = df[detect_blinks(df)].copy()
    
    if len(df_clean) < 2:
        return 0
    
    timestamps = df_clean['timestamp'].values
    left_x = df_clean['left_x'].values
    left_y = df_clean['left_y'].values
    
    dt = np.diff(timestamps)
    dt = np.where(dt > 0, dt, 1/30)
    
    dx = np.diff(left_x)
    dy = np.diff(left_y)
    velocity = np.sqrt(dx**2 + dy**2) / dt
    
    microsaccades = []
    i = 0
    
    while i < len(velocity):
        if velocity[i] > velocity_threshold:
            start = i
            while i < len(velocity) and velocity[i] > velocity_threshold:
                i += 1
            duration = i - start
            
            if min_duration <= duration <= max_duration:
                microsaccades.append((start, i))
        else:
            i += 1
    
    return len(microsaccades)


def detect_square_wave_jerks(df: pd.DataFrame, min_interval_ms: float = 200, 
                            max_interval_ms: float = 300) -> int:
    df_clean = df[detect_blinks(df)].copy()
    
    if len(df_clean) < 2:
        return 0
    
    timestamps = df_clean['timestamp'].values
    left_x = df_clean['left_x'].values
    
    dt = np.diff(timestamps)
    dt = np.where(dt > 0, dt, 1/30)
    
    dx = np.diff(left_x)
    velocity = dx / dt
    
    microsaccades = []
    i = 0
    velocity_threshold = 5.0
    
    while i < len(velocity):
        if abs(velocity[i]) > velocity_threshold:
            direction = np.sign(velocity[i])
            start = i
            while i < len(velocity) and np.sign(velocity[i]) == direction and abs(velocity[i]) > velocity_threshold:
                i += 1
            if i - start >= 2:
                microsaccades.append((start, i, direction))
        else:
            i += 1
    
    swj_count = 0
    for i in range(len(microsaccades) - 1):
        ms1 = microsaccades[i]
        ms2 = microsaccades[i + 1]
        
        if ms1[2] != ms2[2]:
            interval_start = timestamps[ms1[1]]
            interval_end = timestamps[ms2[0]]
            interval_ms = (interval_end - interval_start) * 1000
            
            if min_interval_ms <= interval_ms <= max_interval_ms:
                swj_count += 1
    
    return swj_count


def compute_nystagmus_frequency(df: pd.DataFrame, sampling_rate: float = 30.0) -> float:
    df_clean = df[detect_blinks(df)].copy()
    
    if len(df_clean) < 10:
        return np.nan
    
    left_x = df_clean['left_x'].ffill().bfill().values
    left_y = df_clean['left_y'].ffill().bfill().values
    
    if len(left_x) < 10:
        return np.nan
    
    n = len(left_x)
    fft_x = fft(left_x - np.mean(left_x))
    fft_y = fft(left_y - np.mean(left_y))
    
    freqs = fftfreq(n, 1/sampling_rate)
    
    power_x = np.abs(fft_x)**2
    power_y = np.abs(fft_y)**2
    
    freq_mask = (freqs >= 1) & (freqs <= 8)
    
    if np.sum(freq_mask) == 0:
        return np.nan
    
    power_x_filtered = power_x[freq_mask]
    power_y_filtered = power_y[freq_mask]
    freqs_filtered = freqs[freq_mask]
    
    combined_power = power_x_filtered + power_y_filtered
    max_idx = np.argmax(combined_power)
    
    return abs(freqs_filtered[max_idx])


def extract_fixation_features(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    if 'ipd_px' in df.columns:
        ipd = df['ipd_px'].median()
    else:
        ipd = 100.0
    
    if ipd == 0 or np.isnan(ipd):
        ipd = 100.0
    
    stability = compute_fixation_stability(df, ipd)
    drift = compute_drift_magnitude(df, ipd)
    microsaccades = detect_microsaccades(df)
    swj = detect_square_wave_jerks(df)
    nystagmus_hz = compute_nystagmus_frequency(df)
    
    features = {
        "fix_std_x": stability["fix_std_x"],
        "fix_std_y": stability["fix_std_y"],
        "fix_drift": drift,
        "fix_microsaccades": microsaccades,
        "fix_swj": swj,
        "fix_nystagmus_hz": nystagmus_hz
    }
    
    return features

