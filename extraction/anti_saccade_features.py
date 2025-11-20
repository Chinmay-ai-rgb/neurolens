import numpy as np
import pandas as pd
from pathlib import Path


def extract_anti_saccade_features(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    csv_path_obj = Path(csv_path)
    stem_parts = csv_path_obj.stem.split('_')
    if len(stem_parts) >= 3:
        timestamp = '_'.join(stem_parts[-2:])
    else:
        timestamp = stem_parts[-1] if stem_parts else ''
    trial_file = csv_path_obj.parent / f"anti_saccade_trials_{timestamp}.csv"
    
    if not trial_file.exists():
        return {
            "anti_error_rate": np.nan,
            "anti_corr_latency": np.nan,
            "anti_reflexive_count": np.nan
        }
    
    trial_df = pd.read_csv(trial_file)
    
    if 'error' not in trial_df.columns:
        return {
            "anti_error_rate": np.nan,
            "anti_corr_latency": np.nan,
            "anti_reflexive_count": np.nan
        }
    
    errors = trial_df['error'].dropna()
    total_trials = len(errors)
    
    if total_trials == 0:
        error_rate = np.nan
        reflexive_count = 0
    else:
        error_count = np.sum(errors == True)
        error_rate = error_count / total_trials
        reflexive_count = int(error_count)
    
    correction_latencies = trial_df['correction_latency'].dropna()
    
    if len(correction_latencies) > 0:
        mean_correction_latency = np.mean(correction_latencies)
    else:
        mean_correction_latency = np.nan
    
    features = {
        "anti_error_rate": error_rate,
        "anti_corr_latency": mean_correction_latency,
        "anti_reflexive_count": reflexive_count
    }
    
    return features

