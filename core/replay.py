"""Replay module for NeuroLens+ eye tracking system."""

import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path

from .tracker import FrameData
from .mapping import GazeMapper
from .calibration import CalibrationResult


class ReplayEngine:
    """
    Replay engine for reprocessing frame logs.
    
    Allows recomputation of biomarkers from recorded frame data
    for validation and analysis.
    """
    
    def __init__(
        self,
        calibration: Optional[CalibrationResult] = None,
        screen_width: int = 1920,
        screen_height: int = 1080
    ):
        """
        Initialize replay engine.
        
        Args:
            calibration: Optional calibration to use for mapping
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.calibration = calibration
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize mapper if calibration provided
        self.mapper = None
        if calibration is not None:
            self.mapper = GazeMapper(screen_width, screen_height, calibration)
    
    def load_frame_log(self, filepath: str) -> pd.DataFrame:
        """
        Load frame log CSV into DataFrame.
        
        Args:
            filepath: Path to frame log CSV
        
        Returns:
            DataFrame with frame data
        """
        df = pd.read_csv(filepath)
        return df
    
    def iterate_frames(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """
        Iterate over frames in DataFrame.
        
        Args:
            df: DataFrame with frame data
        
        Yields:
            Tuple of (index, frame_dict)
        """
        for idx, row in df.iterrows():
            yield idx, row.to_dict()
    
    def iterate_trials(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[int, pd.DataFrame], None, None]:
        """
        Iterate over trials in DataFrame.
        
        Args:
            df: DataFrame with frame data
        
        Yields:
            Tuple of (trial_id, trial_df)
        """
        if 'trial_id' not in df.columns:
            yield 0, df
            return
        
        for trial_id in df['trial_id'].unique():
            trial_df = df[df['trial_id'] == trial_id].copy()
            yield trial_id, trial_df
    
    def extract_trial_data(
        self,
        trial_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract common trial data from DataFrame.
        
        Args:
            trial_df: DataFrame for single trial
        
        Returns:
            Dictionary with extracted data
        """
        data = {
            'trial_id': trial_df['trial_id'].iloc[0] if 'trial_id' in trial_df.columns else 0,
            'trial_start_s': trial_df['t_epoch_s'].iloc[0],
            'trial_end_s': trial_df['t_epoch_s'].iloc[-1],
            'n_frames': len(trial_df),
            'duration_s': trial_df['t_epoch_s'].iloc[-1] - trial_df['t_epoch_s'].iloc[0]
        }
        
        # Target position
        if 'target_x' in trial_df.columns:
            data['target_x'] = trial_df['target_x'].iloc[0]
            data['target_y'] = trial_df['target_y'].iloc[0]
        
        # Quality metrics
        if 'face_present' in trial_df.columns:
            data['face_presence_rate'] = trial_df['face_present'].mean()
        
        if 'blink_flag' in trial_df.columns:
            data['blink_rate'] = trial_df['blink_flag'].mean()
        
        if 'valid_sample' in trial_df.columns:
            data['valid_fraction'] = trial_df['valid_sample'].mean()
        
        if 'fps_est' in trial_df.columns:
            data['fps_median'] = trial_df['fps_est'].median()
        
        if 'clamp_flag_x' in trial_df.columns and 'clamp_flag_y' in trial_df.columns:
            data['clamp_rate'] = (trial_df['clamp_flag_x'].sum() + trial_df['clamp_flag_y'].sum()) / (2 * len(trial_df))
        
        return data
    
    def compute_fixation_biomarkers(
        self,
        trial_df: pd.DataFrame,
        target_x: float,
        target_y: float,
        target_radius_px: float = 100.0
    ) -> Dict[str, Any]:
        """
        Compute fixation biomarkers from trial data.
        
        Args:
            trial_df: DataFrame for fixation trial
            target_x: Target x position in pixels
            target_y: Target y position in pixels
            target_radius_px: Radius for on-target calculation
        
        Returns:
            Dictionary with fixation biomarkers
        """
        from .utils import compute_rms, compute_bcea, compute_microsaccade_rate
        
        # Get valid samples
        valid_mask = trial_df['valid_sample'] == 1
        
        if valid_mask.sum() < 3:
            return {
                'fixation_stability_rms_px': np.nan,
                'fixation_bcea_px2': np.nan,
                'microsaccade_rate_per_min': np.nan,
                'drift_velocity_px_s': np.nan,
                'percent_time_on_target': np.nan,
                'blink_rate_per_min': np.nan
            }
        
        gaze_x = trial_df.loc[valid_mask, 'gaze_x_px_comp'].values
        gaze_y = trial_df.loc[valid_mask, 'gaze_y_px_comp'].values
        timestamps = trial_df.loc[valid_mask, 't_epoch_s'].values
        
        # Deviation from target
        dev_x = gaze_x - target_x
        dev_y = gaze_y - target_y
        
        # RMS stability
        rms_x = compute_rms(dev_x)
        rms_y = compute_rms(dev_y)
        rms_total = np.sqrt(rms_x**2 + rms_y**2)
        
        # BCEA
        bcea = compute_bcea(gaze_x, gaze_y)
        
        # Microsaccade rate
        if 'vel_x_px_s' in trial_df.columns and 'vel_y_px_s' in trial_df.columns:
            vel_x = trial_df.loc[valid_mask, 'vel_x_px_s'].values
            vel_y = trial_df.loc[valid_mask, 'vel_y_px_s'].values
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            microsaccade_rate = compute_microsaccade_rate(vel_mag, timestamps)
        else:
            microsaccade_rate = np.nan
        
        # Drift velocity (linear trend)
        if len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            if duration > 0:
                drift_x = (gaze_x[-1] - gaze_x[0]) / duration
                drift_y = (gaze_y[-1] - gaze_y[0]) / duration
                drift_velocity = np.sqrt(drift_x**2 + drift_y**2)
            else:
                drift_velocity = np.nan
        else:
            drift_velocity = np.nan
        
        # Percent time on target
        distances = np.sqrt(dev_x**2 + dev_y**2)
        on_target = distances <= target_radius_px
        percent_on_target = on_target.sum() / len(on_target) * 100
        
        # Blink rate
        total_duration = trial_df['t_epoch_s'].iloc[-1] - trial_df['t_epoch_s'].iloc[0]
        if total_duration > 0:
            blink_count = (trial_df['blink_flag'].diff() == 1).sum()
            blink_rate_per_min = blink_count / (total_duration / 60)
        else:
            blink_rate_per_min = np.nan
        
        return {
            'fixation_stability_rms_px': rms_total,
            'fixation_bcea_px2': bcea,
            'microsaccade_rate_per_min': microsaccade_rate,
            'drift_velocity_px_s': drift_velocity,
            'percent_time_on_target': percent_on_target,
            'blink_rate_per_min': blink_rate_per_min
        }
    
    def compute_saccade_biomarkers(
        self,
        trial_df: pd.DataFrame,
        target_x: float,
        center_x: float,
        velocity_threshold: float = 30.0
    ) -> Dict[str, Any]:
        """
        Compute saccade biomarkers from trial data.
        
        Args:
            trial_df: DataFrame for saccade trial
            target_x: Target x position in pixels
            center_x: Center x position in pixels
            velocity_threshold: Velocity threshold for saccade detection
        
        Returns:
            Dictionary with saccade biomarkers
        """
        from .utils import detect_saccade_onset, find_peak_velocity
        
        # Find JUMP state onset
        if 'state' in trial_df.columns:
            jump_mask = trial_df['state'] == 'JUMP'
            if not jump_mask.any():
                jump_mask = trial_df['state'] == 'POST'
            
            if jump_mask.any():
                jump_idx = jump_mask.idxmax()
                jump_time = trial_df.loc[jump_idx, 't_epoch_s']
            else:
                return self._empty_saccade_biomarkers()
        else:
            return self._empty_saccade_biomarkers()
        
        # Get post-jump data
        post_df = trial_df.loc[jump_idx:].copy()
        
        if len(post_df) < 3:
            return self._empty_saccade_biomarkers()
        
        # Get valid samples
        valid_mask = post_df['valid_sample'] == 1
        
        if valid_mask.sum() < 3:
            return self._empty_saccade_biomarkers()
        
        gaze_x = post_df['gaze_x_px_comp'].values
        timestamps = post_df['t_epoch_s'].values
        
        # Compute velocity if not present
        if 'vel_x_px_s' in post_df.columns:
            vel_x = post_df['vel_x_px_s'].values
        else:
            vel_x = np.gradient(gaze_x, timestamps)
        
        vel_mag = np.abs(vel_x)
        
        # Detect saccade onset
        onset_idx = detect_saccade_onset(vel_mag, timestamps, velocity_threshold)
        
        if onset_idx is None:
            return self._empty_saccade_biomarkers()
        
        onset_time = timestamps[onset_idx]
        onset_gaze = gaze_x[onset_idx]
        
        # Find peak velocity and landing
        peak_vel, peak_idx = find_peak_velocity(vel_mag, onset_idx, len(vel_mag))
        
        # Landing is when velocity drops below threshold after peak
        landing_idx = peak_idx
        for i in range(peak_idx, len(vel_mag)):
            if vel_mag[i] < velocity_threshold:
                landing_idx = i
                break
        
        landing_time = timestamps[landing_idx]
        landing_gaze = gaze_x[landing_idx]
        
        # Compute biomarkers
        eccentricity = target_x - center_x
        amplitude = landing_gaze - onset_gaze
        
        latency_ms = (onset_time - jump_time) * 1000
        duration_ms = (landing_time - onset_time) * 1000
        
        # Direction check
        expected_direction = np.sign(eccentricity)
        actual_direction = np.sign(amplitude)
        direction_correct = expected_direction == actual_direction
        
        # Gain
        gain = amplitude / eccentricity if abs(eccentricity) > 1 else np.nan
        
        # Landing error
        landing_error = abs(landing_gaze - target_x)
        
        # Over/undershoot
        if direction_correct:
            if abs(landing_gaze - center_x) > abs(eccentricity):
                overshoot = abs(landing_gaze - target_x)
                undershoot = 0
            else:
                overshoot = 0
                undershoot = abs(landing_gaze - target_x)
        else:
            overshoot = 0
            undershoot = abs(eccentricity)
        
        # Corrective saccades (velocity peaks after landing)
        corrective_count = 0
        if landing_idx < len(vel_mag) - 1:
            post_landing_vel = vel_mag[landing_idx+1:]
            in_saccade = False
            for v in post_landing_vel:
                if v > velocity_threshold and not in_saccade:
                    corrective_count += 1
                    in_saccade = True
                elif v < velocity_threshold:
                    in_saccade = False
        
        return {
            'saccade_latency_ms': latency_ms,
            'saccade_duration_ms': duration_ms,
            'peak_velocity_px_s': peak_vel,
            'saccade_amplitude_px': amplitude,
            'gain': gain,
            'landing_error_px': landing_error,
            'overshoot_px': overshoot,
            'undershoot_px': undershoot,
            'corrective_saccade_count': corrective_count,
            'direction_correct': direction_correct
        }
    
    def _empty_saccade_biomarkers(self) -> Dict[str, Any]:
        """Return empty saccade biomarkers."""
        return {
            'saccade_latency_ms': np.nan,
            'saccade_duration_ms': np.nan,
            'peak_velocity_px_s': np.nan,
            'saccade_amplitude_px': np.nan,
            'gain': np.nan,
            'landing_error_px': np.nan,
            'overshoot_px': np.nan,
            'undershoot_px': np.nan,
            'corrective_saccade_count': 0,
            'direction_correct': False
        }
    
    def compute_pursuit_biomarkers(
        self,
        trial_df: pd.DataFrame,
        target_velocity: float
    ) -> Dict[str, Any]:
        """
        Compute smooth pursuit biomarkers from trial data.
        
        Args:
            trial_df: DataFrame for pursuit trial
            target_velocity: Target velocity in px/s
        
        Returns:
            Dictionary with pursuit biomarkers
        """
        from .utils import compute_pursuit_gain, cross_correlation_lag, compute_rms
        
        valid_mask = trial_df['valid_sample'] == 1
        
        if valid_mask.sum() < 10:
            return self._empty_pursuit_biomarkers()
        
        gaze_x = trial_df.loc[valid_mask, 'gaze_x_px_comp'].values
        target_x = trial_df.loc[valid_mask, 'target_x'].values
        timestamps = trial_df.loc[valid_mask, 't_epoch_s'].values
        
        # Eye velocity
        if 'vel_x_px_s' in trial_df.columns:
            eye_vel = trial_df.loc[valid_mask, 'vel_x_px_s'].values
        else:
            eye_vel = np.gradient(gaze_x, timestamps)
        
        # Target velocity array
        target_vel = np.full_like(eye_vel, target_velocity)
        
        # Pursuit gain
        gain = compute_pursuit_gain(eye_vel, target_vel, ~np.isnan(eye_vel))
        
        # Position error
        position_error = gaze_x - target_x
        error_mean = np.nanmean(np.abs(position_error))
        error_rmse = compute_rms(position_error)
        
        # Phase lag via cross-correlation
        fps_est = 30.0
        if 'fps_est' in trial_df.columns:
            fps_est = trial_df['fps_est'].median()
        
        corr, lag_samples = cross_correlation_lag(gaze_x, target_x)
        phase_lag_ms = (lag_samples / fps_est) * 1000 if fps_est > 0 else np.nan
        
        # Catch-up saccades (velocity spikes)
        velocity_threshold = 100  # px/s
        catch_up_count = 0
        in_saccade = False
        for v in np.abs(eye_vel):
            if v > velocity_threshold and not in_saccade:
                catch_up_count += 1
                in_saccade = True
            elif v < velocity_threshold * 0.5:
                in_saccade = False
        
        duration_s = timestamps[-1] - timestamps[0]
        catch_up_rate = catch_up_count / duration_s if duration_s > 0 else np.nan
        
        # Pursuit latency (time until eye starts following)
        # Simplified: first time eye velocity matches target direction
        latency_ms = np.nan
        for i, v in enumerate(eye_vel):
            if np.sign(v) == np.sign(target_velocity) and abs(v) > 10:
                latency_ms = (timestamps[i] - timestamps[0]) * 1000
                break
        
        return {
            'pursuit_gain': gain,
            'pursuit_latency_ms': latency_ms,
            'catch_up_saccade_count': catch_up_count,
            'catch_up_saccade_rate_per_s': catch_up_rate,
            'position_error_mean_px': error_mean,
            'position_error_rmse_px': error_rmse,
            'phase_lag_ms': phase_lag_ms
        }
    
    def _empty_pursuit_biomarkers(self) -> Dict[str, Any]:
        """Return empty pursuit biomarkers."""
        return {
            'pursuit_gain': np.nan,
            'pursuit_latency_ms': np.nan,
            'catch_up_saccade_count': 0,
            'catch_up_saccade_rate_per_s': np.nan,
            'position_error_mean_px': np.nan,
            'position_error_rmse_px': np.nan,
            'phase_lag_ms': np.nan
        }
    
    def compute_plr_biomarkers(
        self,
        trial_df: pd.DataFrame,
        flash_time: float
    ) -> Dict[str, Any]:
        """
        Compute PLR biomarkers from trial data.
        
        Args:
            trial_df: DataFrame for PLR trial
            flash_time: Time of light flash
        
        Returns:
            Dictionary with PLR biomarkers
        """
        valid_mask = trial_df['valid_sample'] == 1
        
        if valid_mask.sum() < 10:
            return self._empty_plr_biomarkers()
        
        timestamps = trial_df.loc[valid_mask, 't_epoch_s'].values
        pupil = trial_df.loc[valid_mask, 'pupil_proxy_mean'].values
        pupil_left = trial_df.loc[valid_mask, 'pupil_proxy_left'].values
        pupil_right = trial_df.loc[valid_mask, 'pupil_proxy_right'].values
        
        # Find baseline (before flash)
        pre_flash_mask = timestamps < flash_time
        if pre_flash_mask.sum() < 3:
            baseline = np.nanmean(pupil[:5])
        else:
            baseline = np.nanmean(pupil[pre_flash_mask])
        
        # Find minimum pupil (constriction)
        post_flash_mask = timestamps >= flash_time
        if post_flash_mask.sum() < 3:
            return self._empty_plr_biomarkers()
        
        post_flash_pupil = pupil[post_flash_mask]
        post_flash_times = timestamps[post_flash_mask]
        
        min_idx = np.nanargmin(post_flash_pupil)
        min_pupil = post_flash_pupil[min_idx]
        min_time = post_flash_times[min_idx]
        
        # Constriction amplitude
        constriction_amplitude = baseline - min_pupil
        
        # Constriction latency
        constriction_latency_ms = (min_time - flash_time) * 1000
        
        # Constriction velocity (max negative derivative)
        pupil_deriv = np.gradient(post_flash_pupil, post_flash_times)
        constriction_velocity = np.nanmin(pupil_deriv)
        
        # Dilation recovery time (to 75% of baseline)
        recovery_threshold = min_pupil + 0.75 * constriction_amplitude
        recovery_time_ms = np.nan
        for i in range(min_idx, len(post_flash_pupil)):
            if post_flash_pupil[i] >= recovery_threshold:
                recovery_time_ms = (post_flash_times[i] - min_time) * 1000
                break
        
        # Asymmetry
        baseline_left = np.nanmean(pupil_left[pre_flash_mask]) if pre_flash_mask.sum() > 0 else np.nan
        baseline_right = np.nanmean(pupil_right[pre_flash_mask]) if pre_flash_mask.sum() > 0 else np.nan
        
        if not np.isnan(baseline_left) and not np.isnan(baseline_right):
            asymmetry = abs(baseline_left - baseline_right) / ((baseline_left + baseline_right) / 2)
        else:
            asymmetry = np.nan
        
        return {
            'constriction_latency_ms': constriction_latency_ms,
            'constriction_amplitude': constriction_amplitude,
            'constriction_velocity': constriction_velocity,
            'dilation_recovery_time_ms': recovery_time_ms,
            'baseline_pupil_proxy': baseline,
            'min_pupil_proxy': min_pupil,
            'asymmetry_left_right': asymmetry
        }
    
    def _empty_plr_biomarkers(self) -> Dict[str, Any]:
        """Return empty PLR biomarkers."""
        return {
            'constriction_latency_ms': np.nan,
            'constriction_amplitude': np.nan,
            'constriction_velocity': np.nan,
            'dilation_recovery_time_ms': np.nan,
            'baseline_pupil_proxy': np.nan,
            'min_pupil_proxy': np.nan,
            'asymmetry_left_right': np.nan
        }
    
    def compute_blink_biomarkers(
        self,
        trial_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute blink biomarkers from trial data.
        
        Args:
            trial_df: DataFrame for blink monitoring
        
        Returns:
            Dictionary with blink biomarkers
        """
        timestamps = trial_df['t_epoch_s'].values
        blink_flags = trial_df['blink_flag'].values
        
        duration_s = timestamps[-1] - timestamps[0]
        if duration_s <= 0:
            return self._empty_blink_biomarkers()
        
        # Detect blink events
        blink_starts = []
        blink_ends = []
        in_blink = False
        
        for i, flag in enumerate(blink_flags):
            if flag == 1 and not in_blink:
                blink_starts.append(i)
                in_blink = True
            elif flag == 0 and in_blink:
                blink_ends.append(i)
                in_blink = False
        
        # Handle blink at end
        if in_blink:
            blink_ends.append(len(blink_flags) - 1)
        
        n_blinks = len(blink_starts)
        
        if n_blinks == 0:
            return {
                'blink_rate_per_min': 0.0,
                'blink_duration_mean_ms': np.nan,
                'blink_duration_p95_ms': np.nan,
                'interblink_interval_mean_s': np.nan,
                'blink_irregularity_cv': np.nan
            }
        
        # Blink rate
        blink_rate = n_blinks / (duration_s / 60)
        
        # Blink durations
        durations_ms = []
        for start, end in zip(blink_starts, blink_ends):
            dur = (timestamps[end] - timestamps[start]) * 1000
            durations_ms.append(dur)
        
        duration_mean = np.mean(durations_ms)
        duration_p95 = np.percentile(durations_ms, 95) if len(durations_ms) > 1 else duration_mean
        
        # Interblink intervals
        if n_blinks > 1:
            intervals = []
            for i in range(1, len(blink_starts)):
                interval = timestamps[blink_starts[i]] - timestamps[blink_ends[i-1]]
                intervals.append(interval)
            
            interval_mean = np.mean(intervals)
            interval_std = np.std(intervals)
            irregularity_cv = interval_std / interval_mean if interval_mean > 0 else np.nan
        else:
            interval_mean = np.nan
            irregularity_cv = np.nan
        
        return {
            'blink_rate_per_min': blink_rate,
            'blink_duration_mean_ms': duration_mean,
            'blink_duration_p95_ms': duration_p95,
            'interblink_interval_mean_s': interval_mean,
            'blink_irregularity_cv': irregularity_cv
        }
    
    def _empty_blink_biomarkers(self) -> Dict[str, Any]:
        """Return empty blink biomarkers."""
        return {
            'blink_rate_per_min': np.nan,
            'blink_duration_mean_ms': np.nan,
            'blink_duration_p95_ms': np.nan,
            'interblink_interval_mean_s': np.nan,
            'blink_irregularity_cv': np.nan
        }
    
    def replay_and_compute(
        self,
        frame_log_path: str,
        task_type: str
    ) -> List[Dict[str, Any]]:
        """
        Replay frame log and compute biomarkers for all trials.
        
        Args:
            frame_log_path: Path to frame log CSV
            task_type: Type of task ('fixation', 'saccade', 'pursuit', 'plr', 'blink')
        
        Returns:
            List of trial summary dictionaries
        """
        df = self.load_frame_log(frame_log_path)
        results = []
        
        for trial_id, trial_df in self.iterate_trials(df):
            trial_data = self.extract_trial_data(trial_df)
            
            if task_type == 'fixation':
                target_x = trial_data.get('target_x', self.screen_width / 2)
                target_y = trial_data.get('target_y', self.screen_height / 2)
                biomarkers = self.compute_fixation_biomarkers(trial_df, target_x, target_y)
            elif task_type == 'saccade':
                target_x = trial_data.get('target_x', self.screen_width / 2)
                center_x = self.screen_width / 2
                biomarkers = self.compute_saccade_biomarkers(trial_df, target_x, center_x)
            elif task_type == 'pursuit':
                target_velocity = 200  # Default
                biomarkers = self.compute_pursuit_biomarkers(trial_df, target_velocity)
            elif task_type == 'plr':
                flash_time = trial_data['trial_start_s'] + 2.0  # Assume flash at 2s
                biomarkers = self.compute_plr_biomarkers(trial_df, flash_time)
            elif task_type == 'blink':
                biomarkers = self.compute_blink_biomarkers(trial_df)
            else:
                biomarkers = {}
            
            trial_data.update(biomarkers)
            results.append(trial_data)
        
        return results
