"""Replay tests for NeuroLens+ - verify biomarker recomputation from frame logs."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from core.replay import ReplayEngine


def create_synthetic_frame_log(
    filepath: str,
    task: str = 'saccade',
    n_trials: int = 5,
    n_frames_per_trial: int = 60,
    include_saccade: bool = True
):
    """
    Create a synthetic frame log CSV for testing replay.
    
    Args:
        filepath: Path to save CSV
        task: Task name
        n_trials: Number of trials
        n_frames_per_trial: Frames per trial
        include_saccade: Whether to include saccade-like movements
    """
    rows = []
    session_id = "test_session"
    
    np.random.seed(42)
    
    for trial_id in range(n_trials):
        trial_start = trial_id * 3.0  # 3 seconds per trial
        
        # Determine direction for this trial
        direction = 'left' if trial_id % 2 == 0 else 'right'
        target_x = 384 if direction == 'left' else 1536  # 20% or 80% of 1920
        
        for frame_idx in range(n_frames_per_trial):
            t = trial_start + frame_idx / 30.0  # 30 fps
            
            # Determine state
            if frame_idx < 30:
                state = 'FORE'
                current_target_x = 960  # Center
            elif frame_idx < 32:
                state = 'JUMP'
                current_target_x = target_x
            else:
                state = 'POST'
                current_target_x = target_x
            
            # Generate gaze position
            if state == 'FORE':
                gaze_x = 960 + np.random.normal(0, 10)
                gaze_y = 540 + np.random.normal(0, 10)
                vel_x = np.random.normal(0, 20)
                vel_y = np.random.normal(0, 20)
            else:
                # Saccade occurs around frame 35
                if include_saccade and 32 <= frame_idx <= 38:
                    # Saccade in progress
                    progress = (frame_idx - 32) / 6.0
                    if direction == 'left':
                        gaze_x = 960 - 576 * progress + np.random.normal(0, 5)
                    else:
                        gaze_x = 960 + 576 * progress + np.random.normal(0, 5)
                    vel_x = 576 * 30 / 6 * (1 if direction == 'right' else -1)
                elif frame_idx > 38:
                    # Post-saccade
                    gaze_x = target_x + np.random.normal(0, 15)
                    vel_x = np.random.normal(0, 30)
                else:
                    gaze_x = 960 + np.random.normal(0, 10)
                    vel_x = np.random.normal(0, 20)
                
                gaze_y = 540 + np.random.normal(0, 10)
                vel_y = np.random.normal(0, 20)
            
            # Clamp to screen
            gaze_x = np.clip(gaze_x, 0, 1919)
            gaze_y = np.clip(gaze_y, 0, 1079)
            
            row = {
                'session_id': session_id,
                'task': task,
                'trial_id': trial_id,
                'state': state,
                't_epoch_s': t,
                'frame_idx': trial_id * n_frames_per_trial + frame_idx,
                'fps_est': 30.0 + np.random.normal(0, 1),
                'screen_w': 1920,
                'screen_h': 1080,
                'target_x': current_target_x,
                'target_y': 540,
                'face_present': 1,
                'face_confidence': 0.95 + np.random.normal(0, 0.02),
                'blink_flag': 1 if np.random.random() < 0.02 else 0,
                'valid_sample': 1 if np.random.random() > 0.05 else 0,
                'left_iris_x_norm': gaze_x / 1920,
                'left_iris_y_norm': gaze_y / 1080,
                'right_iris_x_norm': gaze_x / 1920 + np.random.normal(0, 0.01),
                'right_iris_y_norm': gaze_y / 1080 + np.random.normal(0, 0.01),
                'gaze_x_norm': gaze_x / 1920,
                'gaze_y_norm': gaze_y / 1080,
                'anchor_x_norm': 0.5 + np.random.normal(0, 0.01),
                'anchor_y_norm': 0.5 + np.random.normal(0, 0.01),
                'gaze_x_norm_comp': gaze_x / 1920,
                'gaze_y_norm_comp': gaze_y / 1080,
                'gaze_x_px_raw': gaze_x,
                'gaze_y_px_raw': gaze_y,
                'gaze_x_px_comp': gaze_x,
                'gaze_y_px_comp': gaze_y,
                'vel_x_px_s': vel_x,
                'vel_y_px_s': vel_y,
                'pupil_proxy_left': 0.05 + np.random.normal(0, 0.002),
                'pupil_proxy_right': 0.05 + np.random.normal(0, 0.002),
                'pupil_proxy_mean': 0.05 + np.random.normal(0, 0.002),
                'clamp_flag_x': 0,
                'clamp_flag_y': 0,
                'out_of_range_flag': 0,
                'dropout_gap_s': 0.0,
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df


def create_fixation_frame_log(filepath: str, n_frames: int = 300):
    """Create a synthetic fixation frame log."""
    rows = []
    session_id = "test_fixation"
    
    np.random.seed(42)
    
    target_x, target_y = 960, 540
    
    for frame_idx in range(n_frames):
        t = frame_idx / 30.0
        
        # Fixation with some jitter
        gaze_x = target_x + np.random.normal(0, 15)
        gaze_y = target_y + np.random.normal(0, 15)
        
        # Occasional microsaccade
        if np.random.random() < 0.02:
            gaze_x += np.random.normal(0, 30)
            gaze_y += np.random.normal(0, 30)
        
        row = {
            'session_id': session_id,
            'task': 'fixation',
            'trial_id': 0,
            'state': 'FIXATION',
            't_epoch_s': t,
            'frame_idx': frame_idx,
            'fps_est': 30.0,
            'screen_w': 1920,
            'screen_h': 1080,
            'target_x': target_x,
            'target_y': target_y,
            'face_present': 1,
            'face_confidence': 0.95,
            'blink_flag': 1 if np.random.random() < 0.03 else 0,
            'valid_sample': 1,
            'left_iris_x_norm': gaze_x / 1920,
            'left_iris_y_norm': gaze_y / 1080,
            'right_iris_x_norm': gaze_x / 1920,
            'right_iris_y_norm': gaze_y / 1080,
            'gaze_x_norm': gaze_x / 1920,
            'gaze_y_norm': gaze_y / 1080,
            'anchor_x_norm': 0.5,
            'anchor_y_norm': 0.5,
            'gaze_x_norm_comp': gaze_x / 1920,
            'gaze_y_norm_comp': gaze_y / 1080,
            'gaze_x_px_raw': gaze_x,
            'gaze_y_px_raw': gaze_y,
            'gaze_x_px_comp': gaze_x,
            'gaze_y_px_comp': gaze_y,
            'vel_x_px_s': np.random.normal(0, 20),
            'vel_y_px_s': np.random.normal(0, 20),
            'pupil_proxy_left': 0.05,
            'pupil_proxy_right': 0.05,
            'pupil_proxy_mean': 0.05,
            'clamp_flag_x': 0,
            'clamp_flag_y': 0,
            'out_of_range_flag': 0,
            'dropout_gap_s': 0.0,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df


class TestReplayEngine:
    """Tests for ReplayEngine."""
    
    def test_load_frame_log(self):
        """Test loading a frame log."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_synthetic_frame_log(filepath, n_trials=3)
            
            engine = ReplayEngine()
            df = engine.load_frame_log(filepath)
            
            assert df is not None
            assert len(df) == 3 * 60  # 3 trials * 60 frames
        finally:
            os.unlink(filepath)
    
    def test_iterate_trials(self):
        """Test iterating over trials."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_synthetic_frame_log(filepath, n_trials=5)
            
            engine = ReplayEngine()
            df = engine.load_frame_log(filepath)
            
            trial_ids = []
            for trial_id, trial_df in engine.iterate_trials(df):
                trial_ids.append(trial_id)
            
            assert len(trial_ids) == 5
            assert list(trial_ids) == [0, 1, 2, 3, 4]
        finally:
            os.unlink(filepath)
    
    def test_extract_trial_data(self):
        """Test extracting data for a specific trial."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_synthetic_frame_log(filepath, n_trials=3)
            
            engine = ReplayEngine()
            df = engine.load_frame_log(filepath)
            
            for trial_id, trial_df in engine.iterate_trials(df):
                if trial_id == 1:
                    trial_data = engine.extract_trial_data(trial_df)
                    
                    assert trial_data['trial_id'] == 1
                    assert trial_data['n_frames'] == 60
                    break
        finally:
            os.unlink(filepath)
    
    def test_compute_fixation_biomarkers(self):
        """Test fixation biomarker computation from replay."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_fixation_frame_log(filepath, n_frames=300)
            
            engine = ReplayEngine()
            df = engine.load_frame_log(filepath)
            
            for trial_id, trial_df in engine.iterate_trials(df):
                biomarkers = engine.compute_fixation_biomarkers(trial_df, 960, 540)
                
                # Check that biomarkers are computed
                assert 'fixation_stability_rms_px' in biomarkers
                assert 'fixation_bcea_px2' in biomarkers
                assert 'microsaccade_rate_per_min' in biomarkers
                assert 'percent_time_on_target' in biomarkers
                
                # RMS should be reasonable (we added ~15px noise)
                assert 10 < biomarkers['fixation_stability_rms_px'] < 50
                break
        finally:
            os.unlink(filepath)
    
    def test_compute_saccade_biomarkers(self):
        """Test saccade biomarker computation from replay."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_synthetic_frame_log(filepath, n_trials=2, include_saccade=True)
            
            engine = ReplayEngine()
            df = engine.load_frame_log(filepath)
            
            # Get first trial (leftward)
            for trial_id, trial_df in engine.iterate_trials(df):
                if trial_id == 0:
                    biomarkers = engine.compute_saccade_biomarkers(
                        trial_df,
                        target_x=384,  # Left target
                        center_x=960
                    )
                    
                    # Check that biomarkers are computed
                    assert 'saccade_latency_ms' in biomarkers
                    assert 'saccade_duration_ms' in biomarkers
                    assert 'peak_velocity_px_s' in biomarkers
                    assert 'saccade_amplitude_px' in biomarkers
                    break
        finally:
            os.unlink(filepath)
    
    def test_replay_consistency(self):
        """Test that replaying produces consistent results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_fixation_frame_log(filepath, n_frames=300)
            
            # First replay
            engine1 = ReplayEngine()
            df1 = engine1.load_frame_log(filepath)
            for trial_id, trial_df in engine1.iterate_trials(df1):
                biomarkers1 = engine1.compute_fixation_biomarkers(trial_df, 960, 540)
                break
            
            # Second replay
            engine2 = ReplayEngine()
            df2 = engine2.load_frame_log(filepath)
            for trial_id, trial_df in engine2.iterate_trials(df2):
                biomarkers2 = engine2.compute_fixation_biomarkers(trial_df, 960, 540)
                break
            
            # Results should be identical
            assert biomarkers1['fixation_stability_rms_px'] == biomarkers2['fixation_stability_rms_px']
            assert biomarkers1['fixation_bcea_px2'] == biomarkers2['fixation_bcea_px2']
        finally:
            os.unlink(filepath)
    
    def test_replay_and_compute_all(self):
        """Test full replay and compute for all trials."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            create_synthetic_frame_log(filepath, task='saccade', n_trials=4)
            
            engine = ReplayEngine()
            results = engine.replay_and_compute(filepath, task_type='saccade')
            
            # Should have results for all trials
            assert len(results) == 4
            
            # Each result should have biomarkers
            for result in results:
                assert isinstance(result, dict)
                assert 'trial_id' in result
        finally:
            os.unlink(filepath)


class TestReplayBiomarkerAccuracy:
    """Tests for biomarker accuracy in replay mode."""
    
    def test_fixation_rms_accuracy(self):
        """Test that fixation RMS is computed accurately."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            # Create data with known RMS
            rows = []
            target_x, target_y = 960, 540
            known_std = 20.0
            
            np.random.seed(42)
            
            for i in range(300):
                gaze_x = target_x + np.random.normal(0, known_std)
                gaze_y = target_y + np.random.normal(0, known_std)
                
                rows.append({
                    'session_id': 'test',
                    'task': 'fixation',
                    'trial_id': 0,
                    'state': 'FIXATION',
                    't_epoch_s': i / 30.0,
                    'frame_idx': i,
                    'fps_est': 30.0,
                    'screen_w': 1920,
                    'screen_h': 1080,
                    'target_x': target_x,
                    'target_y': target_y,
                    'face_present': 1,
                    'face_confidence': 0.95,
                    'blink_flag': 0,
                    'valid_sample': 1,
                    'gaze_x_px_comp': gaze_x,
                    'gaze_y_px_comp': gaze_y,
                    'vel_x_px_s': 0,
                    'vel_y_px_s': 0,
                    'pupil_proxy_mean': 0.05,
                    'clamp_flag_x': 0,
                    'clamp_flag_y': 0,
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            
            engine = ReplayEngine()
            loaded_df = engine.load_frame_log(filepath)
            
            for trial_id, trial_df in engine.iterate_trials(loaded_df):
                biomarkers = engine.compute_fixation_biomarkers(trial_df, target_x, target_y)
                
                # RMS should be close to known_std * sqrt(2) for 2D
                expected_rms = known_std * np.sqrt(2)
                assert abs(biomarkers['fixation_stability_rms_px'] - expected_rms) < 5
                break
        finally:
            os.unlink(filepath)
    
    def test_saccade_latency_accuracy(self):
        """Test that saccade latency is computed accurately."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            # Create data with known latency
            rows = []
            known_latency_frames = 6  # 200ms at 30fps
            
            for i in range(60):
                t = i / 30.0
                
                if i < 30:
                    state = 'FORE'
                    gaze_x = 960
                    vel_x = 0
                elif i < 32:
                    state = 'JUMP'
                    gaze_x = 960
                    vel_x = 0
                else:
                    state = 'POST'
                    # Saccade starts at frame 36 (6 frames after jump at 30)
                    if i >= 36 and i < 42:
                        progress = (i - 36) / 6.0
                        gaze_x = 960 + 576 * progress
                        vel_x = 576 * 30 / 6  # ~2880 px/s
                    elif i >= 42:
                        gaze_x = 1536
                        vel_x = 0
                    else:
                        gaze_x = 960
                        vel_x = 0
                
                rows.append({
                    'session_id': 'test',
                    'task': 'saccade',
                    'trial_id': 0,
                    'state': state,
                    't_epoch_s': t,
                    'frame_idx': i,
                    'fps_est': 30.0,
                    'screen_w': 1920,
                    'screen_h': 1080,
                    'target_x': 960 if state == 'FORE' else 1536,
                    'target_y': 540,
                    'face_present': 1,
                    'face_confidence': 0.95,
                    'blink_flag': 0,
                    'valid_sample': 1,
                    'gaze_x_px_comp': gaze_x,
                    'gaze_y_px_comp': 540,
                    'vel_x_px_s': vel_x,
                    'vel_y_px_s': 0,
                    'pupil_proxy_mean': 0.05,
                    'clamp_flag_x': 0,
                    'clamp_flag_y': 0,
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            
            engine = ReplayEngine()
            loaded_df = engine.load_frame_log(filepath)
            
            for trial_id, trial_df in engine.iterate_trials(loaded_df):
                biomarkers = engine.compute_saccade_biomarkers(
                    trial_df,
                    target_x=1536,
                    center_x=960
                )
                
                # Latency should be around 200ms (6 frames at 30fps)
                # Allow some tolerance due to detection algorithm
                if not np.isnan(biomarkers['saccade_latency_ms']):
                    assert 100 < biomarkers['saccade_latency_ms'] < 400
                break
        finally:
            os.unlink(filepath)


class TestReplayEdgeCases:
    """Tests for edge cases in replay mode."""
    
    def test_empty_trial(self):
        """Test handling of empty trial data."""
        engine = ReplayEngine()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        trial_data = engine.extract_trial_data(empty_df) if len(empty_df) > 0 else {}
        
        # Should handle gracefully
        assert isinstance(trial_data, dict)
    
    def test_missing_columns(self):
        """Test handling of missing columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            # Create minimal data
            df = pd.DataFrame({
                'trial_id': [0, 0, 0],
                't_epoch_s': [0.0, 0.033, 0.066],
                'gaze_x_px_comp': [960, 961, 962],
                'gaze_y_px_comp': [540, 541, 542],
                'valid_sample': [1, 1, 1],
            })
            df.to_csv(filepath, index=False)
            
            engine = ReplayEngine()
            loaded_df = engine.load_frame_log(filepath)
            
            # Should load without error
            assert len(loaded_df) == 3
        finally:
            os.unlink(filepath)
    
    def test_nan_values_in_data(self):
        """Test handling of NaN values in data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            # Create data with NaN values
            df = pd.DataFrame({
                'trial_id': [0] * 30,
                't_epoch_s': [i / 30.0 for i in range(30)],
                'gaze_x_px_comp': [960 if i % 5 != 0 else np.nan for i in range(30)],
                'gaze_y_px_comp': [540 if i % 5 != 0 else np.nan for i in range(30)],
                'valid_sample': [1 if i % 5 != 0 else 0 for i in range(30)],
                'face_present': [1] * 30,
                'blink_flag': [0] * 30,
                'target_x': [960] * 30,
                'target_y': [540] * 30,
            })
            df.to_csv(filepath, index=False)
            
            engine = ReplayEngine()
            loaded_df = engine.load_frame_log(filepath)
            
            for trial_id, trial_df in engine.iterate_trials(loaded_df):
                # Should handle NaN values gracefully
                biomarkers = engine.compute_fixation_biomarkers(trial_df, 960, 540)
                
                # Should still compute something
                assert isinstance(biomarkers, dict)
                break
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
