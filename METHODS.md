# NeuroLens+ Research Methods Documentation

This document provides research-ready documentation of the biomarkers, validity criteria, calibration procedures, and known limitations of the NeuroLens+ eye tracking platform.

## 1. Eye Tracking Technology

### 1.1 Hardware

NeuroLens+ uses standard laptop webcams (typically 720p or 1080p resolution) operating at approximately 30 frames per second. No specialized eye tracking hardware is required.

### 1.2 Software Pipeline

Eye tracking is performed using Google's MediaPipe FaceMesh model with iris refinement enabled (`refine_landmarks=True`). This provides 478 facial landmarks including detailed iris landmarks for both eyes.

**Gaze Estimation:**
- Iris center positions are extracted from MediaPipe landmarks (indices 468-477)
- Gaze position is computed as the mean of left and right iris centers
- Positions are normalized to [0, 1] range relative to the face bounding box
- Screen coordinates are derived via linear calibration mapping

### 1.3 Sampling Rate

The effective sampling rate depends on webcam capabilities and system performance. NeuroLens+ monitors frame rate continuously and flags sessions where FPS drops below 15 Hz for more than 20% of the recording.

## 2. Calibration Procedure

### 2.1 Calibration Modes

NeuroLens+ supports multiple calibration modes:

| Mode | Points | Use Case |
|------|--------|----------|
| Center | 1 | Baseline only |
| 3-point | 3 | Horizontal tasks (saccade) |
| 5-point | 5 | General use |
| 9-point | 9 | Grid accuracy assessment |

### 2.2 Calibration Process

1. **Baseline Collection (2 seconds)**
   - Participant fixates on center target
   - Baseline gaze position (norm) is recorded
   - Baseline anchor position (nose tip) is recorded

2. **Point Collection**
   - Each calibration point is displayed for 2 seconds
   - Gaze samples are collected during stable fixation
   - Minimum 20 valid samples required per point

3. **Mapping Computation**
   - Linear regression fits gaze_norm to screen_px
   - Separate slopes and intercepts for X and Y axes
   - R² values computed for quality assessment

### 2.3 Degeneracy Detection

Calibration is rejected if:
- Gaze range < 0.01 normalized units between any adjacent points
- Computed slope magnitude < 100 (insufficient sensitivity)
- Mean calibration error > 120 pixels
- R² < 0.5 for either axis

### 2.4 Head Motion Compensation

To account for head movement during tasks:

```
gaze_compensated = gaze_raw - (anchor_current - anchor_baseline)
```

Where `anchor` is the nose tip position (landmark 1) or mid-eye corner average.

## 3. Biomarker Definitions

### 3.1 Fixation Task Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| fixation_stability_rms_px | pixels | Root mean square of gaze deviation from target |
| fixation_bcea_px2 | pixels² | Bivariate Contour Ellipse Area (68% confidence) |
| microsaccade_rate_per_min | events/min | Small rapid eye movements (velocity > 50 px/s) |
| drift_velocity_px_s | pixels/s | Linear trend in gaze position over time |
| percent_time_on_target | % | Proportion of samples within target radius |
| blink_rate_per_min | blinks/min | Spontaneous blink frequency |

**BCEA Calculation:**
```
BCEA = 2π × k × σx × σy × √(1 - ρ²)
```
Where k=1.14 for 68% confidence, σx/σy are standard deviations, and ρ is correlation.

### 3.2 Saccade Task Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| saccade_latency_ms | milliseconds | Time from target jump to saccade onset |
| saccade_duration_ms | milliseconds | Time from saccade onset to landing |
| peak_velocity_px_s | pixels/s | Maximum velocity during saccade |
| saccade_amplitude_px | pixels | Displacement from onset to landing (signed) |
| gain | ratio | amplitude / target_eccentricity |
| landing_error_px | pixels | Distance from landing position to target |
| overshoot_px | pixels | Positive landing error (past target) |
| undershoot_px | pixels | Negative landing error (short of target) |
| corrective_saccade_count | count | Secondary saccades within 500ms of landing |

**Saccade Detection Algorithm:**
1. Onset: First sample where velocity exceeds threshold (30 px/s) in correct direction
2. Peak: Maximum velocity between onset and landing
3. Landing: First sample where velocity drops below threshold after peak

### 3.3 Anti-Saccade Task Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| antisaccade_latency_ms | milliseconds | Time to correct (opposite) saccade |
| direction_error | binary | 1 if initial movement toward stimulus |
| correction_time_ms | milliseconds | Time to correct after direction error |
| inhibition_success | binary | 1 if no direction error |
| peak_velocity_px_s | pixels/s | Maximum velocity of antisaccade |
| amplitude_px | pixels | Antisaccade amplitude |
| landing_error_px | pixels | Distance from correct target |

### 3.4 Smooth Pursuit Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| pursuit_gain | ratio | eye_velocity / target_velocity |
| pursuit_latency_ms | milliseconds | Time from target motion to pursuit onset |
| catch_up_saccade_count | count | Corrective saccades during pursuit |
| catch_up_saccade_rate_per_s | events/s | Catch-up saccade frequency |
| position_error_mean_px | pixels | Mean absolute position error |
| position_error_rmse_px | pixels | Root mean square position error |
| phase_lag_ms | milliseconds | Temporal lag from cross-correlation |

**Gain Calculation:**
```
gain = mean(|eye_velocity|) / mean(|target_velocity|)
```
Computed over steady-state pursuit segments (excluding catch-up saccades).

### 3.5 Pupillary Light Reflex (PLR) Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| constriction_latency_ms | milliseconds | Time from flash to minimum pupil |
| constriction_amplitude | normalized | baseline_pupil - min_pupil |
| constriction_velocity | units/s | Maximum negative derivative |
| dilation_recovery_time_ms | milliseconds | Time to 75% recovery |
| baseline_pupil_proxy | normalized | Pre-flash pupil size |
| min_pupil_proxy | normalized | Minimum pupil during constriction |
| asymmetry_left_right | ratio | |left - right| / mean |

**Note:** NeuroLens+ uses iris diameter as a pupil proxy since webcams cannot directly measure pupil size. This provides relative changes but not absolute measurements.

### 3.6 9-Point Grid Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| mean_gaze_error_px | pixels | Mean distance from target per point |
| rmse_gaze_error_px | pixels | RMSE of gaze error per point |
| dwell_stability_rms_px | pixels | RMS of gaze during fixation |
| grid_accuracy_mean_px | pixels | Mean error across all points |
| grid_accuracy_max_px | pixels | Maximum error across points |
| systematic_bias_x_px | pixels | Mean X offset |
| systematic_bias_y_px | pixels | Mean Y offset |
| gaze_map_linearity_r2 | ratio | R² of gaze-to-target mapping |

### 3.7 Blink Monitoring Biomarkers

| Biomarker | Unit | Definition |
|-----------|------|------------|
| blink_rate_per_min | blinks/min | Spontaneous blink frequency |
| blink_duration_mean_ms | milliseconds | Mean blink duration |
| blink_duration_p95_ms | milliseconds | 95th percentile duration |
| interblink_interval_mean_s | seconds | Mean time between blinks |
| blink_irregularity_cv | ratio | Coefficient of variation of intervals |

**Blink Detection:**
Blinks are detected using Eye Aspect Ratio (EAR):
```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```
Where p1-p6 are eye landmarks. Blink detected when EAR < 0.21.

## 4. Validity Criteria

### 4.1 Per-Sample Validity

A sample is marked valid if:
- Face is detected (face_present = 1)
- Not during blink (blink_flag = 0)
- Frame interval is reasonable (dt > 0.005s and dt < 0.2s)

### 4.2 Per-Trial Validity

| Criterion | Threshold | Reason Code |
|-----------|-----------|-------------|
| Face presence | ≥ 70% | NO_FACE |
| Blink rate | ≤ 30% | TOO_MANY_BLINKS |
| Median FPS | ≥ 15 Hz | LOW_FPS |
| Clamp rate | ≤ 5% | CLAMP_RATE_HIGH |
| Valid samples | ≥ 50% | INSUFFICIENT_SAMPLES |

### 4.3 Saccade-Specific Validity

| Criterion | Range | Reason Code |
|-----------|-------|-------------|
| Latency | 50-900 ms | LATENCY_OUT_OF_RANGE |
| Duration | 15-400 ms | DURATION_OUT_OF_RANGE |
| Amplitude | ≥ 15% of eccentricity | NO_MEANINGFUL_MOVEMENT |
| Direction | Matches target | WRONG_DIRECTION |
| Peak velocity | < 5000 px/s | PEAK_VEL_SPIKE |
| Critical dropout | No dropout in first 250ms | DROPOUT_DURING_CRITICAL_WINDOW |

### 4.4 Reason Codes

```
VALID                        - Trial passed all checks
NO_FACE                      - Face not detected
TOO_MANY_BLINKS             - Excessive blinking (>30%)
LOW_FPS                      - Frame rate below 15 Hz
CALIBRATION_REJECTED         - Calibration failed
CLAMP_RATE_HIGH             - >5% samples clamped
NO_MEANINGFUL_MOVEMENT       - Amplitude too small
WRONG_DIRECTION             - Movement opposite to target
LATENCY_OUT_OF_RANGE        - Response too fast/slow
DURATION_OUT_OF_RANGE       - Movement too short/long
PEAK_VEL_SPIKE              - Unrealistic velocity
DROPOUT_DURING_CRITICAL_WINDOW - Face lost during response
INSUFFICIENT_SAMPLES         - Too few valid samples
TARGET_NOT_REACHED          - Did not reach target zone
ANTICIPATORY_RESPONSE       - Response before stimulus
SKIPPED                     - User skipped trial
OTHER                       - Unspecified reason
```

## 5. Quality Metrics

### 5.1 Quality Score

Each trial receives a quality score (0-1) computed as:

```
quality_score = w1×valid_fraction + w2×calibration_quality + 
                w3×(1-clamp_rate) + w4×fps_ratio + w5×face_presence
```

Default weights: w1=0.3, w2=0.2, w3=0.2, w4=0.15, w5=0.15

### 5.2 Session Quality

Session-level metrics include:
- Mean quality score across trials
- Valid trial rate per task
- Calibration acceptance status
- FPS statistics (mean, median, min, max)
- Overall clamp rate

## 6. Known Limitations

### 6.1 Spatial Accuracy

Webcam-based eye tracking provides lower spatial accuracy than research-grade systems:
- Typical accuracy: 2-5° visual angle
- Precision: ~1° visual angle
- Affected by: lighting, head position, glasses, eye makeup

### 6.2 Temporal Resolution

Standard webcams operate at 30 Hz, limiting:
- Minimum detectable saccade duration: ~33ms
- Velocity estimation precision
- Fine temporal dynamics

### 6.3 Pupil Measurement

NeuroLens+ cannot directly measure pupil size. The iris diameter proxy:
- Correlates with pupil changes
- Does not provide absolute measurements
- May be affected by iris color and lighting

### 6.4 Environmental Factors

Performance is affected by:
- Ambient lighting (too bright or too dark)
- Screen glare
- Glasses reflections
- Head movement range
- Distance from camera

## 7. Mitigation Strategies

### 7.1 Calibration Quality

- Reject degenerate calibrations automatically
- Provide recalibration option (R key)
- Monitor calibration error in real-time

### 7.2 Head Motion

- Continuous anchor point tracking
- Real-time compensation
- Flag excessive drift

### 7.3 Data Quality

- Per-sample validity flags
- Explicit reason codes for invalid trials
- Quality score for confidence assessment

### 7.4 Recommendations

For optimal results:
1. Ensure consistent, diffuse lighting
2. Position face 50-70cm from camera
3. Minimize head movement
4. Remove glasses if possible
5. Recalibrate if quality drops

## 8. Data Output Format

### 8.1 Frame Log (CSV)

33 columns per frame including:
- Timing: t_epoch_s, frame_idx, fps_est
- Gaze: gaze_x_norm, gaze_y_norm, gaze_x_px_comp, gaze_y_px_comp
- Velocity: vel_x_px_s, vel_y_px_s
- Quality: valid_sample, clamp_flag_x, clamp_flag_y
- Pupil: pupil_proxy_left, pupil_proxy_right, pupil_proxy_mean

### 8.2 Summary Log (CSV)

Task-specific biomarkers plus:
- trial_id, trial_start_s, trial_end_s
- valid, invalid_reason, quality_score
- calibration_quality, fps_median, clamp_rate

### 8.3 Metadata (JSON)

Session information including:
- Timestamp, device info, screen size
- Calibration parameters and quality
- FPS statistics
- Valid rates per task

## 9. Reproducibility

### 9.1 Replay Mode

Frame logs can be replayed to recompute biomarkers:
```python
from core.replay import ReplayEngine
engine = ReplayEngine()
results = engine.replay_and_compute('frame_log.csv', task='saccade')
```

### 9.2 Deterministic Processing

Given identical frame logs, biomarker computation is deterministic. Random elements (foreperiod timing, trial order) are logged for reproducibility.

## 10. References

1. MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh
2. Eye Aspect Ratio for blink detection: Soukupová & Čech (2016)
3. BCEA calculation: Steinman (1965)
4. Saccade main sequence: Bahill et al. (1975)

## 11. Version Information

- NeuroLens+ Version: 1.0.0
- MediaPipe Version: ≥0.10.0
- Python Version: ≥3.11

---

*This documentation is intended for research use. For clinical applications, additional validation against gold-standard eye tracking systems is recommended.*
