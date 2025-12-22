# NeuroLens+ Feature Dictionary

This document describes all biomarkers collected by NeuroLens+ and their clinical relevance for neurological assessment.

## Overview

NeuroLens+ collects oculomotor and blink biomarkers across 6 standardized tasks. These biomarkers, when combined, can provide insights into neurological function. However, this is a research tool and NOT a medical diagnostic device.

## Task-Biomarker Matrix

| Task | Primary Biomarkers | Neurological Relevance |
|------|-------------------|----------------------|
| Fixation | Stability, Drift, BCEA | Cerebellar function, attention |
| Smooth Pursuit | Gain, Phase Lag, Catch-up Saccades | Cerebellar/brainstem function |
| Saccade | Latency, Peak Velocity, Amplitude | Frontal/brainstem function |
| Anti-Saccade | Error Rate, Inhibition Success | Frontal lobe inhibitory control |
| 9-Point Grid | Accuracy, Asymmetry | Calibration quality, visual field |
| Visual Search | Blink Rate, Variability | Dopaminergic function, fatigue |

---

## Fixation Task Biomarkers

### fixation_stability_rms_px
- **Units**: pixels
- **Computation**: Root mean square of gaze position deviation from target
- **Formula**: `sqrt(mean((gaze_x - target_x)^2 + (gaze_y - target_y)^2))`
- **Normal Range**: < 50 px (with good calibration)
- **Clinical Relevance**: Elevated instability may indicate cerebellar dysfunction, nystagmus, or attention deficits

### fixation_bcea_px2
- **Units**: pixels squared
- **Computation**: Bivariate Contour Ellipse Area - 68% confidence ellipse of gaze distribution
- **Formula**: `2.291 * pi * std_x * std_y * sqrt(1 - correlation^2)`
- **Normal Range**: < 2000 px^2
- **Clinical Relevance**: Larger BCEA indicates poorer fixation control

### microsaccade_rate_per_min
- **Units**: count per minute
- **Computation**: Number of small saccades (< 1 degree) during fixation
- **Normal Range**: 1-3 per second (60-180 per minute)
- **Clinical Relevance**: Abnormal rates may indicate attention or cerebellar issues

### drift_velocity_px_s
- **Units**: pixels per second
- **Computation**: Mean velocity of slow eye movements during fixation
- **Normal Range**: < 5 px/s
- **Clinical Relevance**: Elevated drift may indicate vestibular or cerebellar dysfunction

### percent_time_on_target
- **Units**: percentage (0-100)
- **Computation**: Fraction of time gaze was within threshold of target
- **Normal Range**: > 80%
- **Clinical Relevance**: Low values indicate poor sustained attention or tracking

---

## Smooth Pursuit Task Biomarkers

### pursuit_gain
- **Units**: dimensionless ratio
- **Computation**: Ratio of eye velocity to target velocity
- **Formula**: `mean(eye_velocity) / target_velocity`
- **Normal Range**: 0.85 - 1.0
- **Clinical Relevance**: Low gain indicates pursuit deficit (cerebellar, brainstem, or medication effects)

### pursuit_latency_ms
- **Units**: milliseconds
- **Computation**: Time from target motion onset to pursuit initiation
- **Normal Range**: 100-150 ms
- **Clinical Relevance**: Prolonged latency may indicate processing delays

### catch_up_saccade_count
- **Units**: count
- **Computation**: Number of saccades made to catch up with moving target
- **Normal Range**: 0-3 per trial
- **Clinical Relevance**: Excessive catch-up saccades indicate poor pursuit

### position_error_rmse_px
- **Units**: pixels
- **Computation**: Root mean square error between gaze and target position
- **Normal Range**: < 50 px
- **Clinical Relevance**: Higher error indicates poorer tracking

### phase_lag_ms
- **Units**: milliseconds
- **Computation**: Temporal delay between gaze and target (via cross-correlation)
- **Normal Range**: 0-50 ms
- **Clinical Relevance**: Increased lag may indicate processing delays

---

## Saccade Task Biomarkers

### saccade_latency_ms
- **Units**: milliseconds
- **Computation**: Time from target appearance to saccade onset
- **Detection**: Displacement-based onset detection with velocity confirmation
- **Normal Range**: 150-250 ms (visually-guided)
- **Clinical Relevance**: Prolonged latency may indicate frontal lobe or attention deficits

### saccade_duration_ms
- **Units**: milliseconds
- **Computation**: Time from saccade onset to landing
- **Normal Range**: 30-100 ms (amplitude-dependent)
- **Clinical Relevance**: Abnormal duration may indicate brainstem dysfunction

### peak_velocity_px_s
- **Units**: pixels per second
- **Computation**: 95th percentile of velocity during saccade (noise-robust)
- **Normal Range**: Amplitude-dependent (main sequence relationship)
- **Clinical Relevance**: Reduced peak velocity may indicate fatigue or neurological issues

### saccade_amplitude_px
- **Units**: pixels
- **Computation**: Distance from saccade onset to landing position
- **Normal Range**: Should match target eccentricity
- **Clinical Relevance**: Hypometric/hypermetric saccades indicate cerebellar dysfunction

### gain
- **Units**: dimensionless ratio
- **Computation**: `amplitude / eccentricity`
- **Normal Range**: 0.9 - 1.1
- **Clinical Relevance**: Gain < 0.9 (hypometric) or > 1.1 (hypermetric) is abnormal

### landing_error_px
- **Units**: pixels
- **Computation**: Distance from landing position to target
- **Normal Range**: < 50 px
- **Clinical Relevance**: Large errors indicate poor saccade accuracy

---

## Anti-Saccade Task Biomarkers

### antisaccade_latency_ms
- **Units**: milliseconds
- **Computation**: Time from stimulus to correct saccade onset
- **Normal Range**: 200-350 ms (longer than pro-saccades)
- **Clinical Relevance**: Prolonged latency may indicate frontal dysfunction

### direction_error
- **Units**: binary (0/1)
- **Computation**: 1 if initial saccade was toward stimulus (error), 0 if correct
- **Normal Range**: Error rate 5-20% in healthy adults
- **Clinical Relevance**: High error rate indicates poor inhibitory control (frontal lobe)

### inhibition_success
- **Units**: binary (0/1)
- **Computation**: 1 if first saccade was away from stimulus (correct)
- **Normal Range**: > 80% success rate
- **Clinical Relevance**: Low success indicates frontal lobe dysfunction

### correction_time_ms
- **Units**: milliseconds
- **Computation**: Time from error saccade to corrective saccade (if error occurred)
- **Normal Range**: 100-300 ms
- **Clinical Relevance**: Slow correction may indicate monitoring deficits

---

## 9-Point Grid Task Biomarkers

### mean_gaze_error_px
- **Units**: pixels
- **Computation**: Mean Euclidean distance from gaze to target per point
- **Normal Range**: < 100 px (calibration-dependent)
- **Clinical Relevance**: Primarily a calibration quality metric

### rmse_gaze_error_px
- **Units**: pixels
- **Computation**: Root mean square error of gaze from target
- **Normal Range**: < 120 px
- **Clinical Relevance**: Calibration quality and gaze stability

### dwell_stability_rms_px
- **Units**: pixels
- **Computation**: RMS of gaze position during dwell at each point
- **Normal Range**: < 50 px
- **Clinical Relevance**: Fixation stability at different gaze angles

### grid_accuracy_mean_px
- **Units**: pixels
- **Computation**: Mean error across all 9 points
- **Normal Range**: < 100 px
- **Clinical Relevance**: Overall gaze mapping accuracy

### gaze_map_linearity_r2
- **Units**: dimensionless (0-1)
- **Computation**: R-squared of linear regression (gaze vs target)
- **Normal Range**: > 0.9
- **Clinical Relevance**: Linearity of gaze mapping

---

## Visual Search Task Biomarkers (Covert Blink Measurement)

### blink_rate_per_min
- **Units**: blinks per minute
- **Computation**: Count of blinks / duration in minutes
- **Normal Range**: 15-20 blinks/min at rest
- **Clinical Relevance**: Reduced rate may indicate Parkinson's disease; elevated rate may indicate fatigue or dry eye

### blink_duration_mean_ms
- **Units**: milliseconds
- **Computation**: Mean duration of eye closure during blinks
- **Normal Range**: 100-400 ms
- **Clinical Relevance**: Prolonged blinks may indicate fatigue or neurological issues

### interblink_interval_mean_s
- **Units**: seconds
- **Computation**: Mean time between consecutive blinks
- **Normal Range**: 3-6 seconds
- **Clinical Relevance**: Regularity of blink timing

### interblink_interval_cv
- **Units**: dimensionless (coefficient of variation)
- **Computation**: `std(intervals) / mean(intervals)`
- **Normal Range**: 0.3-0.6
- **Clinical Relevance**: High variability may indicate attention fluctuations

### blink_burstiness
- **Units**: dimensionless (0-1)
- **Computation**: Fraction of intervals shorter than 50% of mean
- **Normal Range**: < 0.3
- **Clinical Relevance**: High burstiness indicates clustered blinking

### gaze_presence_pct
- **Units**: percentage (0-100)
- **Computation**: Fraction of time with valid gaze tracking
- **Normal Range**: > 80%
- **Clinical Relevance**: Engagement and tracking quality metric

---

## Quality Control Metrics

### valid_fraction
- **Units**: dimensionless (0-1)
- **Computation**: Fraction of samples with valid tracking
- **Threshold**: > 0.7 for valid trial

### calibration_quality
- **Units**: dimensionless (0-1)
- **Computation**: Quality score from calibration procedure
- **Threshold**: > 0.5 recommended

### fps_median
- **Units**: frames per second
- **Computation**: Median frame rate during trial
- **Threshold**: > 15 fps required

### clamp_rate
- **Units**: dimensionless (0-1)
- **Computation**: Fraction of samples hitting screen boundaries
- **Threshold**: < 0.25 per trial

---

## Neurological Condition Relevance Matrix

| Condition | Key Biomarkers | Expected Pattern |
|-----------|---------------|------------------|
| Parkinson's Disease | Blink rate, Saccade latency, Pursuit gain | Reduced blink rate, prolonged latency, reduced gain |
| Cerebellar Ataxia | Pursuit gain, Saccade accuracy, Fixation stability | Poor pursuit, dysmetric saccades, unstable fixation |
| Frontal Lobe Lesion | Anti-saccade errors, Saccade latency | High error rate, prolonged latency |
| ADHD | Fixation stability, Anti-saccade errors | Unstable fixation, elevated errors |
| Fatigue/Drowsiness | Blink rate, Saccade velocity | Increased blink rate, reduced velocity |
| Multiple Sclerosis | Pursuit gain, Saccade latency | Variable deficits depending on lesion location |

---

## Important Disclaimers

1. **Research Tool Only**: NeuroLens+ is NOT a medical diagnostic device. Results should not be used for clinical diagnosis.

2. **Webcam Limitations**: Webcam-based tracking has lower precision (~2-5 degrees) than clinical eye trackers (~0.5 degrees). Spatial metrics (amplitude, landing error) are less reliable than temporal metrics (latency).

3. **Calibration Dependency**: All spatial biomarkers depend heavily on calibration quality. Poor calibration invalidates spatial measurements.

4. **Individual Variation**: Normal ranges vary significantly between individuals. Within-subject comparisons are more reliable than absolute thresholds.

5. **Environmental Factors**: Lighting, head position, and screen distance affect tracking quality. Standardize conditions for reliable measurements.
