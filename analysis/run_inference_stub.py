#!/usr/bin/env python3
"""
NeuroLens+ Inference Stub

Reads a session and outputs results.json with standardized format.
This is a stub for future ML model integration.

Usage:
    python analysis/run_inference_stub.py --session data/sessions/20251220_123456
    python analysis/run_inference_stub.py --session data/sessions/20251220_123456 --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.validate_session import validate_session, ValidationStatus
from tools.build_dataset import (
    load_session_metadata,
    compute_session_biomarkers,
    compute_derived_features,
    TASKS
)


# Results status levels
class ResultStatus:
    ALL_CLEAR = "ALL_CLEAR"
    POSSIBLE_CONCERN = "POSSIBLE_CONCERN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# Disclaimer text
DISCLAIMER = (
    "DISCLAIMER: NeuroLens+ is a research tool, NOT a medical device. "
    "These results are research-grade approximations based on webcam eye tracking "
    "and should NOT be used for clinical diagnosis. Consult a healthcare professional "
    "for any medical concerns. Results may be affected by lighting, camera quality, "
    "calibration accuracy, and individual variation."
)


def compute_confidence_score(
    validation_result,
    session_biomarkers: Dict[str, Any]
) -> float:
    """
    Compute overall confidence score for the session.
    
    Based on:
    - Number of tasks completed
    - Valid trial rates
    - Calibration quality
    - FPS stability
    """
    scores = []
    
    # Task completion score
    n_tasks = len(validation_result.tasks_found)
    task_score = n_tasks / len(TASKS)
    scores.append(task_score)
    
    # Valid rate scores per task
    for task_name in validation_result.tasks_found:
        valid_rate = validation_result.qc_summary.get(f'{task_name}_valid_rate', 0)
        scores.append(valid_rate)
    
    # Calibration score
    calib_error = validation_result.qc_summary.get('calibration_error_px', 200)
    calib_score = max(0, 1 - calib_error / 300)
    scores.append(calib_score)
    
    # FPS score
    fps = validation_result.qc_summary.get('fps_mean', 30)
    fps_score = min(1, fps / 30)
    scores.append(fps_score)
    
    return np.mean(scores) if scores else 0.0


def compute_biomarker_flags(
    session_biomarkers: Dict[str, Any],
    norms: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compute flags for biomarkers that may warrant attention.
    
    This is a stub - in production, this would use normative data
    and ML models to identify concerning patterns.
    """
    flags = []
    
    # Check fixation stability
    fix_rms = session_biomarkers.get('fixation_fixation_stability_rms_px')
    if fix_rms is not None and fix_rms > 200:
        flags.append({
            'biomarker': 'fixation_stability_rms_px',
            'value': fix_rms,
            'threshold': 200,
            'direction': 'high',
            'severity': 'mild',
            'description': 'Elevated fixation instability',
            'evidence': f'RMS deviation of {fix_rms:.1f}px exceeds typical range'
        })
    
    # Check saccade latency
    sacc_lat = session_biomarkers.get('saccade_saccade_latency_ms')
    if sacc_lat is not None and sacc_lat > 350:
        flags.append({
            'biomarker': 'saccade_latency_ms',
            'value': sacc_lat,
            'threshold': 350,
            'direction': 'high',
            'severity': 'mild',
            'description': 'Elevated saccade latency',
            'evidence': f'Mean latency of {sacc_lat:.0f}ms exceeds typical range'
        })
    
    # Check anti-saccade error rate
    anti_err = session_biomarkers.get('antisaccade_direction_error')
    if anti_err is not None and anti_err > 0.3:
        flags.append({
            'biomarker': 'antisaccade_direction_error',
            'value': anti_err,
            'threshold': 0.3,
            'direction': 'high',
            'severity': 'moderate' if anti_err > 0.5 else 'mild',
            'description': 'Elevated anti-saccade error rate',
            'evidence': f'Error rate of {anti_err:.1%} suggests reduced inhibitory control'
        })
    
    # Check pursuit gain
    pursuit_gain = session_biomarkers.get('pursuit_pursuit_gain')
    if pursuit_gain is not None:
        if pursuit_gain < 0.7:
            flags.append({
                'biomarker': 'pursuit_gain',
                'value': pursuit_gain,
                'threshold': 0.7,
                'direction': 'low',
                'severity': 'mild',
                'description': 'Low pursuit gain',
                'evidence': f'Gain of {pursuit_gain:.2f} indicates eyes not keeping up with target'
            })
        elif pursuit_gain > 1.3:
            flags.append({
                'biomarker': 'pursuit_gain',
                'value': pursuit_gain,
                'threshold': 1.3,
                'direction': 'high',
                'severity': 'mild',
                'description': 'High pursuit gain',
                'evidence': f'Gain of {pursuit_gain:.2f} indicates overshooting target'
            })
    
    # Check blink rate
    blink_rate = session_biomarkers.get('visual_search_blink_rate_per_min')
    if blink_rate is not None:
        if blink_rate < 8:
            flags.append({
                'biomarker': 'blink_rate_per_min',
                'value': blink_rate,
                'threshold': 8,
                'direction': 'low',
                'severity': 'mild',
                'description': 'Low spontaneous blink rate',
                'evidence': f'Rate of {blink_rate:.1f}/min is below typical range'
            })
        elif blink_rate > 35:
            flags.append({
                'biomarker': 'blink_rate_per_min',
                'value': blink_rate,
                'threshold': 35,
                'direction': 'high',
                'severity': 'mild',
                'description': 'High spontaneous blink rate',
                'evidence': f'Rate of {blink_rate:.1f}/min is above typical range'
            })
    
    return flags


def determine_overall_status(
    validation_result,
    flags: List[Dict[str, Any]],
    confidence: float
) -> str:
    """Determine overall result status."""
    
    # Insufficient data if validation failed or low confidence
    if validation_result.status == ValidationStatus.FAIL:
        return ResultStatus.INSUFFICIENT_DATA
    
    if confidence < 0.4:
        return ResultStatus.INSUFFICIENT_DATA
    
    # Possible concern if any moderate+ flags
    moderate_flags = [f for f in flags if f.get('severity') in ['moderate', 'severe']]
    if moderate_flags:
        return ResultStatus.POSSIBLE_CONCERN
    
    # Possible concern if many mild flags
    if len(flags) >= 3:
        return ResultStatus.POSSIBLE_CONCERN
    
    return ResultStatus.ALL_CLEAR


def run_inference(
    session_dir: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run inference on a session and generate results.json.
    
    Args:
        session_dir: Path to session directory
        output_file: Optional path to output file
    
    Returns:
        Results dictionary
    """
    session_path = Path(session_dir)
    
    # Validate session
    validation = validate_session(session_dir, strict=False)
    
    # Load metadata
    meta = load_session_metadata(session_path)
    if meta is None:
        meta = {}
    
    # Compute session biomarkers
    session_biomarkers = {
        'session_id': meta.get('session_id', session_path.name),
        'timestamp': meta.get('timestamp', ''),
    }
    
    for task_name in TASKS.keys():
        biomarkers = compute_session_biomarkers(session_path, task_name)
        for key, value in biomarkers.items():
            session_biomarkers[f'{task_name}_{key}'] = value
    
    # Compute derived features
    derived = compute_derived_features(session_biomarkers)
    session_biomarkers.update(derived)
    
    # Compute confidence score
    confidence = compute_confidence_score(validation, session_biomarkers)
    
    # Compute biomarker flags
    flags = compute_biomarker_flags(session_biomarkers)
    
    # Determine overall status
    overall_status = determine_overall_status(validation, flags, confidence)
    
    # Build per-task metrics
    per_task_metrics = {}
    for task_name in validation.tasks_found:
        task_metrics = {
            'n_trials': validation.qc_summary.get(f'{task_name}_n_trials', 0),
            'n_valid': validation.qc_summary.get(f'{task_name}_n_valid', 0),
            'valid_rate': validation.qc_summary.get(f'{task_name}_valid_rate', 0),
            'qc_status': validation.qc_summary.get(f'{task_name}_status', 'UNKNOWN'),
        }
        
        # Add task-specific biomarkers
        for key, value in session_biomarkers.items():
            if key.startswith(f'{task_name}_') and not key.endswith('_status'):
                biomarker_name = key[len(task_name)+1:]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    task_metrics[biomarker_name] = value
        
        per_task_metrics[task_name] = task_metrics
    
    # Build results
    results = {
        'version': '1.0.0',
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'session_id': meta.get('session_id', session_path.name),
        
        'overall': {
            'status': overall_status,
            'confidence_score': round(confidence, 3),
            'tasks_completed': len(validation.tasks_found),
            'tasks_required': len(TASKS),
        },
        
        'qc_summary': {
            'validation_status': validation.status.value,
            'calibration_accepted': validation.qc_summary.get('calibration_accepted', False),
            'calibration_error_px': validation.qc_summary.get('calibration_error_px'),
            'fps_mean': validation.qc_summary.get('fps_mean'),
            'failed_checks': validation.errors,
            'warnings': validation.warnings,
        },
        
        'per_task_metrics': per_task_metrics,
        
        'derived_scores': {
            'oculomotor_health_score': derived.get('oculomotor_health_score'),
            'inhibitory_control_index': derived.get('inhibitory_control_index'),
            'pursuit_quality': derived.get('pursuit_quality'),
            'fixation_quality': derived.get('fixation_quality'),
            'blink_regularity': derived.get('blink_regularity'),
        },
        
        'flags': flags,
        
        'disclaimer': DISCLAIMER,
    }
    
    # Clean up None values
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [clean_dict(i) for i in d]
        elif isinstance(d, float) and np.isnan(d):
            return None
        return d
    
    results = clean_dict(results)
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on NeuroLens+ session and generate results.json"
    )
    
    parser.add_argument(
        '--session', '-s',
        type=str,
        required=True,
        help='Path to session directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: {session_dir}/results.json)'
    )
    
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print results to stdout'
    )
    
    args = parser.parse_args()
    
    # Default output path
    output_file = args.output
    if output_file is None:
        output_file = str(Path(args.session) / 'results.json')
    
    results = run_inference(args.session, output_file)
    
    if args.print:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code based on status
    if results['overall']['status'] == ResultStatus.INSUFFICIENT_DATA:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
