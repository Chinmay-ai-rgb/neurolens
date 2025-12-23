#!/usr/bin/env python3
"""
NeuroLens+ Session Validator

Validates that a session meets quality thresholds for ML readiness.
Checks:
- All required tasks completed
- QC thresholds met
- Data integrity

Usage:
    python tools/validate_session.py --session data/sessions/20251220_123456
    python tools/validate_session.py --session data/sessions/20251220_123456 --strict
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ValidationStatus(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class ValidationResult:
    """Result of session validation."""
    status: ValidationStatus = ValidationStatus.FAIL
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    tasks_found: List[str] = field(default_factory=list)
    tasks_missing: List[str] = field(default_factory=list)
    qc_summary: Dict[str, Any] = field(default_factory=dict)


# Required tasks for a complete session
REQUIRED_TASKS = ['fixation', 'pursuit', 'saccade', 'antisaccade', 'grid9', 'visual_search']

# Task summary file patterns
TASK_FILES = {
    'fixation': 'summary_fixation.csv',
    'pursuit': 'summary_pursuit.csv',
    'saccade': 'summary_saccade.csv',
    'antisaccade': 'summary_antisaccade.csv',
    'grid9': 'summary_grid9.csv',
    'visual_search': 'summary_visual_search.csv',
}

# QC thresholds
QC_THRESHOLDS = {
    'min_valid_rate': 0.5,  # Minimum valid trial rate per task
    'min_fps': 15.0,  # Minimum acceptable FPS
    'max_calibration_error_px': 200.0,  # Maximum calibration error
    'min_tasks_required': 3,  # Minimum tasks for partial session
    'min_tasks_strict': 6,  # All tasks for strict mode
}


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_csv_summary(path: Path) -> List[Dict[str, Any]]:
    """Load CSV summary file."""
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def check_task_quality(task_name: str, summary_rows: List[Dict[str, Any]]) -> Tuple[ValidationStatus, List[str]]:
    """Check quality of a single task."""
    issues = []
    
    if not summary_rows:
        return ValidationStatus.FAIL, ["No trial data found"]
    
    n_trials = len(summary_rows)
    n_valid = sum(1 for row in summary_rows if row.get('valid') == '1' or row.get('valid') == 1)
    valid_rate = n_valid / n_trials if n_trials > 0 else 0
    
    if valid_rate < QC_THRESHOLDS['min_valid_rate']:
        issues.append(f"Low valid rate: {valid_rate:.1%} (threshold: {QC_THRESHOLDS['min_valid_rate']:.0%})")
    
    # Check for specific QC flags in visual_search
    if task_name == 'visual_search':
        fail_count = sum(1 for row in summary_rows if row.get('qc_status') == 'FAIL')
        if fail_count > n_trials * 0.5:
            issues.append(f"High QC failure rate: {fail_count}/{n_trials} trials failed QC")
    
    if issues:
        return ValidationStatus.WARN if valid_rate >= 0.3 else ValidationStatus.FAIL, issues
    
    return ValidationStatus.PASS, []


def validate_session(
    session_dir: str,
    strict: bool = False
) -> ValidationResult:
    """
    Validate a session for ML readiness.
    
    Args:
        session_dir: Path to session directory
        strict: If True, require all tasks; if False, allow partial sessions
    
    Returns:
        ValidationResult with status and details
    """
    result = ValidationResult()
    session_path = Path(session_dir)
    
    # Check session directory exists
    if not session_path.exists():
        result.errors.append(f"Session directory not found: {session_dir}")
        return result
    
    # Check metadata
    meta_path = session_path / 'meta.json'
    if not meta_path.exists():
        result.errors.append("Missing meta.json")
    else:
        try:
            meta = load_json(meta_path)
            result.info.append(f"Session ID: {meta.get('session_id', 'unknown')}")
            result.info.append(f"Timestamp: {meta.get('timestamp', 'unknown')}")
            
            # Check calibration
            if not meta.get('calibration_accepted', False):
                result.warnings.append("Calibration was not accepted")
            
            calib_error = meta.get('calibration_error_mean_px', 0)
            if calib_error > QC_THRESHOLDS['max_calibration_error_px']:
                result.warnings.append(f"High calibration error: {calib_error:.1f}px")
            
            # Check FPS
            fps_mean = meta.get('fps_mean', 30)
            if fps_mean < QC_THRESHOLDS['min_fps']:
                result.warnings.append(f"Low FPS: {fps_mean:.1f}")
            
            result.qc_summary['calibration_accepted'] = meta.get('calibration_accepted', False)
            result.qc_summary['calibration_error_px'] = calib_error
            result.qc_summary['fps_mean'] = fps_mean
            
        except Exception as e:
            result.errors.append(f"Error reading meta.json: {e}")
    
    # Check each task
    task_statuses = {}
    for task_name, summary_file in TASK_FILES.items():
        summary_path = session_path / summary_file
        
        if not summary_path.exists():
            result.tasks_missing.append(task_name)
            task_statuses[task_name] = ValidationStatus.FAIL
            continue
        
        result.tasks_found.append(task_name)
        
        try:
            summary_rows = load_csv_summary(summary_path)
            status, issues = check_task_quality(task_name, summary_rows)
            task_statuses[task_name] = status
            
            if issues:
                for issue in issues:
                    result.warnings.append(f"{task_name}: {issue}")
            
            # Compute task-level stats
            n_trials = len(summary_rows)
            n_valid = sum(1 for row in summary_rows if row.get('valid') == '1' or row.get('valid') == 1)
            result.qc_summary[f'{task_name}_n_trials'] = n_trials
            result.qc_summary[f'{task_name}_n_valid'] = n_valid
            result.qc_summary[f'{task_name}_valid_rate'] = n_valid / n_trials if n_trials > 0 else 0
            result.qc_summary[f'{task_name}_status'] = status.value
            
        except Exception as e:
            result.errors.append(f"Error reading {summary_file}: {e}")
            task_statuses[task_name] = ValidationStatus.FAIL
    
    # Determine overall status
    n_tasks_found = len(result.tasks_found)
    n_tasks_pass = sum(1 for s in task_statuses.values() if s == ValidationStatus.PASS)
    n_tasks_warn = sum(1 for s in task_statuses.values() if s == ValidationStatus.WARN)
    n_tasks_fail = sum(1 for s in task_statuses.values() if s == ValidationStatus.FAIL)
    
    result.qc_summary['tasks_found'] = n_tasks_found
    result.qc_summary['tasks_pass'] = n_tasks_pass
    result.qc_summary['tasks_warn'] = n_tasks_warn
    result.qc_summary['tasks_fail'] = n_tasks_fail
    
    min_tasks = QC_THRESHOLDS['min_tasks_strict'] if strict else QC_THRESHOLDS['min_tasks_required']
    
    if n_tasks_found < min_tasks:
        result.errors.append(f"Insufficient tasks: {n_tasks_found}/{min_tasks} required")
        result.status = ValidationStatus.FAIL
    elif result.errors:
        result.status = ValidationStatus.FAIL
    elif n_tasks_fail > 0 or result.warnings:
        result.status = ValidationStatus.WARN
    else:
        result.status = ValidationStatus.PASS
    
    return result


def print_result(result: ValidationResult, verbose: bool = False):
    """Print validation result."""
    status_colors = {
        ValidationStatus.PASS: '\033[92m',  # Green
        ValidationStatus.WARN: '\033[93m',  # Yellow
        ValidationStatus.FAIL: '\033[91m',  # Red
    }
    reset = '\033[0m'
    
    color = status_colors.get(result.status, '')
    print(f"\n{'='*60}")
    print(f"VALIDATION STATUS: {color}{result.status.value}{reset}")
    print(f"{'='*60}")
    
    if result.info:
        print("\nInfo:")
        for info in result.info:
            print(f"  {info}")
    
    print(f"\nTasks found: {len(result.tasks_found)}/{len(REQUIRED_TASKS)}")
    if result.tasks_found:
        print(f"  Found: {', '.join(result.tasks_found)}")
    if result.tasks_missing:
        print(f"  Missing: {', '.join(result.tasks_missing)}")
    
    if result.errors:
        print(f"\n{status_colors[ValidationStatus.FAIL]}Errors:{reset}")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\n{status_colors[ValidationStatus.WARN]}Warnings:{reset}")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if verbose and result.qc_summary:
        print("\nQC Summary:")
        for key, value in result.qc_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate NeuroLens+ session for ML readiness"
    )
    
    parser.add_argument(
        '--session', '-s',
        type=str,
        required=True,
        help='Path to session directory'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Require all tasks to be completed'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed QC summary'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output result as JSON'
    )
    
    args = parser.parse_args()
    
    result = validate_session(args.session, args.strict)
    
    if args.json:
        output = {
            'status': result.status.value,
            'errors': result.errors,
            'warnings': result.warnings,
            'info': result.info,
            'tasks_found': result.tasks_found,
            'tasks_missing': result.tasks_missing,
            'qc_summary': result.qc_summary,
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result, args.verbose)
    
    # Exit with appropriate code
    if result.status == ValidationStatus.FAIL:
        sys.exit(1)
    elif result.status == ValidationStatus.WARN:
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
