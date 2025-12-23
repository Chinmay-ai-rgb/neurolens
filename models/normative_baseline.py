#!/usr/bin/env python3
"""
NeuroLens+ Normative Baseline

Computes z-scores for biomarkers based on normative data.
Enables immediate results scoring without labeled training data.

Usage:
    from models.normative_baseline import NormativeBaseline
    
    baseline = NormativeBaseline()
    z_scores = baseline.compute_z_scores(session_biomarkers)
    flags = baseline.flag_outliers(session_biomarkers, threshold=2.0)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ZScoreResult:
    """Result of z-score computation for a single biomarker."""
    biomarker: str
    value: float
    z_score: float
    mean: float
    std: float
    direction: str
    is_outlier: bool
    severity: str  # 'normal', 'mild', 'moderate', 'severe'


class NormativeBaseline:
    """
    Normative baseline for computing z-scores and flagging outliers.
    
    Uses population norms from data/norms.json to compute standardized
    scores for each biomarker. This enables immediate interpretation
    without requiring labeled training data.
    """
    
    def __init__(self, norms_path: Optional[str] = None):
        """
        Initialize normative baseline.
        
        Args:
            norms_path: Path to norms.json file. If None, uses default location.
        """
        if norms_path is None:
            # Default to schemas/norms.json relative to project root
            project_root = Path(__file__).parent.parent
            norms_path = project_root / 'schemas' / 'norms.json'
        
        self.norms_path = Path(norms_path)
        self.norms = self._load_norms()
    
    def _load_norms(self) -> Dict[str, Any]:
        """Load normative data from JSON file."""
        if not self.norms_path.exists():
            raise FileNotFoundError(f"Norms file not found: {self.norms_path}")
        
        with open(self.norms_path, 'r') as f:
            return json.load(f)
    
    def get_norm(self, task: str, biomarker: str) -> Optional[Dict[str, Any]]:
        """
        Get normative data for a specific biomarker.
        
        Args:
            task: Task name (e.g., 'fixation', 'saccade')
            biomarker: Biomarker name (e.g., 'latency_ms', 'gain')
        
        Returns:
            Dictionary with mean, std, unit, direction, etc. or None if not found.
        """
        task_norms = self.norms.get(task, {})
        return task_norms.get(biomarker)
    
    def compute_z_score(
        self,
        value: float,
        mean: float,
        std: float
    ) -> float:
        """
        Compute z-score for a value.
        
        Args:
            value: Observed value
            mean: Population mean
            std: Population standard deviation
        
        Returns:
            Z-score (number of standard deviations from mean)
        """
        if std <= 0:
            return 0.0
        return (value - mean) / std
    
    def classify_severity(
        self,
        z_score: float,
        direction: str
    ) -> str:
        """
        Classify severity based on z-score and direction.
        
        Args:
            z_score: Computed z-score
            direction: 'lower_is_better', 'higher_is_better', 'closer_to_1_is_better', 'neutral'
        
        Returns:
            Severity level: 'normal', 'mild', 'moderate', 'severe'
        """
        abs_z = abs(z_score)
        
        # For neutral direction, both extremes are equally concerning
        if direction == 'neutral':
            if abs_z < 1.5:
                return 'normal'
            elif abs_z < 2.0:
                return 'mild'
            elif abs_z < 3.0:
                return 'moderate'
            else:
                return 'severe'
        
        # For directional biomarkers, only one direction is concerning
        if direction == 'lower_is_better':
            # High values are bad
            if z_score < 1.5:
                return 'normal'
            elif z_score < 2.0:
                return 'mild'
            elif z_score < 3.0:
                return 'moderate'
            else:
                return 'severe'
        
        elif direction == 'higher_is_better':
            # Low values are bad
            if z_score > -1.5:
                return 'normal'
            elif z_score > -2.0:
                return 'mild'
            elif z_score > -3.0:
                return 'moderate'
            else:
                return 'severe'
        
        elif direction == 'closer_to_1_is_better':
            # Distance from 1.0 matters
            if abs_z < 1.5:
                return 'normal'
            elif abs_z < 2.0:
                return 'mild'
            elif abs_z < 3.0:
                return 'moderate'
            else:
                return 'severe'
        
        elif direction == 'closer_to_0_is_better':
            # Distance from 0 matters
            if abs_z < 1.5:
                return 'normal'
            elif abs_z < 2.0:
                return 'mild'
            elif abs_z < 3.0:
                return 'moderate'
            else:
                return 'severe'
        
        return 'normal'
    
    def compute_biomarker_z_score(
        self,
        task: str,
        biomarker: str,
        value: float,
        outlier_threshold: float = 2.0
    ) -> Optional[ZScoreResult]:
        """
        Compute z-score for a single biomarker.
        
        Args:
            task: Task name
            biomarker: Biomarker name
            value: Observed value
            outlier_threshold: Z-score threshold for outlier detection
        
        Returns:
            ZScoreResult or None if norm not found
        """
        if np.isnan(value):
            return None
        
        norm = self.get_norm(task, biomarker)
        if norm is None:
            return None
        
        mean = norm.get('mean', 0)
        std = norm.get('std', 1)
        direction = norm.get('direction', 'neutral')
        
        z_score = self.compute_z_score(value, mean, std)
        severity = self.classify_severity(z_score, direction)
        is_outlier = abs(z_score) >= outlier_threshold
        
        return ZScoreResult(
            biomarker=f"{task}_{biomarker}",
            value=value,
            z_score=z_score,
            mean=mean,
            std=std,
            direction=direction,
            is_outlier=is_outlier,
            severity=severity
        )
    
    def compute_z_scores(
        self,
        session_biomarkers: Dict[str, Any],
        outlier_threshold: float = 2.0
    ) -> Dict[str, ZScoreResult]:
        """
        Compute z-scores for all biomarkers in a session.
        
        Args:
            session_biomarkers: Dictionary of biomarker values
                Keys should be in format 'task_biomarker' (e.g., 'saccade_latency_ms')
            outlier_threshold: Z-score threshold for outlier detection
        
        Returns:
            Dictionary mapping biomarker names to ZScoreResults
        """
        results = {}
        
        for key, value in session_biomarkers.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Parse task and biomarker from key
            parts = key.split('_', 1)
            if len(parts) != 2:
                continue
            
            task, biomarker = parts
            
            # Skip internal fields
            if biomarker.startswith('_'):
                continue
            
            result = self.compute_biomarker_z_score(
                task, biomarker, value, outlier_threshold
            )
            
            if result is not None:
                results[key] = result
        
        return results
    
    def flag_outliers(
        self,
        session_biomarkers: Dict[str, Any],
        threshold: float = 2.0
    ) -> List[ZScoreResult]:
        """
        Flag biomarkers that are outliers.
        
        Args:
            session_biomarkers: Dictionary of biomarker values
            threshold: Z-score threshold for outlier detection
        
        Returns:
            List of ZScoreResults for outliers only
        """
        z_scores = self.compute_z_scores(session_biomarkers, threshold)
        return [r for r in z_scores.values() if r.is_outlier]
    
    def compute_composite_score(
        self,
        session_biomarkers: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite health score from z-scores.
        
        Args:
            session_biomarkers: Dictionary of biomarker values
            weights: Optional weights for each biomarker (default: equal weights)
        
        Returns:
            Tuple of (composite_score, per_task_scores)
        """
        z_scores = self.compute_z_scores(session_biomarkers)
        
        if not z_scores:
            return 0.0, {}
        
        # Group by task
        task_scores = {}
        for key, result in z_scores.items():
            task = key.split('_')[0]
            if task not in task_scores:
                task_scores[task] = []
            
            # Convert z-score to 0-1 score (higher is better)
            # Use sigmoid-like transformation
            score = 1.0 / (1.0 + np.exp(abs(result.z_score) - 2))
            task_scores[task].append(score)
        
        # Average per task
        per_task_scores = {
            task: np.mean(scores) for task, scores in task_scores.items()
        }
        
        # Weighted average across tasks
        if weights:
            weighted_sum = sum(
                per_task_scores.get(task, 0) * weight
                for task, weight in weights.items()
            )
            total_weight = sum(weights.values())
            composite = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            composite = np.mean(list(per_task_scores.values()))
        
        return composite, per_task_scores
    
    def generate_report(
        self,
        session_biomarkers: Dict[str, Any],
        outlier_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive normative report for a session.
        
        Args:
            session_biomarkers: Dictionary of biomarker values
            outlier_threshold: Z-score threshold for outlier detection
        
        Returns:
            Report dictionary with z-scores, outliers, and composite scores
        """
        z_scores = self.compute_z_scores(session_biomarkers, outlier_threshold)
        outliers = [r for r in z_scores.values() if r.is_outlier]
        composite, per_task = self.compute_composite_score(session_biomarkers)
        
        # Count severities
        severity_counts = {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 0}
        for result in z_scores.values():
            severity_counts[result.severity] += 1
        
        return {
            'composite_score': round(composite, 3),
            'per_task_scores': {k: round(v, 3) for k, v in per_task.items()},
            'n_biomarkers': len(z_scores),
            'n_outliers': len(outliers),
            'severity_counts': severity_counts,
            'outliers': [
                {
                    'biomarker': r.biomarker,
                    'value': r.value,
                    'z_score': round(r.z_score, 2),
                    'severity': r.severity,
                    'direction': r.direction,
                }
                for r in outliers
            ],
            'z_scores': {
                k: {
                    'value': r.value,
                    'z_score': round(r.z_score, 2),
                    'severity': r.severity,
                }
                for k, r in z_scores.items()
            }
        }


def main():
    """Demo usage of normative baseline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute z-scores for NeuroLens+ biomarkers"
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with sample data'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Sample session biomarkers
        sample_biomarkers = {
            'fixation_stability_rms_px': 120.0,
            'fixation_microsaccade_rate_per_min': 45.0,
            'saccade_latency_ms': 280.0,
            'saccade_gain': 0.85,
            'antisaccade_direction_error_rate': 0.35,
            'pursuit_gain': 0.75,
            'visual_search_blink_rate_per_min': 22.0,
        }
        
        baseline = NormativeBaseline()
        report = baseline.generate_report(sample_biomarkers)
        
        print("Normative Baseline Report")
        print("=" * 50)
        print(f"Composite Score: {report['composite_score']:.3f}")
        print(f"Biomarkers Analyzed: {report['n_biomarkers']}")
        print(f"Outliers Detected: {report['n_outliers']}")
        print()
        
        print("Per-Task Scores:")
        for task, score in report['per_task_scores'].items():
            print(f"  {task}: {score:.3f}")
        print()
        
        print("Severity Distribution:")
        for severity, count in report['severity_counts'].items():
            print(f"  {severity}: {count}")
        print()
        
        if report['outliers']:
            print("Outliers:")
            for outlier in report['outliers']:
                print(f"  {outlier['biomarker']}: z={outlier['z_score']:.2f} ({outlier['severity']})")
        else:
            print("No outliers detected.")


if __name__ == '__main__':
    main()
