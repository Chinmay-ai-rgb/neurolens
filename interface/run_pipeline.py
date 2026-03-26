# interface/run_pipeline.py
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]

def _run(cmd: List[str]) -> Tuple[int, str]:
    """Run a command and return (code, combined_output)."""
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout

def run_analysis_pipeline() -> Tuple[bool, str]:
    """
    Runs the full analysis pipeline after tasks are completed.
    Returns (ok, log_text).
    """
    steps = [
        [sys.executable, "analysis/build_neurosignatures.py"],
        [sys.executable, "analysis/build_ML_matrix.py"],
        [sys.executable, "analysis/train_iforest.py"],
        [sys.executable, "analysis/compute_profiles.py"],
    ]

    logs = []
    for cmd in steps:
        code, out = _run(cmd)
        logs.append(f"$ {' '.join(cmd)}\n{out}")
        if code != 0:
            return False, "\n\n".join(logs)

    return True, "\n\n".join(logs)
