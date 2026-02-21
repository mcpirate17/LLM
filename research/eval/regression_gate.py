"""
Benchmark Regression Gate

Ensures that new architectures do not regress on fundamental capabilities
compared to historical baselines or the standard Transformer reference.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from .diagnostic_tasks import DiagnosticSuiteResult

logger = logging.getLogger(__name__)

class RegressionGate:
    """Gates candidate promotion based on benchmark performance."""
    
    def __init__(self, 
                 min_score_delta: float = -0.05, 
                 absolute_threshold: float = 0.3):
        self.min_score_delta = min_score_delta
        self.absolute_threshold = absolute_threshold

    def check(self, 
              current_result: DiagnosticSuiteResult, 
              baseline_result: Optional[DiagnosticSuiteResult] = None) -> Dict[str, Any]:
        """
        Check if current result passes the gate.
        Returns a verdict and explanation.
        """
        curr_score = current_result.diagnostic_score
        
        # 1. Absolute floor check
        if curr_score < self.absolute_threshold:
            return {
                "pass": False,
                "reason": f"Diagnostic score {curr_score:.3f} below absolute floor {self.absolute_threshold}",
                "delta": 0.0
            }
            
        # 2. Regression check against baseline
        if baseline_result:
            base_score = baseline_result.diagnostic_score
            delta = curr_score - base_score
            if delta < self.min_score_delta:
                return {
                    "pass": False,
                    "reason": f"Regression detected: delta {delta:.3f} exceeds limit {self.min_score_delta} (base={base_score:.3f})",
                    "delta": delta
                }
            return {
                "pass": True,
                "reason": f"Gate passed with score {curr_score:.3f} (delta {delta:+.3f} vs baseline)",
                "delta": delta
            }
            
        return {
            "pass": True,
            "reason": f"Gate passed with score {curr_score:.3f} (no baseline provided)",
            "delta": 0.0
        }
