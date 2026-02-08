"""
Benchmarks module for ConforMix.

This module provides an automated benchmarking framework for evaluating
ConforMix on standard protein conformational datasets.
"""

from .run_benchmark import run_benchmark, BenchmarkConfig
from .evaluate_metrics import compute_metrics, MetricsResult
from .generate_report import generate_report

__version__ = "0.1.0"
__all__ = [
    "run_benchmark",
    "BenchmarkConfig",
    "compute_metrics",
    "MetricsResult",
    "generate_report",
]
