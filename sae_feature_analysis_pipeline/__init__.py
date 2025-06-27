"""
SAE Feature Analysis Pipeline

A clean, modular pipeline for analyzing misalignment-related features in 
Sparse Autoencoders (SAEs) by comparing pre- and post-SFT models.
"""

from .pipeline import run_misalignment_pipeline
from .autointerp import analyze_features_with_claude
from .steering_eval import evaluate_feature_steering, analyze_steering_results
from .constants import DEFAULT_CONFIG, MC_QUESTIONS, MC_OPTIONS, DEFAULT_FEATURE_MINING_PATH_TEMPLATE, DEFAULT_CLAUDE_MODEL

__all__ = [
    "run_misalignment_pipeline",
    "analyze_features_with_claude", 
    "evaluate_feature_steering",
    "analyze_steering_results",
    "DEFAULT_CONFIG"
] 