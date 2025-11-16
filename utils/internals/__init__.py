"""
utils.internals - Clean API for model activation extraction and analysis.

This package provides a structured interface for:
- Loading and managing language models
- Formatting conversations and extracting token indices
- Extracting hidden state activations
- Batch processing conversations efficiently
- Analyzing activation patterns (contrast vectors, projections)
"""

from .exceptions import StopForward
from .model import ProbingModel
from .conversation import ConversationEncoder
from .activations import ActivationExtractor
from .spans import SpanMapper
from .analysis import ActivationAnalyzer
from .batch import BatchProcessor, BatchProcessorConfig, process_batch_conversations

__all__ = [
    "StopForward",
    "ProbingModel",
    "ConversationEncoder",
    "ActivationExtractor",
    "SpanMapper",
    "ActivationAnalyzer",
    "BatchProcessor",
    "BatchProcessorConfig",
    "process_batch_conversations",
]

__version__ = "1.0.0"
