"""BatchProcessor - High-level orchestration for batch conversation processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ProbingModel

from .conversation import ConversationEncoder
from .activations import ActivationExtractor
from .spans import SpanMapper


Conversation = List[Dict[str, str]]
BatchConversations = List[Conversation]


@dataclass
class BatchProcessorConfig:
    """
    Configuration for batch processing of conversations.

    This is intentionally small; add fields here instead of
    growing the method signatures.
    """
    exclude_code: bool = False
    max_length: int = 4096
    layer: Optional[Union[int, List[int]]] = None  # None = all layers / default behavior


class BatchProcessor:
    """
    High-level orchestration over:
      1) building token-level spans for turns
      2) extracting activations for a batch of conversations
      3) mapping spans to turn-level activations
    """

    def __init__(
        self,
        probing_model: 'ProbingModel',
        config: Optional[BatchProcessorConfig] = None,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            probing_model: ProbingModel instance with loaded model and tokenizer
            config: Optional configuration (uses defaults if not provided)
        """
        self.model = probing_model
        self.config = config or BatchProcessorConfig()

        # Core components
        self.encoder = ConversationEncoder(
            probing_model.tokenizer,
            probing_model.model_name,
        )
        self.extractor = ActivationExtractor(
            probing_model,
            self.encoder,
        )
        self.mapper = SpanMapper(probing_model.tokenizer)

    def process(
        self,
        conversations: BatchConversations,
        **chat_kwargs: Any,
    ) -> List:
        """
        Process a batch of conversations, returning mapped activations.

        Args:
            conversations: List of conversations, each being a list of {"role", "content"} dicts
            **chat_kwargs: Additional arguments for apply_chat_template

        Returns:
            List of per-conversation per-turn activations, each with shape (num_turns, num_layers, hidden_size)
        """

        # 1) Build spans and metadata
        batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
            conversations,
            **chat_kwargs,
        )

        # 2) Extract activations
        batch_activations, batch_metadata = self.extractor.batch_conversations(
            conversations,
            layer=self.config.layer,
            max_length=self.config.max_length,
            **chat_kwargs,
        )

        # 3) Map spans to activations
        if self.config.exclude_code:
            mapped = self.mapper.map_spans_no_code(
                batch_activations,
                batch_spans,
                batch_metadata,
            )
        else:
            mapped = self.mapper.map_spans(
                batch_activations,
                batch_spans,
                batch_metadata,
            )

        return mapped


def process_batch_conversations(
    probing_model: 'ProbingModel',
    conversations: BatchConversations,
    exclude_code: bool = False,
    max_length: int = 4096,
    layer: Optional[Union[int, List[int]]] = None,
    **chat_kwargs: Any,
) -> List:
    """
    Convenience wrapper around BatchProcessor.

    This function mirrors the old probing_utils.process_batch_conversations API
    as closely as possible for easy migration.

    Args:
        probing_model: ProbingModel instance
        conversations: List of conversations to process
        exclude_code: Whether to exclude code blocks from averaging
        max_length: Maximum sequence length
        layer: Optional layer specification
        **chat_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of per-conversation per-turn activations
    """
    config = BatchProcessorConfig(
        exclude_code=exclude_code,
        max_length=max_length,
        layer=layer,
    )
    processor = BatchProcessor(probing_model, config=config)
    return processor.process(conversations, **chat_kwargs)
