"""ActivationAnalyzer - Pure functions for analyzing activation patterns."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class ActivationAnalyzer:
    """
    Static methods for analyzing activation patterns.

    Provides:
    - Contrast vector computation
    - Projection onto direction vectors
    - Mean response activations
    """

    @staticmethod
    def contrast(
        positive_activations: Union[torch.Tensor, Dict[int, torch.Tensor]],
        negative_activations: Union[torch.Tensor, Dict[int, torch.Tensor]],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Compute contrast vector: positive_mean - negative_mean

        Args:
            positive_activations: torch.Tensor or dict {layer_idx: torch.Tensor}
            negative_activations: torch.Tensor or dict {layer_idx: torch.Tensor}

        Returns:
            If inputs are tensors: (contrast_vector, positive_mean, negative_mean)
            If inputs are dicts: dict {layer_idx: (contrast_vector, positive_mean, negative_mean)}
        """
        # Check if inputs are dictionaries (multi-layer mode)
        if isinstance(positive_activations, dict) and isinstance(negative_activations, dict):
            # Multi-layer mode
            results = {}

            # Get all layers (should be the same in both dictionaries)
            layers = set(positive_activations.keys()) & set(negative_activations.keys())

            for layer_idx in layers:
                print(f"Layer {layer_idx} has {positive_activations[layer_idx].shape} and {negative_activations[layer_idx].shape}")
                if positive_activations[layer_idx] is not None and negative_activations[layer_idx] is not None:
                    positive_mean = positive_activations[layer_idx].mean(dim=0)
                    negative_mean = negative_activations[layer_idx].mean(dim=0)
                    contrast_vector = positive_mean - negative_mean
                    results[layer_idx] = (contrast_vector, positive_mean, negative_mean)
                else:
                    results[layer_idx] = None

            return results

        else:
            # Single layer mode - maintain original behavior
            positive_mean = positive_activations.mean(dim=0)
            negative_mean = negative_activations.mean(dim=0)
            contrast_vector = positive_mean - negative_mean
            return contrast_vector, positive_mean, negative_mean

    @staticmethod
    def project(
        activations: torch.Tensor,
        contrast_vector: torch.Tensor,
    ) -> Union[float, np.ndarray]:
        """
        Project activations onto contrast vector.

        Args:
            activations: torch.Tensor - either single activation (1D) or batch of activations (2D)
            contrast_vector: torch.Tensor - contrast vector to project onto

        Returns:
            float (if single activation) or np.array (if batch of activations)
        """
        # Handle single activation case
        if activations.ndim == 1:
            # Ensure tensors are on same device and dtype
            activations = activations.to(contrast_vector.device).float()
            contrast_vector = contrast_vector.float()

            # Scalar projection: (h · v) / ||v||
            contrast_norm = torch.norm(contrast_vector)
            if contrast_norm == 0:
                return 0.0

            projection = torch.dot(activations, contrast_vector) / contrast_norm
            return projection.item()

        # Handle batch case
        else:
            # Normalize contrast vector
            contrast_norm = torch.norm(contrast_vector)
            if contrast_norm == 0:
                return torch.zeros(activations.shape[0])

            # Project each activation
            projections = []
            for activation in activations:
                projection = torch.dot(activation, contrast_vector) / contrast_norm
                projections.append(projection.item())

            return np.array(projections)

    @staticmethod
    def mean_response(
        activations: torch.Tensor,
        conversation: List[Dict[str, str]],
        tokenizer,
        model_name: Optional[str] = None,
        **chat_kwargs,
    ) -> torch.Tensor:
        """
        Get the mean activation of the model's response to the user's message.

        Args:
            activations: torch.Tensor with shape (layers, tokens, features)
            conversation: List of dict with 'role' and 'content' keys
            tokenizer: Tokenizer to apply chat template and tokenize
            model_name: Model name to determine which extraction method to use
            **chat_kwargs: additional arguments for apply_chat_template

        Returns:
            Mean activation tensor over all response tokens
        """
        # Import here to avoid circular dependency
        from .conversation import ConversationEncoder

        # Create encoder to get response indices
        encoder = ConversationEncoder(tokenizer, model_name)
        response_indices = encoder.response_indices(conversation, per_turn=False, **chat_kwargs)

        # Get the mean activation of the model's response to the user's message
        mean_activation = activations[:, response_indices, :].mean(dim=1)
        return mean_activation

    @staticmethod
    def mean_response_per_turn(
        activations: torch.Tensor,
        conversation: List[Dict[str, str]],
        tokenizer,
        model_name: Optional[str] = None,
        **chat_kwargs,
    ) -> List[torch.Tensor]:
        """
        Get the mean activation for each of the model's response turns.

        Args:
            activations: Tensor with shape (layers, tokens, features)
            conversation: List of dict with 'role' and 'content' keys
            tokenizer: Tokenizer to apply chat template and tokenize
            model_name: Model name to determine which extraction method to use
            **chat_kwargs: additional arguments for apply_chat_template

        Returns:
            List[torch.Tensor]: List of mean activations, one per assistant turn
        """
        # Import here to avoid circular dependency
        from .conversation import ConversationEncoder

        # Create encoder to get response indices per turn
        encoder = ConversationEncoder(tokenizer, model_name)
        response_indices_per_turn = encoder.response_indices(conversation, per_turn=True, **chat_kwargs)

        # Calculate mean activation for each turn
        mean_activations_per_turn = []

        for turn_indices in response_indices_per_turn:
            if len(turn_indices) > 0:
                # Get mean activation for this turn's tokens
                turn_mean_activation = activations[:, turn_indices, :].mean(dim=1)
                mean_activations_per_turn.append(turn_mean_activation)

        return mean_activations_per_turn
